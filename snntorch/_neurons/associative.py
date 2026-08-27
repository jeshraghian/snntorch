import math

import torch
from torch import nn

from snntorch._neurons.neurons import SpikingNeuron


def _validate_inputs(
    d_value,
    d_key,
    num_spiking_neurons,
    in_dim,
):
    # dims are positive integers
    if in_dim <= 0:
        raise ValueError("in_dim must be a positive integer")
    if d_value <= 0 or d_key <= 0:
        raise ValueError("d_value and d_key must be positive integers")

    # ensure num_spiking_neurons is consistent with d_value * d_key
    inferred = d_value * d_key
    if num_spiking_neurons is None:
        num_spiking_neurons = inferred
    elif num_spiking_neurons != inferred:
        raise ValueError(
            f"num_spiking_neurons={num_spiking_neurons} must equal d_value * d_key={inferred}"
        )

    return


class AssociativeLeaky(SpikingNeuron):
    def __init__(
        self,
        in_dim,
        d_value,
        d_key,
        num_spiking_neurons,
        use_q_projection: bool = True,
    ):
        """
        Base initializer where you specify d_value and d_key directly.

        This implements an associative-memory, SSM-based spiking model in which
        the projection computed from the matrix-valued hidden state S_t maps
        back into the same dimensionality as S_t.

        Unlike Leaky and StateLeaky, we do not return a (spike, membrane) tuple
        when output=True. AssociativeLeaky operates on the matrix state S_t and
        an optional readout projection; materializing and returning both
        representations is ambiguous (different spaces/shapes depending on
        use_q_projection) and unnecessarily expensive. Therefore, forward
        returns a single tensor in the chosen output space.

        Args:
            in_dim:               input feature dimension
            d_value (int):        d (rows of S_t, also v_t dim)
            d_key (int):          n (cols of S_t, also k_t/alpha_t dim)
            num_spiking_neurons:  total neurons; if None, set to d_value * d_key.
                                   If provided, must equal d_value * d_key.
            use_q_projection:     if True, use S_t @ Q_t readout;
                                   if False, return flattened S_t (no q)
        """
        _validate_inputs(
            d_value,
            d_key,
            num_spiking_neurons,
            in_dim,
        )

        super().__init__(output=True)

        self.d_value = d_value  # d
        self.d_key = d_key  # n
        self.num_spiking_neurons = num_spiking_neurons
        self.use_q_projection = use_q_projection

        self.to_v = nn.Linear(in_dim, d_value)  # (T,B,in_dim) -> (T,B,d)
        self.to_k = nn.Linear(in_dim, d_key)  # (T,B,in_dim) -> (T,B,n)
        self.to_alpha = nn.Linear(in_dim, d_key)  # (T,B,in_dim) -> (T,B,n)

        # q projection is optional
        if self.use_q_projection:
            # q projects into "neuron space": flattened (n × d) = num_spiking_neurons
            self.to_q = nn.Linear(
                in_dim, self.num_spiking_neurons
            )  # (T,B,in_dim)->(T,B,d*n)
        else:
            self.to_q = None  # not used

        self.eps = 1e-8

    @classmethod
    def from_num_spiking_neurons(
        cls,
        in_dim,
        num_spiking_neurons,
        use_q_projection: bool = True,
    ):
        """
        Convenience constructor:
        - takes num_spiking_neurons
        - enforces d = n = sqrt(num_spiking_neurons).

        Args:
            in_dim:               input feature dimension
            num_spiking_neurons:  total neurons = d * n, must be a perfect square
            use_q_projection:     if True, use S_t @ Q_t readout;
                                   if False, return flattened S_t
        """
        if num_spiking_neurons <= 0:
            raise ValueError("num_spiking_neurons must be positive")

        # ensure it's a perfect square so we can set d = n = sqrt(n)
        m = math.isqrt(num_spiking_neurons)
        if m * m != num_spiking_neurons:
            raise ValueError(
                f"num_spiking_neurons={num_spiking_neurons} must be a perfect square to set d = n = sqrt(N)"
            )

        d_value = m
        d_key = m
        return cls(
            in_dim=in_dim,
            d_value=d_value,
            d_key=d_key,
            num_spiking_neurons=num_spiking_neurons,
            use_q_projection=use_q_projection,
        )

    def forward(self, x):
        """
        x: (T, B, in_dim)

        Returns:
            y: (T, B, num_spiking_neurons)
               where num_spiking_neurons = d_value * d_key

            - If use_q_projection:
                y[t] is flattened (S_t @ Q_t) ∈ R^{d*d}
            - Else:
                y[t] is flattened S_t ∈ R^{d*n}
        """
        T, B, _ = x.shape
        d = self.d_value
        n = self.d_key

        # projections
        v = self.to_v(x)  # (T,B,d)
        k = self.to_k(x)  # (T,B,n)
        alpha = torch.sigmoid(self.to_alpha(x))  # (T,B,n)

        # optional q projection
        if self.use_q_projection:
            q_flat = self.to_q(x)  # (T,B,N_spike) = (T,B,d*n)
        else:
            q_flat = None

        # outer-product writes per step: v_t k_t^T -> (T,B,d,n)
        outer = torch.einsum("tbd,tbn->tbdn", v, k)  # (T,B,d,n)

        # decay weights via cumprod in log-space
        log_a = (alpha.clamp_min(self.eps)).log()  # (T,B,n)
        log_cumprod = torch.cumsum(log_a, dim=0)  # (T,B,n)
        cumprod = log_cumprod.exp()  # (T,B,n)

        # divide (per write r) by its own prefix p_r
        inv_prefix = 1.0 / (cumprod + 1e-8)  # (T,B,n)
        inv_prefix = inv_prefix.unsqueeze(2).expand(-1, -1, d, -1)  # (T,B,d,n)
        scaled_writes = outer * inv_prefix  # (T,B,d,n)

        # state accumulation
        S_scaled = torch.cumsum(scaled_writes, dim=0)  # (T,B,d,n)
        S_local = S_scaled * cumprod.unsqueeze(2).expand(
            -1, -1, d, -1
        )  # (T,B,d,n)

        # capture S-level mem/spk
        S_flat = S_local.reshape(T, B, d * n)  # (T,B,d*n)
        mem_S = self.state_quant(S_flat) if self.state_quant else S_flat
        if self.output:
            S = self.fire(mem_S)
        else:
            S = mem_S

        # readout
        if self.use_q_projection:
            q_matrix = q_flat.view(T, B, n, d)  # (T,B,n,d)
            S_2d = S.reshape(T * B, d, n)
            q_2d = q_matrix.reshape(T * B, n, d)
            Y_block = torch.bmm(S_2d, q_2d).reshape(T, B, d * d)  # (T,B,d*d)
        else:
            # if not using q projection, return flattened S_t
            # else return spiked S
            Y_block = S  # (T,B,d*n)

        y = Y_block  # (T,B,N_spike)
        return y
