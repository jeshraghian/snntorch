import math

import torch
from torch import nn
from torch.autograd import Function

from snntorch._neurons.neurons import SpikingNeuron


def __validate_inputs(
    d_value,
    d_key,
    num_spiking_neurons,
    input_topk,
    key_topk,
    input_topk_tau,
    key_topk_tau,
):
    # dims are positive integers
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

    # topk is in [1, dim]
    if input_topk is not None:
        if input_topk <= 0 or input_topk > in_dim:
            raise ValueError(
                f"input_topk must be in [1, in_dim={in_dim}] when provided; got {input_topk}"
            )
    if key_topk is not None:
        if key_topk <= 0 or key_topk > d_key:
            raise ValueError(
                f"key_topk must be in [1, d_key={d_key}] when provided; got {key_topk}"
            )

    # tau is positive
    if input_topk_tau <= 0.0:
        raise ValueError("input_topk_tau must be > 0")
    if key_topk_tau <= 0.0:
        raise ValueError("key_topk_tau must be > 0")


class Gen2SingleInputReadout(SpikingNeuron):
    def __init__(
        self,
        in_dim,
        d_value,
        d_key,
        num_spiking_neurons=None,
        use_q_projection: bool = True,
        input_topk: int | None = None,
        key_topk: int | None = None,
        input_topk_tau: float = 1.0,
        key_topk_tau: float = 1.0,
    ):
        """
        Base initializer where you specify d_value and d_key directly.

        The spirit of this implementation is to implement an associative-memory
        based SSM-based SNN, where the projection after the matrix-value hidden
        state (S_t), projects to the same dimensionality as (S_t).

        Args:
            in_dim:               input feature dimension
            d_value (int):        d (rows of S_t, also v_t dim)
            d_key (int):          n (cols of S_t, also k_t/alpha_t dim)
            num_spiking_neurons:  total neurons; if None, set to d_value * d_key.
                                   If provided, must equal d_value * d_key.
            use_q_projection:     if True, use S_t @ Q_t readout;
                                   if False, return flattened S_t (no q)
            input_topk:           if set, keep only top-k entries of input x along
                                   its feature dimension per (t, b). Others are zeroed.
                                   Must satisfy 1 <= input_topk < in_dim.
            key_topk:             if set, keep only top-k entries of k_t along the
                                   key dimension per (t, b). Others are zeroed.
                                   Must satisfy 1 <= key_topk < d_key.
            input_topk_tau:       temperature (>0) for the input soft surrogate
                                   used in training for straight-through estimation.
            key_topk_tau:         temperature (>0) for the k soft surrogate used in
                                   training for straight-through estimation.
        """
        super().__init__(output=True)

        __validate_inputs(
            d_value,
            d_key,
            num_spiking_neurons,
            input_topk,
            key_topk,
            input_topk_tau,
            key_topk_tau,
        )

        self.d_value = d_value  # d
        self.d_key = d_key  # n
        self.num_spiking_neurons = num_spiking_neurons
        self.use_q_projection = use_q_projection
        self.input_topk = input_topk
        self.key_topk = key_topk
        self.input_topk_tau = input_topk_tau
        self.key_topk_tau = key_topk_tau

        # Projections for the Gen-2 update
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
        input_topk: int | None = None,
        key_topk: int | None = None,
        input_topk_tau: float = 1.0,
        key_topk_tau: float = 1.0,
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
            input_topk:           if set, keep only top-k entries of input x along
                                   its feature dimension per (t, b).
            key_topk:             if set, keep only top-k entries of k_t along the
                                   key dimension per (t, b).
            input_topk_tau:       temperature (>0) for input soft surrogate in STE.
            key_topk_tau:         temperature (>0) for k soft surrogate in STE.
        """
        if num_spiking_neurons <= 0:
            raise ValueError("num_spiking_neurons must be positive")

        # Ensure it's a perfect square so we can set d = n = sqrt(N)
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
            input_topk=input_topk,
            key_topk=key_topk,
            input_topk_tau=input_topk_tau,
            key_topk_tau=key_topk_tau,
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

        # top k on input x
        if self.input_topk is not None:
            vals_x, idx_x = torch.topk(x, self.input_topk, dim=-1)
            x_hard = torch.zeros_like(x).scatter(-1, idx_x, vals_x)
            if (
                self.training and False
            ):  # TODO: once we finish benching topk, enable this
                # Straight-through: forward = hard, backward = soft surrogate
                m_soft = torch.softmax(x / self.input_topk_tau, dim=-1)
                x_soft = x * m_soft
                x = x_hard.detach() + x_soft - x_soft.detach()
            else:
                x = x_hard

        # projections
        v = self.to_v(x)  # (T,B,d)
        k = self.to_k(x)  # (T,B,n)
        alpha = torch.sigmoid(self.to_alpha(x))  # (T,B,n)

        # top k on key
        if self.key_topk is not None:
            vals, idx = torch.topk(k, self.key_topk, dim=-1)
            k_hard = torch.zeros_like(k).scatter(-1, idx, vals)
            if (
                self.training and False
            ):  # TODO: once we finish benching topk, enable this
                m_soft_k = torch.softmax(k / self.key_topk_tau, dim=-1)
                k_soft = k * m_soft_k
                k = k_hard.detach() + k_soft - k_soft.detach()
            else:
                k = k_hard

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
        spk_S = self.fire(mem_S) * self.graded_spikes_factor

        # readout
        if self.use_q_projection:
            q_matrix = q_flat.view(T, B, n, d)  # (T,B,n,d)
            # if spiking, apply q on spiked S
            if self.output:
                S_2d = spk_S.reshape(T * B, d, n)
            # if not spiking, apply q on membrane S
            else:
                S_2d = S_local.reshape(T * B, d, n)
            q_2d = q_matrix.reshape(T * B, n, d)
            Y_block = torch.bmm(S_2d, q_2d).reshape(T, B, d * d)  # (T,B,d*d)
        else:
            # if not using q projection, return flattened S_t
            # else return spiked S
            Y_block = spk_S if self.output else S_flat  # (T,B,d*n)

        y = Y_block  # (T,B,N_spike)
        return y


# TODO: remove this dunder
if __name__ == "__main__":
    T, B, in_dim = 16, 2, 32  # time, batch, input dim
    num_spiking_neurons = 16  # must be a perfect square -> d = n = 4

    model = Gen2SingleInputReadout.from_num_spiking_neurons(
        in_dim=in_dim,
        num_spiking_neurons=num_spiking_neurons,
        use_q_projection=True,  # or False to just flatten S_t
    )

    x = torch.randn(T, B, in_dim)
    y = model(x)

    print("x.shape:", x.shape)
    print(
        "y.shape:", y.shape
    )  # expect (T, B, num_spiking_neurons) = (16, 2, 16)
