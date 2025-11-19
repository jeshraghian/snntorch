import math
import torch
from torch import nn
from torch.autograd import Function

from snntorch._neurons.neurons import SpikingNeuron


class Gen2SingleInputReadout(SpikingNeuron):
    def __init__(
        self,
        in_dim,
        d_value,
        d_key,
        num_spiking_neurons=None,
        time_chunk_size=None,
        use_q_projection: bool = True,
        input_topk: int | None = None,
        key_topk: int | None = None,
    ):
        """
        Base initializer where you specify d_value and d_key directly.

        Args:
            in_dim:               input feature dimension
            d_value (int):        d (rows of S_t, also v_t dim)
            d_key (int):          n (cols of S_t, also k_t/alpha_t dim)
            num_spiking_neurons:  total neurons; if None, set to d_value * d_key.
                                   If provided, must equal d_value * d_key.
            time_chunk_size:      optional time chunk size (Tc); if None, use full T
            use_q_projection:     if True, use S_t @ Q_t readout;
                                   if False, return flattened S_t (no q)
            input_topk:           if set, keep only top-k entries of input x along
                                   its feature dimension per (t, b). Others are zeroed.
                                   Must satisfy 1 <= input_topk < in_dim.
            key_topk:             if set, keep only top-k entries of k_t along the
                                   key dimension per (t, b). Others are zeroed.
                                   Must satisfy 1 <= key_topk < d_key.
        """
        super().__init__()
        if d_value <= 0 or d_key <= 0:
            raise ValueError("d_value and d_key must be positive integers")

        inferred = d_value * d_key
        if num_spiking_neurons is None:
            num_spiking_neurons = inferred
        elif num_spiking_neurons != inferred:
            raise ValueError(
                f"num_spiking_neurons={num_spiking_neurons} must equal d_value * d_key={inferred}"
            )

        self.d_value = d_value  # d
        self.d_key = d_key  # n
        self.num_spiking_neurons = num_spiking_neurons
        self.use_q_projection = use_q_projection
        if input_topk is not None:
            if input_topk <= 0 or input_topk >= in_dim:
                raise ValueError(
                    f"input_topk must be in [1, in_dim={in_dim}-1] when provided; got {input_topk}"
                )
        if key_topk is not None:
            if key_topk <= 0 or key_topk >= d_key:
                raise ValueError(
                    f"key_topk must be in [1, d_key={d_key}-1] when provided; got {key_topk}"
                )
        self.input_topk = input_topk
        self.key_topk = key_topk

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

        self.time_chunk_size = time_chunk_size
        self.eps = 1e-8

    # -----------------------------------------------------------
    # Alternative constructor: from num_spiking_neurons
    # -----------------------------------------------------------
    @classmethod
    def from_num_spiking_neurons(
        cls,
        in_dim,
        num_spiking_neurons,
        time_chunk_size=None,
        use_q_projection: bool = True,
        input_topk: int | None = None,
        key_topk: int | None = None,
    ):
        """
        Convenience constructor:
        - takes num_spiking_neurons
        - enforces d = n = sqrt(num_spiking_neurons).

        Args:
            in_dim:               input feature dimension
            num_spiking_neurons:  total neurons = d * n, must be a perfect square
            time_chunk_size:      optional time chunk size
            use_q_projection:     if True, use S_t @ Q_t readout;
                                   if False, return flattened S_t
            input_topk:           if set, keep only top-k entries of input x along
                                   its feature dimension per (t, b).
            key_topk:             if set, keep only top-k entries of k_t along the
                                   key dimension per (t, b).
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
            time_chunk_size=time_chunk_size,
            use_q_projection=use_q_projection,
            input_topk=input_topk,
            key_topk=key_topk,
        )

    # -----------------------------------------------------------
    # Forward
    # -----------------------------------------------------------
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
        N_spike = self.num_spiking_neurons

        # Optional top-k masking on input x along feature dimension per (t, b)
        if self.input_topk is not None:
            vals_x, idx_x = torch.topk(x, self.input_topk, dim=-1)
            x_masked = torch.zeros_like(x)
            x = x_masked.scatter(-1, idx_x, vals_x)

        # Projections for Gen-2 update
        v = self.to_v(x)  # (T,B,d)
        k = self.to_k(x)  # (T,B,n)
        alpha = torch.sigmoid(self.to_alpha(x))  # (T,B,n)

        # Optional top-k masking on k along key dimension per (t, b)
        if self.key_topk is not None:
            vals, idx = torch.topk(k, self.key_topk, dim=-1)
            k_masked = torch.zeros_like(k)
            k = k_masked.scatter(-1, idx, vals)

        # Optional q projection
        if self.use_q_projection:
            q_flat = self.to_q(x)  # (T,B,N_spike) = (T,B,d*n)
        else:
            q_flat = None

        chunk = self.time_chunk_size or T
        y_chunks = []
        S_prev = None  # carry: (B,d,n)

        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            Tc = t1 - t0

            v_ch = v[t0:t1]  # (Tc,B,d)
            k_ch = k[t0:t1]  # (Tc,B,n)
            a_ch = alpha[t0:t1]  # (Tc,B,n)

            if self.use_q_projection:
                q_flat_ch = q_flat[t0:t1]  # (Tc,B,N_spike)

            # Outer-product writes per step: v_t k_t^T -> (Tc,B,d,n)
            outer = torch.einsum("tbd,tbn->tbdn", v_ch, k_ch)  # (Tc,B,d,n)

            # Decay weights inside the chunk via cumprod in log-space
            log_a = (a_ch.clamp_min(self.eps)).log()  # (Tc,B,n)
            log_cumprod = torch.cumsum(log_a, dim=0)  # (Tc,B,n)
            cumprod = log_cumprod.exp()  # (Tc,B,n) = p_t = ∏_{u<=t} a_u

            # Divide (per write r) by its own prefix p_r
            inv_prefix = 1.0 / (cumprod + 1e-8)  # (Tc,B,n)
            inv_prefix = inv_prefix.unsqueeze(2).expand(
                -1, -1, d, -1
            )  # (Tc,B,d,n)
            scaled_writes = outer * inv_prefix  # (Tc,B,d,n)

            # State accumulation
            S_scaled = torch.cumsum(scaled_writes, dim=0)  # (Tc,B,d,n)
            S_local = S_scaled * cumprod.unsqueeze(2).expand(
                -1, -1, d, -1
            )  # (Tc,B,d,n)

            # --- add carry from previous chunk BEFORE readout ---
            if S_prev is not None:
                # S_prev: (B,d,n), cumprod: (Tc,B,n)
                S_prev_time = S_prev.unsqueeze(0).expand(
                    Tc, -1, -1, -1
                )  # (Tc,B,d,n)
                decay_time = cumprod.unsqueeze(2).expand(
                    -1, -1, d, -1
                )  # (Tc,B,d,n)
                carry = S_prev_time * decay_time  # (Tc,B,d,n)
                S_local = S_local + carry

            # ------- Readout -------
            if self.use_q_projection:
                # Readout: S_t (d×n) @ Q_t (n×d) -> (d×d) -> flatten to N_spike

                # q_flat_ch: (Tc,B,N_spike) = (Tc,B,d*n) -> (Tc,B,n,d)
                q_matrix = q_flat_ch.view(Tc, B, n, d)  # (Tc,B,n,d)

                # Reshape for batched matmul: (Tc*B,d,n) @ (Tc*B,n,d) -> (Tc*B,d,d)
                S_2d = S_local.reshape(Tc * B, d, n)  # (Tc*B,d,n)
                q_2d = q_matrix.reshape(Tc * B, n, d)  # (Tc*B,n,d)

                Y_block = torch.bmm(S_2d, q_2d)  # (Tc*B,d,d)
                Y_block = Y_block.reshape(Tc, B, d * d)  # (Tc,B,N_spike)
            else:
                # Readout: just flatten S_t itself → (Tc,B,d*n)
                Y_block = S_local.reshape(Tc, B, d * n)  # (Tc,B,N_spike)

            y_chunks.append(Y_block)

            # Update carry to end-of-chunk state
            S_prev = S_local[-1]  # (B,d,n)

        y = torch.cat(y_chunks, dim=0)  # (T,B,N_spike)
        mem = y

        if self.state_quant:
            mem = self.state_quant(mem)

        if self.output:
            self.spk = self.fire(mem) * self.graded_spikes_factor
            return self.spk, mem

        else:
            return mem


if __name__ == "__main__":
    # Simple smoke test for from_num_spiking_neurons

    T, B, in_dim = 16, 2, 32  # time, batch, input dim
    num_spiking_neurons = 16  # must be a perfect square -> d = n = 4

    model = Gen2SingleInputReadout.from_num_spiking_neurons(
        in_dim=in_dim,
        num_spiking_neurons=num_spiking_neurons,
        time_chunk_size=None,  # or e.g. 8
        use_q_projection=True,  # or False to just flatten S_t
    )

    x = torch.randn(T, B, in_dim)
    y = model(x)

    print("x.shape:", x.shape)
    print(
        "y.shape:", y.shape
    )  # expect (T, B, num_spiking_neurons) = (16, 2, 16)
