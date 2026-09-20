import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["SynapticDelay"]


class SynapticDelay(nn.Module):
    """
    Per-channel (axonal / synaptic) transmission delay for spike or current
    signals, with an optionally **learnable** delay time.

    A spike or current on channel :math:`c` is delayed by :math:`d_c \\ge 0`
    time steps before it is passed on. ``SynapticDelay`` is a plain
    ``nn.Module`` that is dropped into a network wherever a lag is wanted,
    typically between a connection and a neuron::

        self.fc1   = nn.Linear(num_in, num_hidden)
        self.delay = snn.SynapticDelay(max_delay=16, delay=1.0,
                                       channels=num_hidden, learn_delay=True)
        self.lif1  = snn.Leaky(beta=0.9)
        ...
        cur1 = self.delay(self.fc1(x))      # step mode: cur1, x are (batch, N)
        spk1, mem1 = self.lif1(cur1, mem1)

    The fractional delay is realised by interpolation, so :math:`d_c` is a
    continuous, trainable quantity. Two interpolation kernels are provided:

    * ``kernel="linear"`` -- a 2-tap kernel,
      :math:`w[\\lfloor d \\rfloor] = 1 - \\mathrm{frac}(d)`,
      :math:`w[\\lfloor d \\rfloor + 1] = \\mathrm{frac}(d)`. Exactly
      differentiable w.r.t. :math:`d`. This is the default.
    * ``kernel="gaussian"`` -- a normalised
      :math:`\\exp(-(\\tau - d)^2 / 2\\sigma^2)` kernel. Smoother gradients
      over a wide delay range; anneal ``sigma`` toward a small value during
      training (the DCLS trick, Hammouamri et al., ICLR 2024).

    **Two call modes, one kernel:**

    * **Step mode** (``step_mode=True``, the default) -- input
      ``(batch, channels)``, called once per time step (the standard
      snntorch loop). An internal ring buffer holds the last
      ``max_delay + 1`` inputs. Call :meth:`reset_delay` at the start of
      each sequence, exactly as you call ``Leaky.reset_mem()``.
    * **Sequence mode** (``step_mode=False``) -- input
      ``(time, batch, channels)`` (or ``(time, channels)``), the whole
      sequence at once, to pair with :class:`snntorch.LeakyParallel`.
      Implemented as a causal depthwise convolution along time; no state,
      no reset needed.

    Both modes use the same delay kernel and, for the same delay, produce
    the same output (up to the ring buffer's ramp-up at the start).

    The delay is clamped to ``[0, max_delay]`` on every forward pass, in the
    same spirit as ``beta`` being clamped to ``[0, 1]``.

    :param max_delay: maximum delay in time steps; sets the ring-buffer /
        convolution-kernel length. Must satisfy ``max_delay >= ceil(delay)``.
    :type max_delay: int

    :param delay: initial delay in time steps. Scalar, or a tensor of shape
        ``(channels,)`` for a per-channel initialisation. Defaults to 0.0
        (pass-through).
    :type delay: float or torch.Tensor, optional

    :param channels: number of channels ``C``. If given, the delay is a
        length-``C`` vector (one learnable delay per channel). If ``None``,
        a single delay value is shared by all channels. Defaults to None.
    :type channels: int, optional

    :param learn_delay: if ``True``, the delay is an ``nn.Parameter`` and is
        trained by backprop; otherwise it is a fixed buffer. Defaults to
        False.
    :type learn_delay: bool, optional

    :param kernel: ``"linear"`` (2-tap, default) or ``"gaussian"``.
    :type kernel: str, optional

    :param sigma: width of the Gaussian kernel (ignored for ``"linear"``).
        Defaults to 1.0.
    :type sigma: float, optional

    :param step_mode: if ``True`` (default) the module is called once per
        time step with ``(batch, channels)`` input and keeps a ring buffer.
        If ``False`` it takes a whole ``(time, batch, channels)`` sequence
        at once. Defaults to True.
    :type step_mode: bool, optional

    Inputs: input\\_
        - **input_** of shape ``(batch, channels)`` (step mode) or
          ``(time, batch, channels)`` / ``(time, channels)`` (sequence
          mode).

    Outputs: delayed
        - **delayed** -- same shape as ``input_``; each channel delayed by
          its own ``delay`` steps. The first ``ceil(delay)`` step-mode
          outputs after a :meth:`reset_delay` are ramp-up (buffer still
          filling), as with any real delay line.

    Learnable Parameters:
        - **SynapticDelay.delay** (torch.Tensor) -- the per-channel (or
          scalar) delay, present as a parameter when ``learn_delay=True``.
    """

    def __init__(
        self,
        max_delay,
        delay=0.0,
        channels=None,
        learn_delay=False,
        kernel="linear",
        sigma=1.0,
        step_mode=True,
    ):
        super().__init__()

        self.step_mode = bool(step_mode)

        if int(max_delay) != max_delay or max_delay < 1:
            raise ValueError("max_delay must be a positive integer.")
        self.max_delay = int(max_delay)

        if kernel not in ("linear", "gaussian"):
            raise ValueError("kernel must be 'linear' or 'gaussian'.")
        self.kernel = kernel
        self.sigma = float(sigma)
        self.channels = None if channels is None else int(channels)

        if not isinstance(delay, torch.Tensor):
            delay = torch.as_tensor(delay, dtype=torch.float)
        delay = delay.float()
        if self.channels is not None and delay.ndim == 0:
            delay = delay.expand(self.channels).clone()
        if delay.ndim == 0:
            pass
        elif self.channels is not None and delay.shape != (self.channels,):
            raise ValueError(
                f"delay tensor must have shape ({self.channels},), "
                f"got {tuple(delay.shape)}."
            )

        if float(delay.max()) > self.max_delay or float(delay.min()) < 0:
            raise ValueError(
                "every delay must lie in [0, max_delay] "
                f"(got range [{float(delay.min())}, {float(delay.max())}], "
                f"max_delay={self.max_delay})."
            )

        if learn_delay:
            self.delay = nn.Parameter(delay)
        else:
            self.register_buffer("delay", delay)
        self.learn_delay = learn_delay

        # ring buffer for step mode; lazily shaped on first step-mode call
        self.register_buffer("_buffer", torch.zeros(0), persistent=False)

    # ------------------------------------------------------------------ #
    #  delay kernel
    # ------------------------------------------------------------------ #
    def _lag_weights(self, n_channels, device, dtype):
        """Return the per-lag interpolation weights, shape (C, max_delay+1).
        ``w[c, l]`` multiplies the input from ``l`` steps ago for channel c.
        """
        d = self.delay.clamp(0.0, self.max_delay)
        if d.ndim == 0:
            d = d.expand(n_channels)
        elif d.shape[0] != n_channels:
            raise ValueError(
                f"SynapticDelay configured for {d.shape[0]} channels but got "
                f"input with {n_channels}."
            )
        d = d.to(device=device, dtype=dtype)

        lags = torch.arange(
            self.max_delay + 1, device=device, dtype=dtype
        ).unsqueeze(
            0
        )  # (1, K+1)
        diff = lags - d.unsqueeze(1)  # (C, K+1)

        if self.kernel == "linear":
            w = (1.0 - diff.abs()).clamp(min=0.0)
            # exact 2-tap; renormalise only the boundary case d==max_delay
            w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
        else:  # gaussian
            w = torch.exp(-(diff**2) / (2.0 * self.sigma**2))
            w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return w  # (C, K+1)

    # ------------------------------------------------------------------ #
    #  state (step mode)
    # ------------------------------------------------------------------ #
    def reset_delay(self):
        """Clear the step-mode ring buffer. Call once per sequence, like
        ``Leaky.reset_mem()``."""
        self._buffer = torch.zeros_like(self._buffer)
        return self._buffer

    # ------------------------------------------------------------------ #
    #  forward
    # ------------------------------------------------------------------ #
    def forward(self, input_):
        if self.step_mode:
            if input_.dim() != 2:
                raise ValueError(
                    "step_mode=True expects (batch, channels); got "
                    f"{tuple(input_.shape)}. Use step_mode=False for a "
                    "(time, batch, channels) sequence."
                )
            return self._forward_step(input_)
        if input_.dim() not in (2, 3):
            raise ValueError(
                "step_mode=False expects (time, batch, channels) or "
                f"(time, channels); got {tuple(input_.shape)}."
            )
        return self._forward_sequence(input_)

    def _forward_step(self, x):
        # x: (batch, channels)
        b, c = x.shape
        k = self.max_delay + 1
        if self._buffer.shape != (k, b, c):
            self._buffer = torch.zeros(k, b, c, device=x.device, dtype=x.dtype)
        # push current input at lag 0, drop the oldest
        self._buffer = torch.cat(
            (x.unsqueeze(0), self._buffer[:-1]), dim=0
        )  # (K+1, B, C)
        w = self._lag_weights(c, x.device, x.dtype)  # (C, K+1)
        w = w.transpose(0, 1).unsqueeze(1)  # (K+1, 1, C)
        return (self._buffer * w).sum(dim=0)  # (B, C)

    def _forward_sequence(self, x):
        squeeze_batch = x.dim() == 2
        if squeeze_batch:
            x = x.unsqueeze(1)  # (T, 1, C)
        t, b, c = x.shape
        w = self._lag_weights(c, x.device, x.dtype)  # (C, K+1)
        # causal cross-correlation: out[t] = sum_l x[t-l] * w[:, l]
        weight = w.flip(-1).unsqueeze(1)  # (C, 1, K+1)
        xin = x.permute(1, 2, 0)  # (B, C, T)
        xin = F.pad(xin, (self.max_delay, 0))  # left-pad K
        out = F.conv1d(xin, weight, groups=c)  # (B, C, T)
        out = out.permute(2, 0, 1)  # (T, B, C)
        return out.squeeze(1) if squeeze_batch else out

    def extra_repr(self):
        d = self.delay.detach()
        dstr = (
            f"{float(d):.3g}" if d.ndim == 0 else f"tensor[{tuple(d.shape)}]"
        )
        return (
            f"max_delay={self.max_delay}, delay={dstr}, "
            f"channels={self.channels}, learn_delay={self.learn_delay}, "
            f"kernel={self.kernel}"
            + (f", sigma={self.sigma}" if self.kernel == "gaussian" else "")
        )
