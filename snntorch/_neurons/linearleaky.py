from torch import nn
from torch.nn import functional as F
from profilehooks import profile

from .stateleaky import StateLeaky


class LinearLeaky(StateLeaky):
    r"""
    Per-timestep linear projection followed by :class:`StateLeaky` dynamics.

    LinearLeaky applies ``nn.Linear(in_features, out_features)`` at each time
    step of the input sequence, then delegates state computation, spike
    generation, surrogate gradients, and ergonomics to :class:`StateLeaky`.

    Refer to :class:`StateLeaky` for the full description of the causal
    exponential decay kernel, :math:`\beta/\tau` semantics, spike thresholds,
    surrogate gradients, ``kernel_truncation_steps``, input/output conventions,
    and learnable parameters.

    Differences vs :class:`StateLeaky`:
    - Adds an internal linear layer; the processed channel count equals
      ``out_features``.
    - Behavior is equivalent to ``StateLeaky(..., channels=out_features)
      (Linear(x))`` when parameters match.
    - ``kernel_truncation_steps`` is forwarded to :class:`StateLeaky`.

    Minimal example::

        T, B, Fin, Fout = 16, 2, 8, 4
        x = torch.randn(T, B, Fin)
        lif = LinearLeaky(beta=0.9, in_features=Fin, out_features=Fout, output=True)
        spk, mem = lif(x)

    Parameters specific to LinearLeaky:
    - ``in_features`` (int): input feature dimension
    - ``out_features`` (int): output feature dimension (channels after linear)
    - ``bias``/``device``/``dtype``: forwarded to the internal ``nn.Linear``

    All other arguments are accepted and interpreted as in :class:`StateLeaky`.
    """

    def __init__(
        self,
        beta,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        learn_beta=False,
        learn_threshold=False,
        state_quant=False,
        output=True,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        kernel_truncation_steps=None,
    ):
        super().__init__(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            state_quant=state_quant,
            output=output,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
            channels=out_features,
            kernel_truncation_steps=kernel_truncation_steps,
        )

        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            device=device,
            dtype=dtype,
            bias=bias,
        )

    @property
    def beta(self):
        return (self.tau - 1) / self.tau

    def forward(self, input_):
        num_steps, batch, channels = input_.shape

        input_ = self.linear(input_.reshape(-1, self.linear.in_features))

        input_ = input_.reshape(num_steps, batch, self.linear.out_features)
        self.mem = self._base_state_function(input_)

        if self.state_quant:
            self.mem = self.state_quant(self.mem)

        if self.output:
            self.spk = self.fire(self.mem) * self.graded_spikes_factor
            return self.spk, self.mem

        else:
            return self.mem
