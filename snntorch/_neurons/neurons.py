import math
from warnings import warn
from snntorch.surrogate import atan
import torch
import torch.nn as nn


__all__ = [
    "SpikingNeuron",
    "LIF",
]

dtype = torch.float


class SpikingNeuron(nn.Module):
    """Parent class for spiking neuron models."""

    instances = []
    """Each :mod:`snntorch.SpikingNeuron` neuron
    (e.g., :mod:`snntorch.Synaptic`) will populate the
    :mod:`snntorch.SpikingNeuron.instances` list with a new entry.
    The list is used to initialize and clear neuron states when the
    argument `init_hidden=True`."""

    reset_dict = {
        "subtract": 0,
        "zero": 1,
        "none": 2,
    }

    def __init__(
        self,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
    ):
        super().__init__()

        SpikingNeuron.instances.append(self)

        if surrogate_disable:
            self.spike_grad = self._surrogate_bypass
        elif spike_grad == None:
            self.spike_grad = atan()
        else:
            self.spike_grad = spike_grad

        self.init_hidden = init_hidden
        self.inhibition = inhibition
        self.output = output
        self.surrogate_disable = surrogate_disable

        self._snn_cases(reset_mechanism, inhibition)
        self._snn_register_buffer(
            threshold=threshold,
            learn_threshold=learn_threshold,
            reset_mechanism=reset_mechanism,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
        )
        self._reset_mechanism = reset_mechanism

        self.state_quant = state_quant

        # Homeostatic (adaptive, time-varying) threshold -- opt-in, off by
        # default. See enable_homeostasis(). Disabled, `fire()` /
        # `fire_inhibition()` are byte-identical to before this was added.
        self.register_buffer(
            "_threshold_adapt", torch.zeros(0), persistent=False
        )
        self._homeostasis_enabled = False
        self._adapt_increment = 0.0
        self._adapt_decay = 0.0
        self._adapt_max = None

    def enable_homeostasis(self, increment=0.5, tau=20.0, max_adapt=None):
        """Turn on an adaptive, time-varying component added to
        :attr:`threshold`. Every call to :meth:`fire` (i.e. every time
        step), the adaptive component decays toward 0 with time-constant
        ``tau``, then -- after the firing decision -- jumps up by
        ``increment`` for every neuron that just spiked. The net effect is
        spike-frequency adaptation / intrinsic homeostatic plasticity: a
        neuron that has been firing raises its own effective threshold,
        making it progressively harder to fire, and that extra threshold
        relaxes back to the static :attr:`threshold` once firing stops.

        This mechanism is available on every neuron model built on
        :class:`SpikingNeuron` (i.e. all of them) without any change to the
        individual neuron classes, since they all call :meth:`fire` /
        :meth:`fire_inhibition`.

        The adaptive component is intentionally **not** learned by
        backprop -- it is homeostatic bookkeeping updated from the
        detached spike output, the same way :meth:`mem_reset` is detached.
        Use ``learn_threshold`` for a backprop-trained *static* threshold.

        :param increment: how much the effective threshold rises for each
            spike emitted. Defaults to 0.5
        :type increment: float, optional

        :param tau: time-constant (in time steps) over which the adaptive
            component decays back toward 0. Defaults to 20.0
        :type tau: float, optional

        :param max_adapt: if given, the adaptive component is clamped to
            this value so a neuron under sustained strong drive does not
            have its firing suppressed indefinitely. Defaults to None (no
            cap).
        :type max_adapt: float, optional
        """
        self._adapt_increment = float(increment)
        self._adapt_decay = math.exp(-1.0 / float(tau))
        self._adapt_max = None if max_adapt is None else float(max_adapt)
        self._homeostasis_enabled = True

    def disable_homeostasis(self):
        """Turn off the adaptive threshold. The static :attr:`threshold`
        is used as-is, exactly as if :meth:`enable_homeostasis` had never
        been called. The accumulated adaptive component is kept (not
        zeroed) so re-enabling resumes where it left off; call
        :meth:`reset_homeostasis` to clear it."""
        self._homeostasis_enabled = False

    def reset_homeostasis(self):
        """Clear the accumulated adaptive threshold component back to 0."""
        self._threshold_adapt = torch.zeros_like(self._threshold_adapt)
        return self._threshold_adapt

    def _homeostasis_pre(self, mem):
        """Decay the adaptive component and return this step's effective
        threshold. A no-op (returns the static threshold) unless
        :meth:`enable_homeostasis` has been called."""
        if not self._homeostasis_enabled:
            return self.threshold
        if self._threshold_adapt.shape != mem.shape:
            self._threshold_adapt = torch.zeros_like(mem)
        self._threshold_adapt = self._threshold_adapt * self._adapt_decay
        return self.threshold + self._threshold_adapt

    def _homeostasis_post(self, spk):
        """After the firing decision: bump the adaptive component for
        every neuron that just spiked. Detached -- homeostasis is
        bookkeeping driven by the (already detached-in-effect) spike
        count, not a backprop-trained quantity."""
        if not self._homeostasis_enabled:
            return
        self._threshold_adapt = (
            self._threshold_adapt + self._adapt_increment * spk.detach()
        )
        if self._adapt_max is not None:
            self._threshold_adapt = self._threshold_adapt.clamp(
                max=self._adapt_max
            )
        self._threshold_adapt = self._threshold_adapt.detach()

    def fire(self, mem):
        """Generates spike if mem > threshold.
        Returns spk."""

        if self.state_quant:
            mem = self.state_quant(mem)

        threshold_eff = self._homeostasis_pre(mem)
        mem_shift = mem - threshold_eff
        spk = self.spike_grad(mem_shift)
        self._homeostasis_post(spk)

        spk = spk * self.graded_spikes_factor

        return spk

    def fire_inhibition(self, batch_size, mem):
        """Generates spike if mem > threshold, only for the largest membrane.
        All others neurons will be inhibited for that time step.
        Returns spk."""
        threshold_eff = self._homeostasis_pre(mem)
        mem_shift = mem - threshold_eff
        index = torch.argmax(mem_shift, dim=1)
        spk_tmp = self.spike_grad(mem_shift)

        mask_spk1 = torch.zeros_like(spk_tmp)
        mask_spk1[torch.arange(batch_size), index] = 1
        spk = spk_tmp * mask_spk1
        self._homeostasis_post(spk)
        # reset = spk.clone().detach()

        return spk

    def mem_reset(self, mem):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        mem_shift = mem - self.threshold
        reset = self.spike_grad(mem_shift).clone().detach()

        return reset

    def _snn_cases(self, reset_mechanism, inhibition):
        self._reset_cases(reset_mechanism)

        if inhibition:
            warn(
                "Inhibition is an unstable feature that has only been tested "
                "for dense (fully-connected) layers. Use with caution!",
                UserWarning,
            )

    def _reset_cases(self, reset_mechanism):
        if (
            reset_mechanism != "subtract"
            and reset_mechanism != "zero"
            and reset_mechanism != "none"
        ):
            raise ValueError(
                "reset_mechanism must be set to either 'subtract', "
                "'zero', or 'none'."
            )

    def _snn_register_buffer(
        self,
        threshold,
        learn_threshold,
        reset_mechanism,
        graded_spikes_factor,
        learn_graded_spikes_factor,
    ):
        """Set variables as learnable parameters else register them in the
        buffer."""

        self._threshold_buffer(threshold, learn_threshold)
        self._graded_spikes_buffer(
            graded_spikes_factor, learn_graded_spikes_factor
        )

        # reset buffer
        try:
            # if reset_mechanism_val is loaded from .pt, override
            # reset_mechanism
            if torch.is_tensor(self.reset_mechanism_val):
                self.reset_mechanism = list(SpikingNeuron.reset_dict)[
                    self.reset_mechanism_val
                ]
        except AttributeError:
            # reset_mechanism_val has not yet been created, create it
            self._reset_mechanism_buffer(reset_mechanism)

    def _graded_spikes_buffer(
        self, graded_spikes_factor, learn_graded_spikes_factor
    ):
        if not isinstance(graded_spikes_factor, torch.Tensor):
            graded_spikes_factor = torch.as_tensor(graded_spikes_factor)
        if learn_graded_spikes_factor:
            self.graded_spikes_factor = nn.Parameter(graded_spikes_factor)
        else:
            self.register_buffer("graded_spikes_factor", graded_spikes_factor)

    def _threshold_buffer(self, threshold, learn_threshold):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.as_tensor(threshold)
        if learn_threshold:
            self.threshold = nn.Parameter(threshold)
        else:
            self.register_buffer("threshold", threshold)

    def _reset_mechanism_buffer(self, reset_mechanism):
        """Assign mapping to each reset mechanism state.
        Must be of type tensor to store in register buffer. See reset_dict
        for mapping."""
        reset_mechanism_val = torch.as_tensor(
            SpikingNeuron.reset_dict[reset_mechanism]
        )
        self.register_buffer("reset_mechanism_val", reset_mechanism_val)

    def _V_register_buffer(self, V, learn_V):
        if not isinstance(V, torch.Tensor):
            V = torch.as_tensor(V)
        if learn_V:
            self.V = nn.Parameter(V)
        else:
            self.register_buffer("V", V)

    @property
    def reset_mechanism(self):
        """If reset_mechanism is modified, reset_mechanism_val is triggered
        to update.
        0: subtract, 1: zero, 2: none."""
        return self._reset_mechanism

    @reset_mechanism.setter
    def reset_mechanism(self, new_reset_mechanism):
        self._reset_cases(new_reset_mechanism)
        self.reset_mechanism_val = torch.as_tensor(
            SpikingNeuron.reset_dict[new_reset_mechanism]
        )
        self._reset_mechanism = new_reset_mechanism

    @classmethod
    def init(cls):
        """Removes all items from :mod:`snntorch.SpikingNeuron.instances`
        when called."""
        cls.instances = []

    @staticmethod
    def detach(*args):
        """Used to detach input arguments from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are global variables."""
        for state in args:
            state.detach_()

    @staticmethod
    def zeros(*args):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are global variables."""
        for state in args:
            # `state = torch.zeros_like(state)` only rebinds the local
            # loop name, leaving the caller's tensor untouched. Zero the
            # storage in place instead. `.detach()` (a storage-sharing
            # view) is used rather than a bare `state.zero_()` so this
            # also works on a grad-requiring leaf -- e.g. an
            # `init_leaky()` hidden state reset before the first forward
            # pass, where `state.zero_()` would raise "a leaf Variable
            # that requires grad is being used in an in-place operation".
            state.detach().zero_()

    @staticmethod
    def _surrogate_bypass(input_):
        return (input_ > 0).to(input_.dtype)


class LIF(SpikingNeuron):
    """Parent class for leaky integrate and fire neuron models."""

    def __init__(
        self,
        beta,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
    ):
        super().__init__(
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )

        self._lif_register_buffer(
            beta,
            learn_beta,
        )
        self._reset_mechanism = reset_mechanism

    def _lif_register_buffer(
        self,
        beta,
        learn_beta,
    ):
        """Set variables as learnable parameters else register them in the
        buffer."""
        self._beta_buffer(beta, learn_beta)

    def _beta_buffer(self, beta, learn_beta):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)  # TODO: or .tensor() if no copy
        if learn_beta:
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer("beta", beta)

    def _V_register_buffer(self, V, learn_V):
        if V is not None:
            if not isinstance(V, torch.Tensor):
                V = torch.as_tensor(V)
        if learn_V:
            self.V = nn.Parameter(V)
        else:
            self.register_buffer("V", V)
