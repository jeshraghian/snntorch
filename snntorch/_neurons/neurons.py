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

    def fire(self, mem):
        """Generates spike if mem > threshold.
        Returns spk."""

        if self.state_quant:
            mem = self.state_quant(mem)

        mem_shift = mem - self.threshold
        spk = self.spike_grad(mem_shift)

        spk = spk * self.graded_spikes_factor

        return spk

    def fire_inhibition(self, batch_size, mem):
        """Generates spike if mem > threshold, only for the largest membrane.
        All others neurons will be inhibited for that time step.
        Returns spk."""
        mem_shift = mem - self.threshold
        index = torch.argmax(mem_shift, dim=1)
        spk_tmp = self.spike_grad(mem_shift)

        mask_spk1 = torch.zeros_like(spk_tmp)
        mask_spk1[torch.arange(batch_size), index] = 1
        spk = spk_tmp * mask_spk1
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
            state = torch.zeros_like(state)

    @staticmethod
    def _surrogate_bypass(input_):
        return (input_ > 0).float()


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
