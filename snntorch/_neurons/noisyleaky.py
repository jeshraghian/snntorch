#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: 2023-07-26 18:11:31
# Author: Gehua Ma
# -----
# Last Modified: 2023-08-11 14:12:38
# Modified By: Gehua Ma
# -----
###
from .neurons import _SpikeTensor, _SpikeTorchConv, NoisyLIF
import torch

class NoisyLeaky(NoisyLIF):
    """
    Noisy leaky integrate-and-fire neuron model with noisy neuronal dynamics and probabilistic firing.
    Input is assumed to be a current injection. 
    Membrane potential decays exponentially with rate beta.
    
    Refer to `[1] <https://arxiv.org/abs/2305.16044>`. This study introduces the noisy spiking 
    neural network (NSNN) and the noise-driven learning rule (NDL) by incorporating noisy neuronal 
    dynamics to exploit the computational advantages of noisy neural processing. NSNN provides a 
    theoretical SNN framework that yields scalable, flexible, and reliable computation. It demonstrates
    that NSNN leads to spiking neural models with competitive performance, improved robustness 
    against challenging perturbations than deterministic SNNs, and better reproducing probabilistic 
    computations in neural coding. 

    [1] Ma et al. Exploiting Noise as a Resource for Computation and Learning in Spiking Neural 
    Networks. Patterns. Cell Press. 2023. 

    If `reset_mechanism = "subtract"`, then :math:`U[t+1]` will have
    `threshold` subtracted from it whenever the neuron emits a spike:

    .. math::

            U[t+1] = βU[t] + I_{\\rm in}[t+1] - RU_{\\rm thr} + \\epsilon

    If `reset_mechanism = "zero"`, then :math:`U[t+1]` will be set to `0`
    whenever the neuron emits a spike:

    .. math::

            U[t+1] = βU[t] + I_{\\rm syn}[t+1] - R(βU[t] + I_{\\rm in}[t+1]) + \\epsilon

    * :math:`I_{\\rm in}` - Input current
    * :math:`U` - Membrane potential
    * :math:`U_{\\rm thr}` - Membrane threshold
    * :math:`R` - Reset mechanism: if active, :math:`R = 1`, otherwise \
        :math:`R = 0`
    * :math:`β` - Membrane potential decay rate
    * :math:`\\epsilon` - Membrane noise term

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        beta = 0.5
        # noise type
        nt = 'gaussian'
        # noise scale, e.g. std for gaussian noise, scale for logstic noise, etc. 
        ns = 0.3

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                # initialize layers
                self.fc1 = nn.Linear(num_inputs, num_hidden)
                self.nlif1 = snn.NoisyLeaky(beta=beta, noise_type=nt, noise_scale=ns)
                self.fc2 = nn.Linear(num_hidden, num_outputs)
                self.nlif2 = snn.NoisyLeaky(beta=beta, noise_type=nt, noise_scale=ns)

            def forward(self, x, mem1, spk1, mem2):
                cur1 = self.fc1(x)
                spk1, mem1 = self.nlif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.nlif2(cur2, mem2)
                return mem1, spk1, mem2, spk2


    :param beta: membrane potential decay rate. Clipped between 0 and 1
        during the forward-pass. May be a single-valued tensor (i.e., equal
        decay rate for all neurons in a layer), or multi-valued (one weight per
        neuron).
    :type beta: float or torch.tensor

    :param threshold: Threshold for :math:`mem` to reach in order to
        generate a spike `S=1`. Defaults to 1
    :type threshold: float, optional

    :param noise_type: Neuronal membrane noise (ε) type.  
        Implemented types are: "gaussian", "logistic", "triangular", and "uniform". 
        For developers who want to add their own implementations of other kinds of noise: 
        The noise must be continuous, zero-mean, and its probability density function is symmetric 
        about the y-axis to meet the assumptions in the original literature 
        (doi.org/10.48550/arXiv.2305.16044).
    :type noise_type: str, optional 

    :param noise_scale: The noise scale is a parameter of the noise distribution. The larger the 
        noise scale, the more spread out the noise distribution will be. For example, if you are 
        using the "gaussian" noise type, the noise scale represents its standard deviation in our 
        implementation.
    :type noise_scale: float, optional

    :param init_hidden: Instantiates state variables as instance variables.
        Defaults to False
    :type init_hidden: bool, optional

    :param inhibition: If `True`, suppresses all spiking other than the
        neuron with the highest state. Defaults to False
    :type inhibition: bool, optional

    :param learn_beta: Option to enable learnable beta. Defaults to False
    :type learn_beta: bool, optional

    :param learn_threshold: Option to enable learnable threshold. Defaults
        to False
    :type learn_threshold: bool, optional

    :param reset_mechanism: Defines the reset mechanism applied to \
    :math:`mem` each time the threshold is met. Reset-by-subtraction: \
        "subtract", reset-to-zero: "zero", none: "none". Defaults to "subtract"
    :type reset_mechanism: str, optional

    :param state_quant: If specified, hidden state :math:`mem` is quantized
        to a valid state for the forward pass. Defaults to False
    :type state_quant: quantization function from snntorch.quant, optional

    :param output: If `True` as well as `init_hidden=True`, states are
        returned when neuron is called. Defaults to False
    :type output: bool, optional


    Inputs: \\input_, mem_0
        - **input_** of shape `(batch, input_size)`: tensor containing input
            features
        - **mem_0** of shape `(batch, input_size)`: tensor containing the
            initial membrane potential for each element in the batch.

    Outputs: spk, mem_1
        - **spk** of shape `(batch, input_size)`: tensor containing the
            output spikes.
        - **mem_1** of shape `(batch, input_size)`: tensor containing the
            next membrane potential for each element in the batch

    Learnable Parameters:
        - **Leaky.beta** (torch.Tensor) - optional learnable weights must be
            manually passed in, of shape `1` or (input_size).
        - **Leaky.threshold** (torch.Tensor) - optional learnable thresholds
            must be manually passed in, of shape `1` or`` (input_size).

    """
    def __init__(
        self,
        beta,
        threshold=1.0,
        noise_type='gaussian',
        noise_scale=0.3, 
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
        super(NoisyLeaky, self).__init__(
            beta,
            threshold,
            noise_type, 
            noise_scale, 
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )

        if self.init_hidden:
            self.mem = self.init_noisyleaky()

    def forward(self, input_, mem=False):

        if hasattr(mem, "init_flag"):  # only triggered on first-pass
            mem = _SpikeTorchConv(mem, input_=input_)
        elif mem is False and hasattr(
            self.mem, "init_flag"
        ):  # init_hidden case
            self.mem = _SpikeTorchConv(self.mem, input_=input_)

        # TO-DO: alternatively, we could do torch.exp(-1 /
        # self.beta.clamp_min(0)),
        # giving actual time constants instead of values in [0, 1] as
        # initial beta
        # beta = self.beta.clamp(0, 1)

        if not self.init_hidden:
            self.reset = self.mem_reset(mem)
            mem = self._build_state_function(input_, mem)

            if self.state_quant:
                mem = self.state_quant(mem)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)  # batch_size
            else:
                spk = self.fire(mem)

            return spk, mem

        # intended for truncated-BPTT where instance variables are hidden
        # states
        if self.init_hidden:
            self._leaky_forward_cases(mem)
            self.reset = self.mem_reset(self.mem)
            self.mem = self._build_state_function_hidden(input_)

            if self.state_quant:
                self.mem = self.state_quant(self.mem)

            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)

            if self.output:  # read-out layer returns output+states
                return self.spk, self.mem
            else:  # hidden layer e.g., in nn.Sequential, only returns output
                return self.spk
            
    def fire(self, mem):
        r"""
        Generate a spike using the probabilistic firing mechanism, i.e., if we still use mem to denote 
        the noise-free membrane potential, the firing probability is given by
        
        P(firing) = P(mem+noise > threshold) = P(noise < mem-threshold) = CDF_noise(mem-threshold)

        spk ~ Bernoulli(P(firing))
        :param mem: membrane voltage 
        
        Returns spk
        """
        if self.state_quant:
            mem = self.state_quant(mem)

        mem_shift = mem - self.threshold
        # the spike_grad function for noisy lif is called using (mem_shift, mean=0, noise_scale)
        spk = self.spike_grad(mem_shift, 0, self._noise_scale)
        spk = spk * self.graded_spikes_factor

        return spk

    def fire_inhibition(self, batch_size, mem):
        """Generates spike if mem > threshold, only for the largest membrane.
        All others neurons will be inhibited for that time step.
        Returns spk."""
        mem_shift = mem - self.threshold
        index = torch.argmax(mem_shift, dim=1)
        spk_tmp = self.spike_grad(mem_shift, 0, self._noise_scale)

        mask_spk1 = torch.zeros_like(spk_tmp)
        mask_spk1[torch.arange(batch_size), index] = 1
        spk = spk_tmp * mask_spk1
        # reset = spk.clone().detach()

        return spk

    def _base_state_function(self, input_, mem):
        base_fn = self.beta.clamp(0, 1) * mem + input_
        return base_fn

    def _build_state_function(self, input_, mem):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = self._base_state_function(
                input_, mem - self.reset * self.threshold
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = self._base_state_function(
                input_, mem
            ) - self.reset * self._base_state_function(input_, mem)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function(input_, mem)
        return state_fn

    def _base_state_function_hidden(self, input_):
        base_fn = self.beta.clamp(0, 1) * self.mem + input_
        return base_fn

    def _build_state_function_hidden(self, input_):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = (
                self._base_state_function_hidden(input_)
                - self.reset * self.threshold
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            self.mem = (1 - self.reset) * self.mem
            state_fn = self._base_state_function_hidden(input_)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function_hidden(input_)
        return state_fn

    def _leaky_forward_cases(self, mem):
        if mem is not False:
            raise TypeError(
                "When `init_hidden=True`, Leaky expects 1 input argument."
            )

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], NoisyLeaky):
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], NoisyLeaky):
                cls.instances[layer].mem = _SpikeTensor(init_flag=False)
