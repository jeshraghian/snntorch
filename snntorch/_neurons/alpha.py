import torch
import torch.nn as nn

from .neurons import LIF


class Alpha(LIF):
    """
    A variant of the leaky integrate and fire neuron where membrane
    potential follows an alpha function.
    The time course of the membrane potential response depends on a
    combination of exponentials.
    In general, this causes the change in membrane potential to
    experience a delay with respect to an input spike.
    For :math:`U[T] > U_{\\rm thr} ⇒ S[T+1] = 1`.

    .. warning:: For a positive input current to induce a positive membrane \
    response, ensure :math:`α > β`.

    If `reset_mechanism = "zero"`, then :math:`I_{\\rm exc}, I_{\\rm inh}`
    will both be set to `0` whenever the neuron emits a spike:

    .. math::

            I_{\\rm exc}[t+1] = (αI_{\\rm exc}[t] + I_{\\rm in}[t+1]) -
            R(αI_{\\rm exc}[t] + I_{\\rm in}[t+1]) \\\\
            I_{\\rm inh}[t+1] = (βI_{\\rm inh}[t] - I_{\\rm in}[t+1]) -
            R(βI_{\\rm inh}[t] - I_{\\rm in}[t+1]) \\\\
            U[t+1] = τ_{\\rm α}(I_{\\rm exc}[t+1] + I_{\\rm inh}[t+1])

    * :math:`I_{\\rm exc}` - Excitatory current
    * :math:`I_{\\rm inh}` - Inhibitory current
    * :math:`I_{\\rm in}` - Input current
    * :math:`U` - Membrane potential
    * :math:`U_{\\rm thr}` - Membrane threshold
    * :math:`R` - Reset mechanism, :math:`R = 1` if spike occurs, otherwise \
        :math:`R = 0`
    * :math:`α` - Excitatory current decay rate
    * :math:`β` - Inhibitory current decay rate
    * :math:`τ_{\\rm α} = \\frac{log(α)}{log(β)} - log(α) + 1`

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        alpha = 0.9
        beta = 0.8

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                # initialize layers
                self.fc1 = nn.Linear(num_inputs, num_hidden)
                self.lif1 = snn.Alpha(alpha=alpha, beta=beta)
                self.fc2 = nn.Linear(num_hidden, num_outputs)
                self.lif2 = snn.Alpha(alpha=alpha, beta=beta)

            def forward(self, x, syn_exc1, syn_inh1, mem1, spk1, syn_exc2,
            syn_inh2, mem2):
                cur1 = self.fc1(x)
                spk1, syn_exc1, syn_inh1, mem1 = self.lif1(cur1, syn_exc1,
                syn_inh1, mem1)
                cur2 = self.fc2(spk1)
                spk2, syn_exc2, syn_inh2, mem2 = self.lif2(cur2, syn_exc2,
                syn_inh2, mem2)
                return syn_exc1, syn_inh1, mem1, spk1, syn_exc2, syn_inh2,
                mem2, spk2

        # Too many state variables which becomes cumbersome, so the
        # following is also an option:

        alpha = 0.9
        beta = 0.8

        net = nn.Sequential(nn.Linear(num_inputs, num_hidden),
                            snn.Alpha(alpha=alpha, beta=beta,
                            init_hidden=True),
                            nn.Linear(num_hidden, num_outputs),
                            snn.Alpha(alpha=alpha, beta=beta,
                            init_hidden=True, output=True))


    """

    def __init__(
        self,
        alpha,
        beta,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_alpha=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="zero",
        state_quant=False,
        output=False,
    ):
        super().__init__(
            beta,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
        )

        self._alpha_register_buffer(alpha, learn_alpha)
        self._alpha_cases()

        self._init_mem()

        if self.reset_mechanism_val == 0:  # reset by subtraction
            self.state_function = self._base_sub
        elif self.reset_mechanism_val == 1:  # reset to zero
            self.state_function = self._base_zero
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            self.state_function = self._base_int

    def _init_mem(self):
        syn_exc = torch.zeros(1)
        syn_inh = torch.zeros(1)
        mem = torch.zeros(1)

        self.register_buffer("syn_exc", syn_exc)
        self.register_buffer("syn_inh", syn_inh)
        self.register_buffer("mem", mem)

    def reset_mem(self):
        self.syn_exc = torch.zeros_like(
            self.syn_exc, device=self.syn_exc.device
        )
        self.syn_inh = torch.zeros_like(
            self.syn_inh, device=self.syn_inh.device
        )
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)

    def init_alpha(self):
        """Deprecated, use :class:`Alpha.reset_mem` instead"""
        self.reset_mem()
        return self.syn_exc, self.syn_inh, self.mem

    def forward(self, input_, syn_exc=None, syn_inh=None, mem=None):

        if not syn_exc == None:
            self.syn_exc = syn_exc

        if not syn_inh == None:
            self.syn_inh = syn_inh

        if not mem == None:
            self.mem = mem

        if self.init_hidden and (
            not mem == None or not syn_exc == None or not syn_inh == None
        ):
            raise TypeError(
                "When `init_hidden=True`, Alpha expects 1 input argument."
            )

        if not self.syn_exc.shape == input_.shape:
            self.syn_exc = torch.zeros_like(input_, device=self.syn_exc.device)

        if not self.syn_inh.shape == input_.shape:
            self.syn_inh = torch.zeros_like(input_, device=self.syn_inh.device)

        if not self.mem.shape == input_.shape:
            self.mem = torch.zeros_like(input_, device=self.mem.device)

        self.reset = self.mem_reset(self.mem)
        self.syn_exc, self.syn_inh, self.mem = self.state_function(input_)

        if self.state_quant:
            self.syn_exc = self.state_quant(self.syn_exc)
            self.syn_inh = self.state_quant(self.syn_inh)
            self.mem = self.state_quant(self.mem)

        if self.inhibition:
            spk = self.fire_inhibition(self.mem.size(0), self.mem)
        else:
            spk = self.fire(self.mem)

        if self.output:
            return spk, self.syn_exc, self.syn_inh, self.mem
        elif self.init_hidden:
            return spk
        else:
            return spk, self.syn_exc, self.syn_inh, self.mem

    def _base_state_function(self, input_):
        base_fn_syn_exc = self.alpha.clamp(0, 1) * self.syn_exc + input_
        base_fn_syn_inh = self.beta.clamp(0, 1) * self.syn_inh - input_
        tau_alpha = (
            torch.log(self.alpha.clamp(0, 1))
            / (
                torch.log(self.beta.clamp(0, 1))
                - torch.log(self.alpha.clamp(0, 1))
            )
            + 1
        )
        base_fn_mem = tau_alpha * (base_fn_syn_exc + base_fn_syn_inh)
        return base_fn_syn_exc, base_fn_syn_inh, base_fn_mem

    def _base_state_reset_sub_function(self, input_):
        syn_exc_reset = self.threshold
        syn_inh_reset = self.beta.clamp(0, 1) * self.syn_inh - input_
        mem_reset = -self.syn_inh
        return syn_exc_reset, syn_inh_reset, mem_reset

    def _base_sub(self, input_):
        syn_exec, syn_inh, mem = self._base_state_function(input_)
        syn_exec2, syn_inh2, mem2 = self._base_state_reset_sub_function(input_)

        syn_exec -= syn_exec2 * self.reset
        syn_inh -= syn_inh2 * self.reset
        mem -= mem2 * self.reset

        return syn_exec, syn_inh, mem

    def _base_zero(self, input_):
        syn_exec, syn_inh, mem = self._base_state_function(input_)
        syn_exec2, syn_inh2, mem2 = self._base_state_function(input_)

        syn_exec -= syn_exec2 * self.reset
        syn_inh -= syn_inh2 * self.reset
        mem -= mem2 * self.reset

        return syn_exec, syn_inh, mem

    def _base_int(self, input_):
        return self._base_state_function(input_)

    def _alpha_register_buffer(self, alpha, learn_alpha):
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.as_tensor(alpha)
        if learn_alpha:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)

        self.alpha = self.alpha.clamp(0, 1)

    def _alpha_cases(self):
        if (self.alpha <= self.beta).any():
            raise ValueError("alpha must be greater than beta.")

        if (self.beta == 1).any():
            raise ValueError(
                "beta cannot be '1' otherwise ZeroDivisionError occurs: "
                "tau_alpha = log(alpha)/log(beta) - log(alpha) + 1"
            )

    @classmethod
    def detach_hidden(cls):
        """Used to detach hidden states from the current graph.
        Intended for use in truncated backpropagation through
        time where hidden state variables are instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Alpha):
                cls.instances[layer].syn_exc.detach_()
                cls.instances[layer].syn_inh.detach_()
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance
        variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Alpha):
                cls.instances[layer].syn_exc = torch.zeros_like(
                    cls.instances[layer].syn_exc,
                    device=cls.instances[layer].syn_exc.device,
                )
                cls.instances[layer].syn_inh = torch.zeros_like(
                    cls.instances[layer].syn_inh,
                    device=cls.instances[layer].syn_inh.device,
                )
                cls.instances[layer].mem = torch.zeros_like(
                    cls.instances[layer].mem,
                    device=cls.instances[layer].mem.device,
                )
