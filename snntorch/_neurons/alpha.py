import torch
import torch.nn as nn

from .lif import *


class Alpha(LIF):
    """
    A variant of the leaky integrate and fire neuron where membrane potential follows an alpha function.
    The time course of the membrane potential response depends on a combination of exponentials.
    In general, this causes the change in membrane potential to experience a delay with respect to an input spike.
    For :math:`U[T] > U_{\\rm thr} ⇒ S[T+1] = 1`.

    .. warning:: For a positive input current to induce a positive membrane response, ensure :math:`α > β`.

    If `reset_mechanism = "subtract"`, then :math:`I_{\\rm exc}, I_{\\rm inh}` will both have `threshold` subtracted from them whenever the neuron emits a spike:

    .. math::

            I_{\\rm exc}[t+1] = (αI_{\\rm exc}[t] + I_{\\rm in}[t+1]) - R(αI_{\\rm exc}[t] + I_{\\rm in}[t+1]) \\\\
            I_{\\rm inh}[t+1] = (βI_{\\rm inh}[t] - I_{\\rm in}[t+1]) - R(βI_{\\rm inh}[t] - I_{\\rm in}[t+1]) \\\\
            U[t+1] = τ_{\\rm SRM}(I_{\\rm exc}[t+1] + I_{\\rm inh}[t+1])

    If `reset_mechanism = "zero"`, then :math:`I_{\\rm exc}, I_{\\rm inh}` will both be set to `0` whenever the neuron emits a spike:

    .. math::

            I_{\\rm exc}[t+1] = (αI_{\\rm exc}[t] + I_{\\rm in}[t+1]) - R(αI_{\\rm exc}[t] + I_{\\rm in}[t+1]) \\\\
            I_{\\rm inh}[t+1] = (βI_{\\rm inh}[t] - I_{\\rm in}[t+1]) - R(βI_{\\rm inh}[t] - I_{\\rm in}[t+1]) \\\\
            U[t+1] = τ_{\\rm SRM}(I_{\\rm exc}[t+1] + I_{\\rm inh}[t+1])

    * :math:`I_{\\rm exc}` - Excitatory current
    * :math:`I_{\\rm inh}` - Inhibitory current
    * :math:`I_{\\rm in}` - Input current
    * :math:`U` - Membrane potential
    * :math:`U_{\\rm thr}` - Membrane threshold
    * :math:`R` - Reset mechanism, :math:`R = 1` if spike occurs, otherwise :math:`R = 0`
    * :math:`α` - Excitatory current decay rate
    * :math:`β` - Inhibitory current decay rate
    * :math:`τ_{\\rm SRM} = \\frac{log(α)}{log(β)} - log(α) + 1`

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

            def forward(self, x, syn_exc1, syn_inh1, mem1, spk1, syn_exc2, syn_inh2, mem2):
                cur1 = self.fc1(x)
                spk1, syn_exc1, syn_inh1, mem1 = self.lif1(cur1, syn_exc1, syn_inh1, mem1)
                cur2 = self.fc2(spk1)
                spk2, syn_exc2, syn_inh2, mem2 = self.lif2(cur2, syn_exc2, syn_inh2, mem2)
                return syn_exc1, syn_inh1, mem1, spk1, syn_exc2, syn_inh2, mem2, spk2


    """

    def __init__(
        self,
        alpha,
        beta,
        threshold=1.0,
        spike_grad=None,
        init_hidden=False,
        inhibition=False,
        learn_alpha=False,
        learn_beta=False,
        reset_mechanism="subtract",
        output=False,
    ):
        super(Alpha, self).__init__(
            beta,
            threshold,
            spike_grad,
            init_hidden,
            inhibition,
            learn_beta,
            reset_mechanism,
            output,
        )

        if not isinstance(alpha, torch.Tensor):
            alpha = torch.as_tensor(alpha)
        if learn_alpha:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)

        if self.init_hidden:
            self.syn_exc, self.syn_inh, self.mem = self.init_alpha()

        if (self.alpha <= self.beta).any():
            raise ValueError("alpha must be greater than beta.")

        if (self.beta == 1).any():
            raise ValueError(
                "beta cannot be '1' otherwise ZeroDivisionError occurs: tau_srm = log(alpha)/log(beta) - log(alpha) + 1"
            )

        # if reset_mechanism == "subtract":
        #     self.mem_residual = False

    def forward(self, input_, syn_exc=False, syn_inh=False, mem=False):

        if (
            hasattr(syn_exc, "init_flag")
            or hasattr(syn_inh, "init_flag")
            or hasattr(mem, "init_flag")
        ):  # only triggered on first-pass
            syn_exc, syn_inh, mem = _SpikeTorchConv(
                syn_exc, syn_inh, mem, input_=input_
            )
        elif mem is False and hasattr(self.mem, "init_flag"):  # init_hidden case
            self.syn_exc, self.syn_inh, self.mem = _SpikeTorchConv(
                self.syn_exc, self.syn_inh, self.mem, input_=input_
            )

        alpha = self.alpha.clamp(0, 1)
        beta = self.beta.clamp(0, 1)
        tau_srm = torch.log(alpha) / (torch.log(beta) - torch.log(alpha)) + 1

        # if hidden states are passed externally
        if not self.init_hidden:
            reset = self.mem_reset(mem)

            # if neuron fires, subtract threhsold from neuron
            if self.reset_mechanism == "subtract":

                # if self.mem_residual is False:
                #     self.mem_residual = torch.zeros_like(mem)

                syn_exc = (alpha * syn_exc + input_) - reset * (
                    alpha * syn_exc + input_
                )
                syn_inh = (beta * syn_inh - input_) - reset * (beta * syn_inh - input_)
                #  #The residual of (mem - threshold) decays separately
                # self.mem_residual = reset * (mem - self.threshold) + (
                #     self.mem_residual / self.tau_srm
                # )
                mem = tau_srm * (syn_exc + syn_inh)  # + self.mem_residual

            # if neuron fires, reset membrane to zero
            elif self.reset_mechanism == "zero":
                syn_exc = (alpha * syn_exc + input_) - reset * (
                    alpha * syn_exc + input_
                )
                syn_inh = (beta * syn_inh - input_) - reset * (beta * syn_inh - input_)
                mem = tau_srm * (syn_exc + syn_inh)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)

            else:
                spk = self.fire(mem)

            return spk, syn_exc, syn_inh, mem

        # if hidden states and outputs are instance variables
        if self.init_hidden and not mem:

            self.reset = self.mem_reset(self.mem)

            # if neuron fires, subtract threhsold from neuron
            if self.reset_mechanism == "subtract":

                # if self.mem_residual is False:
                #     self.mem_residual = torch.zeros_like(self.mem)

                self.syn_exc = (alpha * self.syn_exc + input_) - self.reset * (
                    alpha * self.syn_exc + input_
                )
                syn_inh = (beta * self.syn_inh - input_) - self.reset * (
                    beta * self.syn_inh - input_
                )
                # # The residual of (mem - threshold) decays separately
                # self.mem_residual = self.reset * (self.mem - self.threshold) + (
                #     self.mem_residual / self.tau_srm
                # )
                self.mem = tau_srm * (
                    self.syn_exc + self.syn_inh
                )  # + self.mem_residual

            # if neuron fires, reset membrane to zero
            elif self.reset_mechanism == "zero":

                syn_exc = (alpha * syn_exc + input_) - self.reset * (
                    alpha * syn_exc + input_
                )
                syn_inh = (beta * syn_inh - input_) - self.reset * (
                    beta * syn_inh - input_
                )
                self.mem = tau_srm * (syn_exc + syn_inh)

            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)

            else:
                self.spk = self.fire(self.mem)

            if self.output:
                return self.spk, self.syn_exc, self.syn_inh, self.mem
            else:
                return self.spk

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
        Intended for use where hidden state variables are instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Alpha):
                cls.instances[layer].syn_exc = _SpikeTensor(init_flag=False)
                cls.instances[layer].syn_inh = _SpikeTensor(init_flag=False)
                cls.instances[layer].mem = _SpikeTensor(init_flag=False)
