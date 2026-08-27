import torch

from .leaky import Leaky


class DeltaLeaky(Leaky):
    '''
    A variant of the leaky integrate-and-fire neuron model that fires when the
    change in membrane potential exceeds a certain threshold. It accepts the
    same arguments as the standard Leaky class.
    '''
    def __init__(self, delta_threshold=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.delta_threshold = delta_threshold
        self.mem = None
        self.mem_prev = None


    def forward(self, input_, mem=None):
        if self.init_hidden:
            if self.mem is None:
                self.mem = torch.zeros_like(input_)
                self.mem_prev = torch.zeros_like(input_)
        else:
            if mem is None:
                raise TypeError("`mem` must be provided when init_hidden=False")

            self.mem, self.mem_prev = mem

            if self.mem is None:
                self.mem = torch.zeros_like(input_)
            if self.mem_prev is None:
                self.mem_prev = torch.zeros_like(input_)

        # update membrane potential
        mem_next = self.beta * self.mem + input_

        # compute change in membrane potential
        delta_v = mem_next - self.mem
        spk = self.spike_grad(delta_v.abs() - self.delta_threshold)

        # don't do a hard reset because delta SNNs encode change
        self.mem_prev = self.mem
        self.mem = mem_next

        return spk, (self.mem, self.mem_prev)

    def reset_mem(self):
        self.mem = None
        self.mem_prev = None
        return (self.mem, self.mem_prev)
