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

        self.mem = None
        self.delta_threshold = delta_threshold
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

        # update membrane potential
        mem_next = self.beta * self.mem + input_

        # compute change in membrane potential
        delta_v = mem_next - self.mem
        spk = self.spike_grad(delta_v.abs() - self.delta_threshold)

        # don't do a hard reset because delta SNNs encode change
        self.mem_prev = self.mem
        self.mem = mem_next

        if self.output:
            return spk, self.mem
        else:
            return spk
