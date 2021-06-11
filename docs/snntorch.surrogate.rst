snntorch.surrogate
-------------------------

By default, PyTorch's autodifferentiation tools are unable to calculate the analytical derivative of the spiking neuron graph. 
The discrete nature of spikes makes it difficult for ``torch.autograd`` to calculate a gradient that facilitates learning.
:mod:`snntorch` overrides the default gradient by using :mod:`snntorch.LIF.Heaviside`.

Alternative gradients are also available in the :mod:`snntorch.surrogate` module. 
These represent either approximations of the backward pass or probabilistic models of firing as a function of the membrane potential.

For further reading, see:

    *E. O. Neftci, H. Mostafa, F. Zenke (2019) Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient-Based Optimization to Spiking Neural Networks. IEEE Signal Processing Magazine, pp. 51-63.*

How to use surrogate
^^^^^^^^^^^^^^^^^^^^^^^^

The surrogate gradient must be passed as the ``spike_grad`` argument to the neuron model. 
If ``spike_grad`` is left unspecified, it defaults to :mod:`snntorch.LIF.Heaviside`.
In the following example, we apply the fast sigmoid surrogate to :mod:`snntorch.Cond`.

Example::

   import snntorch as snn
   from snntorch import surrogate
   import torch
   import torch.nn as nn

   alpha = 0.9
   beta = 0.85

   # Initialize surrogate gradient
   spike_grad1 = surrogate.fast_sigmoid()  # passes default parameters from a closure
   spike_grad2 = surrogate.FastSigmoid.apply  # passes default parameters, equivalent to above
   spike_grad3 = surrogate.fast_sigmoid(slope=50)  # custom parameters from a closure

   # Define Network
   class Net(nn.Module):
    def __init__(self):
        super().__init__()

    # Initialize layers, specify the ``spike_grad`` argument
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Cond(alpha=alpha, beta=beta, spike_grad=spike_grad1)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Cond(alpha=alpha, beta=beta, spike_grad=spike_grad3)

    def forward(self, x, syn1, mem1, spk1, syn2, mem2):
        cur1 = self.fc1(x)
        spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
        cur2 = self.fc2(spk1)
        spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
        return syn1, mem1, spk1, syn2, mem2, spk2

    net = Net().to(device)


.. automodule:: snntorch.surrogate
   :members:
   :undoc-members:
   :show-inheritance: