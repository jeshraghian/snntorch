snntorch
================


snnTorch Neurons
---------------------
:mod:`snntorch` is designed to be intuitively used with PyTorch, as though each spiking neuron were simply another activation in a sequence of layers. 

A variety of spiking neuron classes are available which can simply be treated as activation units with PyTorch. 
Each layer of spiking neurons are therefore agnostic to fully-connected layers, convolutional layers, residual connections, etc. 

The neuron models are represented by recursive functions which removes the need to store membrane potential traces in order to calculate the gradient. 
The lean requirements of :mod:`snntorch` enable small and large networks to be viably trained on CPU, where needed. 
Being deeply integrated with ``torch.autograd``, :mod:`snntorch` is able to take advantage of GPU acceleration in the same way as PyTorch.

By default, PyTorch's autodifferentiation mechanism in ``torch.autograd`` nulls the gradient signal of the spiking neuron graph due to non-differentiable spiking threshold functions.
:mod:`snntorch` overrides the default gradient by using :mod:`snntorch.neurons.Heaviside`. Alternative options exist in :mod:`snntorch.surrogate`.

At present, the neurons available in :mod:`snntorch` are variants of the Leaky Integrate-and-Fire neuron model:

* **Leaky** - 1st-Order Leaky Integrate-and-Fire Neuron
* **RLeaky** - As above, with recurrent connections for output spikes
* **Synaptic** - 2nd-Order Integrate-and-Fire Neuron (including synaptic conductance)
* **RSynaptic** - As above, with recurrent connections for output spikes
* **Lapicque** - Lapicque's RC Neuron Model
* **Alpha** - Alpha Membrane Model

Additional models include spiking-LSTMs and spiking-ConvLSTMs:

* **SLSTM** - Spiking long short-term memory cell with state-thresholding 
* **SConv2dLSTM** - Spiking 2d convolutional short-term memory cell with state thresholding



How to use snnTorch's neuron models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following arguments are common across all neuron models:

* **threshold** - firing threshold of the neuron
* **spike_grad** - surrogate gradient function (see :mod:`snntorch.surrogate`)
* **init_hidden** - setting to ``True`` hides all neuron states as instance variables to reduce code complexity
* **inhibition** - setting to ``True`` enables only the neuron with the highest membrane potential to fire in a dense layer (not for use in convs etc.)
* **learn_beta** - setting to ``True`` enables the decay rate to be a learnable parameter
* **learn_threshold** - setting to ``True`` enables the threshold to be a learnable parameter 
* **reset_mechanism** - options include ``subtract`` (reset-by-subtraction), ``zero`` (reset-to-zero), and ``none`` (no reset mechanism: i.e., leaky integrator neuron)
* **output** - if ``init_hidden=True``, the spiking neuron will only return the output spikes. Setting ``output=True`` enables the hidden state(s) to be returned as well. Useful when using ``torch.nn.sequential``. 

Leaky integrate-and-fire neuron models also include:

* **beta** - decay rate of membrane potential, clipped between 0 and 1 during the forward-pass. Can be a single-value tensor (same decay for all neurons in a layer), or can be multi-valued (individual weights p/neuron in a layer. More complex neurons include additional parameters, such as **alpha**.

Recurrent spiking neuron models, such as :mod:`snntorch.RLeaky` and :mod:`snntorch.RSynaptic` explicitly pass the output spike back to the input. 
Such neurons include additional arguments:

* **V** - Recurrent weight. Can be a single-valued tensor (same weight across all neurons in a layer), or multi-valued tensor (individual weights p/neuron in a layer).
* **learn_V** - defaults to ``True``, which enables **V** to be a learnable parameter. 

Spiking neural networks can be constructed using a combination of the ``snntorch`` and ``torch.nn`` packages.

Example::

      import torch
      import torch.nn as nn
      import snntorch as snn

      alpha = 0.9
      beta = 0.85

      num_steps = 100


      # Define Network
      class Net(nn.Module):
         def __init__(self):
            super().__init__()

            # initialize layers
            self.fc1 = nn.Linear(num_inputs, num_hidden)
            self.lif1 = snn.Leaky(beta=beta)
            self.fc2 = nn.Linear(num_hidden, num_outputs)
            self.lif2 = snn.Leaky(beta=beta)

         def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()

            spk2_rec = []  # Record the output trace of spikes
            mem2_rec = []  # Record the output trace of membrane potential

            for step in range(num_steps):
                  cur1 = self.fc1(x.flatten(1))
                  spk1, mem1 = self.lif1(cur1, mem1)
                  cur2 = self.fc2(spk1)
                  spk2, mem2 = self.lif2(cur2, mem2)

                  spk2_rec.append(spk2)
                  mem2_rec.append(mem2)

            return torch.stack(spk2_rec), torch.stack(mem2_rec)

      net = Net().to(device)

      output, mem_rec = net(data)

In the above example, the hidden state ``mem`` must be manually initialized for each layer.
This can be overcome by automatically instantiating neuron hidden states by invoking ``init_hidden=True``. 

In some cases (e.g., truncated backprop through time), it might be necessary to perform backward passes before all time steps have completed processing.
This requires moving the time step for-loop out of the network and into the training-loop. 

An example of this is shown below::

      import torch
      import torch.nn as nn
      import snntorch as snn

      num_steps = 100

      lif1 = snn.Leaky(beta=0.9, init_hidden=True) # only returns spk
      lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True) # returns mem and spk if output=True


      #  Initialize Network
      net = nn.Sequential(nn.Flatten(),
                          nn.Linear(784,1000), 
                          lif1,
                          nn.Linear(1000, 10),
                          lif2).to(device)

      for step in range(num_steps):
         spk_out, mem_out = net(data)


Setting the hidden states to instance variables is necessary for calling the backpropagation methods available in :mod:`snntorch.backprop`, or for calling :mod:`nn.Sequential` from PyTorch.

Whenever a neuron is instantiated, it is added as a list item to the class variable :mod:`LIF.instances`. 
This helps the functions in :mod:`snntorch.backprop` keep track of what neurons are being used in the network, and when they must be detached from the computation graph. 

In the above examples, the decay rate of membrane potential :mod:`beta` is treated as a hyperparameter. 
But it can also be configured as a learnable parameter, as shown below::

      import torch
      import torch.nn as nn
      import snntorch as snn

      num_steps = 100

      lif1 = snn.Leaky(beta=0.9, learn_beta=True, init_hidden=True) # only returns spk
      lif2 = snn.Leaky(beta=0.5, learn_beta=True, init_hidden=True, output=True) # returns mem and spk if output=True


      #  Initialize Network
      net = nn.Sequential(nn.Flatten(),
                          nn.Linear(784,1000), 
                          lif1,
                          nn.Linear(1000, 10),
                          lif2).to(device)

      for step in range(num_steps):
         spk_out, mem_out = net(data.view(batch_size, -1))

Here, :mod:`beta` is initialized to 0.9 for the first layer, and 0.5 for the second layer.
Each layer then treats it as a learnable parameter, just like all the other network weights.
In the event you wish to have a learnable decay rate for each neuron rather than each layer, the following example shows how::


      import torch
      import torch.nn as nn
      import snntorch as snn

      num_steps = 100
      num_hidden = 1000
      num_output = 10

      beta1 = torch.rand(num_hidden)  # randomly initialize beta as a vector
      beta2 = torch.rand(num_output)

      lif1 = snn.Leaky(beta=beta1, learn_beta=True, init_hidden=True) # only returns spk
      lif2 = snn.Leaky(beta=beta2 learn_beta=True, init_hidden=True, output=True) # returns mem and spk if output=True


      #  Initialize Network
      net = nn.Sequential(nn.Flatten(),
                          nn.Linear(784, num_hidden), 
                          lif1,
                          nn.Linear(1000, num_output),
                          lif2).to(device)

      for step in range(num_steps):
         spk_out, mem_out = net(data.view(batch_size, -1))


The same approach as above can be used for implementing learnable thresholds, using ``learn_threshold=True``. 

Each neuron has the option to inhibit other neurons within the same dense layer from firing. 
This can be invoked by setting ``inhibition=True`` when instantiating the neuron layer. It has not yet been implemented for networks other than fully-connected layers, so use with caution.


Neuron List
---------------------

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :glob:

    snn.neurons_*


Neuron Parent Classes
---------------------

.. automodule:: snntorch._neurons.neurons 
   :members:
   :undoc-members:
   :show-inheritance: