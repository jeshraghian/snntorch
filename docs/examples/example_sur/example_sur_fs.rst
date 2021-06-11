==================================================================
Fast Sigmoid
==================================================================

There are two ways to apply the Fast Sigmoid surrogate gradient:

.. code-block:: python
        
        import torch.nn as nn
        import snntorch as snn
        from snntorch import surrogate

        alpha = 0.6
        beta = 0.5
      
        num_inputs = 784
        num_hidden = 1000
        num_outputs = 10

Example::

        # Method 1 uses a closure to wrap around FastSigmoid, bundling it with the specified slope before calling it

        # initialize layers
        fc1 = nn.Linear(num_inputs, num_hidden)
        lif1 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=surrogate.fast_sigmoid(slope=50))
        fc2 = nn.Linear(num_hidden, num_outputs)
        lif2 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=surrogate.fast_sigmoid(slope=50))

Example::

        # Method 2 applies the autograd inherited method directly, using the default value of slope=25
        # The default value could also be called by specifying ``fast_sigmoid()`` instead

        # initialize layers
        fc1 = nn.Linear(num_inputs, num_hidden)
        lif1 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=surrogate.FastSigmoid.apply)
        fc2 = nn.Linear(num_hidden, num_outputs)
        lif2 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=surrogate.FastSigmoid.apply)
