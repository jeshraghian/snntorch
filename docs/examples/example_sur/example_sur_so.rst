==================================================================
Spike Operator
==================================================================

There are two ways to apply the Spike Operator surrogate gradient:

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

        # Method 1 uses a closure to wrap around SpikeOperator, bundling it with the specified threshold before calling it

        # initialize layers
        fc1 = nn.Linear(num_inputs, num_hidden)
        lif1 = snn.Stein(alpha=alpha, beta=beta, spike_grad=surrogate.spike_operator(threshold=2))
        fc2 = nn.Linear(num_hidden, num_outputs)
        lif2 = snn.Stein(alpha=alpha, beta=beta, spike_grad=surrogate.spike_operator(threshold=2))

Example::

        # Method 2 applies the autograd inherited method directly, using the default value of threshold=1
        # The default value could also be called by specifying ``spike_operator()`` instead

        # initialize layers
        fc1 = nn.Linear(num_inputs, num_hidden)
        lif1 = snn.Stein(alpha=alpha, beta=beta, spike_grad=surrogate.SpikeOperator.apply)
        fc2 = nn.Linear(num_hidden, num_outputs)
        lif2 = snn.Stein(alpha=alpha, beta=beta, spike_grad=surrogate.SpikeOperator.apply)


.. warning:: 
        
        ``threshold`` should match the threshold of the neuron, which defaults to 1 as well.
        If ``threshold`` < 1, this method is known to converge poorly. 
