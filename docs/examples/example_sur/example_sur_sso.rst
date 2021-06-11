==================================================================
Stochastic Spike Operator
==================================================================

There are two ways to apply the Stochastic Spike Operator surrogate gradient:

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

        # Method 1 uses a closure to wrap around StochasticSpikeOperator, bundling it with the specified mean and variance before calling it

        # initialize layers
        fc1 = nn.Linear(num_inputs, num_hidden)
        lif1 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=surrogate.SSO(mean=0.1, variance=0.1))
        fc2 = nn.Linear(num_hidden, num_outputs)
        lif2 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=surrogate.SSO(mean=0.1, variance=0.1))

Example::

        # Method 2 applies the autograd inherited method directly, using the default values of mean=0, variance=0.2
        # The default value could also be called by specifying ``SSO()`` instead

        # initialize layers
        fc1 = nn.Linear(num_inputs, num_hidden)
        lif1 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=surrogate.StochasticSpikeOperator.apply)
        fc2 = nn.Linear(num_hidden, num_outputs)
        lif2 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=surrogate.StochasticSpikeOperator.apply)
