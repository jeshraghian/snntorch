==================================================================
Spike Rate Escape
==================================================================

There are two ways to apply the Spike Rate Escape surrogate gradient:

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

        # Method 1 uses a closure to wrap around SpikeRateEscape, bundling it with the specified beta and slope before calling it

        # initialize layers
        fc1 = nn.Linear(num_inputs, num_hidden)
        lif1 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=surrogate.spike_rate_escape(beta=2, slope=50))
        fc2 = nn.Linear(num_hidden, num_outputs)
        lif2 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=surrogate.spike_rate_escape(beta=2, slope=25))

Example::

        # Method 2 applies the autograd inherited method directly, using the default values of beta=1 and slope=25
        # The default value could also be called by specifying ``spike_rate_escape()`` instead

        # initialize layers
        fc1 = nn.Linear(num_inputs, num_hidden)
        lif1 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=surrogate.SpikeRateEscape.apply)
        fc2 = nn.Linear(num_hidden, num_outputs)
        lif2 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=surrogate.SpikeRateEscape.apply)
