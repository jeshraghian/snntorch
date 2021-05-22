snntorch.backprop
------------------------
:mod:`snntorch.backprop` is a module implementing various time-variant backpropagation algorithms. Each method will perform the forward-pass, backward-pass, and parameter update across all time steps in a single line of code. 


How to use backprop
^^^^^^^^^^^^^^^^^^^^^^^^
To use :mod:`snntorch.backprop` you must first construct a network, determine a loss criterion, and select an optimizer. When initializing neurons, set ``hidden_init=True``. This enables the methods in :mod:`snntorch.backprop` to automatically clear the hidden state variables, as well as detach them from the computational graph when necessary.

.. note:: The first dimension of input ``data`` is assumed to be time. The built-in backprop functions iterate through the first dimension of ``data`` by default. For time-invariant inputs, set ``time_varying_data=False``.

Example::

      net = Net().to(device)
      optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=betas)
      criterion = nn.CrossEntropyLoss()

      # Time-variant input data 
      for input, target in dataset:
         loss = BPTT(net, input, target, num_steps, batch_size, optimizer, criterion)
      
      # Time-invariant input data
      for input, targets in dataset:
         loss = BPTT(net, input, target, num_steps, batch_size, optimizer, criterion, time_varying_data=False)

.. automodule:: snntorch.backprop
   :members:
   :undoc-members:
   :show-inheritance: