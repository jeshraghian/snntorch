snntorch.functional
------------------------
:mod:`snntorch.functional` implements common arithmetic operations applied to spiking neurons, such as loss and regularization functions, and state quantization etc.


How to use functional
^^^^^^^^^^^^^^^^^^^^^^^^
To use :mod:`snntorch.functional` you assign the function state to a variable, and then call that variable.

Example::

      import snntorch as snn
      import snntorch.functional as SF

      net = Net().to(device)
      optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=betas)
      criterion = SF.ce_count_loss()  # apply cross-entropy to spike count

      spk_rec, mem_rec = net(input_data)
      loss = loss_fn(spk_rec, targets)

      optimizer.zero_grad()
      loss.backward()

      # Weight Update
      optimizer.step()


Accuracy Functions
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: snntorch.functional.acc
   :members:
   :undoc-members:
   :show-inheritance:

Loss Functions
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: snntorch.functional.loss
   :members:
   :undoc-members:
   :show-inheritance:

Regularization Functions
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: snntorch.functional.reg
   :members:
   :undoc-members:
   :show-inheritance:

State Quantization
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: snntorch.functional.quant
   :members:
   :undoc-members:
   :show-inheritance: