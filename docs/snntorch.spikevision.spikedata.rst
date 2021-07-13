snntorch.spikevision.spikedata
===============================

All datasets are subclasses of :code:`torch.utils.data.Dataset` i.e., they have :code:`__getitem__` and :code:`__len__` methods implemented. 
Hence, they can all be passed to a :code:`torch.utils.data.DataLoader` which can load multiple samples in parallel using :code:`torch.multiprocessing` workers. 
For example::

   nmnist_data = spikevision.data.NMNIST('path/to/nmnist_root/')
   data_loader = DataLoader(nmnist_data, 
                            batch_size=4,
                            shuffle=True, 
                            num_workers=args.nThreads)


The docstrings for the following datasets are currently not compiling correctly below.
For further information on each dataset and its use, please refer to the help function or `examples <https://snntorch.readthedocs.io/en/latest/examples/examples_svision.html>`_.

NMNIST
^^^^^^^^

.. autoclass:: snntorch.spikevision.spikedata.nmnist.NMNIST
   :members:


DVSGesture
^^^^^^^^^^^

.. autoclass:: snntorch.spikevision.spikedata.dvs_gesture.DVSGesture
   :members:



SHD
^^^^^^^^^^^

.. autoclass:: snntorch.spikevision.spikedata.shd.SHD
   :members:

