snntorch.spikevision.data
===========================

All datasets are subclasses of :code:`torch.utils.data.Dataset` i.e., they have :code:`__getitem__` and :code:`__len__` methods implemented. 
Hence, they can all be passed to a :code:`torch.utils.data.DataLoader` which can load multiple samples in parallel using :code:`torch.multiprocessing` workers. 
For example::

   nmnist_data = spikevision.data.NMNIST('path/to/nmnist_root/')
   data_loader = DataLoader(nmnist_data, 
                            batch_size=4,
                            shuffle=True, 
                            num_workers=args.nThreads)


NMNIST
^^^^^^^^

.. automodule:: snntorch.spikevision.spikedata.nmnist.NMNIST
   :members:
   :undoc-members:
   :show-inheritance:


DVSGesture
^^^^^^^^^^^

.. automodule:: snntorch.spikevision.spikedata.dvs_gesture.DVSGesture
   :members:
   :undoc-members:
   :show-inheritance:




SHD
^^^^^^^^^^^

.. automodule:: snntorch.spikevision.spikedata.shd.SHD
   :members:
   :undoc-members:
   :show-inheritance:

