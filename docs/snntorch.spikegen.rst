snntorch.spikegen
------------------------
:mod:`snntorch.spikegen` is a module that provides a variety of common spike generation and conversion methods, including spike-rate and latency coding.

How to use spikegen
^^^^^^^^^^^^^^^^^^^^^^^^
In general, tensors containing non-spiking data can simply be passed into one of the functions in :mod:`snntorch.spikegen` to convert them into discrete spikes.
There are a variety of methods to achieve this conversion. At present, `snntorch` supports:

* rate coding
* latency coding

There are also options for converting targets into time-varying spikes.

.. automodule:: snntorch.spikegen
   :members:
   :undoc-members:
   :show-inheritance: