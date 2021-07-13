================
Introduction
================


.. image:: https://img.shields.io/pypi/v/snntorch.svg
        :target: https://pypi.python.org/pypi/snntorch

.. image:: https://github.com/jeshraghian/snntorch/actions/workflows/build.yml/badge.svg
        :target: https://snntorch.readthedocs.io/en/latest/?badge=latest

.. image:: https://readthedocs.org/projects/snntorch/badge/?version=latest
        :target: https://snntorch.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

        
.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/snntorch_alpha_scaled.png?raw=true
        :align: center
        :width: 700


The brain is the perfect place to look for inspiration to develop more efficient neural networks. One of the main differences with modern deep learning is that the brain encodes information in spikes rather than continuous activations. 
snnTorch is a Python package for performing gradient-based learning with spiking neural networks.
It extends the capabilities of PyTorch, taking advantage of its GPU accelerated tensor 
computation and applying it to networks of spiking neurons. Pre-designed spiking neuron models are seamlessly integrated within the PyTorch framework and can be treated as recurrent activation units. 


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/spike_excite_alpha_ps2.gif?raw=true
        :align: center
        :width: 800

snnTorch Structure
^^^^^^^^^^^^^^^^^^^^^^^^
snnTorch contains the following components: 

.. list-table::
   :widths: 20 60
   :header-rows: 1

   * - Component
     - Description
   * - `snntorch <https://snntorch.readthedocs.io/en/latest/snntorch.html>`_
     - a spiking neuron library like torch.nn, deeply integrated with autograd
   * - `snntorch.backprop <https://snntorch.readthedocs.io/en/latest/snntorch.backprop.html>`_
     - variations of backpropagation commonly used with SNNs
   * - `snntorch.spikegen <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html>`_
     - a library for spike generation and data conversion
   * - `snntorch.spikeplot <https://snntorch.readthedocs.io/en/latest/snntorch.spikeplot.html>`_
     - visualization tools for spike-based data using matplotlib and celluloid
   * - `snntorch.spikevision <https://snntorch.readthedocs.io/en/latest/snntorch.spikevision.html>`_
     - contains popular neuromorphic datasets
   * - `snntorch.surrogate <https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html>`_
     - optional surrogate gradient functions
   * - `snntorch.utils <https://snntorch.readthedocs.io/en/latest/snntorch.utils.html>`_
     - dataset utility functions

snnTorch is designed to be intuitively used with PyTorch, as though each spiking neuron were simply another activation in a sequence of layers. 
It is therefore agnostic to fully-connected layers, convolutional layers, residual connections, etc. 

At present, the neuron models are represented by recursive functions which removes the need to store membrane potential traces for all neurons in a system in order to calculate the gradient. 
The lean requirements of snnTorch enable small and large networks to be viably trained on CPU, where needed. 
Provided that the network models and tensors are loaded onto CUDA, snnTorch takes advantage of GPU acceleration in the same way as PyTorch. 

Citation 
^^^^^^^^^^^^^^^^^^^^^^^^
Under preparation.

Requirements 
^^^^^^^^^^^^^^^^^^^^^^^^
The following packages need to be installed to use snnTorch:

* torch >= 1.2.0
* numpy >= 1.17
* pandas
* matplotlib
* math
* celluloid

Installation
^^^^^^^^^^^^^^^^^^^^^^^^

Run the following to install::

  $ python
  $ pip install snntorch

To install snnTorch from source instead::

  $ git clone https://github.com/jeshraghian/snnTorch
  $ cd snnTorch
  $ python setup.py install

API & Examples 
^^^^^^^^^^^^^^^^^^^^^^^^
A complete API is available `here`_. Examples, tutorials and Colab notebooks are provided.

.. _here: https://snntorch.readthedocs.io/

Getting Started
^^^^^^^^^^^^^^^^^^^^^^^^
Here are a few ways you can get started with snnTorch:

* `The API Reference`_ 

* `Examples`_

* `Tutorials`_

.. _The API Reference: https://snntorch.readthedocs.io/
.. _Examples: https://snntorch.readthedocs.io/en/latest/examples.html
.. _Tutorials: https://snntorch.readthedocs.io/en/latest/tutorials/index.html


Contributing
^^^^^^^^^^^^^^^^^^^^^^^^
If you're ready to contribute to snnTorch, instructions to do so can be `found here`_.

.. _found here: https://snntorch.readthedocs.io/en/latest/contributing.html

Acknowledgments
^^^^^^^^^^^^^^^^^^^^^^^^
snnTorch was initially developed by `Jason K. Eshraghian`_ in the `Lu Group (University of Michigan)`_.

Additional contributions were made by Xinxin Wang, Vincent Sun, and Emre Neftci.

Several features in snnTorch were inspired by the work of Friedemann Zenke, Emre Neftci, Doo Seok Jeong, Sumit Bam Shrestha and Garrick Orchard.

.. _Jason K. Eshraghian: https://jasoneshraghian.com
.. _Lu Group (University of Michigan): https://lugroup.engin.umich.edu/


License & Copyright
^^^^^^^^^^^^^^^^^^^^^^^^
snnTorch is licensed under the GNU General Public License v3.0: https://www.gnu.org/licenses/gpl-3.0.en.html.
