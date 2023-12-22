snntorch.spikevision
======================

.. warning::
    The spikevision module has been deprecated.
    To load neuromorphic datasets, we recommend using the `Tonic project <https://github.com/neuromorphs/tonic>`_.
    For examples on how to use snnTorch together with Tonic, please refer to `Tutorial 7 in the snnTorch Tutorial Series <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html>`_.


The :code:`spikevision` module consists of neuromorphic datasets and common image transformations.
It is the neuromorphic analog to `torchvision <https://pytorch.org/vision/stable/index.html>`_. 


:code:`spikevision` contains the following neuromorphic datasets:

.. list-table::
   :widths: 20 60 20
   :header-rows: 1

   * - Dataset
     - Description
     - Author URL
   * - `NMNIST <https://snntorch.readthedocs.io/en/latest/snntorch.spikevision.spikedata.html#nmnist>`_
     - | A spiking version of the original 
       | frame-based `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset.
     - `G. Orchard <https://www.garrickorchard.com/datasets/n-mnist>`_
   * - `DVSGesture <https://snntorch.readthedocs.io/en/latest/snntorch.spikevision.spikedata.html#dvsgesture>`_
     - | 11 hand gestures recorded from 29 subjects 
       | under 3 illumination conditions using a DVS128.
     - `IBM Research <https://www.research.ibm.com/dvsgesture/>`_
   * - `SHD <https://snntorch.readthedocs.io/en/latest/snntorch.spikevision.spikedata.html#shd>`_
     - | Spikes in 700 input channels were generated 
       | using an artificial cochlea model listening 
       | to studio recordings of spoken digits from 
       | 0 to 9 in both German and English languages.
     - `Zenke Lab <https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/>`_


**Module Reference:**

.. toctree::
    :maxdepth: 2
    :glob:

    snntorch.spikevision.spikedata
