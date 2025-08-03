================
Introduction
================

.. |build| image:: https://github.com/jeshraghian/snntorch/actions/workflows/build.yml/badge.svg
   :target: https://snntorch.readthedocs.io/en/latest/?badge=latest

.. |docs| image:: https://readthedocs.org/projects/snntorch/badge/?version=latest
   :target: https://snntorch.readthedocs.io/en/latest/?badge=latest

.. |discord| image:: https://img.shields.io/discord/906036932725841941
   :target: https://discord.gg/cdZb5brajb

.. |pypi| image:: https://img.shields.io/pypi/v/snntorch.svg
   :target: https://pypi.python.org/pypi/snntorch

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/snntorch.svg
   :target: https://anaconda.org/conda-forge/snntorch

.. |downloads| image:: https://static.pepy.tech/personalized-badge/snntorch?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads
   :target: https://pepy.tech/project/snntorch

.. |neuromorphiccomputing| image:: https://img.shields.io/badge/Collaboration_Network-Open_Neuromorphic-blue
   :target: https://open-neuromorphic.org/neuromorphic-computing/

|build| |docs| |discord| |pypi| |conda| |downloads| |neuromorphiccomputing|


The brain is the perfect place to look for inspiration to develop more efficient neural networks. One of the main differences with modern deep learning is that the brain encodes information in spikes rather than continuous activations. 
snnTorch is a Python package for performing gradient-based learning with spiking neural networks.
It extends the capabilities of PyTorch, taking advantage of its GPU accelerated tensor 
computation and applying it to networks of spiking neurons. Pre-designed spiking neuron models are seamlessly integrated within the PyTorch framework and can be treated as recurrent activation units. 


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/spike_excite_alpha_ps2.gif?raw=true
        :align: center
        :width: 800

If you like this project, please consider starring ⭐ this repo as it is the easiest and best way to support it.

If you have issues, comments, or are looking for advice on training spiking neural networks, you can open an issue, a discussion, or chat in our `discord <https://discord.gg/cdZb5brajb>`_ channel.

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
   * - `snntorch.export_nir <https://snntorch.readthedocs.io/en/latest/snntorch.export_nir.html>`_
     - enables exporting to other SNN libraries via `NIR <https://nnir.readthedocs.io/en/latest/>`_
   * - `snntorch.functional <https://snntorch.readthedocs.io/en/latest/snntorch.functional.html>`_
     - common arithmetic operations on spikes, e.g., loss, regularization etc.
   * - `snntorch.import_nir <https://snntorch.readthedocs.io/en/latest/snntorch.import_nir.html>`_
     - enables importing from other SNN libraries via `NIR <https://nnir.readthedocs.io/en/latest/>`_
   * - `snntorch.spikegen <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html>`_
     - a library for spike generation and data conversion
   * - `snntorch.spikeplot <https://snntorch.readthedocs.io/en/latest/snntorch.spikeplot.html>`_
     - visualization tools for spike-based data using matplotlib and celluloid
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
If you find snnTorch useful in your work, please cite the following source:

`Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu “Training
Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9)
September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. code-block:: bash

  @article{eshraghian2021training,
          title   =  {Training spiking neural networks using lessons from deep learning},
          author  =  {Eshraghian, Jason K and Ward, Max and Neftci, Emre and Wang, Xinxin 
                      and Lenz, Gregor and Dwivedi, Girish and Bennamoun, Mohammed and 
                     Jeong, Doo Seok and Lu, Wei D},
          journal = {Proceedings of the IEEE},
          volume  = {111},
          number  = {9},
          pages   = {1016--1054},
          year    = {2023}
  }

Let us know if you are using snnTorch in any interesting work, research or blogs, as we would love to hear more about it! Reach out at snntorch@gmail.com.

Requirements 
^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch should be installed to use snnTorch. Ensure the correct version of torch is installed for your system to enable CUDA compatibility.

The following packages are automatically installed if using the pip command:

* numpy
* pandas

The following packages are required for using `export_nir` and `import_nir`:

* nir>=1.0.6
* nirtorch>=2.0.5

The following packages are required for using `spikeplot`:

* matplotlib

Installation
^^^^^^^^^^^^^^^^^^^^^^^^

Run the following to install:

.. code-block:: bash

  $ python
  $ pip install snntorch

To install snnTorch from source instead::

  $ git clone https://github.com/jeshraghian/snnTorch
  $ cd snntorch
  $ python setup.py install


To install snntorch with conda::

    $ conda install -c conda-forge snntorch

To install for an Intelligent Processing Units (IPU) based build using Graphcore's accelerators::

  $ pip install snntorch-ipu
    

API & Examples 
^^^^^^^^^^^^^^^^^^^^^^^^
A complete API is available `here <https://snntorch.readthedocs.io/>`__. Examples, tutorials and Colab notebooks are provided.



Quickstart 
^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/quickstart.ipynb


Here are a few ways you can get started with snnTorch:


* `Quickstart Notebook (Opens in Colab)`_

* `The API Reference`_ 

* `Examples`_

* `Tutorials`_

.. _Quickstart Notebook (Opens in Colab): https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/quickstart.ipynb
.. _The API Reference: https://snntorch.readthedocs.io/
.. _Examples: https://snntorch.readthedocs.io/en/latest/examples.html
.. _Tutorials: https://snntorch.readthedocs.io/en/latest/tutorials/index.html


For a quick example to run snnTorch, see the following snippet, or test the quickstart notebook:


.. code-block:: python

  import torch, torch.nn as nn
  import snntorch as snn
  from snntorch import surrogate
  from snntorch import utils

  num_steps = 25 # number of time steps
  batch_size = 1 
  beta = 0.5  # neuron decay rate 
  spike_grad = surrogate.fast_sigmoid() # surrogate gradient

  net = nn.Sequential(
        nn.Conv2d(1, 8, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
        nn.Conv2d(8, 16, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 10),
        snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True)
        )

  data_in = torch.rand(num_steps, batch_size, 1, 28, 28) # random input data
  spike_recording = [] # record spikes over time
  utils.reset(net) # reset/initialize hidden states for all neurons

  for step in range(num_steps): # loop over time
      spike, state = net(data_in[step]) # one time step of forward-pass
      spike_recording.append(spike) # record spikes in list


A Deep Dive into SNNs
^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you wish to learn all the fundamentals of training spiking neural networks, from neuron models, to the neural code, up to backpropagation, the snnTorch tutorial series is a great place to begin.
It consists of interactive notebooks with complete explanations that can get you up to speed.


.. list-table::
   :widths: 20 60 30
   :header-rows: 1

   * - Tutorial
     - Title
     - Colab Link
   * - `Tutorial 1 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html>`_
     - Spike Encoding with snnTorch
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb

   * - `Tutorial 2 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html>`_
     - The Leaky Integrate and Fire Neuron
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_2_lif_neuron.ipynb

   * - `Tutorial 3 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html>`_
     -  A Feedforward Spiking Neural Network
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_3_feedforward_snn.ipynb


   * - `Tutorial 4 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_4.html>`_
     -  2nd Order Spiking Neuron Models (Optional)
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_4_advanced_neurons.ipynb

  
   * - `Tutorial 5 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html>`_
     -  Training Spiking Neural Networks with snnTorch
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_5_FCN.ipynb
   

   * - `Tutorial 6 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html>`_
     - Surrogate Gradient Descent in a Convolutional SNN
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_6_CNN.ipynb

   * - `Tutorial 7 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html>`_
     - Neuromorphic Datasets with Tonic + snnTorch
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_7_neuromorphic_datasets.ipynb

.. list-table::
   :widths: 70 40
   :header-rows: 1

   * - Advanced Tutorials
     - Colab Link

   * - `Population Coding <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_pop.html>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_pop.ipynb

   * - `Regression: Part I - Membrane Potential Learning with LIF Neurons <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_1.html>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_1.ipynb

   * - `Regression: Part II - Regression-based Classification with Recurrent LIF Neurons <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_2.html>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_2.ipynb

   * - `Accelerating snnTorch on IPUs <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_ipu_1.html>`_
     -       —

Contributing
^^^^^^^^^^^^^^^^^^^^^^^^
If you're ready to contribute to snnTorch, instructions to do so can be `found here`_.

.. _found here: https://snntorch.readthedocs.io/en/latest/contributing.html

Acknowledgments
^^^^^^^^^^^^^^^^^^^^^^^^
snnTorch is currently maintained by the `UCSC Neuromorphic Computing Group <https://ncg.ucsc.edu>`_. It was initially developed by `Jason K. Eshraghian`_ in the `Lu Group (University of Michigan)`_. 

Additional contributions were made by `Vincent Sun <https://github.com/vinniesun>`_, `Peng Zhou <https://github.com/pengzhouzp>`_, `Ridger Zhu <https://github.com/ridgerchu>`_, `Alexander Henkes <https://github.com/ahenkes1>`_, `Steven Abreu <https://github.com/stevenabreu7>`_, Xinxin Wang, Sreyes Venkatesh, `gekkom <https://github.com/gekkom>`_, and Emre Neftci.

.. _Jason K. Eshraghian: https://jasoneshraghian.com
.. _Lu Group (University of Michigan): https://lugroup.engin.umich.edu/


License & Copyright
^^^^^^^^^^^^^^^^^^^^^^^^
snnTorch source code is published under the terms of the MIT License. 
snnTorch's documentation is licensed under a Creative Commons Attribution-Share Alike 3.0 Unported License (`CC BY-SA 3.0 <https://creativecommons.org/licenses/by-sa/3.0/>`_).
