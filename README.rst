================
Introduction
================


.. image:: https://github.com/jeshraghian/snntorch/actions/workflows/build.yml/badge.svg
        :target: https://snntorch.readthedocs.io/en/latest/?badge=latest

.. image:: https://readthedocs.org/projects/snntorch/badge/?version=latest
        :target: https://snntorch.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/discord/906036932725841941
        :target: https://discord.gg/cdZb5brajb
        :alt: Discord

.. image:: https://img.shields.io/pypi/v/snntorch.svg
         :target: https://pypi.python.org/pypi/snntorch

.. image:: https://img.shields.io/conda/vn/conda-forge/snntorch.svg
        :target: https://anaconda.org/conda-forge/snntorch

.. image:: https://static.pepy.tech/personalized-badge/snntorch?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads
        :target: https://pepy.tech/project/snntorch

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
   * - `snntorch.backprop <https://snntorch.readthedocs.io/en/latest/snntorch.backprop.html>`_
     - variations of backpropagation commonly used with SNNs
   * - `snntorch.functional <https://snntorch.readthedocs.io/en/latest/snntorch.functional.html>`_
     - common arithmetic operations on spikes, e.g., loss, regularization etc.
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
If you find snnTorch useful in your work, please cite the following source:

`Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu “Training
Spiking Neural Networks Using Lessons From Deep Learning”. arXiv preprint arXiv:2109.12894,
September 2021. <https://arxiv.org/abs/2109.12894>`_

.. code-block:: bash

  @article{eshraghian2021training,
          title   =  {Training spiking neural networks using lessons from deep learning},
          author  =  {Eshraghian, Jason K and Ward, Max and Neftci, Emre and Wang, Xinxin 
                      and Lenz, Gregor and Dwivedi, Girish and Bennamoun, Mohammed and 
                     Jeong, Doo Seok and Lu, Wei D},
          journal = {arXiv preprint arXiv:2109.12894},
          year    = {2021}
  }

Let us know if you are using snnTorch in any interesting work, research or blogs, as we would love to hear more about it! Reach out at snntorch@gmail.com.

Requirements 
^^^^^^^^^^^^^^^^^^^^^^^^
The following packages need to be installed to use snnTorch:

* torch >= 1.1.0
* numpy >= 1.17
* pandas
* matplotlib
* math

They are automatically installed if snnTorch is installed using the pip command. Ensure the correct version of torch is installed for your system to enable CUDA compatibility. 

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
    

API & Examples 
^^^^^^^^^^^^^^^^^^^^^^^^
A complete API is available `here`_. Examples, tutorials and Colab notebooks are provided.

.. _here: https://snntorch.readthedocs.io/


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

  num_steps = 25 # number of time steps
  batch_size = 1 
  beta = 0.5  # neuron decay rate 
  spike_grad = surrogate.fast_sigmoid()

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

  # random input data
  data_in = torch.rand(num_steps, batch_size, 1, 28, 28)

  spike_recording = []

  for step in range(num_steps):
      spike, state = net(data_in[step])
      spike_recording.append(spike)


If you're feeling lazy and want the training process to be taken care of:

.. code-block:: python
    
    import snntorch.functional as SF
    from snntorch import backprop

    # correct class should fire 80% of the time
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))

    # train for one epoch using the backprop through time algorithm
    # assume train_loader is a DataLoader with time-varying input
    avg_loss = backprop.BPTT(net, train_loader, optimizer=optimizer, 
                            num_steps=num_steps, criterion=loss_fn)  


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
