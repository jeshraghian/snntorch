==================================================================
Tutorial 1 - Spike Encoding and Visualization
==================================================================

Tutorial written by Jason K. Eshraghian (`www.jasoneshraghian.com <https://www.jasoneshraghian.com>`_)

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_

Spike Encoding and Visualization
-------------------------------------------------------------------

Introduction
--------------

In this tutorial, you will learn how to use snnTorch to:

  * convert datasets into spiking datasets using various encoding methods, 
  * how to visualise them, 
  * and how to generate random spike trains.

Install the latest PyPi distribution of snnTorch::

  $ pip install snntorch 

1. Setting up the MNIST Dataset
---------------------------------

1.1 Import packages and setup environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::
  
  import snntorch as snn
  import torch

Let's define a few variables:


:code:`data_path` will be used as the target directory for downloading the training set.

:code:`valid_split` will be used to assign data from the training set to the validation set.
*E.g., for a split of 0.1, the validation set will be made up of 10% of the train set.*

:code:`subset` is used to partition the training and test sets down by the given factor.
*E.g., for a subset of 10, a training set of 60,000 will be reduced to 6,000.*

:code:`num_steps` is the number of time steps to simulate.

::

  # Training Parameters
  batch_size=128
  data_path='/data/mnist'
  val_split = 0.1
  subset = 10
  num_classes = 10

  # Temporal Dynamics
  num_steps = 100

  # Torch Variables
  dtype = torch.float
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

1.2 Download Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MNIST does not have a specified validation set by default. So we can make a copy of the training set in :code:`mnist_val`.
We won't use :code:`mnist_val` or :code:`mnist_test` here - they're only to demonstrate creating a train-validation split.

::

  from torchvision import datasets, transforms

  # Define a transform
  transform = transforms.Compose([
              transforms.Resize((28,28)),
              transforms.Grayscale(),
              transforms.ToTensor(),
              transforms.Normalize((0,), (1,))])

  mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
  mnist_val = datasets.MNIST(data_path, train=True, download=True, transform=transform)
  mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

:code:`snntorch.utils` contains a few useful functions for modifying datasets.
A train-validation split can be created by calling :code:`valid_split`:

::

  from snntorch import utils

  mnist_train, mnist_val = utils.valid_split(mnist_train, mnist_val, val_split)


Until we actually start doing some training, we won't need large datasets.
So let's make our life simpler by reducing the size of the MNIST dataset.
We can apply :code:`data_subset` to reduce the dataset by the factor given in the argument :code:`subset`.

::

  mnist_train = utils.data_subset(mnist_train, subset)
  mnist_val = utils.data_subset(mnist_val, subset)
  mnist_test = utils.data_subset(mnist_test, subset)

To verify, we can take a look at the length of each of our datasets:

::

  >>> print(f"The size of mnist_train is {len(mnist_train)}")
  >>> print(f"The size of mnist_val is {len(mnist_val)}")
  >>> print(f"The size of mnist_test is {len(mnist_test)}")

  The size of mnist_train is 5400
  The size of mnist_val is 600
  The size of mnist_test is 1000


1.3 Create DataLoaders 
^^^^^^^^^^^^^^^^^^^^^^^^

The Dataset objects we created above load training/validation/test data into memory, and the DataLoader will fetch data from this dataset and serve it up in batches. 

DataLoaders in PyTorch are a handy interface for passing data into a network. They return an iterator divided up into mini-batches of size :code:`batch_size`.

::

  from torch.utils.data import DataLoader

  train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

2. Spike Encoding
---------------------------------

Spiking Neural Networks (SNNs) are made to exploit time-varying data. And yet, MNIST is not a time-varying dataset. 
This means that we have one of two options for passing input data into an SNN:


1. For a single training sample :math:`x^{(i)}`, directly feed the same static input features at each time step, where each element of :math:`x^{(i)}` takes on an analog value :math:`x^{(i)}_n ∈ [0, 1]. n` spans the number of inputs. 
 This is like converting MNIST into a static, unchanging video.

   .. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_1_static.png?raw=true
            :align: center
            :width: 800


2. Convert the input into a spike train of sequence length :code:`num_steps`, where :math:`x^{(i)}` takes on a discrete value :math:`x^{(i)} ∈ {0, 1}`.
In this case, MNIST would become a time-varying sequence of spikes that are related to the original image.

    .. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_2_spikeinput.png?raw=true
              :align: center
              :width: 800

The first method is quite straightforward, so let's consider (2) in more detail.

The module :code:`snntorch.spikegen` contains a series of functions that simplify the conversion of data into spikes. There are currently three options available for spike generation in :code:`snntorch`:

1. Rate coding: `spikegen.rate <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html#snntorch.spikegen.rate>`_
2. Latency coding: `spikegen.latency <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html#snntorch.spikegen.latency>`_
3. Delta modulation: `spikegen.delta <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html#snntorch.spikegen.delta>`_

*Rate coding* uses input features to determine spiking **frequency**. *Latency coding* uses input features to determine spike **timing**. *Delta modulation* uses the temporal **change** of input features to generate spikes.

2.1 Rate coding of MNIST
^^^^^^^^^^^^^^^^^^^^^^^^^

Each input feature is used as the probability an event occurs, sampled from a binomial distribution. Formally, **X** is a matrix of random variables and each element of **X**, :math:`X^{(i)}`, is sampled from the distribution using the original feature as the probability that a '1' occurs: :math:`X^{(i)}\sim B(n=1, p=x^{(i)})` where the
**expected value** :math:`𝔼[X^{(i)}]=x^{(i)}` is simply the probability that a spike is generated at any given time step.

For an MNIST image, this probability corresponds to the pixel value. A white pixel corresponds to a 100% probability of spiking, and a black pixel will never generate a spike.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_3_spikeconv.png?raw=true
        :align: center
        :width: 1000


::

  from snntorch import spikegen

  # Iterate through minibatches
  data = iter(train_loader)
  data_it, targets_it = next(data)
  data_it = data_it.to(device)
  targets_it = targets_it.to(device)

  # Spiking Data
  spike_data = spikegen.rate(data_it, num_steps=num_steps, gain=1, offset=0)
      

As you can see, :code:`spikegen.rate` takes a few arguments that can modify spiking probability:

* :code:`gain` multiplies the input by the given factor, and
* :code:`offset` applies a level-shift to the input.

If the result falls outside of [0,1], this no longer represents a probability. The result will automatically be clipped such that the feature represents a probability.

.. note::
  
There are numerous other options available for data conversion available. Fore more detail on converting input features (and targets) to spikes, please [refer to the documentation of `snntorch.spikegen` here](https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html#snntorch.spikegen.targets_to_spikes).

The structure of the input data is :code:`[num_steps x batch_size x input dimensions]`:

::

  >>> print(spike_data.size())

  torch.Size([100, 128, 1, 28, 28])

2.2 Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^

2.2.1 Animations
""""""""""""""""""

snnTorch contains a module :code:`snntorch.spikeplot` that can simplify the process of visualizing, plotting, and animating spiking neurons.

::

  import matplotlib.pyplot as plt
  import snntorch.spikeplot as splt
  from IPython.display import HTML

To plot one sample of data, we have to index into the batch (B) dimension of :code:`spike_data`, :code:`[T x B x 1 x 28 x 28]`:

::

  >>> spike_data_sample = spike_data[:, 0, 0]
  >>> print(spike_data_sample.size())

  torch.Size([100, 28, 28])

:code:`spikeplot.animator` makes it super simple to animate 2-D data:

::

  >>> fig, ax = plt.subplots()
  >>> anim = splt.animator(spike_data_sample, fig, ax)

  >>> HTML(anim.to_html5_video())

.. raw:: html

  <center>
    <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/splt.animator.mp4?raw=true"></video>
  </center>

::

  # If you're feeling sentimental, you can save the animation: .gif, .mp4 etc.
  anim.save("spike_mnist_test.mp4")

The associated target label can be indexed as follows:

::

  >>> print(f"The corresponding target is: {targets_it[0]}")

  The corresponding target is: 3

As a matter of interest, let's do that again but with 25% of the gain to promote sparsity. This time, we won't bother passing the targets into :code:`spikegen.rate`, as we don't need it.

::

  spike_data = spikegen.rate(data_it, num_steps, gain=0.25)

  spike_data_sample2 = spike_data[:, 0, 0]
  fig, ax = plt.subplots()
  anim = splt.animator(spike_data_sample2, fig, ax)
  HTML(anim.to_html5_video())

.. raw:: html

  <center>
    <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/splt.animator-25.mp4?raw=true"></video>
  </center>



:: 

  # Uncomment for optional save
  # anim.save("spike_mnist_test2.mp4")

Now let's average the spikes out over time and reconstruct the input images.

::

  plt.figure(facecolor="w")
  plt.subplot(1,2,1)
  plt.imshow(spike_data_sample.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
  plt.axis('off')
  plt.title('Gain = 1')

  plt.subplot(1,2,2)
  plt.imshow(spike_data_sample2.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
  plt.axis('off')
  plt.title('Gain = 0.25')

  plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/gain.png?raw=true
        :align: center
        :width: 300

The case where :code:`gain=0.25` is lighter than where :code:`gain=1`, as spiking probability has been reduced by a factor of x4.

2.2.2 Raster Plots
"""""""""""""""""""

Alternatively, we can generate a raster plot of an input sample. This requires reshaping our sample into a 2-D tensor, where the number of steps is the first dimension. We then pass this sample into the function :code:`spikeplot.raster`. 

::

  # Reshape
  spike_data_sample2 = spike_data_sample2.reshape((num_steps, -1))

  # raster plot
  fig = plt.figure(facecolor="w", figsize=(10, 5))
  ax = fig.add_subplot(111)
  splt.raster(spike_data_sample2, ax, s=1.5, c="black")

  plt.title("Input Layer")
  plt.xlabel("Time step")
  plt.ylabel("Neuron Number")
  plt.show()


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster.png?raw=true
        :align: center
        :width: 600

We can also index into one single neuron. Below, we are indexing into the 210th neuron.
Depending on your input data, you may need to index into a few different neurons between 0 & 784 before finding one that spikes.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster1.png?raw=true
        :align: center
        :width: 400

The idea of rate coding is actually quite controversial. Multiple spikes are needed to achieve any sort of task, and each spike consumes power. It is unlikely to be the only mechanism within the brain, which is both resource-constrained and highly efficient.

We know that the reaction time of a human is around 250ms. If the averaging firing rate of a neuron in the human brain is on the order of 10Hz, then we can only process about 2 spikes within our reaction timescale.

On the other hand, biological neurons are somewhat stochastic. In fact,  neurons fail to fire around 70% of the time that our idealized models would have us believe. Spike rate coding offsets the power disadvantage by showing huge noise robustness: it's fine if some of the spikes fail to generate, because there will be plenty more where they came from.

Rate coding is almost certainly working in conjunction with other encoding schemes in the brain. We'll consider these other encoding mechanisms in the following sections. 

This covers the :code:`spikegen.rate` function. Further information `can be found in the documentation here <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html>`_.

2.3 Latency Coding of MNIST
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Temporal codes capture information about the precise firing time of neurons; a single spike carries much more meaning than in rate codes which rely on firing frequency.

While this opens up more susceptibility to noise, it can also decrease the power consumed by the hardware running SNN algorithms by orders of magnitude. 

:code:`spikegen.latency` is a function that allows each input to fire at most **once** during the full time sweep.
Features closer to :code:`1` will fire earlier and features closer to :code:`0` will fire later. I.e., in our MNIST case, bright pixels will fire earlier and dark pixels will fire later. 

By default, spike timing is calculated by setting the input feature as a current injection :math:`I_{in}` into an RC circuit. This current moves charge onto the capacitor, which increases :math:`V(t)`. We assume that there is a trigger voltage, :math:`V_{thr}`, which once reached, generates a spike. The question then becomes: *for a given input current (and equivalently, input feature), how long does it take for a spike to be generated?*

Starting with Kirchhoff's current law, :math:`I_{in} = I_R + I_C`, the rest of the derivation leads us to a logarithmic relationship between time and the input. 

If you've forgotten circuit theory and/or the following means nothing to you, then don't worry! All that matters is: **big** input means **fast** spike; **small** input means **late** spike.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_4_latencyrc.png?raw=true
        :align: center
        :width: 500

::

  spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01)

Some of the arguments include:

* :code:`tau`: by default, the input features are treated as a constant current injected into an RC circuit. :code:`tau` is the RC time constant of the circuit. A higher :code:`tau` will induce slower firing.
* :code:`threshold`: the membrane potential the RC circuit must charge to before it can fire. All features below the threshold are saturated.


2.3.1 Raster Plot
"""""""""""""""""""
We'll start with a raster this time.

::

  fig = plt.figure(facecolor="w", figsize=(10, 5))
  ax = fig.add_subplot(111)
  splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")

  plt.title("Input Layer")
  plt.xlabel("Time step")
  plt.ylabel("Neuron Number")
  plt.show()

  # optional save
  # fig.savefig('destination_path.png', format='png', dpi=300)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster2.png?raw=true
        :align: center
        :width: 600

To make sense of your raster plot, you'll notice that high intensity features fire first, whereas low intensity features fire last:

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_5_latencyraster.png?raw=true
        :align: center
        :width: 800

The logarithmic code coupled with the lack of diverse input values (i.e., the lack of midtone/grayscale features) causes significant clustering in two areas of the plot.
The bright pixels induce firing at the start of the run, and the dark pixels at the end.
We can increase :code:`tau` to slow down our spike times, or we can linearize the data by setting the optional argument :code:`linear=True`.

::

  spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, linear=True)

  fig = plt.figure(facecolor="w", figsize=(10, 5))
  ax = fig.add_subplot(111)
  splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")
  plt.title("Input Layer")
  plt.xlabel("Time step")
  plt.ylabel("Neuron Number")
  plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster3.png?raw=true
        :align: center
        :width: 600

The spread of firing times is much more evenly distributed now. This is achieved by simply linearizing the logarithmic equation according to the rules shown below. Unlike the RC model, there's no physical basis for the model. It's just simpler.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_6_latencylinear.png?raw=true
        :align: center
        :width: 600

But notice all firing occurs within the first ~5 time steps, whereas the simulation range is 100 time steps.
This indicates that we have a lot of redundant time steps doing nothing. This can be solved by either increasing :code:`tau` to slow down the time constant, or setting the optional argument :code:`normalize=True` to span the full range of :code:`num_steps`.

::

  spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01,
                                normalize=True, linear=True)

  fig = plt.figure(facecolor="w", figsize=(10, 5))
  ax = fig.add_subplot(111)
  splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")

  plt.title("Input Layer")
  plt.xlabel("Time step")
  plt.ylabel("Neuron Number")
  plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster4.png?raw=true
        :align: center
        :width: 600

One major advantage of latency coding over rate coding is the increased sparsity of spikes. If neurons are constrained to firing a maximum of once over the time course of interest, then this promotes low-power operation.

In the scenario shown above, a majority of the spikes occur at the final time step, where the input features fall below the threshold. In a sense, the background of the image holds no useful information to us. 

We can remove these redundant features by setting :code:`clip=True`.

::

  spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, 
                                clip=True, normalize=True, linear=True)

  fig = plt.figure(facecolor="w", figsize=(10, 5))
  ax = fig.add_subplot(111)
  splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")

  plt.title("Input Layer")
  plt.xlabel("Time step")
  plt.ylabel("Neuron Number")
  plt.show()


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster5.png?raw=true
        :align: center
        :width: 600

That looks much better!


2.3.2 Animation
"""""""""""""""""""
We will run the exact same code block as before to create an animation.

::

  >>> spike_data_sample = spike_data[:, 0, 0]
  >>> print(spike_data_sample.size())
  torch.Size([100, 28, 28])

::

  fig, ax = plt.subplots()
  anim = splt.animator(spike_data_sample, fig, ax)
  HTML(anim.to_html5_video())

.. raw:: html

  <center>
    <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/splt.animator2.mp4?raw=true"></video>
  </center>

This animation is obviously much tougher to make out in video form, but a keen eye will be able to catch a glimpse of the initial frame where most of the spikes occur.
We can index into the corresponding target value to check what value it is.

::

  # Save output: .gif, .mp4 etc.
  # anim.save("mnist_latency.gif")

::

  >>> print(targets_it[0])
  tensor(4, device='cuda:0')


2.4 Delta Modulation
^^^^^^^^^^^^^^^^^^^^^


There are theories that the retina is adaptive: it will only process information when there is something new to process. If there is no change in your field of view, then your photoreceptor cells will be much lesss prone to firing. 

That is to say: **biology is event-driven**. Our neurons thrive on change.

As a nifty example, a few researchers have dedicated their lives to designing retina-inspired image sensors, for example, the `Dynamic Vision Sensor <https://ieeexplore.ieee.org/abstract/document/7128412/>`_. Although `the attached link is from over a decade ago, the work in this video <https://www.youtube.com/watch?v=6eOM15U_t1M&ab_channel=TobiDelbruck>`_ was clearly ahead of its time.

Delta modulation is based on event-driven spiking. The :code:`snntorch.delta` function accepts a time-series tensor as input. It takes the difference between each subsequent feature across all time steps. By default, if the difference is both *positive* and *greater* than the threshold :math:`V_{thr}`, a spike is generated:

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_7_delta.png?raw=true
        :align: center
        :width: 600

To illustrate, let's first come up with a contrived example where we create our own input tensor.

::

    # Create a tensor with some fake time-series data
    data = torch.Tensor([0, 1, 0, 2, 8, -20, 20, -5, 0, 1, 0])

    # Plot the tensor
    plt.plot(data)

    plt.title("Some fake time-series data")
    plt.xlabel("Time step")
    plt.ylabel("Voltage (mV)")
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/fake_data.png?raw=true
      :align: center
      :width: 300

Let's pass the above tensor into the :code:`spikegen.delta` function, with an arbitrarily selected :code:`threshold=4`:

::

    # Convert data
    spike_data = spikegen.delta(data, threshold=4)

    # Create fig, ax
    fig = plt.figure(facecolor="w", figsize=(8, 1))
    ax = fig.add_subplot(111)

    # Raster plot of delta converted data
    splt.raster(spike_data, ax, c="black")

    plt.title("Input Neuron")
    plt.xlabel("Time step")
    plt.yticks([])
    plt.xlim(0, len(data))
    plt.show()

    
.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/delta.png?raw=true
        :align: center
        :width: 400

There are three time steps where the difference between :math:`data[T]` and :math:`data[T+1]` is greater than or equal to :math:`V_{thr}=4`. This means there are three on-spikes. 

The large dip to :math:`-20` has not been captured in our spikes. It might be the case that our data cares about negative swings as well, in which case we can enable the optional argument :code:`off_spike=True`.

::

  # Convert data
  spike_data = spikegen.delta(data, threshold=4, off_spike=True)

  # Create fig, ax
  fig = plt.figure(facecolor="w", figsize=(8, 1))
  ax = fig.add_subplot(111)

  # Raster plot of delta converted data
  splt.raster(spike_data, ax, c="black")

  plt.title("Input Neuron")
  plt.xlabel("Time step")
  plt.yticks([])
  plt.xlim(0, len(data))
  plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/delta2.png?raw=true
        :align: center
        :width: 400

We've generated additional spikes, but this isn't actually the full picture! 

If we print out the tensor, we will discover that we have actually generated "off-spikes". These spikes take on a value of :math:`-1`.

::

  >>> print(spike_data)
  tensor([ 0.,  0.,  0.,  0.,  1., -1.,  1., -1.,  1.,  0.,  0.])

Although we have only shown :code:`spikegen.delta` on a fake sample of data, the true intention is to pass in time-series data and only generate an output when there has been a sufficiently large event. 

That wraps up the three main spike conversion functions! There are still additional features to each of the three conversion techniques that have not been detailed in this tutorial. We recommend `referring to the documentation for a deeper dive <https://snntorch.readthedocs.io/en/latest/_modules/snntorch/spikegen.html>`_.

3. Spike Generation
---------------------------------

Now what if we don't actually have any data to start with? 
Say we just want a randomly generated spike train from scratch.
:code:`spikegen.rate` has a nested function, :code:`rate_conv` which takes care of converting features into spikes.

All we have to do is initialize a randomly generated :code:`torch.Tensor` to pass in.

::

  # Create a random spike train
  spike_prob = torch.rand((num_steps, 28, 28), device=device, dtype=dtype) * 0.5  
  spike_rand = spikegen.rate_conv(spike_prob)

3.1 Animation
^^^^^^^^^^^^^^^
  
::

  fig, ax = plt.subplots()
  anim = splt.animator(spike_rand, fig, ax)

  HTML(anim.to_html5_video())


.. raw:: html

  <center>
    <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/rand_spikes.mp4?raw=true"></video>
  </center>


::

  # Save output: .gif, .mp4 etc.
  # anim.save("random_spikes.gif")


3.2 Raster
^^^^^^^^^^^^^

::

  fig = plt.figure(facecolor="w", figsize=(10, 5))
  ax = fig.add_subplot(111)
  splt.raster(spike_rand[:, 0].view(num_steps, -1), ax, s=25, c="black")

  plt.title("Input Layer")
  plt.xlabel("Time step")
  plt.ylabel("Neuron Number")
  plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/rand_raster.png?raw=true
      :align: center
      :width: 600

Conclusion
-----------
That's it for spike conversion and generation. 
This approach generalizes beyond images, to single-dimensional and multi-dimensional tensors.

For reference, the documentation for :code:`spikegen` can be found `at this link <https://snntorch.readthedocs.io/en/latest/_modules/snntorch/spikegen.html>`_ and for :code:`spikeplot`, `at the link here <https://snntorch.readthedocs.io/en/latest/_modules/snntorch/spikeplot.html>`_

In the next tutorial, you will learn the basics of spiking neurons and how to use them. Following that, you will be equipped with the tools to train your own spiking neural network in tutorial 3. 