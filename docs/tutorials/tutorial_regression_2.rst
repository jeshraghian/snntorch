=============================
Regression with SNNs: Part II
=============================

Regression-based Classification with Recurrent Leaky Integrate-and-Fire Neurons
-------------------------------------------------------------------------------

Tutorial written by Alexander Henkes (`ORCID <https://orcid.org/0000-0003-4615-9271>`_) and Jason K. Eshraghian (`ncg.ucsc.edu <https://ncg.ucsc.edu>`_)


.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_2.ipynb


This tutorial is based on the following papers on nonlinear regression
and spiking neural networks. If you find these resources or code useful
in your work, please consider citing the following sources:

   `Alexander Henkes, Jason K. Eshraghian, and Henning Wessels. “Spiking
   neural networks for nonlinear regression”, arXiv preprint
   arXiv:2210.03515, October 2022. <https://arxiv.org/abs/2210.03515>`_

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_2.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


In the regression tutorial series, you will learn how to use snnTorch to
perform regression using a variety of spiking neuron models, including:

-  Leaky Integrate-and-Fire (LIF) Neurons
-  Recurrent LIF Neurons
-  Spiking LSTMs

An overview of the regression tutorial series:

-  Part I will train the membrane potential of a LIF neuron to follow a
   given trajectory over time.
-  Part II (this tutorial) will use LIF neurons with recurrent feedback
   to perform classification using regression-based loss functions
-  Part III will use a more complex spiking LSTM network instead to
   train the firing time of a neuron.


::

    !pip install snntorch --quiet

::

    # imports
    import snntorch as snn
    from snntorch import functional as SF
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torch.nn.functional as F
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    import tqdm

1. Classification as Regression
--------------------------------------------

In conventional deep learning, we often calculate the Cross Entropy Loss
to train a network to do classification. The output neuron with the
highest activation is thought of as the predicted class.

In spiking neural nets, this may be interpreted as the class that fires
the most spikes. I.e., apply cross entropy to the total spike count (or
firing frequency). The effect of this is that the predicted class will
be maximized, while other classes aim to be suppressed.

The brain does not quite work like this. SNNs are sparsely activated,
and while approaching SNNs with this deep learning attitude may lead to
optimal accuracy, it’s important not to ‘overfit’ too much to what the
deep learning folk are doing. After all, we use spikes to achieve better
power efficiency. Good power efficiency relies on sparse spiking
activity.

In other words, training bio-inspired SNNs using deep learning tricks
does not lead to brain-like activity.

So what can we do?

We will focus on recasting classification problems into regression
tasks. This is done by training the predicted neuron to fire a given
number of times, while incorrect neurons are trained to still fire a
given number of times, albeit less frequently.

This contrasts with cross-entropy which would try to drive the correct
class to fire at *all* time steps, and incorrect classes to not fire at
all.

As with the previous tutorial, we can use the mean-square error to
achieve this. Recall the form of the mean-square error loss:

.. math:: \mathcal{L}_{MSE} = \frac{1}{n}\sum_{i=1}^n(y_i-\hat{y_i})^2

where :math:`y` is the target and :math:`\hat{y}` is the predicted
value.

To apply MSE to the spike count, assume we have :math:`n` output neurons
in a classification problem, where :math:`n` is the number of possible
classes. :math:`\hat{y}_i` is now the total number of spikes the
:math:`i^{th}` output neuron emits over the full simulation runtime.

Given that we have :math:`n` neurons, this means that :math:`y` and
:math:`\hat{y}` must be vectors with :math:`n` elements, and our loss
will sum the independent MSE losses of each neuron.

1.1 A Theoretical Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider a simulation of 10 time steps. Say we wish for the correct
neuron class to fire 8 times, and the incorrect classes to fire 2 times.
Assume :math:`y_1` is the correct class:

.. math::  y = \begin{bmatrix} 8 \\ 2 \\ \vdots \\ 2 \end{bmatrix},  \hat{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}

The element-wise MSE is taken to generate :math:`n` loss components,
which are all summed together to generate a final loss.

2. Recurrent Leaky Integrate-and-Fire Neurons
------------------------------------------------

Neurons in the brain have a ton of feedback connections. And so the SNN
community have been exploring the dynamics of networks that feed output
spikes back to the input. This is in addition to the recurrent dynamics
of the membrane potential.

There are a few ways to construct recurrent leaky integrate-and-fire
(``RLeaky``) neurons in snnTorch. Refer to the
`docs <https://snntorch.readthedocs.io/en/latest/snn.neurons_rleaky.html>`__
for an exhaustive description of the neuron’s hyperparameters. Let’s see
a few examples.

2.1 RLIF Neurons with 1-to-1 connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/regression2/reg2-1.jpg?raw=true
        :align: center
        :width: 400

This assumes each neuron feeds back its output spikes into itself, and
only itself. There are no cross-coupled connections between neurons in
the same layer.

::

    beta = 0.9 # membrane potential decay rate
    num_steps = 10 # 10 time steps
    
    rlif = snn.RLeaky(beta=beta, all_to_all=False) # initialize RLeaky Neuron
    spk, mem = rlif.init_rleaky() # initialize state variables
    x = torch.rand(1) # generate random input
    
    spk_recording = []
    mem_recording = []
    
    # run simulation
    for step in range(num_steps):
      spk, mem = rlif(x, spk, mem)
      spk_recording.append(spk)
      mem_recording.append(mem)

By default, ``V`` is a learnable parameter that initializes to :math:`1`
and will be updated during the training process. If you wish to disable
learning, or use your own initialization variables, then you may do so
as follows:

::

    rlif = snn.RLeaky(beta=beta, all_to_all=False, learn_recurrent=False) # disable learning of recurrent connection
    rlif.V = torch.rand(1) # set this to layer size
    print(f"The recurrent weight is: {rlif.V.item()}")

2.2 RLIF Neurons with all-to-all connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.2.1 Linear feedback
............................


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/regression2/reg2-2.jpg?raw=true
        :align: center
        :width: 400

By default, ``RLeaky`` assumes feedback connections where all spikes
from a given layer are first weighted by a feedback layer before being
passed to the input of all neurons. This introduces more parameters, but
it is thought this helps with learning time-varying features in data.

::

    beta = 0.9 # membrane potential decay rate
    num_steps = 10 # 10 time steps
    
    rlif = snn.RLeaky(beta=beta, linear_features=10)  # initialize RLeaky Neuron
    spk, mem = rlif.init_rleaky() # initialize state variables
    x = torch.rand(10) # generate random input
    
    spk_recording = []
    mem_recording = []
    
    # run simulation
    for step in range(num_steps):
      spk, mem = rlif(x, spk, mem)
      spk_recording.append(spk)
      mem_recording.append(mem)

You can disable learning in the feedback layer with
``learn_recurrent=False``.

2.2.2 Convolutional feedback
........................................................

If you are using a convolutional layer, this will throw an error because
it does not make sense for the output spikes (3-dimensional) to be
projected into 1-dimension by a ``nn.Linear`` feedback layer.

To address this, you must specify that you are using a convolutional
feedback layer:

::

    beta = 0.9 # membrane potential decay rate
    num_steps = 10 # 10 time steps
    
    rlif = snn.RLeaky(beta=beta, conv2d_channels=3, kernel_size=(5,5))  # initialize RLeaky Neuron
    spk, mem = rlif.init_rleaky() # initialize state variables
    x = torch.rand(3, 32, 32) # generate random 3D input
    
    spk_recording = []
    mem_recording = []
    
    # run simulation
    for step in range(num_steps):
      spk, mem = rlif(x, spk, mem)
      spk_recording.append(spk)
      mem_recording.append(mem)

To ensure the output spike dimension matches the input dimensions,
padding is automatically applied.

If you have exotically shaped data, you will need to construct your own
feedback layers manually.

3. Construct Model
------------------------

Let’s train a couple of models using ``RLeaky`` layers. For speed, we
will train a model with linear feedback.

::

    class Net(torch.nn.Module):
        """Simple spiking neural network in snntorch."""
    
        def __init__(self, timesteps, hidden, beta):
            super().__init__()
            
            self.timesteps = timesteps
            self.hidden = hidden
            self.beta = beta
    
            # layer 1
            self.fc1 = torch.nn.Linear(in_features=784, out_features=self.hidden)
            self.rlif1 = snn.RLeaky(beta=self.beta, linear_features=self.hidden)
    
            # layer 2
            self.fc2 = torch.nn.Linear(in_features=self.hidden, out_features=10)
            self.rlif2 = snn.RLeaky(beta=self.beta, linear_features=10)
    
        def forward(self, x):
            """Forward pass for several time steps."""
    
            # Initalize membrane potential
            spk1, mem1 = self.rlif1.init_rleaky()
            spk2, mem2 = self.rlif2.init_rleaky()
    
            # Empty lists to record outputs
            spk_recording = []
    
            for step in range(self.timesteps):
                spk1, mem1 = self.rlif1(self.fc1(x), spk1, mem1)
                spk2, mem2 = self.rlif2(self.fc2(spk1), spk2, mem2)
                spk_recording.append(spk2)
    
            return torch.stack(spk_recording)

Instantiate the network below:

::

    hidden = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = Net(timesteps=num_steps, hidden=hidden, beta=0.9).to(device)

4. Construct Training Loop
--------------------------------------------

4.1 Mean Square Error Loss in ``snntorch.functional``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From ``snntorch.functional``, we call ``mse_count_loss`` to set the
target neuron to fire 80% of the time, and incorrect neurons to fire 20%
of the time. What it took 10 paragraphs to explain is achieved in one
line of code:

::

    loss_function = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

4.2 DataLoader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataloader boilerplate. Let’s just do MNIST, and testing this on
temporal data is an exercise left to the reader/coder.

::

    batch_size = 128
    data_path='/tmp/data/mnist'
    
    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
    
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

4.3 Train Network
-----------------

::

    num_epochs = 5
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    loss_hist = []
    
    with tqdm.trange(num_epochs) as pbar:
        for _ in pbar:
            train_batch = iter(train_loader)
            minibatch_counter = 0
            loss_epoch = []
    
            for feature, label in train_batch:
                feature = feature.to(device)
                label = label.to(device)
    
                spk = model(feature.flatten(1)) # forward-pass
                loss_val = loss_function(spk, label) # apply loss
                optimizer.zero_grad() # zero out gradients
                loss_val.backward() # calculate gradients
                optimizer.step() # update weights
    
                loss_hist.append(loss_val.item())
                minibatch_counter += 1
    
                avg_batch_loss = sum(loss_hist) / minibatch_counter
                pbar.set_postfix(loss="%.3e" % avg_batch_loss)

5. Evaluation
----------------------

::

    test_batch = iter(test_loader)
    minibatch_counter = 0
    loss_epoch = []
    
    model.eval()
    with torch.no_grad():
      total = 0
      acc = 0
      for feature, label in test_batch:
          feature = feature.to(device)
          label = label.to(device)
    
          spk = model(feature.flatten(1)) # forward-pass
          acc += SF.accuracy_rate(spk, label) * spk.size(1)
          total += spk.size(1)
    
    print(f"The total accuracy on the test set is: {(acc/total) * 100:.2f}%")

6. Alternative Loss Metric
==========================

In the previous tutorial, we tested membrane potential learning. We can
do the same here by setting the target neuron to reach a membrane
potential greater than the firing threshold, and incorrect neurons to
reach a membrane potential below the firing threshold:

::

    loss_function = SF.mse_membrane_loss(on_target=1.05, off_target=0.2)

In the above case, we are trying to get the correct neuron to constantly
sit above the firing threshold.

Try updating the network and the training loop to make this work.

Hints: 

- You will need to return the output membrane potential instead of spikes. 

- Pass membrane potential to the loss function instead of spikes

Conclusion
------------------------

The next regression tutorial will introduce spiking LSTMs to achieve
precise spike time learning.

If you like this project, please consider starring ⭐ the repo on GitHub
as it is the easiest and best way to support it.

Additional Resources
------------------------

-  `Check out the snnTorch GitHub project
   here. <https://github.com/jeshraghian/snntorch>`__
-  More detail on nonlinear regression with SNNs can be found in our
   corresponding preprint here: `Henkes, A.; Eshraghian, J. K.; and
   Wessels, H. “Spiking neural networks for nonlinear regression”, arXiv
   preprint arXiv:2210.03515,
   Oct. 2022. <https://arxiv.org/abs/2210.03515>`__
