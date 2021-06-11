============================================
Tutorial 3 - Deep Learning with snnTorch
============================================

Tutorial written by Jason K. Eshraghian (`www.jasoneshraghian.com <https://www.jasoneshraghian.com>`_)

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_3_FCN.ipynb

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_3_FCN.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_

Deep Learning with `snnTorch`
-------------------------------------------------------------------

Introduction
--------------

In this tutorial, you will:

  * Learn how spiking neurons are implemented in a recurrent network
  * Understand backpropagation through time, and the associated challenges in SNNs such as target labeling, and the non-differentiability of spikes
  * Train a fully-connected network on the static MNIST dataset

Part of this tutorial was inspired by Friedemann Zenke's extensive work on SNNs. Check out his `repo on surrogate gradients here <https://github.com/fzenke/spytorch>`_, and a favourite paper of mine: E. O. Neftci, H. Mostafa, F. Zenke, `Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient-based optimization to spiking neural networks. <https://ieeexplore.ieee.org/document/8891809>`_ IEEE Signal Processing Magazine 36, 51–63.

As a quick recap, `Tutorial 1 <https://colab.research.google.com/github/jeshraghian/snntorch/blob/tutorials/examples/tutorial_1_spikegen.ipynb>`_ explained how to convert datasets into spikes using three encoding mechanisms:

  * Rate coding
  * Latency coding
  * Delta modulation

`Tutorial 2 <https://colab.research.google.com/github/jeshraghian/snntorch/blob/tutorials/examples/tutorial_2_neuronal_dynamics.ipynb>`_ showed how to build neural networks using three different leaky integrate-and-fire (LIF) neuron models:

  * Lapicque's RC model
  * Synaptic conductance-based model
  * Spike Response model

At the end of the tutorial, a basic supervised learning algorithm will be implemented. We will use the original static MNIST dataset and train a multi-layer fully-connected spiking neural network using gradient descent to perform image classification. 

Install the latest PyPi distribution of snnTorch::

  $ pip install snntorch 

1. A Recurrent Representation of SNNs
----------------------------------------

The following is a summary of the continuous time-domain representation LIF neurons, and applies the result to develop a recurrent representation that is more suitable for use in recurrent neural networks (RNNs). 

We derived the dynamics of the passive membrane using an RC circuit in the time-domain: 

$$\\tau_{\\rm mem} \\frac{dU_{\\rm mem}(t)}{dt} = -U_{\\rm mem}(t) + RI_{\\rm syn}(t),$$

where the general solution of this equation is:

$$U_{\\rm mem}=I_{\\rm syn}(t)R + [U_0 - I_{\\rm syn}(t)R]e^{-t/\\tau_{\\rm mem}}$$

In Lapicque's model, :math:`I_{\rm syn}(t)` is also the input current, :math:`I_{\rm in}(t)`. 

In the Synaptic conductance-based model (which we will loosely refer to as the synaptic model), a more biologically plausible approach is taken that ensures :math:`I_{\rm syn}(t)` follows an exponential decay as a function of the input:


$$I_{\\rm syn}(t) = \\sum_k W_{i,j} S_{in; i,j}(t) e^{-(t-t_k)/\\tau_{syn}}\\Theta(t-t_k)$$

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/3_1_stein_decomp.png?raw=true
        :align: center
        :width: 600

The synaptic model has two exponentially decaying terms: :math:`I_{\rm syn}(t)` and :math:`U_{\rm mem}(t)`. The ratio between subsequent terms (i.e., decay rate) of :math:`I_{\rm syn}(t)` is set to :math:`\alpha`, and that of :math:`U_{\rm mem}(t)` is set to :math:`\beta`:

$$ \\alpha = e^{-1/\\tau_{\\rm syn}}$$

$$ \\beta = e^{-1/\\tau_{\\rm mem}}$$


RNNs will process data sequentially, and so time must be discretised, and the neuron models must be converted into a recursive form. :math:`\alpha` and :math:`\beta` can be used to give a recursive representation of the Synaptic neuron model:

$$I_{\\rm syn}[t+1]=\\underbrace{\\alpha I_{\\rm syn}[t]}_\\text{decay} + \\underbrace{WS_{\\rm in}[t+1]}_\\text{input}$$

$$U[t+1] = \\underbrace{\\beta U[t]}_\\text{decay} + \\underbrace{I_{\\rm syn}[t+1]}_\\text{input} - \\underbrace{R[t+1]}_\\text{reset}$$

**Spiking**

If :math:`U[t] > U_{\rm thr}`, then an output spike is triggered: :math:`S_{\rm out}[t] = 1`. Otherwise, :math:`S_{\rm out}[t] = 0`. 

.. note::

  A variation of this is to set the output spike at the *next* time step to be triggered; i.e., :math:`U[t] > U_{\rm thr} \implies S_{\rm out}[t+1] = 1`. This is the approach taken in snnTorch, and will be explained in following sections.

An alternative way to represent the relationship between :math:`S_{\rm out}` and :math:`U_{\rm mem}`, which is also used to calculate the gradient in the backward pass, is:

$$S_{\\rm out}[t] = \\Theta(U_{\\rm mem}[t] - U_{\\rm thr})$$

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/3_2_spike_descrip.png?raw=true
        :align: center
        :width: 600

        
**Reset**

The reset term is activated only when the neuron triggers a spike. That is to say, if :math:`S_{\rm out}[t+1]=1`:

  * For :code:`reset_mechanism="subtract"`: :math:`R[t+1]=U_{\rm thr}` 
  * For :code:`reset_mechanism="zero"`: :math:`R[t+1]=U[t+1]`

.. note::
  
  In snnTorch, the reset will also take a one time step delay such that :math:`R[t+1]` is activated only when :math:`S_{\rm out}[t+1]=1`

The other neurons follow a similar form, which is `detailed in the documentation <https://snntorch.readthedocs.io/en/latest/snntorch.html>`_. The recursive neuron equations can be mapped into computation graphs, where the recurrent connections take place with a delay of a single time step, from the state at time math:`t` to the state at time :math:`t+1`. 

An alternative way to represent recurrent models is to unfold the computational graph, in which each component is represented by a sequence of different variables, with one variable per time step. The unfolded form of the Synaptic model is shown below:



.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/3_2_unrolled.png?raw=true
        :align: center
        :width: 800


Up until now, the notation used for all variables have had an association with their electrical meanings. As we move from neuronal dynamics to deep learning, we will slightly modify the notation throughout the rest of the tutorial:

* **Input spike:** :math:`S_{\rm in} \rightarrow X`
* **Input current (weighted spike):** :math:`I_{\rm in} \rightarrow Y`
* **Synaptic current:** :math:`I_{\rm syn} \rightarrow I`
* **Membrane potential:** :math:`U_{\rm mem} \rightarrow U`
* **Output spike:** :math:`S_{\rm out} \rightarrow S`

The benefit of an unrolled graph is that we now have an explicit description of how computations are performed. The process of unfolding illustrates the flow of information forward in time (from left to right) to compute outputs and losses, and backward in time to compute gradients. The more time steps that are simulated, the deeper the graph becomes. 

Conventional RNNs treat :math:`\alpha` and :math:`\beta` as learnable parameters. This is also possible for SNNs, but in snnTorch, they are treated as hyperparameters by default. This replaces the vanishing and exploding gradient problems with a parameter search.

2. Setting up the Static MNIST Dataset
----------------------------------------

Much of the following code has already been explained in the first two tutorials. So we'll dive straight in. 

2.1 Import packages and setup the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

  import snntorch as snn
  import torch
  import torch.nn as nn
  from torch.utils.data import DataLoader
  from torchvision import datasets, transforms
  import numpy as np
  import itertools
  import matplotlib.pyplot as plt

::

  # Network Architecture
  num_inputs = 28*28
  num_hidden = 1000
  num_outputs = 10

  # Training Parameters
  batch_size=128
  data_path='/data/mnist'

  # Temporal Dynamics
  num_steps = 25
  alpha = 0.7
  beta = 0.8

  dtype = torch.float
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

2.2 Download MNIST Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

  # Define a transform
  transform = transforms.Compose([
              transforms.Resize((28, 28)),
              transforms.Grayscale(),
              transforms.ToTensor(),
              transforms.Normalize((0,), (1,))])

  mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
  mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

If the above code blocks throws an error, e.g. the MNIST servers are down, then uncomment the following code instead.

::

  # # temporary dataloader if MNIST service is unavailable
  # !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
  # !tar -zxvf MNIST.tar.gz

  # mnist_train = datasets.MNIST(root = './', train=True, download=True, transform=transform)
  # mnist_test = datasets.MNIST(root = './', train=False, download=True, transform=transform)

::

  # Create DataLoaders
  train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)


3. Define the Network
----------------------------------------

The spiking neurons available in snnTorch are designed to be treated as activation units. The only difference is that these spiking neuron activations depend not only on their inputs, but also on their previous state (e.g., :math:`I[t-1]` and :math:`U[t-1]` for the Synaptic neuron). This can be implemented in a for-loop with ease.

If you have a basic understanding of PyTorch, the following code block should look familiar. :code:`nn.Linear` initializes the linear transformation layer, and instead of applying a sigmoid, ReLU or some other nonlinear activation, a spiking neuron is applied instead by calling :code:`snn.Synaptic`:

::

  # Define Network
  class Net(nn.Module):
      def __init__(self):
          super().__init__()

          # Initialize layers
          self.fc1 = nn.Linear(num_inputs, num_hidden)
          self.lif1 = snn.Synaptic(alpha=alpha, beta=beta)
          self.fc2 = nn.Linear(num_hidden, num_outputs)
          self.lif2 = snn.Synaptic(alpha=alpha, beta=beta)

      def forward(self, x):

          # Initialize hidden states and outputs at t=0
          spk1, syn1, mem1 = self.lif1.init_synaptic(batch_size, num_hidden)
          spk2, syn2, mem2 = self.lif2.init_synaptic(batch_size, num_outputs)
          
          # Record the final layer
          spk2_rec = []
          mem2_rec = []

          for step in range(num_steps):
              cur1 = self.fc1(x)
              spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
              cur2 = self.fc2(spk1)
              spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

              spk2_rec.append(spk2)
              mem2_rec.append(mem2)

          return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

The code in the :code:`forward()` function will only be called once the input argument :code:`x` is explicitly passed in:

* :code:`fc1` applies a linear transformation to the input: :math:`:W_{i, j}^{[1]}X_{i}^{[1]}[t] \rightarrow Y_{j}^{[1]}[t]`, i.e., :code:`cur1`
* :code:`lif1` integrates :math:`Y^{[1]}_{j}[t]` over time (with a decay), to generate :math:`I_{j}^{[1]}[t]` and :math:`U_{j}^{[1]}[t]`. An output spike is triggered if :math:`U_{j}^{[1]}[t] > U_{\rm thr}`. Equivalently, :code:`spk1=1` if :code:`mem1` > :code:`threshold=1.0`
* :code:`fc2` applies a linear transformation to :code:`spk1`: :math:`W_{j, k}^{[2]}S_{j}^{[1]}[t] \rightarrow Y_{k}^{[2]}[t]`, i.e., :code:`cur2`
* :code:`lif2` is another spiking neuron layer, and generates output spikes :math:`S_{k}^{[2]}[t]` which are returned in the variable :code:`spk2`

Here, :math:`i` denotes one of 784 input neurons, :math:`j` indexes one of the 1,000 neurons in the hidden layer, and :math:`k` points to one of 10 output neurons.

The layers in :code:`def __init__(self)` are automatically created upon instantiating :code:`Net()`, as is done below:

::

  # Load the network onto CUDA if available
  net = Net().to(device)

4. Backpropagation for SNNs
----------------------------------------

A few questions arise when setting up a backprop-driven learning algorithm:

1.   **Targets**: What should the target of the output layer be?
2.   **Backprop through time**: How might the gradient flow back in time?
3.   **Spike non-differentiability**: If spikes are discrete, instantaneous bursts of information, doesn't that make them non-differentiable? If the output spike has no gradient with respect to the network parameters, wouldn't backprop be impossible?

Let's tackle these one by one. 

4.1 Target Labels
^^^^^^^^^^^^^^^^^^^^^


In `tutorial 1 <https://colab.research.google.com/github/jeshraghian/snntorch/blob/tutorials/examples/tutorial_1_spikegen.ipynb>`_, we learnt about rate and latency coding. Rate coding stores information in the frequency of spikes, and latency coding stores information in the timing of each spike. Previously, we used these encoding strategies to convert datasets into time-varying spikes. Here, they are used as encoding strategies for the output layer of our SNN. I.e., these codes will be used to teach the final layer of the network how to respond to certain inputs. 

The goal of the SNN is to predict a discrete variable with :math:`n` possible values, as is the case with MNIST where :math:`n=10`. 

4.1.1 Rate code
""""""""""""""""""""""""""""""""""

For rate encoding, the most naive implementation is to encourage the correct class to fire at every time step, and the incorrect classes to not fire at all. There are two ways to implement this, one of which is a lot more effective than the other:

* Set the target of the output spike of the correct class :math:`y_{\rm spk} = 1` for all :math:`t`, or
* Set the target of the membrane potential of the correct class :math:`y_{\rm mem} = U_{\rm thr}` for all :math:`t` 

Which is the better approach? 

**Spiking Targets**

Consider the first option. The output spikes are discrete events, and rely on large perturbations of the membrane potential around the threshold to have any infleunce. If the output spiking behavior goes unchanged, the gradient of the output of the network with respect to its parameters would be :math:`0`. This is problematic, because the training process would no longer have a guide for how to improve the weights. It would be an ineffective approach for gradient descent. 

**Membrane Potential Targets**

Instead, it is better to promote spiking by applying the target to the membrane potential. As the membrane potential is a much stronger function of the parameters, (i.e., a small perturbation of the weights would directly perturb the membrane potential), this would ensure there is a strong gradient whenever the network obtains a wrong result. So we set :math:`y_{\rm mem} = U_{\rm thr}`. By default, :code:`threshold=1`. The outputs can then be applied to a softmax unit, which are then used to find the cross-entropy loss:

$$CE = - \\sum^n_{i=1}y_{i,\\rm mem} {\\rm log}(p_i),$$

where :math:`y_{i, \rm mem}` is the target label at a given time step, :math:`n` is the number of classes, and :math:`p_i` is the softmax probability for the :math:`i^{th}` class. 

The accuracy of the network would then be measured by counting up how many times each neuron fired across all time steps. We could then use :code:`torch.max()` to choose the neuron with the most spikes, or somewhat equivalently, the highest average firing rate. 

It is possible to increase the target of membrane potential beyond the threshold to excite the neuron further. While this may be desirable in some instances, it will likely trigger high-conductance pathways for the wrong class when training other samples.

4.1.2 Latency code
""""""""""""""""""""""""""""""""""

In latency encoding, the neuron that fires first is the predicted class. The target may be set to 1 for one of the first few time steps. Depending on the neuron model being used, it will take several time steps before the input can propagate to the output of the network. Therefore, it is inadvisable to set the target to :code:`1` only for the first time step. 

Consider the case of a neuron receiving an input spike. Depending on the neuron model in use, the post-synaptic potential may experience a time delay :math:`t_{\rm psp}` to reach the peak of its membrane potential, and subsequently emit an output spike. If this neuron is connected in a deep neural network, the minimum time before the final layer could generate output spikes *as a result of the input (and not biases)* would thus be :math:`t_{\rm min} = Lt_{\rm psp}`, where :math:`L` is the number of layers in the network. 

For the Synaptic and Lapicque models, the membrane potential will immediately jump as a result of the input. But there is a time delay of one step before the output spike can be triggered as a result. Therefore, we set :math:`t_{\rm psp}=1` time step. For SRM0, it will take a longer time to reach the peak, and is a function of the decay rates, :math:`\alpha` and :math:`\beta`. 


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/3_3_delay.png?raw=true
        :align: center
        :width: 450

In absence of this post-synaptic potential delay, it becomes challenging to control the output layer in terms of spike timing. An input spike of a multi-layer SNN could effectively be transmitted straight to the output instantaneously, without considering the input data at any later time steps. A slight modification is made to the unrolled computational graph, which adds a delay of one time step between :math:`U` and :math:`S`.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/3_4_graphdelay.png?raw=true
        :align: center
        :width: 700

As for the incorrect classes, it is acceptable to set their targets to 0. However, this could result in low conductance pathways that completely inhibit firing. It may be preferable to set their membrane potential target to something slightly higher, e.g., :math:`U_{\rm thr}/5`. The optimal point is a topic of further investigation. Note that all of the above can have a cross-entropy loss applied, just as with rate coding.

A simple example across 4 time steps is provided in the image below, though the values and spiking periodicity should not be taken literally.


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/3_5_targets.png?raw=true
        :align: center
        :width: 700

An alternative approach is to treat the number of time steps as a continuous variable and use a mean square error loss to dictate when firing should occur:

$$MSE = \\sum^n_{t=1}(t_{\\rm spk} - \\hat{t_{\\rm spk}}^2),$$

where :math:`t` is the time step, and :math:`n` is the total number of steps. In such a case, a larger number of time steps are expected to improve performance as it will allow the flow of time to look more 'continuous'.

Is there a preference between latency and rate codes? We briefly touched on this question in the context of data encoding, and the same arguments apply here. Latency codes are desirable because they only rely on a single spike to convey all necessary information. Rate coding spreads out information across many time steps, and there is much less information transfer within each spike. Therefore, latency codes are much more power efficient when running on neuromorphic hardware. On the other hand, the redundant spikes in rate codes makes them much more noise tolerant. 

4.2 Backpropagation Through Time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Computing the gradient through an SNN is mostly the same as that of an RNN. The generalized backpropagation algorithm is applied to the unrolled computational graph. Working backward from the end of the sequence, the gradient flows from the loss to all descendents. Shown below are the various pathways of the gradient :math:`\nabla_W \mathcal{L}` from the parent (:math:`\mathcal{L}`: cross-entropy loss) to its leaf nodes (:math:`W`). 


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/3_6_bptt.png?raw=true
        :align: center
        :width: 800


The learnable parameter :math:`W` is shared across each time step. This means that multiple backprop paths exist between the loss and the same network parameter. To resolve this, all gradients :math:`\nabla_W \mathcal{L}` are simply summed together before applying a weight update.

To find :math:`\nabla_W \mathcal{L}`, the chain rule is applied to each pathway. 

**Shortest Pathway** 

Considering only the shortest pathway at :math:`t=3`, where the superscript :math:`^{<1>}` indicates this is just one of many paths to be summed:

$$\\nabla_W \\mathcal{L}^{<1>} = \\frac{\\partial{\\mathcal{L}}}{\\partial{p_i}} \\frac{\\partial{p_i}}{\\partial{U[3]}} \\frac{\\partial{U[3]}}{\\partial{Y[3]}} \\frac{\\partial{Y[3]}}{\\partial{W}}$$

The first two terms can be analytically solved by taking the derivative of the cross-entropy loss and the softmax function. The third term must be decomposed into the following terms:

$$ \\frac{\\partial{U[3]}}{\\partial{Y[3]}} = \\frac{\\partial{U[3]}}{\\partial{I[3]}} \\frac{\\partial{I[3]}}{\\partial{Y[3]}}$$

Recall the recursive form of the Synaptic neuron model:


$$I[t+1]=\\alpha I[t] + WX[t+1]$$

$$U[t+1] = \\beta U[t] + I[t+1] - R[t+1]$$

:math:`WX=Y` is directly added to :math:`I`, which is directly added to :math:`U`. Therefore, both partial derivative terms evaluate to 1:

$$\\frac{\\partial{U[3]}}{\\partial{Y[3]}} = 1$$

The final term :math:`\frac{\partial{Y[3]}}{\partial{W}}` evaluates to the input at that time step :math:`X[3]`. 

**2nd Shortest Pathways**

Consider the pathway that flows backwards one time step from :math:`t=3` to :math:`t=2` through :math:`\beta`:

$$\\nabla_W \\mathcal{L}^{<2>} = \\frac{\\partial{\\mathcal{L}}}{\\partial{p_i}} \\frac{\\partial{p_i}}{\\partial{U[3]}} \\frac{\\partial{U[3]}}{\\partial{U[2]}} 
\\frac{\\partial{U[2]}}{\\partial{Y[2]}} \\frac{\\partial{Y[2]}}{\\partial{W}}$$

Almost all terms are the same as the shortest pathway calculation, or at least evaluate to the same values. The only major difference is the third term, which signals the backwards flow through time: :math:`U[3] \rightarrow U[2]`. The derivative is simply :math:`:\beta`. 

The parallel pathway flowing through :math:`I[3] \rightarrow I[2]` follows the same method, but instead, :math:`\frac{\partial{I[3]}}{\partial{I[2]}} = \alpha`. 

An interesting result arises: for each additional time step the graph flows through, the smaller that component of the gradient becomes. This is because each backwards path is recursively multiplied by either :math:`\alpha` or :math:`\beta`, which gradually diminish the contribution of earlier states of the network to gradient.

Luckily for you, all of this is automatically taken care of by PyTorch's autodifferentiation framework. Variations of backprop through time are also available within snnTorch, which will be demonstrated in future tutorials.


4.3 Non-differentiability of Spikes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The above analysis only solved for parameter updates for the final layer. This was not an issue as we used membrane potential :math:`U` to calculate the loss, which is a continuous function. If we backpropagate to earlier layers, we need to take the derivative of spikes, i.e., a non-differentiable, non-continuous function.

Let's open up the computational graph of the Synaptic neuron model to identify exactly where this problem occurs.


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/3_7_stein_bptt.png?raw=true
        :align: center
        :width: 800


Backpropagating through the shortest path gives:
$$\\frac{\\partial{S[3]}}{\\partial{Y[2]}} = \\frac{\\partial{S[3]}}{\\partial{U[2]}} \\frac{\\partial{U[2]}}{\\partial{I[2]}}\\frac{\\partial{I[2]}}{\\partial{Y[2]}}$$

The final two terms evaluate to 1 for the same reasons described above. But the first term is non-differentiable. Recall how :math:`S=1` only for :math:`U>U_{\rm thr}`, i.e., a shifted form of the Heaviside step function. The analytical derivative evaluates to 0 everywhere, except at :math:`U_{\rm thr}: \frac{\partial{S[t]}}{\partial{U[t-1]}} \rightarrow \infty`. This is the result generated by PyTorch's default autodifferentiation framework, and will zero out the gradient thus immobilizing the network's ability to learn:

$$W := W - \\eta \\nabla_W \\mathcal{L} $$

where :math:`\nabla_W \mathcal{L} \rightarrow 0`. 

How do we overcome this issue? Several approaches have been taken and yielded great results. Smooth approximations of the Heaviside function have been used, taking gradients of the continuous function instead. Friedemann Zenke's extensive work on surrogate gradients is among the most rigorous on this topic, and is `very well documented here <https://github.com/fzenke/spytorch>`_. The option to use surrogate gradients is available in snnTorch as well, and can be called from the `snntorch.surrogate` library. `More details are available here <https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html>`_.

snnTorch takes a wholly different approach that is simple, yet effective. 

4.3.1 A Time-Evolution Approach to the Spiking Derivative
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

What follows is a simple, intuitive description behind the approach taken. A rigorous mathematical treatment will be made available separately. 

The analytical derivative of :math:`S` with respect to :math:`U` neglects two features of spiking neurons:

* the discrete time representation of SNNs 
* spike-induced reset and refractory periods of neurons

**Discrete Time Representation**

Given that SNNs (and more generally, RNNs) operate in discrete time, we can approximate the derivative to be the relative change across 1 time step:

$$\\frac{\\partial S}{\\partial U} \\rightarrow \\frac{\\Delta S}{\\Delta U}$$

Intuitively, the time derivative cannot be calculated by letting :math:`\Delta t \rightarrow 0`, but rather, it must approach the smallest possible value :math:`\Delta t \rightarrow 1`. It therefore follows that the derivative of a time-varying pair of functions must be treated similarly.

**Spike-induced Reset**

Next, the occurrence of a spike necessarily incurs a membrane potential reset. So when the spike mechanism switches off: :math:`S: 1 \rightarrow 0`, the membrane potential resets by subtraction of the threshold, which is set to one by default: :math:`\Delta U = U_{\rm thr} \rightarrow -1`:

$$\\frac{\\Delta S}{\\Delta U} = \\frac{-1}{-1} = 1$$

This situation is illustrated below:

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/3_8_timevarying.png?raw=true
        :align: center
        :width: 550


If instead there is no spike, then :math:`\Delta S = 0` for a finite change in :math:`U`. Formally:

.. math::

  \begin{equation}
      \frac{\partial S}{\partial U} \approx \Theta(U - U_{\rm thr}) =     
      \begin{cases}
        1  & \text{if $S$ = $1$}\\
        0 & \text{if $S$ = $0$}
      \end{cases}  
  \end{equation}

This is simply the Heaviside step function shifted about the membrane threshold, :math:`U_{\rm thr} = \theta`.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/3_9_spike_grad.png?raw=true
        :align: center
        :width: 550

What this suggests is that learning only takes place when neurons fire. This is generally not a concern, as a large enough network will have sufficient spiking to enable a gradient to flow through the computational graph. Armed with the knowledge that weight updates only take place when neurons fire, this approach echoes a rudimentary form of Hebbian learning.

Importantly, the situation is more nuanced than what has been described above. But this should be sufficient to give you the big picture intuition. As a matter of interest, the Heaviside gradient takes a similar approach to how the gradient flows through a max-pooling unit, and also evaluates to the same derivative as a shifted ReLU activation. 


5. Training on Static MNIST
-------------------------------------------------

Time for training! Let's first define a couple of functions to print out test/train accuracy.

::

  def print_batch_accuracy(data, targets, train=False):
      output, _ = net(data.view(batch_size, -1))
      _, idx = output.sum(dim=0).max(1)
      acc = np.mean((targets == idx).detach().cpu().numpy())

      if train:
          print(f"Train Set Accuracy: {acc}")
      else:
          print(f"Test Set Accuracy: {acc}")

  def train_printer():
      print(f"Epoch {epoch}, Minibatch {minibatch_counter}")
      print(f"Train Set Loss: {loss_hist[counter]}")
      print(f"Test Set Loss: {test_loss_hist[counter]}")
      print_batch_accuracy(data_it, targets_it, train=True)
      print_batch_accuracy(testdata_it, testtargets_it, train=False)
      print("\n")

5.1 Optimizer
^^^^^^^^^^^^^^^^^^^^^

We will apply a softmax to the output of our network, and calculate the loss using the negative log-likelihood.

::

  optimizer = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0.9, 0.999))
  log_softmax_fn = nn.LogSoftmax(dim=-1)
  loss_fn = nn.NLLLoss()

5.2 Training Loop
^^^^^^^^^^^^^^^^^^^^^

We assume some working knowledge of PyTorch. The training loop is fairly standard, with the only exceptions being the following.

**Inputs**

The for-loop that iterates through each time step during the forward pass has already been nested within :code:`net`. This means that the following line of code:

:code:`spk_rec, mem_rec = net(data_it.view(batch_size, -1))`

passes the same sample at each step. That is why we refer to it as static MNIST.


**Targets**

The losses generated at each time steps are summed together in the for-loop that contains:

:code:`loss_val += loss_fn(log_p_y[step], targets_it)`

Also note how :code:`targets_it` is not indexed, because the same value is used as the target for each step. '1' is applied as the target for the correct class for all of time, and '0' is applied as the target for all other classes.

Let's train this across 3 epochs to keep things quick.

::

  loss_hist = []
  test_loss_hist = []
  counter = 0

  # Outer training loop
  for epoch in range(3):
      minibatch_counter = 0
      train_batch = iter(train_loader)

      # Minibatch training loop
      for data_it, targets_it in train_batch:
          data_it = data_it.to(device)
          targets_it = targets_it.to(device)

          spk_rec, mem_rec = net(data_it.view(batch_size, -1))
          log_p_y = log_softmax_fn(mem_rec)
          loss_val = torch.zeros((1), dtype=dtype, device=device)

          # Sum loss over time steps: BPTT
          for step in range(num_steps):
            loss_val += loss_fn(log_p_y[step], targets_it)

          # Gradient calculation
          optimizer.zero_grad()
          loss_val.backward()

          # Weight Update
          optimizer.step()

          # Store loss history for future plotting
          loss_hist.append(loss_val.item())

          # Test set
          test_data = itertools.cycle(test_loader)
          testdata_it, testtargets_it = next(test_data)
          testdata_it = testdata_it.to(device)
          testtargets_it = testtargets_it.to(device)

          # Test set forward pass
          test_spk, test_mem = net(testdata_it.view(batch_size, -1))

          # Test set loss
          log_p_ytest = log_softmax_fn(test_mem)
          log_p_ytest = log_p_ytest.sum(dim=0)
          loss_val_test = loss_fn(log_p_ytest, testtargets_it)
          test_loss_hist.append(loss_val_test.item())

          # Print test/train loss/accuracy
          if counter % 50 == 0:
              train_printer()
          minibatch_counter += 1
          counter += 1

  loss_hist_true_grad = loss_hist
  test_loss_hist_true_grad = test_loss_hist


If this was your first time training an SNN, then congratulations!

6. Results
-------------------------------------------------

6.1 Plot Training/Test Loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

  # Plot Loss
  fig = plt.figure(facecolor="w", figsize=(10, 5))
  plt.plot(loss_hist)
  plt.plot(test_loss_hist)
  plt.legend(["Train Loss", "Test Loss"])
  plt.xlabel("Minibatch")
  plt.ylabel("Loss")
  plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/_static/loss.png?raw=true
        :align: center
        :width: 500

Taking a look at the training / test loss, the process is somewhat noisy. This could be a result of a variety of things: minibatch gradient descent is the obvious one, but the use of improper targets likely also contributes. By encouraging the correct class to fire at every time step, the loss function conflicts with the reset mechanism that tries to prevent this.

6.2 Test Set Accuracy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This function iterates over all minibatches to obtain a measure of accuracy over the full 10,000 samples in the test set.

::

  total = 0
  correct = 0

  # drop_last switched to False to keep all samples
  test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

  with torch.no_grad():
    net.eval()
    for data in test_loader:
      images, labels = data
      images = images.to(device)
      labels = labels.to(device)

      # If current batch matches batch_size, just do the usual thing
      if images.size()[0] == batch_size:
        outputs, _ = net(images.view(batch_size, -1))

      # If current batch does not match batch_size (i.e., is the final batch),
      # modify batch_size in a temp variable and restore it at the end
      else:
        temp_bs = batch_size
        batch_size = images.size()[0]
        outputs, _ = net(images.view(images.size()[0], -1))
        batch_size = temp_bs

      _, predicted = outputs.sum(dim=0).max(1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
::

  >>> print(f"Total correctly classified test set images: {correct}/{total}")
  >>> print(f"Test Set Accuracy: {100 * correct / total}%")
  
  Total correctly classified test set images: 9540/10000
  Test Set Accuracy: 95.4%

Voila! That's it for static MNIST. Feel free to tweak the network parameters, hyperparameters, decay rate, using a learning rate scheduler etc. to see if you can improve the network performance. 

Conclusion
--------------

Now you know how to construct and train a fully-connected network on a static dataset. The spiking neurons can actually be adapted to other layer types, including convolutions and skip connections. Armed with this knowledge, you should now be able to build many different types of SNNs.

In the next tutorial, you will learn how to train a spiking convolutional network using a time-varying spiking dataset.
