===============================
Tutorial 2 - Neuronal Dynamics
===============================

Tutorial written by Jason K. Eshraghian (`www.jasoneshraghian.com <https://www.jasoneshraghian.com>`_)

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/tutorials/examples/tutorial_2_neuronal_dynamics.ipynb

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/tutorials/examples/tutorial_2_neuronal_dynamics.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_

Neuronal Dynamics with `snnTorch`
-------------------------------------------------------------------

Introduction
--------------

In this tutorial, you will:

* Learn the fundamentals of the leaky integrate-and-fire (LIF) neuron model
* Use snnTorch to implement variations of the LIF model: 
  
  * Lapicque's neuron model
  
  * Stein's neuron model
  
  * :math:`0^{th}` Order Spike Response Model

  * Implement a feedforward spiking neural network

Part of this tutorial was inspired by the book `Neuronal Dynamics: From single neurons to networks and models of cognition <https://neuronaldynamics.epfl.ch/index.html>`_ by
Wulfram Gerstner, Werner M. Kistler, Richard Naud and Liam Paninski.

Install the latest PyPi distribution of snnTorch::

  $ pip install snntorch 

1. The Spectrum of Neuron Models
---------------------------------


A large variety of neuron models are out there, ranging from biophysically accurate models (i.e., the Hodgkin-Huxley models) to the extremely simple artificial neuron that pervades all facets of modern deep learning.

**Hodgkin-Huxley Neuron Models** - While biophysical models can reproduce electrophysiological results with a high degree of accuracy, their complexity makes them difficult to use. We expect this to change as more rigorous theories of how neurons contribute to higher-order behaviors in the brain are uncovered.

**Artificial Neuron Model** - On the other end of the spectrum is the artificial neuron. The inputs are multiplied by their corresponding weights and passed through an activation function. This simplification has enabled deep learning researchers to perform incredible feats in computer vision, natural language processing, and many other machine learning-domain tasks.

**Leaky Integrate-and-Fire Neuron Models** - Somewhere in the middle of the divide lies the leaky integrate-and-fire (LIF) neuron model. It takes the sum of weighted inputs, much like the artificial neuron. But rather than passing it directly to an activation function, it will integrate the input over time with a leakage, much like an RC circuit. If the integrated value exceeds a threshold, then the LIF neuron will emit a voltage spike. The LIF neuron abstracts away the shape and profile of the output spike; it is simply treated as a discrete event. As a result, information is not stored within the spike, but rather the timing (or frequency) of spikes. Simple spiking neuron models have produced much insight into the neural code, memory, network dynamics, and more recently, deep learning. The LIF neuron sits in the sweet spot between biological plausibility and practicality. 


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_1_neuronmodels.png?raw=true
        :align: center
        :width: 1000


The different versions of the LIF model each have their own dynamics and use-cases. snnTorch currently supports three types of LIF neurons:

* Lapicque's RC model: ``snntorch.Lapicque``
  
* Stein's neuron model: ``snntorch.Stein``
  
* :math:`0^{th}` Order Spike Response Model: ``snntorch.SRM0``

Before learning how to use them, let's understand how to construct a simple LIF neuron model.

2. The Leaky Integrate-and-Fire Neuron Model
---------------------------------------------

2.1 Spiking Neurons: Intuition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A neuron might be connected to 1,000 - 10,000 other neurons. If one neuron spikes, all of these downhill neurons will feel it. But what determines whether a neuron spikes in the first place? The past century of experiments demonstrate that if a neuron experiences *sufficient* stimulus at its input, then we might expect it to become excited and fire its own spike. 

Where does this stimulus come from? It could be from

* the sensory periphery, 
  
* an invasive electrode artificially stimulating the neuron, or in most cases,
  
* from other pre-synaptic neurons. 


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_2_intuition.png?raw=true
        :align: center
        :width: 800

Given that these spikes are very short bursts of electrical activity, it is quite unlikely for all input spikes to arrive at the neuron body in precise unison. This indicates the presence of temporal dynamics that 'sustain' the input spikes, kind of like a delay.

2.2 The Passive Membrane
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Like all cells, a neuron is surrounded by a thin membrane. This membrane is a lipid bilayer that insulates the conductive saline solution within the neuron from the extracellular medium. Electrically, the two conductors separated by an insulator is a capacitor. 

Another function of this membrane is to control what goes in and out of this cell (e.g., ions such as :math:`Na^+`). The membrane is usually impermeable to ions which blocks them from entering and exiting the neuron body. But there are specific channels in the membrane that are triggered to open by injecting current into the neuron. This charge movement is electrically modelled by a resistor.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_3_passivemembrane.png?raw=true
        :align: center
        :width: 450

Now say some arbitrary time-varying current :math:`I_{\rm in}(t)` is injected into the neuron, be it via electrical stimulation or from other neurons. The total current in the circuit is conserved, so:

$$I_{\\rm in}(t) = I_{R} + I_{C}$$

From Ohm's Law, the membrane potential measured between the inside and outside of the neuron :math:`U_{\rm mem}` is proportional to the current through the resistor:

$$I_{R}(t) = \\frac{U_{\\rm mem}(t)}{R}$$

The capacitance is a proportionality constant between the charge stored on the capacitor :math:`Q` and :math:`U_{\rm mem}(t)`:


$$Q = CU_{\\rm mem}(t)$$

The rate of change of charge gives the capacitive current:

$$\\frac{dQ}{dt}=I_C(t) = C\\frac{dU_{\\rm mem}(t)}{dt}$$

Therefore:

$$I_{\\rm in}(t) = \\frac{U_{\\rm mem}(t)}{R} + C\\frac{dU_{\\rm mem}(t)}{dt}$$

$$\\implies RC \\frac{dU_{\\rm mem}(t)}{dt} = -U_{\\rm mem}(t) + RI_{\\rm in}(t)$$

The right hand side of the equation is of units **\[Voltage]**. On the left hand side of the equation, the term :math:`\frac{dU_{\rm mem}(t)}{dt}` is of units **\[Voltage/Time]**. To equate it to the left hand side (i.e., voltage), :math:`RC` must be of unit **\[Time]**. We refer to :math:`\tau = RC` as the time constant of the circuit:

$$ \\tau \\frac{dU_{\\rm mem}(t)}{dt} = -U_{\\rm mem}(t) + RI_{\\rm in}(t)$$

The passive membrane is therefore described by a linear differential equation.

For a derivative of a function to be of the same form as the original function, i.e., :math:`\frac{dU_{\rm mem}(t)}{dt} \propto U_{\rm mem}(t)`, this implies the solution is exponential with a time constant :math:`\tau`.

Say the neuron starts at some value :math:`U_{0}` with no further input, i.e., :math:`I_{\rm in}(t)=0`. The solution of the linear differential equation is:

$$U_{\\rm mem}(t) = U_0e^{-\\frac{t}{\\tau}}$$

The general solution is shown below.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_4_RCmembrane.png?raw=true
        :align: center
        :width: 450

        
2.3 Lapicque's LIF Neuron Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This similarity between nerve membranes and RC circuits was observed by `Louis Lapicque in 1907 <https://core.ac.uk/download/pdf/21172797.pdf>`_. He stimulated the nerve fiber of a frog with a brief electrical pulse, and found that membranes could be approximated as a capacitor with a leakage. We pay homage to his findings by naming the basic LIF neuron model in snnTorch after him. 

Most of the concepts in Lapicque's model carry forward to other LIF neuron models. Now let's simulate this neuron using snnTorch.

2.3.1 Lapicque: Without Stimulus
""""""""""""""""""""""""""""""""""

First, import the packages needed to run Lapicque's neuron model: snnTorch and PyTorch.

::

  import snntorch as snn
  import torch

The membrane potential has a time constant :math:`\tau = RC` associated with it. This can be equivalently represented by a decay rate :math:`\beta` that specifies the ratio of potential between subsequent time steps:

  $$\\beta = \\frac{U_0e^{-\\frac{1}{\\tau}}}{U_0e^{-\\frac{0}{\\tau}}} = \\frac{U_0e^{-\\frac{2}{\\tau}}}{U_0e^{-\\frac{1}{\\tau}}} = \\frac{U_0e^{-\\frac{3}{\\tau}}}{U_0e^{-\\frac{2}{\\tau}}}=~~...$$
  $$\\implies \\beta = e^{-\\frac{1}{\\tau}}$$
  
Setting :math:`\tau = 5\times 10^{-3} \implies \beta \approx 0.819`:

::

  # RC time constant
  tau_mem = 5e-3
  time_step = 1e-3 # one time step = 1ms

  # decay p/time step
  beta = float(torch.exp(torch.tensor(-time_step/tau_mem)))

  # Number of time steps to simulate
  num_steps = 200

::

  >>> print(f"Membrane decay rate ('beta'): {beta}")

  Membrane decay rate ('beta'): 0.8187307715415955

Instantiating Lapicque's neuron only requires the following line of code:

::

  # leaky integrate and fire neuron
  lif1 = snn.Lapicque(beta=beta)

The same thing can also be accomplished by specifying the parallel RC values:

::

  R = 5
  C = 1e-3

  lif1 = snn.Lapicque(R=R, C=C, time_step=time_step)

::

  >>> 
  print(f"Membrane decay rate ('beta'): {lif1.beta[0]}")
  Membrane decay rate ('beta'): 0.8187307715415955

To use this neuron: 

**Inputs**

* :code:`spk_in`: each element of :math:`I_{\rm in}`, which are all :code:`0` for now, is sequentially passed as an input

* :code:`mem`: the membrane potential at the present time :math:`t` is also passed as input. Initialize it arbitrarily as :math:`U_0 = 0.9~V`.

**Outputs**

* :code:`spk_out`: output spike :math:`S_{\rm out}[t+1]` at the next time step ('1' if there is a spike; '0' if there is no spike)

* :code:`mem`: membrane potential :math:`U_{\rm mem}[t+1]` at the next time step

These all need to be of type :code:`torch.Tensor`.
  
::

  # Initialize membrane, input, and output
  mem = torch.ones(1) * 0.9  # membrane potential of 0.9 at t=0
  cur_in = torch.zeros(num_steps)  # input is 0 for all t 
  spk_out = torch.zeros(1)  # neuron needs somewhere to sequentially dump its output spikes

These values are only for the initial time step :math:`t=0`. We'd like to watch the evolution of :code:`mem` over time. The list :code:`mem_rec` is initialized to record these values at every time step.

::

  # Initialize somewhere to store recordings of membrane potential
  mem_rec = [mem]

  Now it's time to run a simulation! 200 time steps will be simulated, updating :code:`mem` at each step and recording its value in :code:`mem_rec`:

::

  # pass updated value of mem and cur_in[step]=0 at every time step
  for step in range(num_steps):
    spk_out, mem = lif1(cur_in[step], mem)

    # Store recordings of membrane potential
    mem_rec.append(mem)

Let's take a look at how the membrane potential and synaptic current evolved.

::

  import matplotlib.pyplot as plt

  plt.title("Lapicque's Neuron Model Without Stimulus")
  plt.plot(mem_rec, label="Membrane Potential")
  plt.xlabel("Time step")
  plt.ylabel("Membrane Potential")
  plt.xlim([0, 50])
  plt.ylim([0, 1])
  plt.show()


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/rc_decay.png?raw=true
        :align: center
        :width: 450

This matches the dynamics that were previously derived. We've proven to ourselves that the membrane potential will decay over time in the absence of any input stimuli. 

The rest of this tutorial is under construction. Please refer to the Colab version linked at the top of this page for the full version.