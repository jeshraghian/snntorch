======================================================
Tutorial 2 - The Leaky Integrate-and-Fire Neuron
======================================================

Tutorial written by Jason K. Eshraghian (`www.jasoneshraghian.com <https://www.jasoneshraghian.com>`_)

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_2_lif_neuron.ipynb

The snnTorch tutorial series is based on the following paper. If you find these resources or code useful in your work, please consider citing the following source:

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. arXiv preprint arXiv:2109.12894,
    September 2021. <https://arxiv.org/abs/2109.12894>`_

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_2_lif_neuron.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


Introduction
-------------

In this tutorial, you will: 

* Learn the fundamentals of the leaky integrate-and-fire (LIF) neuron model 
* Use snnTorch to implement a first order LIF neuron

Install the latest PyPi distribution of snnTorch:

::

    $ pip install snntorch

::

    # imports
    import snntorch as snn
    from snntorch import spikeplot as splt
    from snntorch import spikegen
    
    import torch
    import torch.nn as nn
    
    import numpy as np
    import matplotlib.pyplot as plt


1. The Spectrum of Neuron Models
---------------------------------------

A large variety of neuron models are out there, ranging from
biophysically accurate models (i.e., the Hodgkin-Huxley models) to the
extremely simple artificial neuron that pervades all facets of modern
deep learning.

**Hodgkin-Huxley Neuron Models**\ :math:`-`\ While biophysical models
can reproduce electrophysiological results with a high degree of
accuracy, their complexity makes them difficult to use at present.

**Artificial Neuron Model**\ :math:`-`\ On the other end of the spectrum
is the artificial neuron. The inputs are multiplied by their
corresponding weights and passed through an activation function. This
simplification has enabled deep learning researchers to perform
incredible feats in computer vision, natural language processing, and
many other machine learning-domain tasks.

**Leaky Integrate-and-Fire Neuron Models**\ :math:`-`\ Somewhere in the
middle of the divide lies the leaky integrate-and-fire (LIF) neuron
model. It takes the sum of weighted inputs, much like the artificial
neuron. But rather than passing it directly to an activation function,
it will integrate the input over time with a leakage, much like an RC
circuit. If the integrated value exceeds a threshold, then the LIF
neuron will emit a voltage spike. The LIF neuron abstracts away the
shape and profile of the output spike; it is simply treated as a
discrete event. As a result, information is not stored within the spike,
but rather the timing (or frequency) of spikes. Simple spiking neuron
models have produced much insight into the neural code, memory, network
dynamics, and more recently, deep learning. The LIF neuron sits in the
sweet spot between biological plausibility and practicality.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_1_neuronmodels.png?raw=true
        :align: center
        :width: 1000

The different versions of the LIF model each have their own dynamics and
use-cases. snnTorch currently supports the following LIF neurons: 

* Lapicque’s RC model: ``snntorch.Lapicque`` 
* 1st-order model: ``snntorch.Leaky`` 
* Synaptic Conductance-based neuron model: ``snntorch.Synaptic``
* Recurrent 1st-order model: ``snntorch.RLeaky``
* Recurrent Synaptic Conductance-based neuron model: ``snntorch.RSynaptic``
* Alpha neuron model: ``snntorch.Alpha``

Several other non-LIF spiking neurons are also available. 
This tutorial focuses on the first of these models. This will
be used to build towards the other models in `subsequent tutorials <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_.

2. The Leaky Integrate-and-Fire Neuron Model
--------------------------------------------------

2.1 Spiking Neurons: Intuition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In our brains, a neuron might be connected to 1,000 :math:`-` 10,000
other neurons. If one neuron spikes, all downhill neurons might
feel it. But what determines whether a neuron spikes in the first place?
The past century of experiments demonstrate that if a neuron experiences
*sufficient* stimulus at its input, then it might become excited and fire its own spike. 

Where does this stimulus come from? It could be from:

* the sensory periphery, 
* an invasive electrode artificially stimulating the neuron, or in most cases, 
* from other pre-synaptic neurons.


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_2_intuition.png?raw=true
        :align: center
        :width: 600

Given that these spikes are very short bursts of electrical activity, it
is quite unlikely for all input spikes to arrive at the neuron body in
precise unison. This indicates the presence of temporal dynamics that
‘sustain’ the input spikes, kind of like a delay.

2.2 The Passive Membrane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like all cells, a neuron is surrounded by a thin membrane. This membrane
is a lipid bilayer that insulates the conductive saline solution within
the neuron from the extracellular medium. Electrically, the two
conductive solutions separated by an insulator act as a capacitor.

Another function of this membrane is to control what goes in and out of
this cell (e.g., ions such as Na\ :math:`^+`). The membrane is usually
impermeable to ions which blocks them from entering and exiting the
neuron body. But there are specific channels in the membrane that are
triggered to open by injecting current into the neuron. This charge
movement is electrically modelled by a resistor.


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_3_passivemembrane.png?raw=true
        :align: center
        :width: 450

The following block will derive the behaviour of a LIF neuron from
scratch. If you’d prefer to skip the math, then feel free to scroll on
by; we’ll take a more hands-on approach to understanding the LIF neuron
dynamics after the derivation.

------------------------

**Optional: Derivation of LIF Neuron Model**

Now say some arbitrary time-varying current :math:`I_{\rm in}(t)` is injected into the neuron, 
be it via electrical stimulation or from other neurons. The total current in the circuit is conserved, so:

.. math:: I_{\rm in}(t) = I_{R} + I_{C}

From Ohm's Law, the membrane potential measured between the inside 
and outside of the neuron :math:`U_{\rm mem}` is proportional to 
the current through the resistor:

.. math:: I_{R}(t) = \frac{U_{\rm mem}(t)}{R}

The capacitance is a proportionality constant between the charge 
stored on the capacitor :math:`Q` and :math:`U_{\rm mem}(t)`:

.. math:: Q = CU_{\rm mem}(t)

The rate of change of charge gives the capacitive current:

.. math:: \frac{dQ}{dt}=I_C(t) = C\frac{dU_{\rm mem}(t)}{dt}

Therefore:

.. math:: I_{\rm in}(t) = \frac{U_{\rm mem}(t)}{R} + C\frac{dU_{\rm mem}(t)}{dt}

.. math:: \implies RC \frac{dU_{\rm mem}(t)}{dt} = -U_{\rm mem}(t) + RI_{\rm in}(t)

The right hand side of the equation is of units 
**\[Voltage]**. On the left hand side of the equation, 
the term :math:`\frac{dU_{\rm mem}(t)}{dt}` is of units 
**\[Voltage/Time]**. To equate it to the left hand side (i.e., voltage), 
:math:`RC` must be of unit **\[Time]**. We refer to :math:`\tau = RC` as the time constant of the circuit:

.. math:: \tau \frac{dU_{\rm mem}(t)}{dt} = -U_{\rm mem}(t) + RI_{\rm in}(t)

The passive membrane is therefore described by a linear differential equation.

For a derivative of a function to be of the same form as the original function, 
i.e., :math:`\frac{dU_{\rm mem}(t)}{dt} \propto U_{\rm mem}(t)`, this implies 
the solution is exponential with a time constant :math:`\tau`.

Say the neuron starts at some value :math:`U_{0}` with no further input, 
i.e., :math:`I_{\rm in}(t)=0.` The solution of the linear differential equation is:

.. math:: U_{\rm mem}(t) = U_0e^{-\frac{t}{\tau}}

The general solution is shown below.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_4_RCmembrane.png?raw=true
        :align: center
        :width: 450

------------------------


**Optional: Forward Euler Method to Solving the LIF Neuron Model**

We managed to find the analytical solution to the LIF neuron, but it is 
unclear how this might be useful in a neural network. This time,
let’s instead use the forward Euler method to solve the previous linear
ordinary differential equation (ODE). This approach might seem
arduous, but it gives us a discrete, recurrent representation of the LIF
neuron. Once we reach this solution, it can be applied directly to a neural
network. As before, the linear ODE describing the RC circuit is:

.. math:: \tau \frac{dU(t)}{dt} = -U(t) + RI_{\rm in}(t)

The subscript from :math:`U(t)` is omitted for simplicity.

First, let’s solve this derivative without taking the limit
:math:`\Delta t \rightarrow 0`:

.. math:: \tau \frac{U(t+\Delta t)-U(t)}{\Delta t} = -U(t) + RI_{\rm in}(t)

For a small enough :math:`\Delta t`, this gives a good enough
approximation of continuous-time integration. Isolating the membrane at
the following time step gives:

.. math:: U(t+\Delta t) = U(t) + \frac{\Delta t}{\tau}\big(-U(t) + RI_{\rm in}(t)\big)

The following function represents this equation:

::

    def leaky_integrate_neuron(U, time_step=1e-3, I=0, R=5e7, C=1e-10):
      tau = R*C
      U = U + (time_step/tau)*(-U + I*R)
      return U

The default values are set to :math:`R=50 M\Omega` and
:math:`C=100pF` (i.e., :math:`\tau=5ms`). These are quite
realistic with respect to biological neurons.

Now loop through this function, iterating one time step at a time.
The membrane potential is initialized at :math:`U=0.9 V`, with the assumption that
there is no injected input current, :math:`I_{\rm in}=0 A`.
The simulation is performed with a millisecond precision
:math:`\Delta t=1\times 10^{-3}`\ s.

::

    num_steps = 100
    U = 0.9
    U_trace = []  # keeps a record of U for plotting
    
    for step in range(num_steps):
      U_trace.append(U)
      U = leaky_integrate_neuron(U)  # solve next step of U
    
    plot_mem(U_trace, "Leaky Neuron Model")


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/leaky1.png?raw=true
        :align: center
        :width: 300

This exponential decay seems to match what we expected!

3 Lapicque’s LIF Neuron Model
--------------------------------

This similarity between nerve membranes and RC circuits was observed by
`Louis Lapicque in
1907 <https://core.ac.uk/download/pdf/21172797.pdf>`__. He stimulated
the nerve fiber of a frog with a brief electrical pulse, and found that neuron
membranes could be approximated as a capacitor with a leakage. We pay
homage to his findings by naming the basic LIF neuron model in snnTorch
after him.

Most of the concepts in Lapicque’s model carry forward to other LIF
neuron models. Now it's time to simulate this neuron using snnTorch.

3.1 Lapicque: Without Stimulus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instantiate Lapicque’s neuron using the following line of code.
R & C are modified to simpler values, while keeping the previous time
constant of :math:`\tau=5\times10^{-3}`\ s.

::

    time_step = 1e-3
    R = 5
    C = 1e-3
    
    # leaky integrate and fire neuron, tau=5e-3
    lif1 = snn.Lapicque(R=R, C=C, time_step=time_step)

The neuron model is now stored in ``lif1``. To use this neuron:

**Inputs** 

* ``spk_in``: each element of :math:`I_{\rm in}` is sequentially passed as an input (0 for now) 
* ``mem``: the membrane potential, previously :math:`U[t]`, is also passed as input. Initialize it arbitrarily as :math:`U[0] = 0.9~V`.

**Outputs** 

* ``spk_out``: output spike :math:`S_{\rm out}[t+\Delta t]` at the next time step (‘1’ if there is a spike; ‘0’ if there is no spike) 
* ``mem``: membrane potential :math:`U_{\rm mem}[t+\Delta t]` at the next time step

These all need to be of type ``torch.Tensor``.

::

    # Initialize membrane, input, and output
    mem = torch.ones(1) * 0.9  # U=0.9 at t=0
    cur_in = torch.zeros(num_steps)  # I=0 for all t 
    spk_out = torch.zeros(1)  # initialize output spikes

These values are only for the initial time step :math:`t=0`. 
To analyze the evolution of ``mem`` over time, create a list ``mem_rec`` to record these values at every time step.

::

    # A list to store a recording of membrane potential
    mem_rec = [mem]

Now it’s time to run a simulation! At each time step, ``mem`` is
updated and stored in ``mem_rec``:

::

    # pass updated value of mem and cur_in[step]=0 at every time step
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in[step], mem)
    
      # Store recordings of membrane potential
      mem_rec.append(mem)
    
    # convert the list of tensors into one tensor
    mem_rec = torch.stack(mem_rec)
    
    # pre-defined plotting function
    plot_mem(mem_rec, "Lapicque's Neuron Model Without Stimulus")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque.png?raw=true
        :align: center
        :width: 300

The membrane potential decays over time in the absence of any input
stimuli.

3.2 Lapicque: Step Input
~~~~~~~~~~~~~~~~~~~~~~~~~~

Now apply a step current :math:`I_{\rm in}(t)` that switches on at
:math:`t=t_0`. Given the linear first-order differential equation:

.. math::  \tau \frac{dU_{\rm mem}}{dt} = -U_{\rm mem} + RI_{\rm in}(t),

the general solution is:

.. math:: U_{\rm mem}=I_{\rm in}(t)R + [U_0 - I_{\rm in}(t)R]e^{-\frac{t}{\tau}}

If the membrane potential is initialized to
:math:`U_{\rm mem}(t=0) = 0 V`, then:

.. math:: U_{\rm mem}(t)=I_{\rm in}(t)R [1 - e^{-\frac{t}{\tau}}]

Based on this explicit time-dependent form, we expect
:math:`U_{\rm mem}` to relax exponentially towards :math:`I_{\rm in}R`.
Let’s visualize what this looks like by triggering a current pulse of
:math:`I_{in}=100mA` at :math:`t_0 = 10ms`.

::

    # Initialize input current pulse
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.1), 0)  # input current turns on at t=10
    
    # Initialize membrane, output and recordings
    mem = torch.zeros(1)  # membrane potential of 0 at t=0
    spk_out = torch.zeros(1)  # neuron needs somewhere to sequentially dump its output spikes
    mem_rec = [mem]

This time, the new values of ``cur_in`` are passed to the neuron:

::

    num_steps = 200
    
    # pass updated value of mem and cur_in[step] at every time step
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in[step], mem)
      mem_rec.append(mem)
    
    # crunch -list- of tensors into one tensor
    mem_rec = torch.stack(mem_rec)
    
    plot_step_current_response(cur_in, mem_rec, 10)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_step.png?raw=true
        :align: center
        :width: 450

As :math:`t\rightarrow \infty`, the membrane potential
:math:`U_{\rm mem}` exponentially relaxes to :math:`I_{\rm in}R`:

::

    >>> print(f"The calculated value of input pulse [A] x resistance [Ω] is: {cur_in[11]*lif1.R} V")
    >>> print(f"The simulated value of steady-state membrane potential is: {mem_rec[200][0]} V")
    
    The calculated value of input pulse [A] x resistance [Ω] is: 0.5 V
    The simulated value of steady-state membrane potential is: 0.4999999403953552 V

Close enough!

3.3 Lapicque: Pulse Input
~~~~~~~~~~~~~~~~~~~~~~~~~~

Now what if the step input was clipped at :math:`t=30ms`?

::

    # Initialize current pulse, membrane and outputs
    cur_in1 = torch.cat((torch.zeros(10), torch.ones(20)*(0.1), torch.zeros(170)), 0)  # input turns on at t=10, off at t=30
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec1 = [mem]

::

    # neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in1[step], mem)
      mem_rec1.append(mem)
    mem_rec1 = torch.stack(mem_rec1)
    
    plot_current_pulse_response(cur_in1, mem_rec1, "Lapicque's Neuron Model With Input Pulse", 
                                vline1=10, vline2=30)


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_pulse1.png?raw=true
        :align: center
        :width: 450

:math:`U_{\rm mem}` rises just as it did for the step input, but now it
decays with a time constant of :math:`\tau` as in our first simulation.

Let’s deliver approximately the same amount of charge
:math:`Q = I \times t` to the circuit in half the time. This means the
input current amplitude must be increased by a little, and the
time window must be decreased.

::

    # Increase amplitude of current pulse; half the time.
    cur_in2 = torch.cat((torch.zeros(10), torch.ones(10)*0.111, torch.zeros(180)), 0)  # input turns on at t=10, off at t=20
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec2 = [mem]
    
    # neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in2[step], mem)
      mem_rec2.append(mem)
    mem_rec2 = torch.stack(mem_rec2)
    
    plot_current_pulse_response(cur_in2, mem_rec2, "Lapicque's Neuron Model With Input Pulse: x1/2 pulse width",
                                vline1=10, vline2=20)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_pulse2.png?raw=true
        :align: center
        :width: 450


Let’s do that again, but with an even faster input pulse and higher
amplitude:

::

    # Increase amplitude of current pulse; quarter the time.
    cur_in3 = torch.cat((torch.zeros(10), torch.ones(5)*0.147, torch.zeros(185)), 0)  # input turns on at t=10, off at t=15
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec3 = [mem]
    
    # neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in3[step], mem)
      mem_rec3.append(mem)
    mem_rec3 = torch.stack(mem_rec3)
    
    plot_current_pulse_response(cur_in3, mem_rec3, "Lapicque's Neuron Model With Input Pulse: x1/4 pulse width",
                                vline1=10, vline2=15)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_pulse3.png?raw=true
        :align: center
        :width: 450


Now compare all three experiments on the same plot:


::

    compare_plots(cur_in1, cur_in2, cur_in3, mem_rec1, mem_rec2, mem_rec3, 10, 15, 
                  20, 30, "Lapicque's Neuron Model With Input Pulse: Varying inputs")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/compare_pulse.png?raw=true
        :align: center
        :width: 450

As the input current pulse amplitude increases, the rise time of the
membrane potential speeds up. In the limit of the input current pulse
width becoming infinitesimally small, :math:`T_W \rightarrow 0s`, the
membrane potential will jump straight up in virtually zero rise time:

::

    # Current spike input
    cur_in4 = torch.cat((torch.zeros(10), torch.ones(1)*0.5, torch.zeros(189)), 0)  # input only on for 1 time step
    mem = torch.zeros(1) 
    spk_out = torch.zeros(1)
    mem_rec4 = [mem]
    
    # neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in4[step], mem)
      mem_rec4.append(mem)
    mem_rec4 = torch.stack(mem_rec4)
    
    plot_current_pulse_response(cur_in4, mem_rec4, "Lapicque's Neuron Model With Input Spike", 
                                vline1=10, ylim_max1=0.6)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_spike.png?raw=true
        :align: center
        :width: 450


The current pulse width is now so short, it effectively looks like a
spike. That is to say, charge is delivered in an infinitely short period
of time, :math:`I_{\rm in}(t) = Q/t_0` where :math:`t_0 \rightarrow 0`.
More formally:

.. math:: I_{\rm in}(t) = Q \delta (t-t_0),

where :math:`\delta (t-t_0)` is the Dirac-Delta function. Physically, it
is impossible to ‘instantaneously’ deposit charge. But integrating
:math:`I_{\rm in}` gives a result that makes physical sense, as we can
obtain the charge delivered:

.. math:: 1 = \int^{t_0 + a}_{t_0 - a}\delta(t-t_0)dt

.. math:: f(t_0) = \int^{t_0 + a}_{t_0 - a}f(t)\delta(t-t_0)dt

Here,
:math:`f(t_0) = I_{\rm in}(t_0=10) = 0.5A \implies f(t) = Q = 0.5C`.

Hopefully you have a good feel of how the membrane potential leaks at
rest, and integrates the input current. That covers the ‘leaky’ and
‘integrate’ part of the neuron. How about the fire?

3.4 Lapicque: Firing
~~~~~~~~~~~~~~~~~~~~~~

So far, we have only seen how a neuron will react to spikes at the
input. For a neuron to generate and emit its own spikes at the output,
the passive membrane model must be combined with a threshold.

If the membrane potential exceeds this threshold, then a voltage spike
will be generated, external to the passive membrane model.


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_4_spiking.png?raw=true
        :align: center
        :width: 400

Modify the ``leaky_integrate_neuron`` function from before to add
a spike response.

::

    # R=5.1, C=5e-3 for illustrative purposes
    def leaky_integrate_and_fire(mem, cur=0, threshold=1, time_step=1e-3, R=5.1, C=5e-3):
      tau_mem = R*C
      spk = (mem > threshold) # if membrane exceeds threshold, spk=1, else, 0
      mem = mem + (time_step/tau_mem)*(-mem + cur*R)
      return mem, spk

Set ``threshold=1``, and apply a step current to get this neuron
spiking.

::

    # Small step current input
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.2), 0)
    mem = torch.zeros(1)
    mem_rec = []
    spk_rec = []
    
    # neuron simulation
    for step in range(num_steps):
      mem, spk = leaky_integrate_and_fire(mem, cur_in[step])
      mem_rec.append(mem)
      spk_rec.append(spk)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, vline=109, ylim_max2=1.3, 
                     title="LIF Neuron Model With Uncontrolled Spiking")


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lif_uncontrolled.png?raw=true
        :align: center
        :width: 450


Oops - the output spikes have gone out of control! This is
because we forgot to add a reset mechanism. In reality, each time a
neuron fires, the membrane potential hyperpolarizes back to its resting
potential.

Implementing this reset mechanism into our neuron:

::

    # LIF w/Reset mechanism
    def leaky_integrate_and_fire(mem, cur=0, threshold=1, time_step=1e-3, R=5.1, C=5e-3):
      tau_mem = R*C
      spk = (mem > threshold)
      mem = mem + (time_step/tau_mem)*(-mem + cur*R) - spk*threshold  # every time spk=1, subtract the threhsold
      return mem, spk

::

    # Small step current input
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.2), 0)
    mem = torch.zeros(1)
    mem_rec = []
    spk_rec = []
    
    # neuron simulation
    for step in range(num_steps):
      mem, spk = leaky_integrate_and_fire(mem, cur_in[step])
      mem_rec.append(mem)
      spk_rec.append(spk)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, vline=109, ylim_max2=1.3, 
                     title="LIF Neuron Model With Reset")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/reset_2.png?raw=true
        :align: center
        :width: 450

Bam. We now have a functional leaky integrate-and-fire neuron model!

Note that if :math:`I_{\rm in}=0.2 A` and :math:`R<5 \Omega`, then
:math:`I\times R < 1 V`. If ``threshold=1``, then no spiking would
occur. Feel free to go back up, change the values, and test it out.

As before, all of that code is condensed by calling the built-in Lapicque neuron model from snnTorch:

::

    # Create the same neuron as before using snnTorch
    lif2 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3)
    
    >>> print(f"Membrane potential time constant: {lif2.R * lif2.C:.3f}s")
    "Membrane potential time constant: 0.025s"

::

    # Initialize inputs and outputs
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.2), 0)
    mem = torch.zeros(1)
    spk_out = torch.zeros(1) 
    mem_rec = [mem]
    spk_rec = [spk_out]
    
    # Simulation run across 100 time steps.
    for step in range(num_steps):
      spk_out, mem = lif2(cur_in[step], mem)
      mem_rec.append(mem)
      spk_rec.append(spk_out)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, vline=109, ylim_max2=1.3, 
                     title="Lapicque Neuron Model With Step Input")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_reset.png?raw=true
        :align: center
        :width: 450

The membrane potential exponentially rises and then hits the threshold,
at which point it resets. We can roughly see this occurs between
:math:`105ms < t_{\rm spk} < 115ms`. As a matter of curiousity, let’s
see what the spike recording actually consists of:

::

    >>> print(spk_rec[105:115].view(-1))
    tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])

The absence of a spike is represented by :math:`S_{\rm out}=0`, and the
occurrence of a spike is :math:`S_{\rm out}=1`. Here, the spike occurs
at :math:`S_{\rm out}[t=109]=1`. If you are wondering why each of these entries is stored as a tensor, it
is because in future tutorials we will simulate large scale neural
networks. Each entry will contain the spike responses of many neurons,
and tensors can be loaded into GPU memory to speed up the training
process.

If :math:`I_{\rm in}` is increased, then the membrane potential
approaches the threshold :math:`U_{\rm thr}` faster:

::

    # Initialize inputs and outputs
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.3), 0)  # increased current
    mem = torch.zeros(1)
    spk_out = torch.zeros(1) 
    mem_rec = [mem]
    spk_rec = [spk_out]
    
    # neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif2(cur_in[step], mem)
      mem_rec.append(mem)
      spk_rec.append(spk_out)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, ylim_max2=1.3, 
                     title="Lapicque Neuron Model With Periodic Firing")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/periodic.png?raw=true
        :align: center
        :width: 450

A similar increase in firing frequency can also be induced by decreasing
the threshold. This requires initializing a new neuron model, but the
rest of the code block is the exact same as above:

::

    # neuron with halved threshold
    lif3 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3, threshold=0.5)
    
    # Initialize inputs and outputs
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.3), 0) 
    mem = torch.zeros(1)
    spk_out = torch.zeros(1) 
    mem_rec = [mem]
    spk_rec = [spk_out]
    
    # Neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif3(cur_in[step], mem)
      mem_rec.append(mem)
      spk_rec.append(spk_out)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=0.5, ylim_max2=1.3, 
                     title="Lapicque Neuron Model With Lower Threshold")


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/threshold.png?raw=true
        :align: center
        :width: 450

That’s what happens for a constant current injection. But in both deep
neural networks and in the biological brain, most neurons will be
connected to other neurons. They are more likely to receive spikes,
rather than injections of constant current.

3.5 Lapicque: Spike Inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s harness some of the skills we learnt in `Tutorial
1 <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb>`__,
and use the ``snntorch.spikegen`` module to create some randomly
generated input spikes.

::

    # Create a 1-D random spike train. Each element has a probability of 40% of firing.
    spk_in = spikegen.rate_conv(torch.ones((num_steps)) * 0.40)

Run the following code block to see how many spikes have been generated.

::

    >>> print(f"There are {int(sum(spk_in))} total spikes out of {len(spk_in)} time steps.")
    There are 85 total spikes out of 200 time steps.

::

    fig = plt.figure(facecolor="w", figsize=(8, 1))
    ax = fig.add_subplot(111)
    
    splt.raster(spk_in.reshape(num_steps, -1), ax, s=100, c="black", marker="|")
    plt.title("Input Spikes")
    plt.xlabel("Time step")
    plt.yticks([])
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/spikes.png?raw=true
        :align: center
        :width: 400

::

    # Initialize inputs and outputs
    mem = torch.ones(1)*0.5
    spk_out = torch.zeros(1)
    mem_rec = [mem]
    spk_rec = [spk_out]
    
    # Neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif3(spk_in[step], mem)
      spk_rec.append(spk_out)
      mem_rec.append(mem)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_spk_mem_spk(spk_in, mem_rec, spk_out, "Lapicque's Neuron Model With Input Spikes")


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/spk_mem_spk.png?raw=true
        :align: center
        :width: 450

3.6 Lapicque: Reset Mechanisms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We already implemented a reset mechanism from scratch, but let’s dive a
little deeper. This sharp drop of membrane potential promotes a
reduction of spike generation, which supplements part of the theory on
how brains are so power efficient. Biologically, this drop of membrane
potential is known as ‘hyperpolarization’. Following that, it is
momentarily more difficult to elicit another spike from the neuron.
Here, we use a reset mechanism to model hyperpolarization.

There are two ways to implement the reset mechanism:

1. *reset by subtraction* (default) :math:`-` subtract the threshold
   from the membrane potential each time a spike is generated;
2. *reset to zero* :math:`-` force the membrane potential to zero each
   time a spike is generated.
3. *no reset* :math:`-` do nothing, and let the firing go potentially uncontrolled.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_5_reset.png?raw=true
        :align: center
        :width: 400

Instantiate another neuron model to demonstrate how to alternate
between reset mechanisms. By default, snnTorch neuron models use ``reset_mechanism = "subtract"``.
This can be explicitly overridden by passing the argument
``reset_mechanism =  "zero"``.

::

    # Neuron with reset_mechanism set to "zero"
    lif4 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3, threshold=0.5, reset_mechanism="zero")
    
    # Initialize inputs and outputs
    spk_in = spikegen.rate_conv(torch.ones((num_steps)) * 0.40)
    mem = torch.ones(1)*0.5
    spk_out = torch.zeros(1)
    mem_rec0 = [mem]
    spk_rec0 = [spk_out]
    
    # Neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif4(spk_in[step], mem)
      spk_rec0.append(spk_out)
      mem_rec0.append(mem)
    
    # convert lists to tensors
    mem_rec0 = torch.stack(mem_rec0)
    spk_rec0 = torch.stack(spk_rec0)
    
    plot_reset_comparison(spk_in, mem_rec, spk_rec, mem_rec0, spk_rec0)


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/comparison.png?raw=true
        :align: center
        :width: 550

Pay close attention to the evolution of the membrane potential,
especially in the moments after it reaches the threshold. You may notice
that for “Reset to Zero”, the membrane potential is forced back to zero
after each spike.

So which one is better? Applying ``"subtract"`` (the default value in
``reset_mechanism``) is less lossy, because it does not ignore how much
the membrane exceeds the threshold by.

On the other hand, applying a hard reset with ``"zero"`` promotes
sparsity and potentially less power consumption when running on
dedicated neuromorphic hardware. Both options are available for you to
experiment with.

That covers the basics of a LIF neuron model!

Conclusion
---------------

In practice, we probably wouldn’t use this neuron model to train a
neural network. The Lapicque LIF model has added a lot of
hyperparameters to tune: :math:`R`, :math:`C`, :math:`\Delta t`,
:math:`U_{\rm thr}`, and the choice of reset mechanism. It’s all a
little bit daunting. So the `next tutorial <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_ will eliminate most of these
hyperparameters, and introduce a neuron model that is better suited for
large-scale deep learning.

If you like this project, please consider starring ⭐ the repo on GitHub as it is the easiest and best way to support it.

For reference, the documentation `can be found
here <https://snntorch.readthedocs.io/en/latest/snntorch.html>`__.

Further Reading
---------------

-  `Check out the snnTorch GitHub project here. <https://github.com/jeshraghian/snntorch>`__
-  `snnTorch
   documentation <https://snntorch.readthedocs.io/en/latest/snntorch.html>`__
   of the Lapicque, Leaky, Synaptic, and Alpha models
-  `Neuronal Dynamics: From single neurons to networks and models of
   cognition <https://neuronaldynamics.epfl.ch/index.html>`__ by Wulfram
   Gerstner, Werner M. Kistler, Richard Naud and Liam Paninski.
-  `Theoretical Neuroscience: Computational and Mathematical Modeling of
   Neural
   Systems <https://mitpress.mit.edu/books/theoretical-neuroscience>`__
   by Laurence F. Abbott and Peter Dayan
