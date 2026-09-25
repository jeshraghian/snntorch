===========================
snntorch.weight_init
===========================

.. automodule:: snntorch.weight_init
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

Weight initialization functions tailored for spiking neural networks.
These functions initialize synaptic weights based on the neuron's membrane
time constant (``tau_mem``), synaptic time constant (``tau_syn``), and
presynaptic firing rate (``nu``), placing neurons in the fluctuation-driven
optimal regime for surrogate gradient learning.

The fluctuation-driven regime places neurons near (but below) the spike
threshold, where they are most sensitive to input changes. This initialization
method accounts for SNN-specific neuron dynamics, unlike standard initializers
(e.g., Kaiming, Xavier) which assume rate-based activations.

Based on:
   Rossbroich, J., Gygax, J. & Zenke, F. (2022).
   *Fluctuation-driven initialization for spiking neural network training.*
   Neuromorphic Computing and Engineering, 2(4), 044016.

Usage Example
-------------

.. code-block:: python

   import torch
   import torch.nn as nn
   import snntorch as snn
   from snntorch import weight_init

   # Create a simple SNN
   net = nn.Sequential(
       nn.Linear(784, 256),
       snn.Leaky(beta=0.95),
       nn.Linear(256, 128),
       snn.Leaky(beta=0.95),
       nn.Linear(128, 10),
       snn.Leaky(beta=0.95),
   )

   # Apply fluctuation-driven initialization to all Linear layers
   for layer in net:
       if isinstance(layer, nn.Linear):
           weight_init.fluctuation_driven_normal_centered_(
               layer.weight,
               tau_mem=0.02,   # 20ms membrane time constant
               tau_syn=0.005,  # 5ms synaptic time constant
               nu=10.0,        # 10 Hz estimated firing rate
               sigma_u=1.0,    # Target membrane std
           )

Mathematical Background
-----------------------

The membrane potential of a current-based LIF neuron driven by Poisson inputs
follows a Gaussian distribution with mean ``mu_u`` and variance ``sigma_u^2``:

.. math::

   \mu_U = n \cdot \mu_W \cdot \nu \cdot \bar{\epsilon}

   \sigma_U^2 = n \cdot (\sigma_W^2 + \mu_W^2) \cdot \nu \cdot \hat{\epsilon}

where:

- ``n`` is the number of presynaptic inputs (fan-in)
- ``nu`` is the presynaptic firing rate (Hz)
- ``epsilon_bar`` is the integral of the PSP kernel
- ``epsilon_hat`` is the integral of the squared PSP kernel

The initialization inverts these equations to compute weight parameters:

.. math::

   \mu_W = \frac{\mu_U}{n \cdot \nu \cdot \bar{\epsilon}}

   \sigma_W = \sqrt{\frac{\sigma_U^2}{n \cdot \nu \cdot \hat{\epsilon}} - \mu_W^2}

For the centered case (balanced excitation/inhibition, ``mu_u = 0``):

.. math::

   \sigma_W = \frac{\sigma_U}{\sqrt{n \cdot \nu \cdot \hat{\epsilon}}}

API Reference
-------------

.. autofunction:: snntorch.weight_init.fluctuation_driven_normal_

.. autofunction:: snntorch.weight_init.fluctuation_driven_normal_centered_

.. autofunction:: snntorch.weight_init.compute_psp_kernel_integrals
