============================
Regression with SNNs: Part I
============================

Learning Membrane Potentials with LIF Neurons
---------------------------------------------

Tutorial written by Alexander Henkes (`ORCID <https://orcid.org/0000-0003-4615-9271>`_) and Jason K. Eshraghian (`ncg.ucsc.edu <https://ncg.ucsc.edu>`_)


.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_1.ipynb


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
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_1.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


In the regression tutorial series, you will learn how to use snnTorch to
perform regression using a variety of spiking neuron models, including:

-  Leaky Integrate-and-Fire (LIF) Neurons
-  Recurrent LIF Neurons
-  Spiking LSTMs

An overview of the regression tutorial series:

-  Part I (this tutorial) will train the membrane potential of a LIF
   neuron to follow a given trajectory over time.
-  Part II will use LIF neurons with recurrent feedback to perform classification using regression-based loss functions
-  Part III will use a more complex spiking LSTM network instead to train the firing time of a neuron.

::

    !pip install snntorch --quiet

::

    # imports
    import snntorch as snn
    from snntorch import surrogate
    from snntorch import functional as SF
    from snntorch import utils
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torch.nn.functional as F
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    import random
    import statistics
    import tqdm

Fix the random seed:

::

    # Seed
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

1. Spiking Regression
----------------------

1.1 A Quick Background on Linear and Nonlinear Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tutorials so far have focused on multi-class classification
problems. But if you’ve made it this far, then it’s probably safe to say
that your brain can do more than distinguish cats and dogs. You’re
amazing and we believe in you.

An alternative problem is regression, where multiple input features
:math:`x_i` are used to estimate an output on a continuous number line
:math:`y \in \mathbb{R}`. A classic example is estimating the price of a
house, given a bunch of inputs such as land size, number of rooms, and
the local demand for avocado toast.

The objective of a regression problem is often the mean-square error:

.. math:: \mathcal{L}_{MSE} = \frac{1}{n}\sum_{i=1}^n(y_i-\hat{y_i})^2

or the mean absolute error:

.. math:: \mathcal{L}_{L1} = \frac{1}{n}\sum_{i=1}^n|y_i-\hat{y_i}|

where :math:`y` is the target and :math:`\hat{y}` is the predicted
value.

One of the challenges of linear regression is that it can only use
linear weightings of input features in predicting the output. Using a
neural network trained using the mean-square error as the cost function
allows us to perform nonlinear regression on more complex data.

1.2 Spiking Neurons in Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spikes are a type of nonlinearity that can also be used to learn more
complex regression tasks. But if spiking neurons only emit spikes that
are represented with 1’s and 0’s, then how might we perform regression?
I’m glad you asked! Here are a few ideas:

-  Use the total number of spikes (a rate-based code)
-  Use the time of the spike (a temporal/latency-based code)
-  Use the distance between pairs of spikes (i.e., using the interspike
   interval)

Or perhaps you pierce the neuron membrane with an electrical probe and
decide to use the membrane potential instead, which is a continuous
value.

   Note: is it cheating to directly access the membrane potential, i.e.,
   something that is meant to be a ‘hidden state’? At this time, there
   isn’t much consensus in the neuromorphic community. Despite being a
   high precision variable in many models (and thus computationally
   expensive), the membrane potential is commonly used in loss functions
   as it is a more ‘continuous’ variable compared to discrete time steps
   or spike counts. While it costs more in terms of power and latency to
   operate on higher-precision values, the impact might be minor if you
   have a small output layer, or if the output does not need to be
   scaled by weights. It really is a task-specific and hardware-specific
   question.

2. Setting up the Regression Problem
------------------------------------------------

2.1 Create Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s construct a simple toy problem. The following class returns the
function we are hoping to learn. If ``mode = "linear"``, a straight line
with a random slope is generated. If ``mode = "sqrt"``, then the square
root of this straight line is taken instead.

Our goal: train a leaky integrate-and-fire neuron such that its membrane
potential follows the sample over time.

::

    class RegressionDataset(torch.utils.data.Dataset):
        """Simple regression dataset."""
    
        def __init__(self, timesteps, num_samples, mode):
            """Linear relation between input and output"""
            self.num_samples = num_samples # number of generated samples
            feature_lst = [] # store each generated sample in a list
    
            # generate linear functions one by one
            for idx in range(num_samples):
                end = float(torch.rand(1)) # random final point
                lin_vec = torch.linspace(start=0.0, end=end, steps=timesteps) # generate linear function from 0 to end
                feature = lin_vec.view(timesteps, 1)
                feature_lst.append(feature) # add sample to list
    
            self.features = torch.stack(feature_lst, dim=1) # convert list to tensor
    
            # option to generate linear function or square-root function
            if mode == "linear":
                self.labels = self.features * 1
    
            elif mode == "sqrt":
                slope = float(torch.rand(1))
                self.labels = torch.sqrt(self.features * slope)
    
            else:
                raise NotImplementedError("'linear', 'sqrt'")
    
        def __len__(self):
            """Number of samples."""
            return self.num_samples
    
        def __getitem__(self, idx):
            """General implementation, but we only have one sample."""
            return self.features[:, idx, :], self.labels[:, idx, :]


To see what a random sample looks like, run the following code-block:

::

    num_steps = 50
    num_samples = 1
    mode = "sqrt" # 'linear' or 'sqrt'
    
    # generate a single data sample
    dataset = RegressionDataset(timesteps=num_steps, num_samples=num_samples, mode=mode)
    
    # plot
    sample = dataset.labels[:, 0, 0]
    plt.plot(sample)
    plt.title("Target function to teach network")
    plt.xlabel("Time")
    plt.ylabel("Membrane Potential")
    plt.show()


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/regression1/reg_1-1.png?raw=true
        :align: center
        :width: 450

2.2 Create DataLoader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Dataset objects created above load data into memory, and the
DataLoader will serve it up in batches. DataLoaders in PyTorch are a
handy interface for passing data into a network. They return an iterator
divided up into mini-batches of size ``batch_size``.

::

    batch_size = 1 # only one sample to learn
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True)

3. Construct Model
------------------------

Let us try a simple network using only leaky integrate-and-fire layers
without recurrence. Subsequent tutorials will show how to use more
complex neuron types with higher-order recurrence. These architectures
should work just fine, if there is no strong time dependency in the
data, i.e., the next time step has weak dependence on the previous one.

A few notes on the architecture below:

-  Setting ``learn_beta=True`` enables the decay rate ``beta`` to be a
   learnable parameter
-  Each neuron has a unique, and randomly initialized threshold and
   decay rate
-  The output layer has the reset mechanism disabled by setting
   ``reset_mechanism="none"`` as we will not use any output spikes

::

    class Net(torch.nn.Module):
        """Simple spiking neural network in snntorch."""
    
        def __init__(self, timesteps, hidden):
            super().__init__()
            
            self.timesteps = timesteps # number of time steps to simulate the network
            self.hidden = hidden # number of hidden neurons 
            spike_grad = surrogate.fast_sigmoid() # surrogate gradient function
            
            # randomly initialize decay rate and threshold for layer 1
            beta_in = torch.rand(self.hidden)
            thr_in = torch.rand(self.hidden)
    
            # layer 1
            self.fc_in = torch.nn.Linear(in_features=1, out_features=self.hidden)
            self.lif_in = snn.Leaky(beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=spike_grad)
            
            # randomly initialize decay rate and threshold for layer 2
            beta_hidden = torch.rand(self.hidden)
            thr_hidden = torch.rand(self.hidden)
    
            # layer 2
            self.fc_hidden = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
            self.lif_hidden = snn.Leaky(beta=beta_hidden, threshold=thr_hidden, learn_beta=True, spike_grad=spike_grad)
    
            # randomly initialize decay rate for output neuron
            beta_out = torch.rand(1)
            
            # layer 3: leaky integrator neuron. Note the reset mechanism is disabled and we will disregard output spikes.
            self.fc_out = torch.nn.Linear(in_features=self.hidden, out_features=1)
            self.li_out = snn.Leaky(beta=beta_out, threshold=1.0, learn_beta=True, spike_grad=spike_grad, reset_mechanism="none")
    
        def forward(self, x):
            """Forward pass for several time steps."""
    
            # Initalize membrane potential
            mem_1 = self.lif_in.init_leaky()
            mem_2 = self.lif_hidden.init_leaky()
            mem_3 = self.li_out.init_leaky()
    
            # Empty lists to record outputs
            mem_3_rec = []
    
            # Loop over 
            for step in range(self.timesteps):
                x_timestep = x[step, :, :]
    
                cur_in = self.fc_in(x_timestep)
                spk_in, mem_1 = self.lif_in(cur_in, mem_1)
                
                cur_hidden = self.fc_hidden(spk_in)
                spk_hidden, mem_2 = self.lif_hidden(cur_hidden, mem_2)
    
                cur_out = self.fc_out(spk_hidden)
                _, mem_3 = self.li_out(cur_out, mem_3)
    
                mem_3_rec.append(mem_3)
    
            return torch.stack(mem_3_rec)

Instantiate the network below:

::

    hidden = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = Net(timesteps=num_steps, hidden=hidden).to(device)


Let’s observe the behavior of the output neuron before it has been
trained and how it compares to the target function:

::

    train_batch = iter(dataloader)
    
    # run a single forward-pass
    with torch.no_grad():
        for feature, label in train_batch:
            feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
            label = torch.swapaxes(input=label, axis0=0, axis1=1)
            feature = feature.to(device)
            label = label.to(device)
            mem = model(feature)
    
    # plot
    plt.plot(mem[:, 0, 0].cpu(), label="Output")
    plt.plot(label[:, 0, 0].cpu(), '--', label="Target")
    plt.title("Untrained Output Neuron")
    plt.xlabel("Time")
    plt.ylabel("Membrane Potential")
    plt.legend(loc='best')
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/regression1/reg_1-2.png?raw=true
        :align: center
        :width: 450

As the network has not yet been trained, it is unsurprising the membrane
potential follows a senseless evolution.

4. Construct Training Loop
------------------------------------------------

We call ``torch.nn.MSELoss()`` to minimize the mean square error between
the membrane potential and the target evolution.

We iterate over the same sample of data.

::

    num_iter = 100 # train for 100 iterations
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    loss_function = torch.nn.MSELoss()
    
    loss_hist = [] # record loss
    
    # training loop
    with tqdm.trange(num_iter) as pbar:
        for _ in pbar:
            train_batch = iter(dataloader)
            minibatch_counter = 0
            loss_epoch = []
            
            for feature, label in train_batch:
                # prepare data
                feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
                label = torch.swapaxes(input=label, axis0=0, axis1=1)
                feature = feature.to(device)
                label = label.to(device)
    
                # forward pass
                mem = model(feature)
                loss_val = loss_function(mem, label) # calculate loss
                optimizer.zero_grad() # zero out gradients
                loss_val.backward() # calculate gradients
                optimizer.step() # update weights
    
                # store loss
                loss_hist.append(loss_val.item())
                loss_epoch.append(loss_val.item())
                minibatch_counter += 1
    
                avg_batch_loss = sum(loss_epoch) / minibatch_counter # calculate average loss p/epoch
                pbar.set_postfix(loss="%.3e" % avg_batch_loss) # print loss p/batch

5. Evaluation
------------------------

::

    loss_function = torch.nn.L1Loss() # Use L1 loss instead
    
     # pause gradient calculation during evaluation
    with torch.no_grad():
        model.eval()
    
        test_batch = iter(dataloader)
        minibatch_counter = 0
        rel_err_lst = []
    
        # loop over data samples
        for feature, label in test_batch:
    
            # prepare data
            feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
            label = torch.swapaxes(input=label, axis0=0, axis1=1)
            feature = feature.to(device)
            label = label.to(device)
    
            # forward-pass
            mem = model(feature)
    
            # calculate relative error
            rel_err = torch.linalg.norm(
                (mem - label), dim=-1
            ) / torch.linalg.norm(label, dim=-1)
            rel_err = torch.mean(rel_err[1:, :])
    
            # calculate loss
            loss_val = loss_function(mem, label)
    
            # store loss
            loss_hist.append(loss_val.item())
            rel_err_lst.append(rel_err.item())
            minibatch_counter += 1
    
        mean_L1 = statistics.mean(loss_hist)
        mean_rel = statistics.mean(rel_err_lst)
    
    print(f"{'Mean L1-loss:':<{20}}{mean_L1:1.2e}")
    print(f"{'Mean rel. err.:':<{20}}{mean_rel:1.2e}")


::

    >> Mean L1-loss:       1.22e-02
    >> Mean rel. err.:     2.84e-02

Let’s plot our results for some visual intuition:

::

    mem = mem.cpu()
    label = label.cpu()
    
    plt.title("Trained Output Neuron")
    plt.xlabel("Time")
    plt.ylabel("Membrane Potential")
    for i in range(batch_size):
        plt.plot(mem[:, i, :].cpu(), label="Output")
        plt.plot(label[:, i, :].cpu(), label="Target")
    plt.legend(loc='best')
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/regression1/reg_1-3.png?raw=true
        :align: center
        :width: 450

It is a little jagged, but it’s not looking too bad.

You might try to improve the curve fit by expanding the size of the
hidden layer, increasing the number of iterations, adding extra time
steps, hyperparameter fine-tuning, or using a completely different
neuron type.

Conclusion
------------------------

The next regression tutorials will test more powerful spiking neurons,
such as Reucrrent LIF neurons and spiking LSTMs, to see how they
compare.

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
