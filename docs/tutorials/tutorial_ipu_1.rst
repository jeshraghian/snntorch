===================================================
Accelerating snnTorch on IPUs
===================================================


Tutorial written by `Jason K. Eshraghian <https://www.jasoneshraghian.com>`_ and Vincent Sun

The snnTorch tutorial series is based on the following paper. If you find these resources or code useful in your work, please consider citing the following source:

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  This tutorial is a static non-editable version. An editable script is available via the following link:
    * `Python Script (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples/tutorial_ipu_1.py>`_


Introduction
============

Spiking neural networks (SNNs) have achieved orders of magnitude improvement in terms of energy consumption and latency when performing inference with deep learning workloads.
But in a twist of irony, using error backpropagation to train SNNs becomes more expensive than non-spiking network when trained on CPUs and GPUs.
The additional temporal dimension must be accounted for, and memory complexity increases lineary with time when a network is trained using the backpropagation-through-time algorithm.

An alternative build of snnTorch has been optimized for `Graphcore's Intelligence Processing Units (IPUs) <https://www.graphcore.ai/>`_.
IPUs are custom accelerators tailored for deep learning workloads, and adopt multi-instruction multi-data (MIMD) parallelism by running individual processing threads on smaller blocks of data.
This is an ideal fit for partitions of spiking neuron dynamical state equations that must be sequentially processed, and cannot be vectorized.


In this tutorial, you will: 

    * Learn how to train a SNN accelerated using IPUs.


Ensure up-to-date versions of :code:`poptorch` and the Poplar SDK are installed. Refer to `Graphcore's documentation <https://github.com/graphcore/poptorch>`_ for installation instructions.

Install :code:`snntorch-ipu` in an environment that does not have :code:`snntorch` pre-installed to avoid package conflicts:

::

    !pip install snntorch-ipu

Import the required Python packages:

::

    import torch, torch.nn as nn
    import popart, poptorch
    import snntorch as snn
    import snntorch.functional as SF

DataLoading
===========

Load in the MNIST dataset.

::

    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

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
    
    # Train using full precision 32-flt
    opts = poptorch.Options()
    opts.Precision.halfFloatCasting(poptorch.HalfFloatCastingBehavior.HalfUpcastToFloat)

    # Create DataLoaders
    train_loader = poptorch.DataLoader(options=opts, dataset=mnist_train, batch_size=batch_size, shuffle=True, num_workers=20)
    test_loader = poptorch.DataLoader(options=opts, dataset=mnist_test, batch_size=batch_size, shuffle=True, num_workers=20)


Define Network
==============

Let's simulate our network for 25 time steps using a slow state-decay rate for our spiking neurons:

::

    num_steps = 25
    beta = 0.9


We will now construct a vanilla SNN model. 
When training on IPUs, note that the loss function must be wrapped within the model class.
The full code will look this:

::

    class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_inputs = 784
        num_hidden = 1000
        num_outputs = 10

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_output)
        self.lif2 = snn.Leaky(beta=beta)

        # Cross-Entropy Spike Count Loss
        self.loss_fn = SF.ce_count_loss()

    def forward(self, x, labels=None):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []
       
        for step in range(num_steps):
            cur1 = self.fc1(x.view(batch_size,-1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        spk2_rec = torch.stack(spk2_rec)
        mem2_rec = torch.stack(mem2_rec)

        if self.training:
            return spk2_rec, poptorch.identity_loss(self.loss_fn(mem2_rec, labels), "none")
        return spk2_rec


Let's quickly break this down. 

Contructing the model is the same as all previous tutorials. We apply spiking neuron nodes at the end of each dense layer:

::

    self.fc1 = nn.Linear(num_inputs, num_hidden)
    self.lif1 = snn.Leaky(beta=beta)
    self.fc2 = nn.Linear(num_hidden, num_output)
    self.lif2 = snn.Leaky(beta=beta)

By default, the surrogate gradient of the spiking neurons will be a straight through estimator.
Fast Sigmoid and Sigmoid options are also available if you prefer to use those:

::

    from snntorch import surrogate

    self.lif1 = snn.Leaky(beta=beta, spike_grad = surrogate.fast_sigmoid())


The loss function will count up the total number of spikes from each output neuron and apply the Cross Entropy Loss:

::

    self.loss_fn = SF.ce_count_loss()

Now we define the forward pass. Initialize the hidden state of each spiking neuron by calling the following functions:

::

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()


Next, run the for-loop to simulate the SNN over 25 time steps.
The input data is flattened using :code:`.view(batch_size, -1)` to make it compatible with a dense input layer.

::

    for step in range(num_steps):
        cur1 = self.fc1(x.view(batch_size,-1))
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)

The loss is applied using the function :code:`poptorch.identity_loss(self.loss_fn(mem2_rec, labels), "none")`.


Training on IPUs
=================

Now, the full training loop is run across 10 epochs. 
Note the optimizer is called from :code:`poptorch`. Otherwise, the training process is much the same as in typical use of snnTorch.

::

    net = Model()
    optimizer = poptorch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

    poptorch_model = poptorch.trainingModel(net, options=opts, optimizer=optimizer)

    epochs = 10
    for epoch in tqdm(range(epochs), desc="epochs"):
        correct = 0.0

        for i, (data, labels) in enumerate(train_loader):
            output, loss = poptorch_model(data, labels)

            if i % 250 == 0:
                _, pred = output.sum(dim=0).max(1)
                correct = (labels == pred).sum().item()/len(labels)

                # Accuracy on a single batch
                print("Accuracy: ", correct)

The model will first be compiled, after which, the training process will commence. 
The accuracy will be printed out for individual minibatches on the training set to keep this tutorial quick and minimal.


Conclusion
==========

Our initial benchmarks on show improvements of up to 10x improvements over CUDA accelerated SNNs in mixed-precision training throughput across a variety of neuron models.
A detailed benchmark and blog highlighting additional features are currently under construction.

-  For a detailed tutorial of spiking neurons, neural nets, encoding,
   and training using neuromorphic datasets, check out the `snnTorch
   tutorial
   series <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__.
-  For more information on the features of snnTorch, check out the
   `documentation at this
   link <https://snntorch.readthedocs.io/en/latest/>`__.
-  If you have ideas, suggestions or would like to find ways to get
   involved, then `check out the snnTorch GitHub project
   here. <https://github.com/jeshraghian/snntorch>`__
