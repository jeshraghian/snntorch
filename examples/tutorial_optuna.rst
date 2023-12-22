` <https://github.com/jeshraghian/snntorch/>`__

Discover SNN Hyperparameters with Optuna
========================================

Tutorial written by Reto Stamm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

` <https://github.com/jeshraghian/snntorch/>`__
` <https://github.com/jeshraghian/snntorch/>`__

*This tutorial demonstrates optimizing Spiking Neural Network
hyperparameters with Optuna, blending advanced neural modeling and
hyperparameter tuning. In this example, we minimize power consumption by
adjusting hyperparameter.*

`Optuna <https://optuna.org>`__ is an efficient, open-source
hyperparameter optimization framework, streamlining ML model tuning with
support for various algorithms and easy integration with key ML
libraries. In other words, a tool that helps automatically figure out
the best settings for machine learning models, making them perform
better.

**Spiked Neural Networks** (SNNs) model neural processing realistically,
efficiently handling time-dependent data with low power. Their
event-driven nature excels in dynamic, real-time tasks, advancing
neuromorphic computing and AI. For an in-depth understanding of SNNs and
their inner workings, check out the `snnTorch tutorial
series <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__.
Please cite the accompanying paper if you find these tutorials or code
beneficial.

For this example, let’s assume that the more the neural network spikes,
the more power it consumes. We want to **minimize power consumption**,
so we adjust its **hyperparameters**, including the shape of the
network, the number of steps run, and the number of epochs it is trained
for.

`Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz,
Girish Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu.
“Training Spiking Neural Networks Using Lessons From Deep Learning”.
Proceedings of the IEEE, 111(9) September
2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`__

.. code:: ipython3

    !pip install optuna snntorch torchvision matplotlib optunacy --quiet

.. code:: ipython3

    # Import all the libraries
    import copy
    import logging
    import random
    import numbers
    import sys
    import time # To see how long each iteration takes
    import multiprocessing # To check how many cores we have
    
    import optuna # the optimizer
    # To abort, or prune inefficient parameter sets
    from optuna.exceptions import TrialPruned
    from concurrent.futures import ThreadPoolExecutor # todo: concurrent execution
    from optuna.trial import TrialState
    
    # Basic torch tools
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    
    # Image processing tools
    import torchvision
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    
    # Extra plotting tools
    from optunacy.oplot import OPlot
    import scipy as scipy
    
    # Spiked Neural Networks!
    import snntorch as snn
    import snntorch.functional as SF
    from snntorch import utils


1. The MNIST Dataset
--------------------

1.1 Dataloading
~~~~~~~~~~~~~~~

Define variables for dataloading.

.. code:: ipython3

    batch_size = 128
    data_path='/tmp/data/mnist'

Load the dataset.

We don’t use the test set for tuning hyperparameters to avoid losing the
ability to generalize and to spot overfitting. Instead, we separate a
validation set for this purpose.

This should be the same subset each time, otherwise we “learn” the
hyperparameters that are good for that set too.

.. code:: ipython3

    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
    
    # The MNIST dataset contains black and white images (28x28) of digits from 0-9
    # 60,000 training images
    mnist_train = datasets.MNIST(data_path, train=True, download=True,
                                 transform=transform)
    
    # Split out a validation subset
    total_size = len(mnist_train)
    val_size = int(total_size * 0.08)  # 8% for validation
    train_size = total_size - val_size  # Remaining for training
    
    # Split the dataset, the same way every time
    mnist_val = Subset(mnist_train, range(train_size, total_size))
    mnist_train = Subset(mnist_train, range(0, train_size))
    
    # 10,000 test images
    mnist_test = datasets.MNIST(data_path, train=False, download=True,
                                transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

.. code:: ipython3

    # See what acceleration hardware we have available on this machine
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Running on {device}")

.. code:: ipython3

    # We need to use the logger so that the messages are in sync with Optunas output
    logger = logging.getLogger('optuna')

1.2 A description of the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MNIST is a collection of 70,000 images of handwritten numbers. It
includes 60,000 training images and 10,000 test images. The images are
simple black and white, showing digits from 0 to 9, and the aim is to
correctly identify these digits.

Key points about these images:

-  Each image is 28 pixels wide and 28 pixels tall.
-  They are in black and white, which means each pixel is just a shade
   of gray, not color.
-  These images don’t change over time; they’re just single, static
   pictures.

It’s good to look at the shape of the datastructure.

.. code:: ipython3

    for data, label in iter(train_loader):
      print(data.size())
      break

2. A parameterizeable network with snnTorch
-------------------------------------------

With this MNIST dataset, some things always remain the same. The input
image size, and the fact that we want to detect one of 10 pixels. Those
are hardwired.

The first layer’s decay rate might be adjustable as a hyperparameter.
There is one neuron for each pixel.

The output layer is fixed to classify digits 0-9. There is one neuron
for each digit.

We also track the average spike activity across the network, so that we
can caluclate how much spiking activity per digit was generated.

The number of timesteps before we get the result is configurable.

Beta is the decay rate, the amount that each neuron remembers from the
previous timestep (0 - no memory, 1 - never forget). *beta1* is the
decay rate for the first layer, and it is a model parameter. All the
other neurons learn their own beta during training.

.. code:: ipython3

    class Net(nn.Module):
    
        def __init__(self, num_steps, num_hidden_neurons=299, num_hidden_layers=1, beta1=0.9):
            super().__init__()
            assert 0 <= beta1 <= 1, "Beta1 must be between 0 and 1"
            assert num_hidden_layers >= 0, "Number of hidden layers must be non-negative"
    
            num_inputs = 28 * 28 # image is 28x28 pixels
            num_outputs = 10 # we want to get digits 0-9
            self.num_steps = num_steps
            self.num_hidden_neurons = num_hidden_neurons
            self.num_hidden_layers = num_hidden_layers
    
            # Initialize layers
            self.layers = []
            for n in range(num_hidden_layers + 1):
                layer = {}
                if n == 0:
                    # First layer
                    layer['fc'] = nn.Linear(num_inputs, num_hidden_neurons)
                    layer['lif'] = snn.Leaky(beta=beta1)
                elif n < num_hidden_layers:
                    # Inner layers
                    layer['fc'] = nn.Linear(num_hidden_neurons, num_hidden_neurons)
                    beta2 = torch.rand((num_hidden_neurons), dtype=torch.float)
                    layer['lif'] = snn.Leaky(beta=beta2, learn_beta=True)
                else:
                    # Output layer
                    layer['fc'] = nn.Linear(num_hidden_neurons, num_outputs)
                    beta2 = torch.rand((num_outputs), dtype=torch.float)
                    layer['lif'] = snn.Leaky(beta=beta2, learn_beta=True)
    
                # Add the layers to the internal representation
                self.add_module(f'fc{n}', layer['fc'])
                self.add_module(f'lif{n}', layer['lif'])
    
                # Add the layers to our layer list.
                self.layers.append(layer)
    
            # Reset spike counter
            self.reset_spikes()
    
        def forward(self, x):
            # The forward pass.
    
            # Initialize all the neurons in all layers
            for layer in self.layers:
                layer['mem'] = layer['lif'].init_leaky()
    
            spk_rec, mem_rec = [], []
    
            # process each timestep
            for step in range(self.num_steps):
                cur = x.flatten(1)
    
                # process each layer
                for index, layer in enumerate(self.layers):
                    # process the data
                    cur, layer['mem'] = layer['lif'](layer['fc'](cur), layer['mem'])
    
                    # update the total spike count
                    self.total_spike_count += cur.sum().item()
                # update the spike records
                spk_rec.append(cur)
                mem_rec.append(self.layers[-1]['mem'])
    
            self.forward_count += 1 # so we can normalize the spike_count later
            return torch.stack(spk_rec), torch.stack(mem_rec)
    
        def get_spikes_per_digit(self):
            # Returns average number of spikes per forward pass
            return self.total_spike_count/self.forward_count
    
        def reset_spikes(self):
            # Reset all the spike counting information
            self.total_spike_count = 0 # How many spikes have been generated, in all layers
            self.forward_count = 0 # How many forward passes have been made, altogether

3. The hyperparameter trainer
-----------------------------

The trainer class is here to define how the training takes place, given
a network and a few training hyperparameters. It makes the objective
below a bit more readable.

It’s a bit of a challange because all sorts of networks are evaluated
here, deep ones, wide ones, many neurons, few neurons, and they all
improve their loss and accuracy at different speeds. That’s why this
class has an automatic early stopping feature. Early stopping stops the
training when there the loss has not significantly improved in the last
*patience=300* batches. That avoids serious overtraing.

We adjust the definition of significant improvement based on the number
of layers. Bcause deep networks take longer to train and the
improvements are on average correspondingly smaller per batch this seems
about right:

.. math:: significane_{actual} = \frac{significance_{base}}{layers^3}

That means if there’s no change for single layer networks (5%
improvement in the last 300 steps).

For a ten layer network, pretty much any tiny improvement is an
improvement of significance. This method has been heuristically
determined, and it seems to work quite well and gets most networks to a
reasonably high training accuracy.

The early stopping parameter is important, if it stops the training of
deep networks too soon, it’ll appear as if it is a bad network, but it
was not trained well enough. If it overtrains a 1 layer network and it
memorized the input, it’ll wrongly appear as a bad net also.

The early stop mechanism could also be optimized for (a
hyper-hyperparameter), and can be as complicated as desired. For our
spike optimizer, we keep this part constant because GPU time is still
expensive.

.. code:: ipython3

    class SNNTrainer:
    
        def __init__(self, net, trial, num_epochs=30, num_steps=25, learning_rate=2e-3, patience = 300, sig_improvement = 0.05):
            self.net = net.to(device)
            self.num_epochs = num_epochs
            self.num_steps = num_steps
            self.learning_rate = learning_rate
            self.trial = trial
            self.patience = patience
            
            # Calculate what we mean by significant improvement
            self.sig_improvement = sig_improvement/(net.num_hidden_layers**3)
    
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=learning_rate,
                                              betas=(0.9, 0.999))
    
            self.loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
            self.loss_hist = []
            self.acc_hist = []
            self.epochs = 0
            self.batches = 0
    
        def train(self, train_loader):
            acc = 0
            best_loss = float("inf")
            loss_counter = 0
    
            for epoch in range(self.num_epochs):
                for i, (data, targets) in enumerate(train_loader):
                    data = data.to(device)
                    targets = targets.to(device)
    
                    self.net.train()
                    spk_rec, _ = self.net(data)
                    loss_val = self.loss_fn(spk_rec, targets)
                    self.optimizer.zero_grad()
                    loss_val.backward()
                    self.optimizer.step()
    
                    current_loss = loss_val.item()
                    self.loss_hist.append(current_loss)
    
                    # Update display every few iterations.
                    if i % 100 == 0 and i != 0:
                        acc = SF.accuracy_rate(spk_rec, targets)
                        self.acc_hist.append(acc)
                        logger.info(f"Trial {self.trial.number}: Training: Epoch {epoch}, Batch {i} "+
                                     f"Loss: {loss_val.item():.4f} (best:{best_loss:.4f} t-{loss_counter}) "+ 
                                     f"Accuracy: {acc * 100:.2f}%")
                    
                    # rudimentary early stop:
                    # After the first epoch, if there is no improvement, call it a day
    
                    if current_loss < (1-self.sig_improvement)*best_loss: # an improvement!
                        best_loss = current_loss
                        loss_counter = 0
                    else: # No improvement
                        loss_counter += 1
                        
                    self.batches += 1
                    
                    if epoch > 0 and loss_counter > self.patience:
                        logger.info("Early stopping.")
                        return
                    
                self.epochs += 1
    
        def get_accuracy(self, test_loader):
            # Get the normal test accuracy for the dataset provided.
            total_acc = 0
            total = 0
            with torch.no_grad():
                self.net.eval()
                for data, targets in test_loader:
                    data = data.to(device)
                    targets = targets.to(device)
                    spk_rec, _ = self.net(data)
    
                    acc = SF.accuracy_rate(spk_rec, targets)
                    total_acc += acc * data.size(0)
                    total += data.size(0)
            return total_acc / total

4. The Objective
----------------

We direct Optuna to optimize specific parameters within carefully chosen
ranges, ensuring they are neither too broad nor too narrow. This balance
is crucial, especially for parameters like the number of steps and
epochs, as overly high values can significantly increase training time.
Even these hyper-hyperparameters have to be chosen in some way.

Because we want to explore many parameters, we are a bit generous here -
the range can be somewhat on the large side for each of them. Before we
do anything further, we then prune the ones that will contain too much
information and pose a risk of overtraining. That, to the delight of the
developer, generally conicides with ones that are very large and take a
very long time to run.

Then, it runs the training.

After that, accuracy is gauged using a separate validation dataset. This
contains data that this network has never seen - so we can see how well
the network can generalize from the training data.

.. code:: ipython3

    def optuna_objective(trial, train_loader, test_loader):
        # Suggest hyperparameters, set the approximate range where we want to optimze
        num_steps = trial.suggest_int('Timesteps', 10, 50)
        num_hidden_layers = trial.suggest_int('Hidden Layers', 1, 10)
        num_hidden_neurons = trial.suggest_int('Neurons per Hidden Layer', 5, 300)
    
        learning_rate = trial.suggest_float('Learning Rate', 1e-4, 1e-2, log=True)
        first_layer_beta = trial.suggest_float('First Layer β', 0.5, 1)
    
        logger.info(f"Trial {trial.number}: Training: Layers={num_hidden_layers} "+
              f"Neurons={num_hidden_neurons} Steps={num_steps} l1Beta={first_layer_beta:2f}")
        
        # Skip large networks with many steps, they take too long to train
        # This cuts off a large corner of the parameter space - and the runtime
        if num_hidden_layers*num_hidden_neurons*num_steps > 300*15:
            raise TrialPruned("Too computationally intensive.")
    
        logger.info(f"Trial {trial.number}: Running")
    
        net = Net(num_steps, num_hidden_neurons, num_hidden_layers, first_layer_beta)
        
        # Run the training!
        trainer = SNNTrainer(net, trial, num_steps=num_steps, learning_rate=learning_rate)
        training_start_time = time.time()
        trainer.train(train_loader)
        
        # Training info - so we can plot it later
        trial.set_user_attr("Training Time [s]", time.time() - training_start_time)
        trial.set_user_attr("Epochs", trainer.epochs)
        trial.set_user_attr("Batches", trainer.batches)
        
        logger.info(f"Trial {trial.number}: Run on validation set")
        net.reset_spikes() # Only consider spikes/digit after training is complete
        validation_accuracy = trainer.get_accuracy(validation_loader)
    
        # The thing we really want to optimize for!
        spikes_per_digit = net.get_spikes_per_digit()
    
        # Define the objective to maximize test accuracy and minimize spike count
        return               validation_accuracy,   spikes_per_digit
    
    # The objectives have a printable name and direction
    # Optuna keeps track of the objectives returned as an ordered array, 
    # so we do, too, all here in one place.
    objective_names      = ["Validation Accuracy", "Spikes per Digit"]
    objective_directions = ["maximize",            "minimize"]


5. The study
------------

Now we can run the things we just defined and see the results!

.. code:: ipython3

    # Define the Optuna study
    # maximize accuracy
    # minimize spikes
    study = optuna.create_study(study_name="Minimize spikes, maximize accuracy",
                                directions=objective_directions)
    
    completed_trials = 0 # Nothing has been done yet.

.. code:: ipython3

    # Helper to figure out how many trials have successfully completed
    def completed_trials(study):
        # Counts the completed, successful trials
        return sum(1 for trial in study.trials if trial.state == TrialState.COMPLETE)

.. code:: ipython3

    # Need at least 3 for the plots below
    additional_trials = 50
    
    # Bookkeeping
    start_time = time.time()
    start_trials = completed_trials(study)
    target_trials = start_trials + additional_trials
    logger.info(f"Running on device={device}.")
    logger.info(f"{start_trials} completed. Running {additional_trials} more to have {target_trials} in total.")
    
    while completed_trials(study) < target_trials:
        # Run trials one at a time so we can stop the code block and keep whatever has been learned
        study.optimize( lambda trial:
                        optuna_objective(trial, train_loader, test_loader),
                        n_trials=additional_trials)
        
        # Bookkeeping and message generation
        elapsed = time.time() - start_time
        total_completed = completed_trials(study)
        completed = total_completed - start_trials
        remaining_trials = target_trials - completed - start_trials
        logger.info(f"#### Remaining trials {remaining_trials} ####")
        if completed > 0:
            rate = elapsed/(completed)
            remaining_time = (target_trials - completed)*rate
            logger.info(f"Completed {total_completed}/{target_trials} studies at {rate/60:.1f}min/trial")
            if total_completed < target_trials:
                logger.info(f"Remaining time: {remaining_time/60:.1f} minutes to do {remaining_trials} trials.")
            
    logger.info(f"DONE")

6. Ponder the Results
---------------------

Now it’s time to actually look at the parameters and think about them!

.. code:: ipython3

    # Initialize the optunacy plotter
    see = OPlot(study, objective_names)

6.1 Cause and Effect
--------------------

We can look at the importance of hyperparameters on outcome metrics, and
see what impact a change in hyperparameter input has on an output.

I’ve already forgotten what we collected; let’s see a list. That also
allows easy copying of the strings because we’ll make lots of plots now.

.. code:: ipython3

    see.parameters()

I am curious about deep networks with many hidden layers and if they are
effective here. Let’s see:

.. code:: ipython3

    see.plot("Spikes per Digit", "Validation Accuracy", "Hidden Layers")

In this plot, each dot is a Network, and the color indicates the hidden
networks in a given area. Spikes per Digit is roughly proportiona to
power consumption, and Validation Accuracy is a measure of how well the
network works. So we want to be in the top left corner. But we can
already see: The top left corner is dominated by one-layered networks.
So my hypothesis was not right, deep networks make lots of spikes.

It’s a bit chaotic, and we absolutely don’t care about accuracies below
60%. So let’s zoom in a bit:

.. code:: ipython3

    see.plot("Spikes per Digit", "Validation Accuracy", "Hidden Layers", y_range=(0.60, 1), z_clip=(1,5))

Deeper networks are defnintely to the right.

I’d guess that network size and spike rate are correlated.

.. code:: ipython3

    see.plot("Neurons per Hidden Layer", "Hidden Layers", "Validation Accuracy")

First, note that there are no datapoints in the top right part of the
graph. That’s because we prune these - lots of deep layers are very
computationally expensive.

In any case, the graph is not very informative, we mostly care about
accuracies that are at the very least 80%.

.. code:: ipython3

    see.plot("Neurons per Hidden Layer", "Hidden Layers", "Validation Accuracy", z_clip=(.8,1))

That’s nice, it seems we need about 100-200 Neurons (if you mouse over a
point you can see the data) on one or two layers, or more on 3 layers.
Also, large networks seem to be not very accurate. Also, networks with
very few neurons (in the bottom left corner) are not accurate.

Let’s look at the spike rate on that same picture. I’ll clip it to see
the interesting parts.

.. code:: ipython3

    see.plot("Neurons per Hidden Layer", "Hidden Layers", "Spikes per Digit", z_clip=(20000,80000))

From this it’s clear that large networks are not power efficient.

What about the other parameters?

.. code:: ipython3

    see.plot("First Layer β", "Validation Accuracy", "Spikes per Digit", z_clip=(30000, 80000), y_range=(0.8,1))

That does not look particularly helpful. It seems like all values for β
can provide high accuracy results, some even with low spike counts. It
appears that there are more low spike count nets with high accuracy
where β is close to 1, so maybe β should be greater than 0.95.

What about timesteps?

.. code:: ipython3

    see.plot("Timesteps", "Validation Accuracy", "Spikes per Digit", z_clip=(30000, 80000), y_range=(0.8,1))

As we can expect, the longer it runs, the more timesteps we get. It
seems that the optimum numer of timesteps is around 15-20.

What about Learning Rate?

.. code:: ipython3

    see.plot("Learning Rate", "Validation Accuracy", "Spikes per Digit", z_clip=(30000, 80000), y_range=(0.8,1))

Here, it seems like most of the results are in the top left corner.
There’s an area in the right top corner that is maybe underexplored.
That’s becuase the learning rate was run with a log distribution:

``learning_rate = trial.suggest_float('Learning Rate', 1e-4, 1e-2, log=True)``

Maybe in the next run, take that off, and explore the top right corner
also!

Optuna has some more built in `plotting
features <https://optuna.readthedocs.io/en/stable/reference/visualization/index.html>`__,
for example, a way to plot the importance of a parameter for a
particular optimization target.

.. code:: ipython3

    optuna.visualization.plot_param_importances(study,
                                      target=lambda t: t.values[0],
                                      target_name = "Validation accuracy").show()

This plot means that the hyperparameter with the longest bar has the
highest impact on accuracy. It does not say wether that number needs to
be large or small.

6.1 Summary
~~~~~~~~~~~

From looking at the data, we’ve found that the optimal network is likely
around

-  1-2 layer deep
-  100-200 total neurons
-  15-20 timesteps long
-  at least 0.9 for the first layer’s β

This drastically reduces our searchspace, and we can re-run the
optimizer with a focus in that area.
