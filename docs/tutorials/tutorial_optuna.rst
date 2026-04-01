===================================================
Hyperparameter Optimization with Optuna
===================================================

Tutorial written by Reto Stamm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_optuna.ipynb

The snnTorch tutorial series is based on the following paper. If you find these resources or code useful in your work, please consider citing the following source:

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. "Training
    Spiking Neural Networks Using Lessons From Deep Learning". Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_optuna.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


Introduction
============

This tutorial demonstrates how to optimize Spiking Neural Network
hyperparameters with `Optuna <https://optuna.org>`_, an efficient
open-source hyperparameter optimization framework.

We frame the problem as a **multi-objective optimization**: maximize
classification accuracy while minimizing spike count (a proxy for power
consumption). Optuna explores network architecture (depth, width),
temporal parameters (timesteps), and training settings (learning rate,
decay) to find Pareto-optimal configurations.

For a comprehensive overview of SNNs and snnTorch, see the
`snnTorch tutorial series <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_.

::

    !pip install optuna snntorch

.. code:: python

    import logging
    import time

    import matplotlib.pyplot as plt
    import optuna
    from optuna.exceptions import TrialPruned
    from optuna.trial import TrialState
    import optuna.visualization.matplotlib as optuna_plt

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset

    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    import snntorch as snn
    import snntorch.functional as SF


1. The MNIST Dataset
====================

1.1 Dataloading
~~~~~~~~~~~~~~~

Define variables for dataloading.

.. code:: python

    batch_size = 128
    data_path = '/tmp/data/mnist'

Load the dataset. We split out a validation set for hyperparameter
evaluation. The test set is never used during tuning to avoid data
leakage.

The split is deterministic (fixed index ranges) so that results are
reproducible across runs.

.. code:: python

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)

    # Deterministic train/validation split (92%/8%)
    total_size = len(mnist_train)
    val_size = int(total_size * 0.08)
    train_size = total_size - val_size

    mnist_val = Subset(mnist_train, range(train_size, total_size))
    mnist_train = Subset(mnist_train, range(0, train_size))

    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

.. code:: python

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Running on {device}")

    logger = logging.getLogger("optuna")


2. A Parameterizable Network
=============================

The network has several fixed aspects:

- **Input**: 28x28 = 784 pixels
- **Output**: 10 neurons (digits 0-9)

Everything else is a tunable hyperparameter:

- Number of hidden layers and neurons per layer
- Number of timesteps
- First-layer decay rate (beta1); all other layers learn their own beta

We also track total spike activity across the network to estimate power
consumption.

.. code:: python

    class Net(nn.Module):

        def __init__(self, num_steps, num_hidden_neurons=299, num_hidden_layers=1, beta1=0.9):
            super().__init__()
            assert 0 <= beta1 <= 1, "Beta1 must be between 0 and 1"
            assert num_hidden_layers >= 0, "Number of hidden layers must be non-negative"

            num_inputs = 28 * 28
            num_outputs = 10
            self.num_steps = num_steps
            self.num_hidden_neurons = num_hidden_neurons
            self.num_hidden_layers = num_hidden_layers

            self.layers = []
            for n in range(num_hidden_layers + 1):
                layer = {}
                if n == 0:
                    layer['fc'] = nn.Linear(num_inputs, num_hidden_neurons)
                    layer['lif'] = snn.Leaky(beta=beta1)
                elif n < num_hidden_layers:
                    layer['fc'] = nn.Linear(num_hidden_neurons, num_hidden_neurons)
                    beta2 = torch.rand(num_hidden_neurons, dtype=torch.float)
                    layer['lif'] = snn.Leaky(beta=beta2, learn_beta=True)
                else:
                    layer['fc'] = nn.Linear(num_hidden_neurons, num_outputs)
                    beta2 = torch.rand(num_outputs, dtype=torch.float)
                    layer['lif'] = snn.Leaky(beta=beta2, learn_beta=True)

                self.add_module(f'fc{n}', layer['fc'])
                self.add_module(f'lif{n}', layer['lif'])
                self.layers.append(layer)

            self.reset_spikes()

        def forward(self, x):
            for layer in self.layers:
                layer['mem'] = layer['lif'].init_leaky()

            spk_rec, mem_rec = [], []

            for step in range(self.num_steps):
                cur = x.flatten(1)
                for layer in self.layers:
                    cur, layer['mem'] = layer['lif'](layer['fc'](cur), layer['mem'])
                    self.total_spike_count += cur.sum().item()
                spk_rec.append(cur)
                mem_rec.append(self.layers[-1]['mem'])

            self.forward_count += 1
            return torch.stack(spk_rec), torch.stack(mem_rec)

        def get_spikes_per_digit(self):
            return self.total_spike_count / self.forward_count

        def reset_spikes(self):
            self.total_spike_count = 0
            self.forward_count = 0


3. The Trainer
==============

The trainer wraps the training loop with **early stopping**. Because
Optuna evaluates networks of wildly different sizes, we adapt the
stopping criterion to the network depth:

.. math:: \text{significance}_{\text{actual}} = \frac{\text{significance}_{\text{base}}}{\text{layers}^3}

A single-layer network stops if loss hasn't improved by 5% in the last
300 batches. A 10-layer network accepts nearly any improvement -- deep
networks train more slowly and make smaller per-batch gains.

This heuristic prevents both overtraining shallow networks and
prematurely killing deep ones.

.. code:: python

    class SNNTrainer:

        def __init__(self, net, trial, num_epochs=30, num_steps=25,
                     learning_rate=2e-3, patience=300, sig_improvement=0.05):
            self.net = net.to(device)
            self.num_epochs = num_epochs
            self.num_steps = num_steps
            self.learning_rate = learning_rate
            self.trial = trial
            self.patience = patience
            self.sig_improvement = sig_improvement / (net.num_hidden_layers ** 3)

            self.optimizer = torch.optim.Adam(
                self.net.parameters(), lr=learning_rate, betas=(0.9, 0.999)
            )
            self.loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
            self.loss_hist = []
            self.acc_hist = []
            self.epochs = 0
            self.batches = 0

        def train(self, train_loader):
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

                    if i % 100 == 0 and i != 0:
                        acc = SF.accuracy_rate(spk_rec, targets)
                        self.acc_hist.append(acc)
                        logger.info(
                            f"Trial {self.trial.number}: Epoch {epoch}, Batch {i} "
                            f"Loss: {current_loss:.4f} (best: {best_loss:.4f} t-{loss_counter}) "
                            f"Acc: {acc * 100:.2f}%"
                        )

                    if current_loss < (1 - self.sig_improvement) * best_loss:
                        best_loss = current_loss
                        loss_counter = 0
                    else:
                        loss_counter += 1

                    self.batches += 1

                    if epoch > 0 and loss_counter > self.patience:
                        logger.info("Early stopping.")
                        return

                self.epochs += 1

        def get_accuracy(self, data_loader):
            total_acc = 0
            total = 0
            with torch.no_grad():
                self.net.eval()
                for data, targets in data_loader:
                    data = data.to(device)
                    targets = targets.to(device)
                    spk_rec, _ = self.net(data)
                    acc = SF.accuracy_rate(spk_rec, targets)
                    total_acc += acc * data.size(0)
                    total += data.size(0)
            return total_acc / total


4. The Objective
================

We have two targets:

- **Maximize** validation accuracy
- **Minimize** spikes per digit (proxy for power consumption)

Optuna explores five hyperparameters within bounded ranges. Networks
that are too computationally expensive (large x deep x many timesteps)
are pruned before training starts.

.. code:: python

    def optuna_objective(trial, train_loader, validation_loader):
        num_steps = trial.suggest_int('Timesteps', 10, 50)
        num_hidden_layers = trial.suggest_int('Hidden Layers', 1, 10)
        num_hidden_neurons = trial.suggest_int('Neurons per Hidden Layer', 5, 300)
        learning_rate = trial.suggest_float('Learning Rate', 1e-4, 1e-2, log=True)
        first_layer_beta = trial.suggest_float('First Layer Beta', 0.5, 1)

        logger.info(
            f"Trial {trial.number}: Layers={num_hidden_layers} "
            f"Neurons={num_hidden_neurons} Steps={num_steps} "
            f"Beta1={first_layer_beta:.2f}"
        )

        # Prune configurations that would take too long
        if num_hidden_layers * num_hidden_neurons * num_steps > 300 * 15:
            raise TrialPruned("Too computationally intensive.")

        net = Net(num_steps, num_hidden_neurons, num_hidden_layers, first_layer_beta)

        trainer = SNNTrainer(net, trial, num_steps=num_steps, learning_rate=learning_rate)
        training_start_time = time.time()
        trainer.train(train_loader)

        trial.set_user_attr("Training Time [s]", time.time() - training_start_time)
        trial.set_user_attr("Epochs", trainer.epochs)
        trial.set_user_attr("Batches", trainer.batches)

        net.reset_spikes()
        validation_accuracy = trainer.get_accuracy(validation_loader)
        spikes_per_digit = net.get_spikes_per_digit()

        return validation_accuracy, spikes_per_digit


    objective_names = ["Validation Accuracy", "Spikes per Digit"]
    objective_directions = ["maximize", "minimize"]


5. Run the Study
================

We run 10 trials by default. This is enough to see clear trends on a
free Colab GPU (typically 10-30 minutes). Increase ``n_trials`` for a
more thorough search.

.. code:: python

    study = optuna.create_study(
        study_name="Minimize spikes, maximize accuracy",
        directions=objective_directions,
    )

    n_trials = 10  # increase for a more thorough search

    start_time = time.time()

    study.optimize(
        lambda trial: optuna_objective(trial, train_loader, validation_loader),
        n_trials=n_trials,
    )

    n_complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
    elapsed = time.time() - start_time
    print(f"\nDone: {n_complete} completed trials in {elapsed / 60:.1f} minutes")


6. Analyze the Results
======================

6.1 Helper: scatter plot
~~~~~~~~~~~~~~~~~~~~~~~~

This utility plots any two metrics on the axes with a third as color,
making it easy to explore three-way relationships between
hyperparameters and objectives.

.. code:: python

    def plot_study(study, x_name, y_name, z_name,
                   y_range=None, z_clip=None):
        """Scatter plot of completed trials: x vs y, colored by z."""
        trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

        def _val(trial, name):
            if name in objective_names:
                return trial.values[objective_names.index(name)]
            return trial.params.get(name)

        xs, ys, zs = [], [], []
        for t in trials:
            x, y, z = _val(t, x_name), _val(t, y_name), _val(t, z_name)
            if None in (x, y, z):
                continue
            if z_clip and not (z_clip[0] <= z <= z_clip[1]):
                continue
            if y_range and not (y_range[0] <= y <= y_range[1]):
                continue
            xs.append(x); ys.append(y); zs.append(z)

        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(xs, ys, c=zs, cmap='viridis', alpha=0.7,
                        edgecolors='k', linewidth=0.5)
        plt.colorbar(sc, label=z_name)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f"{z_name} by {x_name} vs {y_name}")
        if y_range:
            ax.set_ylim(y_range)
        plt.tight_layout()
        plt.show()


6.2 Pareto Front
~~~~~~~~~~~~~~~~

The Pareto front shows the best trade-offs between accuracy and spike
count. Points on the front cannot be improved in one objective without
worsening the other.

.. code:: python

    optuna_plt.plot_pareto_front(
        study,
        target_names=objective_names,
    )
    plt.tight_layout()
    plt.show()


6.3 Accuracy vs Spikes
~~~~~~~~~~~~~~~~~~~~~~~

Each dot is a trained network. We want to be in the **top-left** corner
(high accuracy, few spikes). The color shows the number of hidden
layers.

.. code:: python

    plot_study(study, "Spikes per Digit", "Validation Accuracy", "Hidden Layers")

Let's zoom in to accuracies above 60%:

.. code:: python

    plot_study(study, "Spikes per Digit", "Validation Accuracy", "Hidden Layers",
               y_range=(0.60, 1), z_clip=(1, 5))


6.4 Network Size vs Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

How does the number of neurons and layers affect accuracy?

.. code:: python

    plot_study(study, "Neurons per Hidden Layer", "Hidden Layers",
               "Validation Accuracy", z_clip=(0.8, 1))

Note the empty top-right corner -- those configurations were pruned as
too expensive. The most accurate networks tend to have 100-200 neurons
on 1-2 layers.

Now the same axes, but colored by spike count:

.. code:: python

    plot_study(study, "Neurons per Hidden Layer", "Hidden Layers",
               "Spikes per Digit", z_clip=(20000, 80000))

Larger networks produce far more spikes and are less power-efficient.


6.5 Other Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    plot_study(study, "First Layer Beta", "Validation Accuracy",
               "Spikes per Digit", z_clip=(30000, 80000), y_range=(0.8, 1))

All values of beta can yield high accuracy, but values close to 1 seem
to correlate with lower spike counts.

.. code:: python

    plot_study(study, "Timesteps", "Validation Accuracy",
               "Spikes per Digit", z_clip=(30000, 80000), y_range=(0.8, 1))

More timesteps means more spikes. The sweet spot appears to be around
15-20 timesteps.

.. code:: python

    plot_study(study, "Learning Rate", "Validation Accuracy",
               "Spikes per Digit", z_clip=(30000, 80000), y_range=(0.8, 1))

The log-scale sampling concentrates trials at lower learning rates. The
high end of the range may be worth exploring further.


6.6 Hyperparameter Importance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optuna can estimate which hyperparameters have the most influence on
each objective.

.. code:: python

    optuna_plt.plot_param_importances(
        study,
        target=lambda t: t.values[0],
        target_name="Validation Accuracy",
    )
    plt.tight_layout()
    plt.show()

.. code:: python

    optuna_plt.plot_param_importances(
        study,
        target=lambda t: t.values[1],
        target_name="Spikes per Digit",
    )
    plt.tight_layout()
    plt.show()

The bar length indicates how much each hyperparameter affects the
objective -- it does not indicate whether the value should be large or
small.


6.7 Summary
~~~~~~~~~~~~

From the data, the optimal SNN for this task is likely:

- **1-2 hidden layers**
- **100-200 neurons per layer**
- **15-20 timesteps**
- **First-layer beta >= 0.9**

This drastically reduces the search space. A follow-up study could
focus on this region with more trials for finer resolution.
