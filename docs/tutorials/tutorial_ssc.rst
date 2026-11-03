===============================================================================================
Training on Spiking Speech Commands With Tonic + snnTorch Tutorial
===============================================================================================

Tutorial written by Richard Dao (rqdao@ucsc.edu), Annabel Truong (anptruon@ucsc.edu), Mira Prabhakar (miprabha@ucsc.edu)

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_ssc.ipynb

The snnTorch tutorial series is based on the following paper. If you find these resources or code useful in your work, please consider citing the following source:

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_ssc.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


Introduction
---------------

This example tutorial will cover training spiking neural networks for an audio-based classication dataset.

In this tutorial. you will:

* Learn how to load an audio-based classification dataset using `Tonic <https://github.com/neuromorphs/tonic>`__
* Understand Visualization, transformations and batching
* Train the SNN with the `Spiking Speech Commmands <https://tonic.readthedocs.io/en/latest/generated/tonic.datasets.SSC.html#tonic.datasets.SSC>`__

First, install the snntorch library if it is not already installed on your machine.

::

   !pip install snntorch --quiet
   !pip install tonic --quiet

And then we need to import some libraries to use for this project.

::

   # PyTorch
   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import Dataset, DataLoader
   from tonic import collation, datasets, DiskCachedDataset, transforms
   import torchaudio

   # snnTorch
   import snntorch as snn
   from snntorch import functional, utils

   # Tonic
   import tonic

   # Other
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   import random as rn
   import os

   from sklearn.model_selection import train_test_split
   from IPython.display import HTML, display

1. The Spiking Speech Commands (SSC) Dataset
-------------------------------------------------

The Spiking Speech Commands (SSC) Dataset was generated using Lauscher, an artificial cochlea model. SSC is a spiking version of Google’s Speech Commands dataset and contains over 100,000 samples of waveform audio data.

The dataset contains 35 classes: Yes, No, Up, Down, Left, Right, On, Off, Stop, Go, Backward, Forward, Follow, Learn, Bed, Bird, Cat, Dog, Happy, House, Marvin, Sheila, Tree, Wow, Zero, One, Two, Three, Four, Five, Six, Seven, Eight, Nine

More information about the SSC dataset can be found in the
following paper:

   `Cramer, B., Stradmann, Y., Schemmel, J., and Zenke, F. (2022).
   The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks.
   IEEE Transactions on Neural Networks and Learning Systems 33, 2744–2757.
   https://doi.org/10.1109/TNNLS.2020.3044364.`


2. Data Preprocessing Using Tonic
-------------------------------------------------

``Tonic`` will be used to convert the dataset into a format compatible with PyTorch/snnTorch. The documentation for Tonic can be found `here <https://tonic.readthedocs.io/en/latest/generated/tonic.datasets.SSC.html#tonic.datasets.SSC>`__

After the data is successfully mounted to your drive, we will begin the data visualization using tonic.

The event tuples are formatted of type ``(t, x, p)``:

* ``t`` is time stamp
* ``x`` is the audio channel
* ``p`` is a boolean value that is always 0

::

   dataset = tonic.datasets.SSC(save_to=path, split='train')
   events, target = dataset[2]
   tonic.utils.plot_event_grid(events)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/ssc/ssc_channels_vs_time.png?raw=true
      ::align: center
      ::width: 450

::
   
   print(events.dtype)
   >>> [('t', '<i8'), ('x', '<i8'), ('p', '<i8')]

2.1 Downsampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We decided to downsample our data for efficiency purposes for this tutorial. While this means we could potentially lose important datapoints, it allows us to save significant computation time.

For the purposes of this tutorial, we downsampled to 375 channels, a downsampling factor of 1/2.

Downsampling allows us to turn data like this:

.. image:: https://rockpool.ai/_images/tutorials_rockpool-shd_4_0.png?raw=true
      ::align: center
      ::width: 450

Into this:

.. image:: https://rockpool.ai/_images/tutorials_rockpool-shd_14_0.png?raw=true
      ::align: center
      ::width: 450

::

   sensor_size = datasets.SSC.sensor_size # By default is (700, 1, 1)
   time_step = 12000 # The max time steps
   downsample_factor = 1/2 # Change as needed

   toTensorTransform = transforms.Compose([
      transforms.Downsample(spatial_factor=downsample_factor),
      transforms.ToFrame(sensor_size=(700 // int(1 / downsample_factor), 1, 1), time_window=time_step)
   ])
   
   train_dataset = tonic.datasets.SSC(save_to=path, split='train', transform=toTensorTransform)
   validation_dataset = tonic.datasets.SSC(save_to=path, split='valid', transform=toTensorTransform)
   test_dataset = tonic.datasets.SSC(save_to=path, split='test', transform=toTensorTransform)

3. DataLoading and Batching
-------------------------------------------------

Since the original data is stored in a format that is slow to read, we utilize disk caching and batching. This allows us to wrute files that are loaded from the original dataset to the disk for quick re-use.

Because event recordings will have different lengths, we are going to provide a collation function ``tonic.collation.PadTensors()`` that will pad out shorter recordings to ensure all samples in a batch have the same dimensions for our DataLoaders.

::
   
   batch_size = 128

   train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collation.PadTensors(batch_first=False))
   validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, collate_fn=collation.PadTensors(batch_first=False))
   test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collation.PadTensors(batch_first=False))

   cached_train_dataloader = DiskCachedDataset(train_dataset, cache_path="./cache/dataloader_train")
   train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collation.PadTensors(batch_first=False), num_workers=2)

   cached_validation_dataloader = DiskCachedDataset(validation_dataset, cache_path="./cache/dataloader_validation")
   validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, collate_fn=collation.PadTensors(batch_first=False), num_workers=2)

   cached_test_dataloader = DiskCachedDataset(test_dataset, cache_path="./cache/dataloader_test")
   test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collation.PadTensors(batch_first=False), num_workers=2)

   # Storing all the dataloaders in a dictionary
   dataloaders = {
      'train':train_dataloader,
      'validation':validation_dataloader,
      'test':test_dataloader
   }

Note the shape of our event tensors are: time x batch x dimensions

::

   data_tensor, targets = next(iter(train_dataloader))
   print(data_tensor.shape)
   >>> torch.Size([83, 128, 1, 350])

4. Network Architecture
-------------------------------------------------

We will use snnTorch and PyTorch to construct a Spiking Multi-Layered Perceptron (SMLP).

We use 3 hidden layers separated by leaky neurons and a final dropout layer before our output layer. Note that we also specify a surrogate gradient *atan* with an alpha=2.

::

   # Defining the Network Architecture
   device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

   inputs = int(700 * downsample_factor)
   hidden = 512
   outputs = 35

   beta = 0.95
   lr = 0.0001

   surrogate = snn.surrogate.atan(alpha=2)

   model = nn.Sequential(
      # Define network architecture here
      nn.Linear(inputs, hidden),
      snn.Leaky(beta=beta, spike_grad=surrogate, init_hidden=True),

      nn.Linear(hidden, hidden),
      snn.Leaky(beta=beta, spike_grad=surrogate, init_hidden=True),

      nn.Linear(hidden, hidden),
      snn.Leaky(beta=beta, spike_grad=surrogate, init_hidden=True),

      nn.Linear(hidden, hidden),
      snn.Leaky(beta=beta, spike_grad=surrogate, init_hidden=True),

      nn.Dropout(0.4),

      nn.Linear(hidden, outputs),
      snn.Leaky(beta=beta, spike_grad=surrogate, init_hidden=True, output=True),

   ).to(device)

4.1 Loss Function and optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For this tutorial, we found that Adam and Mean Squared Error Spike Count Loss performed best.

*Mean Squared Error Spike Count Loss* obtains spikes from the correct class a % of the time and spikes from the incorrect classes a % of the time to encourage incorrect neurons to fire and avoid them from dying.

::

   optimizer = optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.999))
   criterion = functional.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

5. Defining the Forward Pass
-------------------------------------------------

The standard forward pass we use for spiking neural networks, we keep track of the total number of spikes and reset the hidden states for all Leaky neurons in our network.

Note that ``data.size(0)`` is the number of time steps at each iteration.

::

   def forward(net, data):
      total_spikes = [] # collect total number of spikes
      utils.reset(net) # reset hidden states of all Leaky neurons

      for i in range(data.size(0)): # loop over number of timesteps
         output_spikes, mem_out = net(data[i])
         total_spikes.append(output_spikes)

      return torch.stack(total_spikes)

For organization, we will store the loss and accuracy histories in dictionaries.

::

   loss_history = {
      'train':[],
      'validation':[],
      'test':[]
   }
   accuracy_history = {
      'train':[],
      'validation':[],
      'test':[]
   }

6. Training
-------------------------------------------------

Training neuromorphic data takes a large amount of computation time as it requires seqeuentially iterating through time steps. In the case of the SSC dataset, there are roughly 600 timesteps that will be run per epoch.

In our own experiments, it took about 10 epochs with 600 iterations each to crack ~55% validation accuracy.

   Warning: the following simulation will take a while. In our own experiments, it took about 2 hours to train 10 epochs of 600 iterations.

::

   # Training Loop
   num_epochs = 10

   from tqdm.autonotebook import tqdm

   with tqdm(range(num_epochs), unit='Epoch', desc='Training') as pbar:
      epoch = 0
      for _ in pbar:
         for phase in ['train', 'validation']:
               if phase == 'train':
                  model.train()
               else:
                  model.eval()
               for i, (events, labels) in enumerate(dataloaders[phase]):
                  events = events.squeeze()
                  events, labels = events.to(device), labels.to(device)

                  optimizer.zero_grad()
                  with torch.set_grad_enabled(phase == 'train'):
                     spk_rec = forward(model, events)
                     loss = criterion(spk_rec, labels)

                     if phase == 'train':
                           loss.backward()
                           optimizer.step()

                     if i % 25 == 0:
                           loss_history[phase].append(loss.item())
                           accuracy = functional.accuracy_rate(spk_rec, labels)
                           accuracy_history[phase].append(accuracy)
                           print(f"Epoch {epoch+1}, Iteration {i} \n{phase} loss: {loss.item():.2f}")
                           print(f"Accuracy: {accuracy * 100:.2f}%\n")

         epoch += 1


The output should look something like this:

::

   >>>
   Epoch 0, Iteration 0 
   train loss: 4.48
   Accuracy: 3.91%

   Epoch 0, Iteration 25 
   train loss: 1.87
   Accuracy: 3.91%

   Epoch 0, Iteration 50 
   train loss: 1.08
   Accuracy: 4.69%

   Epoch 0, Iteration 75 
   train loss: 0.99
   Accuracy: 3.91%

   Epoch 0, Iteration 100 
   train loss: 0.99
   Accuracy: 3.12%

   ...

   Epoch 9, Iteration 500 
   train loss: 0.66
   Accuracy: 52.34%

   Epoch 9, Iteration 525 
   train loss: 0.68
   Accuracy: 46.88%

   Epoch 9, Iteration 550 
   train loss: 0.69
   Accuracy: 45.31%

   Epoch 9, Iteration 575 
   train loss: 0.69
   Accuracy: 42.19%

   Epoch 9, Iteration 0 
   validation loss: 0.70
   Accuracy: 33.59%

   Epoch 9, Iteration 25 
   validation loss: 0.65
   Accuracy: 51.56%

7. Results
-------------

We plot and compare the accuracy and loss of the different splits of the dataset.

7.1 Plot Train and Validation Set Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that we recorded accuracies and losses every **25** iterations. You can see that our graphs, while jumpy, show an upward trend in accuracy with little overfitting.

::

   train_fig = plt.figure(facecolor="w")
   plt.plot(accuracy_history['train'])
   plt.title("Train Set Accuracy")
   plt.xlabel("Iteration x 25")
   plt.ylabel("Accuracy")
   plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/ssc/ssc_train_acc.png?raw=true
      ::align: center
      ::width: 450

::

   val_fig = plt.figure(facecolor="w")
   plt.plot(accuracy_history['validation'])
   plt.title("Validation Set Accuracy")
   plt.xlabel("Iteration x 25")
   plt.ylabel("Accuracy")
   plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/ssc/ssc_val_acc.png?raw=true
      ::align: center
      ::width: 450

7.2 Testing Our Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tonic provides us a test dataset for SSC from which we can use to test the average accuracy of our model.

::

   model.eval()

   with torch.no_grad():
      for i, (events, labels) in enumerate(dataloaders['test']):
         events = events.squeeze()
         events, labels = events.to(device), labels.to(device)

         spk_rec = forward(model, events)
         loss = criterion(spk_rec, labels)
         if i % 50 == 0:
               loss_history['test'].append(loss.item())
               accuracy = functional.accuracy_rate(spk_rec, labels)
               accuracy_history['test'].append(accuracy)

On average, we get around the 40% accuracy range for our test set.

::

   print("Average Accuracy of Test Dataset: ", str(np.mean(accuracy_history['test']) * 100) + "%")
   >>> Average Accuracy of Test Dataset:  43.8380238791423%

Finally, some comparison graphs over the entire training time.

::

   loss_comparison_fig = plt.figure(facecolor="w")
   plt.plot(loss_history['train'], label='train')
   plt.plot(loss_history['validation'], label='validation')
   plt.plot(loss_history['test'], label='test')
   plt.legend(loc='best')
   plt.title(f"Train vs Validation vs Test - Loss Curves , LR = {lr} Batch Size = {batch_size} Epochs = {num_epochs}")
   plt.xlabel("Iteration x 25")
   plt.ylabel("Loss")
   plt.show()

   accuracy_comparison_fig = plt.figure(facecolor="w")
   plt.plot(accuracy_history['train'], label='train')
   plt.plot(accuracy_history['validation'], label='validation')
   plt.plot(accuracy_history['test'], label='test')
   plt.legend(loc='best')
   plt.title(f"Train vs Validation vs Test - Accuracy Curves , LR = {lr} Batch Size = {batch_size} Epochs = {num_epochs}")
   plt.xlabel("Iteration x 25")
   plt.ylabel("Accuracy")
   plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/ssc/ssc_train_val_test_loss.png?raw=true
      ::align: center
      ::width: 450

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/ssc/ssc_train_val_test_acc.png?raw=true
      ::align: center
      ::width: 450

Congratulations!
------------------

You trained a Spiking Neural Network using ``snnTorch`` and ``Tonic`` on SSC!