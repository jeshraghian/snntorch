Install Packages
================

.. code:: ipython3

    !pip install snntorch -q
    !pip install sconce -q
    %pip show sconce


.. parsed-literal::

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m109.0/109.0 kB[0m [31m1.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m153.0/153.0 kB[0m [31m2.0 MB/s[0m eta [36m0:00:00[0m
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Installing backend dependencies ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m11.6/11.6 MB[0m [31m70.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m18.2/18.2 MB[0m [31m69.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.6/3.6 MB[0m [31m114.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.8/2.8 MB[0m [31m113.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.1/53.1 kB[0m [31m5.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m121.1/121.1 kB[0m [31m16.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m69.3 MB/s[0m eta [36m0:00:00[0m
    [?25h  Building wheel for lit (pyproject.toml) ... [?25l[?25hdone
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    lida 0.0.10 requires fastapi, which is not installed.
    lida 0.0.10 requires kaleido, which is not installed.
    lida 0.0.10 requires python-multipart, which is not installed.
    lida 0.0.10 requires uvicorn, which is not installed.
    cupy-cuda11x 11.0.0 requires numpy<1.26,>=1.20, but you have numpy 1.26.2 which is incompatible.
    imageio 2.31.6 requires pillow<10.1.0,>=8.3.2, but you have pillow 10.1.0 which is incompatible.
    tensorflow-probability 0.22.0 requires typing-extensions<4.6.0, but you have typing-extensions 4.8.0 which is incompatible.[0m[31m
    [0m

Import Libraries
================

.. code:: ipython3

    from collections import defaultdict, OrderedDict
    
    import numpy as np
    import torch
    from torch import nn
    from torch.optim import *
    from torch.optim.lr_scheduler import *
    from torch.utils.data import DataLoader
    from torchvision.datasets import *
    from torchvision.transforms import *
    import torch.optim as optim
    from sconce import sconce
    
    assert torch.cuda.is_available(), \
    "The current runtime does not have CUDA support." \
    "Please go to menu bar (Runtime - Change runtime type) and select GPU"

.. code:: ipython3

    from google.colab import drive
    drive.mount('/content/drive')



.. parsed-literal::

    Mounted at /content/drive


**Spiking Neural Network Compression**
======================================

.. code:: ipython3

    # Import snntorch libraries
    import snntorch as snn
    from snntorch import surrogate
    from snntorch import backprop
    from snntorch import functional as SF
    from snntorch import utils
    from snntorch import spikeplot as splt
    from torch import optim
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torch.nn.functional as F
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    



.. parsed-literal::

    <ipython-input-4-b898cb6c07c2>:4: DeprecationWarning: The module snntorch.backprop will be deprecated in  a future release. Writing out your own training loop will lead to substantially faster performance.
      from snntorch import backprop


Dataset
=======

.. code:: ipython3

    
    # Event Drive Data
    
    # dataloader arguments
    batch_size = 128
    data_path = "./data/mnist"
    
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Define a transform
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ]
    )
    
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        mnist_test, batch_size=batch_size, shuffle=True, drop_last=True
    )



.. parsed-literal::

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./data/mnist/MNIST/raw/train-images-idx3-ubyte.gz


.. parsed-literal::

    100%|██████████| 9912422/9912422 [00:00<00:00, 82101508.40it/s]


.. parsed-literal::

    Extracting ./data/mnist/MNIST/raw/train-images-idx3-ubyte.gz to ./data/mnist/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./data/mnist/MNIST/raw/train-labels-idx1-ubyte.gz


.. parsed-literal::

    100%|██████████| 28881/28881 [00:00<00:00, 111748795.04it/s]


.. parsed-literal::

    Extracting ./data/mnist/MNIST/raw/train-labels-idx1-ubyte.gz to ./data/mnist/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./data/mnist/MNIST/raw/t10k-images-idx3-ubyte.gz


.. parsed-literal::

    100%|██████████| 1648877/1648877 [00:00<00:00, 26490461.97it/s]


.. parsed-literal::

    Extracting ./data/mnist/MNIST/raw/t10k-images-idx3-ubyte.gz to ./data/mnist/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./data/mnist/MNIST/raw/t10k-labels-idx1-ubyte.gz


.. parsed-literal::

    100%|██████████| 4542/4542 [00:00<00:00, 6970555.71it/s]


.. parsed-literal::

    Extracting ./data/mnist/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./data/mnist/MNIST/raw
    


Instantiate an Object of sconce
===============================

.. code:: ipython3

    
    sconces = sconce()


Set you Dataloader
==================

.. code:: ipython3

    
    dataloader = {}
    dataloader["train"] = train_loader
    dataloader["test"] = test_loader
    sconces.dataloader = dataloader

#Enable snn in sconce

.. code:: ipython3

    
    sconces.snn = True


Load your snn Model
===================

.. code:: ipython3

    spike_grad = surrogate.fast_sigmoid(slope=25)
    beta = 0.5
    snn_model = nn.Sequential(
        nn.Conv2d(1, 12, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Conv2d(12, 64, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 10),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
    ).to('cuda')
    


Load the pretrained weights
===========================

.. code:: ipython3

    snn_pretrained_model_path = "drive/MyDrive/Efficientml/Efficientml.ai/snn_model.pth"
    snn_model.load_state_dict(torch.load(snn_pretrained_model_path))  # Model Definition
    sconces.model = snn_model

.. code:: ipython3

    
    sconces.optimizer = optim.Adam(sconces.model.parameters(), lr=1e-4)
    sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)
    
    sconces.criterion = SF.ce_rate_loss()
    
    sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sconces.experiment_name = "snn-gmp"  # Define your experiment name here
    sconces.prune_mode = "GMP"
    sconces.num_finetune_epochs = 1


.. code:: ipython3

    sconces.evaluate()


.. parsed-literal::

    



.. parsed-literal::

    97.11538461538461



.. code:: ipython3

    sconces.compress()


.. parsed-literal::

    
    Original Dense Model Size Model=0.11 MiB


.. parsed-literal::

    

.. parsed-literal::

    Original Model Validation Accuracy: 97.11538461538461 %
    Granular-Magnitude Pruning


.. parsed-literal::

    

.. parsed-literal::

    Sensitivity Scan Time(secs): 204.14258646965027
    Sparsity for each Layer: {'0.weight': 0.6500000000000001, '3.weight': 0.5000000000000001, '7.weight': 0.7000000000000002}
    Pruning Time Consumed (mins): 2.8362054
    Total Pruning Time Consumed (mins): 3.402399698893229


.. parsed-literal::

    

.. parsed-literal::

    
    Pruned Model has size=0.05 MiB(non-zeros) = 43.13% of Original model size


.. parsed-literal::

    

.. parsed-literal::

    
    Pruned Model has Accuracy=95.94 MiB(non-zeros) = -1.17% of Original model Accuracy
    
     
    ========== Fine-Tuning ==========


.. parsed-literal::

    

.. parsed-literal::

    Epoch:1 Train Loss: 0.00000 Validation Accuracy: 95.96354


.. parsed-literal::

    

.. parsed-literal::

    
     ................. Comparison Table  .................
                    Original        Pruned          Reduction Ratio
    Latency (ms)    16.7            15.6            1.1            
    MACs (M)        160             160             1.0            
    Param (M)       0.01            0.01            1.0            
    Accuracies (%)  97.115          95.964          -1.152         
    Fine-Tuned Sparse model has size=0.05 MiB = 43.13% of Original model size
    Fine-Tuned Pruned Model Validation Accuracy: 95.96354166666667


.. parsed-literal::

    /usr/local/lib/python3.10/dist-packages/torchprofile/profile.py:22: UserWarning: No handlers found: "prim::pythonop". Skipped.
      warnings.warn('No handlers found: "{}". Skipped.'.format(
    /usr/local/lib/python3.10/dist-packages/torchprofile/profile.py:22: UserWarning: No handlers found: "prim::pythonop". Skipped.
      warnings.warn('No handlers found: "{}". Skipped.'.format(

