![snnTorch Logo](https://github.com/jeshraghian/snntorch/blob/master/docs/source/_static/img/snntorch-logo.png)

--------------------------------------------------------------------------------

snnTorch is a Python package for performing gradient-based learning with spiking neural networks.
Rather than reinventing the wheel, it sits on top of PyTorch and takes advantage of their GPU accelerated tensor 
computation. Pre-designed spiking neuron models are seamlessly integrated within the PyTorch framework and can be treated as recurrent activation units. 

## snnTorch Structure
snnTorch contains the following components: 

| Component | Description |
| ---- | --- |
| [**snntorch**](https://snntorch.org/docs/stable/snntorch.html) | a spiking neuron library like torch.nn, deeply integrated with autograd |
| [**snntorch.backprop**](https://snntorch.org/docs/stable/backprop.html) | variations of backpropagation commonly used with SNNs for convenience |
| [**snntorch.spikegen**](https://snntorch.org/docs/stable/spikegen.html) | a library for spike generation and data conversion using matplotlib and celluloid|
| [**snntorch.spikeplot**](https://snntorch.org/docs/stable/spikeplot.html) | visualization tools for spike-based data |
| [**snntorch.surrogate**](https://snntorch.org/docs/stable/surrogate.html) | optional surrogate gradient functions |
| [**snntorch.utils**](https://snntorch.org/docs/stable/utils.html) | dataset utility functions for convenience |

snnTorch is designed to be intuitively used with PyTorch, as though each spiking neuron were simply another activation in a sequence of layers. 
It is therefore agnostic to fully-connected layers, convolutional layers, etc. 

At present, the neuron models are represented by recursive functions which removes the need to store membrane potential traces for all neurons in a system in order to calculate the gradient. 
The lean requirements of snnTorch enable small and large networks to be viably trained on CPU, where needed. 
Despite that, snnTorch avoids bottlenecking the acceleration libraries used by PyTorch. 
Provided that the network models and tensors are loaded onto CUDA, snnTorch takes advantage of GPU acceleration in the same way as PyTorch. 

## Citation

## Requirements 
The following packages need to be installed to use snnTorch:

* torch >= 1.2.0
* numpy >= 1.17
* pandas
* matplotlib
* math
* celluloid

## Installation

Run the following to install:

```
python
pip install snntorch
```

To install snnTorch from source instead:

```
git clone https://github.com/jeshraghian/snnTorch
cd snnTorch
python setup.py install
```

## Documentation

## API & Examples 
A complete API is available [here](https://snntorch.readthedocs.io/). 
Examples, tutorials and Colab notebooks are provided.

## Developing snnTorch
To install snnTorch, along with the tools you need to develop and run tests, run the following in your virtualenv:
```
bash
$ pip install -e .[dev]
```

## Contribution
snnTorch was developed by Jason K. Eshraghian in the Lu Group (University of Michigan), with additional contributions from Xinxin Wang and Vincent Sun.
Our inspiration comes from the work done by several researchers on this topic, including Friedemann Zenke, Emre Neftci, 
Doo Seok Jeong, Sumit Bam Shrestha, Garrick Orchard, and Bodo Rueckauer. 

## License & Copyright
snnTorch is licensed under the GNU General Public License v3.0: https://www.gnu.org/licenses/gpl-3.0.en.html.
