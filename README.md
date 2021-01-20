![snnTorch Logo](https://github.com/jeshraghian/snntorch/blob/master/docs/source/_static/img/snntorch_logo.png)

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
| [**snntorch.spikegen**](https://snntorch.org/docs/stable/spikegen.html) | a library for spike generation and data conversion |
| [**snntorch.spikeplot**](https://snntorch.org/docs/stable/spikeplot.html) | visualization tools for spike-based data using matplotlib and celluloid |
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
Ready to contribute? Here's how to set up snnTorch for local development.

1. Fork the snnTorch rop on GitHub.
2. Clone your fork locally:
``git clone git@github.com:your_name/snnTorch.git``
3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is 
how you set up your fork for local development:

``$ mkvirtualenv snntorch
$ cd snntorch.
python setup.py develop``

4. Create a branch for local development
``$ git checkout -b name-of-your-bugfix-or-feature``

Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the tests, including 
testing other Python versions with tox. In addition, ensure that your code is formatted using black:

``$ flake8 snntorch tests
$ black sntorch tests
$ pytorch setup.py test or py.test
$ tox``

To get flake8, black, and tox, just pip install them into your virtualenv. If you wish,
you can add pre-commit hooks for both flake8 and black to make all formatting easier.

6. Commit your changes and push your branch to GitHub:

``$ git add .
$ git commit -m "Your detailed description of your changes."
git push origin name-of-your-bugfix-or-feature``

In your commit message:
* Always contain a subject line which briefly describes the changes made. For example: 
"Update CONTRIBUTING.rst".
* Subject lines should not exceed 50 characters.
* The commit body should contain context about the change - how the code worked before, how it works now, and why you decided to solve the issue in the way you did. 

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated.
Put your new functionality into a function with a docstring, and add the feature to the list in 
README.rst.
3. The pull request should work for Python>=3.6.

## Acknowledgments
snnTorch was developed by Jason K. Eshraghian in the Lu Group (University of Michigan), with additional contributions from Xinxin Wang and Vincent Sun.
Our inspiration comes from the work done by several researchers on this topic, including Friedemann Zenke, Emre Neftci, 
Doo Seok Jeong, Sumit Bam Shrestha, Garrick Orchard, and Bodo Rueckauer. 

## License & Copyright
snnTorch is licensed under the GNU General Public License v3.0: https://www.gnu.org/licenses/gpl-3.0.en.html.
