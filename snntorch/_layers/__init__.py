###############################################################
# When adding new neurons, update the following:              #
# i) create neuron in snntorch/_neurons/your_neuron.py        #
# ii) utils.py: reset(); _layer_check(), _layer_reset() etc.  #
# iii) neurons_dict in backprop.py,                           #
# iv) update __neuron__ & import below                        #
# v) unit tests in tests/test_snntorch.py                     #
# vi) update docs: snntorch.rst                               #
###############################################################

__layer__ = [
    "bntt1d",
    "bntt2d"
]

from .bntt import *

# from .slstm import SLSTM
