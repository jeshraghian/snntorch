###############################################################
# When adding new neurons, update the following:              #
# i) neurons_dict in backprop.py,                             #
# ii) init_neuron function in __init__.py                     #
# iii) update __neuron__ & import below                       #
###############################################################

__neuron__ = ["alpha", "lapicque", "leaky", "synaptic"]

from .lif import LIF
from .alpha import Alpha
from .lapicque import Lapicque
from .leaky import Leaky
from .synaptic import Synaptic
