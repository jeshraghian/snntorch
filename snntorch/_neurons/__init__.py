###############################################################
# When adding new neurons, update the following:              #
# i) create neuron in snntorch/_neurons/your_neuron.py        #
# ii) utils.py: reset(); _layer_check(), _layer_reset() etc.  #
# iii) neurons_dict in backprop.py,                           #
# iv) update __neuron__ & import below                        #
# v) unit tests in tests/test_snntorch.py                     #
# vi) update docs: snntorch.rst                               #
###############################################################

__neuron__ = ["alpha", "lapicque", "leaky", "rleaky", "synaptic", "rsynaptic"]

from .lif import LIF
from .alpha import Alpha
from .lapicque import Lapicque
from .leaky import Leaky
from .synaptic import Synaptic

from .rleaky import RLeaky
from .rsynaptic import RSynaptic
