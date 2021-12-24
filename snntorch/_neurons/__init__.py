###############################################################
# When adding new neurons, update the following:              #
# i) utils.py: reset(); _layer_check(), _layer_reset() etc.   #
# ii) neurons_dict in backprop.py,                            #
# iii) update __neuron__ & import below                       #
# iv) unit tests in tests/test_snntorch.py                    #
###############################################################

__neuron__ = ["alpha", "lapicque", "leaky", "rleaky", "synaptic"]

from .lif import LIF
from .alpha import Alpha
from .lapicque import Lapicque
from .leaky import Leaky
from .synaptic import Synaptic

from .rleaky import RLeaky
from .rsynaptic import RSynaptic
