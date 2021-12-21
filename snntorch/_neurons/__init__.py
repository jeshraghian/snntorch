###############################################################
# When adding new neurons, update neurons_dict in backprop.py #
#              and also update __neuron__ below               #
###############################################################

__neuron__ = ["alpha", "lapicque", "leaky", "synaptic"]

from .lif import LIF
from .alpha import Alpha
from .lapicque import Lapicque
from .leaky import Leaky
from .synaptic import Synaptic
