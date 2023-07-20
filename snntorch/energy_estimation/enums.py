from enum import Enum, IntEnum, unique

@unique
class Units(str, Enum):
    """Enum containing all available bytes units."""

    AUTO = "auto"
    MEGABYTES = "M"
    GIGABYTES = "G"
    TERABYTES = "T"
    NONE = ""

@unique
class Verbosity(IntEnum):
    """Contains verbosity levels."""
    QUIET, DEFAULT, VERBOSE = 0, 1, 2

@unique
class RowSettings(str, Enum):
    """Enum containing all available row settings."""

    DEPTH = "depth"
    VAR_NAMES = "var_names"
    ASCII_ONLY = "ascii_only"
    HIDE_RECURSIVE_LAYERS = "hide_recursive_layers"


@unique
class ColumnSettings(str, Enum):
    """Enum containing all available column settings."""

    KERNEL_SIZE = "kernel_size"
    INPUT_SIZE = "input_size"
    OUTPUT_SIZE = "output_size"
    NUM_PARAMS = "num_params"
    PARAMS_PERCENT = "params_percent"
    MULT_ADDS = "mult_adds"
    TRAINABLE = "trainable"
    NUMBER_OF_NEURONS = "neurons"
    NUMBER_OF_SYNAPSES = "synapses"
    AVERAGE_FIRING_RATE = "firing_rate"
    SPIKING_EVENTS = "spiking_events"
    TOTAL_EVENTS = "total_events"
