import torch
from typing import Callable, Tuple, Dict, TypeVar
from .layer_info import LayerInfo

synapse_neuron_count_signature = TypeVar("synapse_neuron_count_signature",
                                         bound=Callable[[LayerInfo, torch.Tensor, torch.Tensor], Tuple[int, int]])
event_counter_signature = TypeVar("event_counter_signature",
                                  bound=Callable[[LayerInfo, torch.Tensor, torch.Tensor], Tuple[int, int]])


class LayerParameterEventCalculator(object):
    _recognized_layers: Dict[str, Tuple[synapse_neuron_count_signature, event_counter_signature]] = {}

    @staticmethod
    def register_new_layer(layer: torch.nn.Module,
                           synapse_neuron_count_lambda: synapse_neuron_count_signature,
                           event_counter_lambda: event_counter_signature,
                           override_layer_info: bool = False):

        if layer.__class__.__name__ in LayerParameterEventCalculator._recognized_layers:
            if override_layer_info:
                pass
            else:
                raise Exception("overriding layer information, and override_layer_info was set to False!")

    @staticmethod
    def get_calculate_synapse_neuron_count_for(name: str, ignore_not_recognized: bool = False) -> synapse_neuron_count_signature | None:
        if name not in LayerParameterEventCalculator._recognized_layers:
            if ignore_not_recognized:
                return None
            else:
                raise Exception(f"couldn't get the function to calculate the synapse neuron count for layer with "
                                f"classname :\'{name}\'")

        return LayerParameterEventCalculator._recognized_layers[name][0]

    @staticmethod
    def get_event_counter_for(name : str, ignore_not_recognized : bool = False) -> event_counter_signature | None:
        if name not in LayerParameterEventCalculator._recognized_layers:
            if ignore_not_recognized:
                return None
            else:
                raise Exception(f"couldn't get the function to count the total and spiking events for layer with "
                                f"classname :\'{name}\'")

        return LayerParameterEventCalculator._recognized_layers[name][1]
