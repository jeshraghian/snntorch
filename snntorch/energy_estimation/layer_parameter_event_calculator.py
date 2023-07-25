import torch
import torch.nn as nn
from typing import Callable, Tuple, Dict, TypeVar, ClassVar, List
from copy import deepcopy

import snntorch
from .utils import *

synapse_neuron_count_signature = TypeVar("synapse_neuron_count_signature",
                                         bound=Callable[[ClassVar, torch.Tensor, torch.Tensor], Tuple[int, int]])
event_counter_signature = TypeVar("event_counter_signature",
                                  bound=Callable[[ClassVar, torch.Tensor, torch.Tensor], Tuple[int, int]])

custom_energy_contribution_signature = TypeVar("custom_energy_contribution_signature",
                                               bound=Callable[[ClassVar, torch.Tensor, torch.Tensor], List[float]])


def synapse_neuron_count_for_linear(layer_info, inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[int, int]:
    # TODO : make sure that's correct
    if len(inputs) == 1:
        val = torch.Tensor(inputs[0])
    else:
        val = torch.Tensor(inputs)
    return layer_info.trainable_params, 0


def count_events_for_linear(layer_info, inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[int, int]:
    current_spiking_events = (float(torch.sum(inputs[0]).detach().cpu()) + layer_info.include_bias_in_events) * \
                             outputs[0].shape[0]
    current_total_events = (prod(inputs[0].size()) + layer_info.include_bias_in_events) * outputs[0].shape[0]
    return int(current_spiking_events), current_total_events


def synapse_neuron_count_for_snn_neuron(layer_info, inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[int, int]:
    val = torch.Tensor(outputs[0])
    layer_info.is_synaptic = False
    return 0, prod(outputs[0].shape[1:])


def count_events_for_snn_neuron(layer_info, inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[int, int]:
    current_spiking_events = int(torch.sum(outputs[0].detach()).cpu())
    current_total_events = prod(outputs[0].size())
    return current_spiking_events, current_total_events


def synapse_neuron_count_for_conv(layer_info, inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[int, int]:
    # TODO : make sure that's correct
    if len(inputs) == 1:
        val = torch.Tensor(inputs[0])
    else:
        val = torch.Tensor(inputs)
    return layer_info.trainable_params, 0


def count_events_for_conv(layer_info, inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[int, int]:
    connections = layer_info.trainable_params - layer_info.include_bias_in_events * prod(layer_info.module.bias.shape)
    current_total_events = connections * prod(outputs.shape)

    # TODO : seems like a very ugly solution, make it more elegant
    ones_conv = deepcopy(layer_info.module)
    # remove all the hooks from a copy of this layer
    ones_conv._forward_hooks = {}
    ones_conv.weight.data = torch.ones_like(ones_conv.weight.data)

    if layer_info.include_bias_in_events:
        ones_conv.bias.data = torch.ones_like(ones_conv.bias.data)
    else:
        ones_conv.bias.data = torch.zeros_like(ones_conv.bias.data)

    with torch.no_grad():
        current_spiking_events = int(torch.sum(ones_conv(inputs[0]).cpu()))

    return current_spiking_events, current_total_events


class LayerParameterEventCalculator(object):
    _recognized_layers: Dict[str, Tuple[
        synapse_neuron_count_signature, event_counter_signature, custom_energy_contribution_signature | None]] = \
        {
            nn.Linear.__name__: (synapse_neuron_count_for_linear, count_events_for_linear, None),
            snntorch.Leaky.__name__: (synapse_neuron_count_for_snn_neuron, count_events_for_snn_neuron, None),
            nn.Conv2d.__name__: (synapse_neuron_count_for_conv, count_events_for_conv, None),
            nn.Conv1d.__name__: (synapse_neuron_count_for_conv, None, None)
        }

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
    def get_calculate_synapse_neuron_count_for(name: str,
                                               ignore_not_recognized: bool = True) -> synapse_neuron_count_signature | None:
        if name not in LayerParameterEventCalculator._recognized_layers:
            if ignore_not_recognized:
                return None
            else:
                raise Exception(f"couldn't get the function to calculate the synapse neuron count for layer with "
                                f"classname :\'{name}\'")

        return LayerParameterEventCalculator._recognized_layers[name][0]

    @staticmethod
    def get_event_counter_for(name: str, ignore_not_recognized: bool = True) -> event_counter_signature | None:
        if name not in LayerParameterEventCalculator._recognized_layers:
            if ignore_not_recognized:
                return None
            else:
                raise Exception(f"couldn't get the function to count the total and spiking events for layer with "
                                f"classname :\'{name}\'")

        return LayerParameterEventCalculator._recognized_layers[name][1]

    @staticmethod
    def get_custom_energy_contribution_for(name: str) -> custom_energy_contribution_signature | None:
        if name not in LayerParameterEventCalculator._recognized_layers:
            return None
        return LayerParameterEventCalculator._recognized_layers[name][2]
