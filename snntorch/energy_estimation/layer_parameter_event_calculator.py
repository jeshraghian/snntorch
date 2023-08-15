# import the | for compatibility with python 3.7.*
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable, Tuple, Dict, TypeVar, ClassVar, List, Type
from .device_profile import DeviceProfile
from copy import deepcopy

import snntorch
from .utils import *

# signature for functions
synapse_neuron_count_signature = TypeVar("synapse_neuron_count_signature",
                                         bound=Callable[[object, torch.Tensor, torch.Tensor], Tuple[int, int]])
#                                                       TODO: passing object is not a good solution probably in typing
#                                                        This most likely should be LayerInfo (or some base class of it)
event_counter_signature = TypeVar("event_counter_signature",
                                  bound=Callable[[object, torch.Tensor, torch.Tensor], Tuple[int, int]])
#                                                       TODO: passing object is not a good solution probably in typing
#                                                        This most likely should be LayerInfo (or some base class of it)

custom_energy_contribution_signature = TypeVar("custom_energy_contribution_signature",
                                               bound=Callable[[object, torch.Tensor, torch.Tensor, DeviceProfile],
                                               float])
#                                                       TODO: passing object is not a good solution probably in typing
#                                                        This most likely should be LayerInfo (or some base class of it)


# calculate the neuron and synapse count for Linear layer
def synapse_neuron_count_for_linear(layer_info, inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[int, int]:
    if len(inputs) == 1:
        val = torch.Tensor(inputs[0])
    else:
        val = torch.Tensor(inputs)

    return layer_info.trainable_params, 0


# calculate the number of events (spiking/total) for a Linear layer
def count_events_for_linear(layer_info, inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[int, int]:
    current_spiking_events = (float(torch.sum(inputs[0]).detach().cpu()) + layer_info.include_bias_in_events) * \
                             prod(outputs[0].shape)
    current_total_events = (prod(inputs[0].size()) + layer_info.include_bias_in_events) * prod(outputs[0].shape)

    # in the above calculations the is included twice (these values scales quadratically rather than linearly)
    # to solve this, get the batch size value (shape[1] or shape[0], depending on whether
    # network_requires_first_dim_as_time is True or False ) and divide both values by it

    if layer_info.network_requires_first_dim_as_time:
        batch_size = inputs[0].shape[1]
    else:
        batch_size = inputs[0].shape[0]

    return int(current_spiking_events / batch_size), int(current_total_events / batch_size)


# calculate the neuron and synapse count for snntorch activation function ( like Leaky)
def synapse_neuron_count_for_snn_neuron(layer_info, inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[int, int]:
    val = torch.Tensor(outputs[0])
    layer_info.is_synaptic = False

    if layer_info.network_requires_first_dim_as_time:
        return 0, prod(outputs[0].shape[2:])
    return 0, prod(outputs[0].shape[1:])


# calculate the number of events (spiking/total) for a snntorch activation function ( like Leaky)
def count_events_for_snn_neuron(layer_info, inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[int, int]:
    current_spiking_events = int(torch.sum(outputs[0].detach()).cpu())
    current_total_events = prod(outputs[0].size())
    return current_spiking_events, current_total_events


# calculate the neuron and synapse count for the convolution layers
def synapse_neuron_count_for_conv(layer_info, inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[int, int]:
    # TODO : make sure that's correct
    if len(inputs) == 1:
        val = torch.Tensor(inputs[0])
    else:
        val = torch.Tensor(inputs)
    return layer_info.trainable_params, 0


# calculate the number of events (spiking/total) for a convolution layers
def count_events_for_conv(layer_info, inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[int, int]:
    connections = layer_info.trainable_params - (1 - layer_info.include_bias_in_events) * prod(
        layer_info.module.bias.shape)

    # get input and output channels
    in_channels, out_channels, kernel_size, bias_size = (layer_info.module.in_channels, layer_info.module.out_channels,
                                                         layer_info.module.kernel_size,
                                                         layer_info.module.bias.data.shape)

    current_total_events = (in_channels * prod(outputs.shape) *
                            (prod(kernel_size) + layer_info.include_bias_in_events * prod(bias_size)))

    # TODO : seems like a very ugly solution, maybe make it more elegant ?
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
    """
        Singleton which provides an API for getting neuron/synapse/event counts for the estimate_energy.
        Every basic pytorch layer (like Linear, Conv2d or activation function ) has a small function which will return
        to synapse/neuron/event counts, which will be calculated in estimate_energy.

        Every layer can also have a `custom_energy_contribution` function (with signature
        `custom_energy_contribution_signature`), which will override default calculations (multiply the number of
        synapses and neurons by energies required to perform these operations ), and developer can specify custom
        energy value (based on input and output tensors + significant amount of information from LayerInfo)

        An example of adding a layer using this singleton :

        >>> class ANewLinearLayer(nn.module):
        >>>     def forward(self, x : torch.Tensor):
        >>>        # ...
        >>>
        >>>
        >>>  LayerParameterEventCalculator.register_new_layer(ANewLinearLayer,
        >>>                                             synapse_neuron_count_for_linear,
        >>>                                             count_events_for_linear)

    """

    # a dictionary mapping a layer class name (like Linear) to tuple, where :
    # - first element is a function returning 2 ints representing number of synapses and neurons (function should
    # have signature `synapse_neuron_count_signature`)
    # - second element is function returning 2 ints representing number of spiking and total events (function should
    # have signature 'event_counter_signature')
    # - third optional element returning a float representing energy required to perform the operation (function, if
    # specified should have signature custom_energy_contribution_signature )
    _recognized_layers: Dict[str, Tuple[
        synapse_neuron_count_signature, event_counter_signature, custom_energy_contribution_signature | None]] = \
        {
            nn.Linear.__name__: (synapse_neuron_count_for_linear, count_events_for_linear, None),
            snntorch.Leaky.__name__: (synapse_neuron_count_for_snn_neuron, count_events_for_snn_neuron, None),
            nn.Conv2d.__name__: (synapse_neuron_count_for_conv, count_events_for_conv, None),
            nn.Conv1d.__name__: (synapse_neuron_count_for_conv, count_events_for_conv, None)
        }

    @staticmethod
    def register_new_layer(layer: Type[torch.nn.Module],
                           synapse_neuron_count_lambda: synapse_neuron_count_signature,
                           event_counter_lambda: event_counter_signature,
                           custom_energy_contribution: custom_energy_contribution_signature = None,
                           override_layer_info: bool = False):

        if layer.__class__.__name__ in LayerParameterEventCalculator._recognized_layers:
            if override_layer_info:
                pass
            else:
                raise Exception("overriding layer information, and override_layer_info was set to False!")

        LayerParameterEventCalculator._recognized_layers[layer.__name__] = (synapse_neuron_count_lambda,
                                                                            event_counter_lambda,
                                                                            custom_energy_contribution)

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
