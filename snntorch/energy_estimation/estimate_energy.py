import numpy as np
import torch
import torch.nn as nn
from typing import Sequence, Union, Iterable, Callable, Mapping, cast, List
from .layer_info import LayerInfo, get_children_layers
from .FormattingOptions import *
from .model_statistics import *
from .device_profile_registry import DeviceProfileRegistry
from snntorch.utils import reset
import sys


def set_children_layers(summary_list: list[LayerInfo]) -> None:
    """Populates the children and depth_index fields of all LayerInfo."""
    idx: dict[int, int] = {}
    for i, layer in enumerate(summary_list):
        idx[layer.depth] = idx.get(layer.depth, 0) + 1
        layer.depth_index = idx[layer.depth]
        layer.children = get_children_layers(summary_list, i)


def prod(num_list: Iterable[int] | torch.Size) -> int:
    result = 1
    if isinstance(num_list, Iterable):
        for item in num_list:
            result *= prod(item) if isinstance(item, Iterable) else item
    return result


def estimate_energy(model: nn.Module,
                    devices: DeviceProfile | List[DeviceProfile | str] | str | None = "cpu",
                    input_size: Union[Sequence[int]] | None = None,
                    input_data: Union[torch.Tensor | np.ndarray | Sequence[float]] | None = None,
                    input_size_proportion_of_one: float = 1.0,
                    verbose: Verbosity = Verbosity.VERBOSE,
                    depth=1,
                    col_width: int = 25,
                    first_dim_is_time=True,
                    include_bias_term_in_events=True):
    # to make forward pass, we need to have either input data or input size
    # assert input_data is not None or input_size is not None, "To run estimate energy"
    validate_user_parameters(model, input_size, input_data, input_size_proportion_of_one)

    input_data = create_input_for_network_from_size(input_data, input_size, input_size_proportion_of_one,
                                                    first_dim_is_time)

    devices = _resolve_devices(devices)
    summary_list = forward_pass(model, input_data, devices, first_dim_is_time, include_bias_term_in_events)

    columns = [ColumnSettings.OUTPUT_SIZE, ColumnSettings.NUMBER_OF_SYNAPSES, ColumnSettings.NUMBER_OF_NEURONS,
               ColumnSettings.AVERAGE_FIRING_RATE, ColumnSettings.TOTAL_EVENTS, ColumnSettings.SPIKING_EVENTS]
    formatting = FormattingOptions(depth, verbose, columns, col_width, {RowSettings.DEPTH}, devices)
    results = ModelStatistics(summary_list, input_data.size(), get_total_memory_used(input_data), formatting,
                              devices)
    return results


def validate_user_parameters(model: nn.Module,
                             input_size: Union[Sequence[int]] | None = None,
                             input_data: Union[torch.Tensor | np.ndarray | Sequence[float]] | None = None,
                             input_size_proportion_of_one: float = 1.0):
    if input_size is not None and input_data is not None:
        raise Exception("only one of input_size and input_data should be specified")

    if input_size is None and input_data is None:
        raise Exception(
            "both input_size and input_data are None ! please specify one in order to do a forward pass and get "
            "energy estimates")


def create_input_for_network_from_size(input_data: Union[torch.Tensor | np.ndarray | Sequence[float]] | None = None,
                                       input_size: Union[Sequence[int]] | None = None,
                                       input_size_proportion_of_one: float = 1.0,
                                       first_dim_is_time: bool = True):
    if input_data is None:
        prob_mat = input_size_proportion_of_one * torch.ones(input_size[1:])
        return torch.bernoulli(prob_mat)
    else:
        # TODO
        return input_data


def forward_pass(model: nn.Module, fwd_inp: torch.Tensor, devices : List[DeviceProfile], first_dim_is_time: bool,
                 include_bias_term_in_events):
    model_name = model.__class__.__name__
    summary_list, global_layer_info, hooks = apply_hooks(model_name, model, fwd_inp, include_bias_term_in_events)
    # TODO: calculations done only for evaluation/inference, as on kerasSpiking (training will be
    #                                                                            much more difficult todo )
    model.eval()
    with torch.no_grad():
        out = model(fwd_inp)

    set_children_layers(summary_list)
    set_the_synapses_neurons_for_recursive_layers(summary_list)

    # remove the pre-hooks
    for module_id, h in hooks.items():
        h[0].remove()

    # flip the settings for layers to estimating the energy
    for module_id, layer_info in global_layer_info.items():
        layer_info.calculate_energy_mode = True

        # set the information about the devices to calculate the contributing energy
        layer_info.setup_device_profiles(devices)

    reset(model)

    # trickier case
    if first_dim_is_time:
        T = fwd_inp.size()[0]
        for t in range(T):
            inp = torch.stack([fwd_inp[t]])
            model(inp)

    # pretty straightforward, just run it once
    else:
        pass
    # assume that the first

    calculate_total_energies_recursively(summary_list)

    return summary_list


def set_the_synapses_neurons_for_recursive_layers(summary_list : List[LayerInfo]):
    # set the synapses correctly
    for layer in reversed(summary_list):
        if layer.neuron_count is None:
            n = 0
            for child in layer.children:
                n += child.neuron_count
            layer.neuron_count = n

        if layer.synapse_count is None:
            s = 0
            for child in layer.children:
                s += child.synapse_count
            layer.synapse_count = s

        # set the firing rate of composed layer to
        # the firing rate of last basic layer
        # TODO: does this make sense ?
        if len(layer.children) > 0:
            layer.firing_rate = layer.children[-1].firing_rate

def calculate_total_energies_recursively(summary_list : List[LayerInfo]):
    # always add the energy contributions for the
    for layer in summary_list:
        total_energy = np.array(layer.total_energy_contributions, dtype=float)
        for child in layer.children:
            total_energy += np.array(child.total_energy_contributions, dtype=float)
        layer.total_energy_contributions = total_energy

def construct_pre_hook(global_layer_info,
                       summary_list,
                       layer_ids,
                       var_name,
                       curr_depth,
                       parent_info,
                       include_bias_term_in_events=True):
    def pre_hook(module: nn.Module, inputs) -> None:
        """Create a Layer info object to aggregate layer information"""
        del inputs

        info = LayerInfo(var_name, module, curr_depth, parent_info, include_bias_in_events=include_bias_term_in_events)
        info.calculate_num_params()
        info.check_recursive(layer_ids)
        summary_list.append(info)
        layer_ids.add(info.layer_id)
        global_layer_info[info.layer_id] = info

    return pre_hook


def construct_hook(global_layer_info, batch_dim):
    def hook(module: nn.Module, inputs, outputs) -> None:
        """Update LayerInfo after forward pass."""
        info = global_layer_info[id(module)]
        if not info.calculate_energy_mode:
            if info.contains_lazy_param:
                info.calculate_num_params()

            info.calculate_synapse_neuron_count(inputs, outputs)
            info.input_size, _ = info.calculate_size(inputs, batch_dim)
            info.output_size, elem_bytes = info.calculate_size(outputs, batch_dim)
            info.output_bytes = elem_bytes * prod(info.output_size)
            info.executed = True
            # TODO: doesn't work
            # info.calculate_macs()
        else:
            info.calculate_events(inputs, outputs)

    return hook


def apply_hooks(model_name: str, model: nn.Module, fwd_inp: torch.Tensor, include_bias_term_in_events=True):
    """
        recursively adds hooks to all layers of the model
    """
    summary_list = []
    layers_ids = set()
    global_layer_info = {}
    hooks = {}
    stack = [(model_name, model, 0, None)]

    while stack:
        module_name, layer, curr_depth, parent_info = stack.pop()
        module_id = id(layer)

        global_layer_info[module_id] = LayerInfo(module_name, layer, curr_depth, parent_info)

        pre_hook = construct_pre_hook(global_layer_info, summary_list, layers_ids, module_name, curr_depth,
                                      parent_info, include_bias_term_in_events=include_bias_term_in_events)

        # register the hook using the last layer that uses this module
        if module_id in hooks:
            for hook in hooks[module_id]:
                hook.remove()

        hooks[module_id] = (layer.register_forward_pre_hook(pre_hook),
                            layer.register_forward_hook(construct_hook(global_layer_info, batch_dim=0)),)

        for name, mod in reversed(layer._modules.items()):
            if mod is not None:
                stack += [(name, mod, curr_depth + 1, global_layer_info[module_id])]

    return summary_list, global_layer_info, hooks


def traverse_input_data(
    data: Any, action_fn: Callable[..., Any], aggregate_fn: Callable[..., Any]
) -> Any:
    """
    Traverses any type of nested input data. On a tensor, returns the action given by
    action_fn, and afterwards aggregates the results using aggregate_fn.
    """
    if isinstance(data, torch.Tensor):
        result = action_fn(data)
    elif isinstance(data, np.ndarray):
        result = action_fn(torch.from_numpy(data))
        # If the result of action_fn is a torch.Tensor, then action_fn was meant for
        #   torch.Tensors only (like calling .to(...)) -> Ignore.
        if isinstance(result, torch.Tensor):
            result = data

    # Recursively apply to collection items
    elif isinstance(data, Mapping):
        aggregate = aggregate_fn(data)
        result = aggregate(
            {
                k: traverse_input_data(v, action_fn, aggregate_fn)
                for k, v in data.items()
            }
        )
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # Named tuple
        aggregate = aggregate_fn(data)
        result = aggregate(
            *(traverse_input_data(d, action_fn, aggregate_fn) for d in data)
        )
    elif isinstance(data, Iterable) and not isinstance(data, str):
        aggregate = aggregate_fn(data)
        result = aggregate(
            [traverse_input_data(d, action_fn, aggregate_fn) for d in data]
        )
    else:
        # Data is neither a tensor nor a collection
        result = data
    return result


def get_total_memory_used(data) -> int:
    """Calculates the total memory of all tensors stored in data."""
    result = traverse_input_data(
        data,
        action_fn=lambda data: sys.getsizeof(
            data.untyped_storage()
            if hasattr(data, "untyped_storage")
            else data.storage()
        ),
        aggregate_fn=(
            # We don't need the dictionary keys in this case
            lambda data: (lambda d: sum(d.values()))
            if isinstance(data, Mapping)
            else sum
        ),
    )
    return cast(int, result)


def _resolve_devices_base_case(device: str | DeviceProfile):
    if isinstance(device, DeviceProfile):
        return device
    elif isinstance(device, str):
        return DeviceProfileRegistry.get_device(device.lower())


def _resolve_devices(devices: str | DeviceProfile | List[str | DeviceProfile]) -> List[DeviceProfile]:
    if devices is None or len(devices) == 0:
        return []

    if isinstance(devices, str) or isinstance(devices, DeviceProfile):
        return [_resolve_devices_base_case(devices)]

    resolved_devices = []
    for device in devices:
        resolved_devices.append(_resolve_devices_base_case(device))
    return resolved_devices
