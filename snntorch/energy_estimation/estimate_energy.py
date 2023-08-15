# import the | for compatibility with python 3.7.*
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence, Union, Iterable, Callable, Mapping, cast, List
from .layer_info import LayerInfo, get_children_layers
from .FormattingOptions import *
from .model_statistics import *
from .device_profile_registry import DeviceProfileRegistry
from .energy_estimation_network_interface import EnergyEstimationNetworkInterface
from snntorch.utils import reset
from .utils import *
import sys

"""
    The code from torchinfo project (https://github.com/TylerYep/torchinfo) has been adjusted to get the estimate energy
    function (quite a bit of it was modified, however overall structure, especially with displaying is the same)
"""


def estimate_energy(model: nn.Module | EnergyEstimationNetworkInterface,
                    devices: DeviceProfile | List[DeviceProfile | str] | str | None = "cpu",
                    input_size: Union[Sequence[int] | None] = None,
                    input_data: Union[torch.Tensor | np.ndarray | Sequence[float]] | None = None,
                    input_size_proportion_of_one: float = 1.0,
                    verbose: Verbosity = Verbosity.VERBOSE,
                    depth=1,
                    col_width: int = 25,
                    first_dim_is_time=True,
                    network_requires_first_dim_as_time=True,
                    include_bias_term_in_events=True,
                    columns: List[ColumnSettings] = (ColumnSettings.NUMBER_OF_SYNAPSES,
                                                     ColumnSettings.NUMBER_OF_NEURONS,
                                                     ColumnSettings.AVERAGE_FIRING_RATE,
                                                     ColumnSettings.TOTAL_EVENTS, ColumnSettings.SPIKING_EVENTS)):
    """
`       The main function handling generating the energy estimation feature. It takes a model, prepared by user input
        (either specify a shape by passing `input_size` or data `input_data` on which we want the network to run ), and
        a list of devices (represented by names or Device Profiles) with information how the energy should be calculated.

        It is expected that the network either accepts input in shape (time, batchsize, ...) when
        `network_requires_first_dim_as_time` is True or (batchsize, ...) when False.

        :param model : a PyTorch architecture for which we want to do the energy estimation. Because the network itself
        can be implemented in many ways (more flexibility than Keras), a helpful assumption is that the network except
        forward method (or overall some method doing computation) it will also have a `reset` method from
        `EnergyEstimationNetworkInterface` which will reset all hidden states from the network. If the module is
        an instance of nn.Module, it will try to reset it using snntorch.utils reset function (TODO : however currently
        I'm probably using it wrong, as I didn't get consistent numbers)

        :param devices : a string or list of strings representing a name of the device (from DeviceProfileRegistry).
        For every specified device, layer-wise energy estimation will be provided

        :param input_size : a tuple of ints representing the shape of sample input. Setting this parameters to something
        else than None is mutually exclusive with `input_data` (only one should be specified), please specify only one.
        A tensor with requested shape will be prepared to pass to the network (each element of that tensor will have
        `input_size_proportion_of_one`probability of being 1.0 (otherwise zero)).

        :param input_data :a pytorch tensor (or python array of floats or numpy float array) that will be passed to the
        Pytorch model `model`. Mutually exclusive with input_size, please specify only one

        :param input_size_proportion_of_one : if `input_size` is provided, each of the elements will have probability
        of being one set to this value.  it is implemented in following way :

            >>> prob_mat = input_size_proportion_of_one * torch.ones(input_size)
            >>> return torch.bernoulli(prob_mat)

        :param first_dim_is_time : boolean flag which when set to true, will let this function know that the first
        dimension in input (either input_data or input_shape) we pass to network is time dimension. It is important
        to set it appropriately, so this function can interpret the data correctly

        :param network_requires_first_dim_as_time : boolean flag which when set to true, will let this function know
        that the PyTorch model requires first dimension to be time ( accepts (time, batchsize, ...)). If set to True,
        an input in format (batchsize, ...) will be passed. It is important to set it appropriately, so the input data
        can be passed correctly to PyTorch model.

        :param include_bias_term_in_events : If set to False, the bias calculations for energy are excluded (makes them
        the same as in kerasSpiking ). If set to True, bias terms are included

        :param columns : a list of Columns to display

        :param col_width : width of the column required for printing

        :param depth : The level of Depth through which we will go to print the model.

        :param verbose : verbosity level set for formatting options
    """

    # to make forward pass, we need to have either input data or input size
    # assert input_data is not None or input_size is not None, "To run estimate energy"
    validate_user_parameters(model, input_size, input_data, input_size_proportion_of_one)

    # based on provided data, prepare the input for passing to the network
    # we want our input to be in format (time, batch size , ... )
    input_data = create_input_for_network_from_size(input_data, input_size, input_size_proportion_of_one,
                                                    first_dim_is_time)

    # get the information about the devices for which we will get the energy estimates
    devices = _resolve_devices(devices)

    # get the layer-wise information about the parameters/events/energies by doing a forward pass with the prepared
    # `input_data` on PyTorch model `model`
    summary_list = forward_pass(model, input_data, devices, first_dim_is_time,
                                network_requires_first_dim_as_time, include_bias_term_in_events)

    # displaying options from https://github.com/TylerYep/torchinfo
    columns = list(columns)
    formatting = FormattingOptions(depth, verbose, columns, col_width, {RowSettings.DEPTH}, devices)
    results = ModelStatistics(summary_list, input_data.size(), get_total_memory_used(input_data), formatting,
                              devices)
    return results


def validate_user_parameters(model: nn.Module | EnergyEstimationNetworkInterface,
                             input_size: Union[Sequence[int]] | None = None,
                             input_data: Union[torch.Tensor | np.ndarray | Sequence[float]] | None = None,
                             input_size_proportion_of_one: float = 1.0):
    """
        Function which takes information about the data (either input_size or input_data) and does some basic sanity
        checks about the data

        TODO: probably add more here, inform user about potential problems
    """
    if input_size is not None and input_data is not None:
        raise Exception("only one of input_size and input_data should be specified")

    if input_size is None and input_data is None:
        raise Exception(
            "both input_size and input_data are None ! please specify one in order to do a forward pass and get "
            "energy estimates")


def create_input_for_network_from_size(input_data: Union[torch.Tensor | np.ndarray | Sequence[float]] | None = None,
                                       input_size: Union[Sequence[int]] | None = None,
                                       input_size_proportion_of_one: float = 1.0,
                                       first_dim_is_time: bool = True) -> torch.Tensor:
    """
        Based on provided input, return a correct input to our network, in format (time, batchsize, ....)
    """

    if input_data is None:
        if first_dim_is_time:
            prob_mat = input_size_proportion_of_one * torch.ones(input_size)
        else:
            prob_mat = input_size_proportion_of_one * torch.ones((1, *input_size))
        return torch.bernoulli(prob_mat)
    else:
        if first_dim_is_time:
            return input_data
        else:
            return torch.stack([input_data])


def forward_pass(model: nn.Module | EnergyEstimationNetworkInterface, fwd_inp: torch.Tensor,
                 devices: List[DeviceProfile], first_dim_is_time: bool, network_requires_first_dim_as_time: bool,
                 include_bias_term_in_events: bool):
    # get the class name of the model
    model_name = model.__class__.__name__

    # apply the forward hooks and pre-hooks
    summary_list, global_layer_info, hooks = apply_hooks(model_name, model, fwd_inp, include_bias_term_in_events,
                                                         network_requires_first_dim_as_time)

    # set the model to evaluation model (no need for gradients)
    model.eval()

    # until this point the hooks has been registered, and input has been preprocessed to be in expected
    # shape of (time, batch size, ...). If the flag `network_requires_first_dim_as_time` is true we will pass
    # (time, batch size, ...) or when false just (batch size, ...)
    if network_requires_first_dim_as_time:
        # pass (timestep, batch size, ...)
        fwd_inp_single = torch.stack([fwd_inp[0]])
        with torch.no_grad():
            model(fwd_inp_single)
    else:
        # pass (timestep, batch size, ...)
        with torch.no_grad():
            model(fwd_inp[0])

    # set the children layer information to any nested layers
    set_children_layers(summary_list)

    # set the synapse and neuron counts for any nested layers
    set_the_synapses_neurons_for_nested_layers(summary_list)

    # remove the pre-hooks, they are no longer needed
    for module_id, h in hooks.items():
        h[0].remove()

    # flip the settings for layers to estimating the energy
    for module_id, layer_info in global_layer_info.items():
        layer_info.calculate_energy_mode = True

        # set the information about the devices to calculate the contributing energy
        layer_info.setup_device_profiles(devices)

    # TODO : this probably needs to be done better
    if isinstance(model, EnergyEstimationNetworkInterface):
        model.reset()
    else:
        # TODO : does this work as expected ? weird results ?
        reset(model)

    # get time number of timesteps
    T = fwd_inp.size()[0]

    # iterate over all timesteps

    # TODO: should I pass the whole input here if the network requires time as first dimension ?
    # something like this :
    # if network_requires_first_dim_as_time:
    #   model(fwd_inp)
    # rather than below
    for t in range(T):
        if network_requires_first_dim_as_time:
            inp = torch.stack([fwd_inp[t]])
        else:
            inp = fwd_inp[t]
        model(inp)

    # the forward passes for whole input has been made, calculate the energies
    calculate_total_energies_nested_layers(summary_list)

    return summary_list


def set_the_synapses_neurons_for_nested_layers(summary_list: List[LayerInfo]):
    # set the synapses correctly
    for layer in summary_list:

        # if neuron count is None (so either a non-basic module or unrecognized module), go through it's children
        # and get the total number of neurons (and set its own neuron count to that value)
        if layer.neuron_count is None:
            n = None
            for child in layer.children:
                if child.neuron_count is not None:
                    if n is None:
                        n = 0
                    n += child.neuron_count
            layer.neuron_count = n

        # do the same calculation for synapse count
        if layer.synapse_count is None:
            s = None
            for child in layer.children:
                if child.synapse_count is not None:
                    if s is None:
                        s = 0
                    s += child.synapse_count
            layer.synapse_count = s


def calculate_total_energies_nested_layers(summary_list: List[LayerInfo]):
    """
        Calculate the total energy, spiking and total events for nested layers
    """

    # always add the energy contributions for the
    for layer in summary_list:

        # convert energy to numpy array (easier to add everything together)
        total_energy = np.array(layer.total_energy_contributions, dtype=float)

        # add the energies of its children to the total energy
        for child in layer.children:
            total_energy += np.array(child.total_energy_contributions, dtype=float)

        # update the total energy contributions
        layer.total_energy_contributions = total_energy

        if layer.total_events is None:
            total_events = 0
            for child in layer.children:

                # watch out for layers/children that were not recognized (i.e. the total events is none)
                if child.total_events is not None:
                    total_events += child.total_events
            layer.total_events = None if total_events == 0 else total_events

        if layer.spiking_events is None:
            spiking_events = None
            for child in layer.children:

                # watch out for layers/children that were not recognized (i.e. the spiking events is none)
                if child.spiking_events is not None:
                    if spiking_events is None:
                        spiking_events = 0
                    spiking_events += child.spiking_events

            layer.spiking_events = None if spiking_events is None else spiking_events

        # set the firing rate to the spiking_events / total_events
        if layer.spiking_events is not None and layer.total_events is not None and layer.firing_rate is None:
            layer.firing_rate = layer.spiking_events / layer.total_events


def construct_pre_hook(global_layer_info,
                       summary_list,
                       layer_ids,
                       var_name,
                       curr_depth,
                       parent_info,
                       include_bias_term_in_events,
                       network_requires_first_dim_as_time):
    def pre_hook(module: nn.Module, inputs) -> None:
        """Create a Layer info object to aggregate layer information"""
        del inputs

        info = LayerInfo(var_name, module, curr_depth, parent_info,
                         include_bias_term_in_events,
                         network_requires_first_dim_as_time)
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
            info.calculate_macs()
        else:
            info.calculate_events(inputs, outputs)

    return hook


def apply_hooks(model_name: str, model: nn.Module, fwd_inp: torch.Tensor, include_bias_term_in_events: bool,
                network_requires_first_dim_as_time: bool):
    """
        recursively adds hooks to all layers of the model
    """
    summary_list = []
    layers_ids = set()
    global_layer_info = {}
    hooks = {}

    #        (modelname, nn.Module, depth, parent)
    stack = [(model_name, model, 0, None)]

    while stack:
        module_name, layer, curr_depth, parent_info = stack.pop()
        module_id = id(layer)

        global_layer_info[module_id] = LayerInfo(module_name, layer, curr_depth, parent_info)

        pre_hook = construct_pre_hook(global_layer_info, summary_list, layers_ids, module_name, curr_depth,
                                      parent_info, include_bias_term_in_events=include_bias_term_in_events,
                                      network_requires_first_dim_as_time=network_requires_first_dim_as_time
                                      )

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
    action_fn, and afterward aggregates the results using aggregate_fn.
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


def set_children_layers(summary_list: list[LayerInfo]) -> None:
    """Populates the children and depth_index fields of all LayerInfo."""
    idx: dict[int, int] = {}
    for i, layer in enumerate(summary_list):
        idx[layer.depth] = idx.get(layer.depth, 0) + 1
        layer.depth_index = idx[layer.depth]
        layer.children = get_children_layers(summary_list, i)
