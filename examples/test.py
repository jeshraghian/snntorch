import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List

import nir
import numpy as np

import spinnaker2
from spinnaker2 import ann2snn_helpers, hardware, snn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HardwareConstraintError(Exception):
    pass


class NotFoundError(Exception):
    pass


NEURON_NODES = (nir.LIF, nir.IF, nir.CubaLIF)
SPIKE_SOURCE_NODES = (nir.LIF, nir.IF, nir.CubaLIF, nir.Input)
WEIGHT_NODES = (nir.Affine, nir.Conv2d, nir.Conv2d, nir.Linear)


class IntegratorMethod(str, Enum):
    FORWARD = "Forward-Euler"
    EXPONENTIAL = "Exponential-Euler"


class ResetMethod(str, Enum):
    SUBTRACT = "subtract"  # subtract threshold
    ZERO = "zero"  # reset voltage to zero


def get_s2_reset_method(reset: ResetMethod):
    """get SpiNNaker2 reset method string."""
    if reset == ResetMethod.SUBTRACT:
        return "reset_by_subtraction"
    elif reset == ResetMethod.ZERO:
        return "reset_to_v_reset"
    else:
        raise Exception("Unsupported ResetMethod")


@dataclass
class ConversionConfig:
    """NIR-to-SpiNNaker2-conversion configuration.

    Attributes:
        dt: discretization timestep in seconds
        output_record: list of variables to record from output populations.
            Supported: ["spikes", "v"]. Default: `["spikes"]`.
        conn_delay: connection delay in timesteps for creating :spinnaker2.snn.Projection:s.
        scale_weights: if True, scale weights to maximum dynamic range [-127, 127],
            else don't scale weights
        integrator: numerical integration method for SpiNNaker2. This affects how
            parameters, especially time constants are translated to the SpiNNaker2
            neuron models.
            Available options:
                `IntegratorMethod.EXPONENTIAL`: First-order exponential Euler (default)
                `IntegratorMethod.FORWARD`: First-order forward Euler
        reset: neuron reset mechanism for SpiNNaker2. This affects how the
            membrane voltage is reset after a spike is detected.
            Available options:
                `ResetMethod.SUBTRACT`: subtract threshold from voltage (default)
                `ResetMethod.ZERO`: reset voltage to 0.0
    """

    dt: float = 1.0
    output_record: List[str] = field(default_factory=lambda: ["spikes"])
    conn_delay: int = 1
    scale_weights: bool = False
    integrator: IntegratorMethod = IntegratorMethod.EXPONENTIAL
    reset: ResetMethod = ResetMethod.SUBTRACT


def add_output_to_node(node_name, nir_model, output_name):
    assert node_name in nir_model.nodes.keys()
    assert output_name not in nir_model.nodes.keys()
    node = nir_model.nodes[node_name]
    output_node = nir.Output(output_type=node.output_type)
    nir_model.nodes[output_name] = output_node
    nir_model.edges.append((node_name, output_name))


def recurse_layer_shapes(name, node, nir_model):
    targets = get_outgoing_nodes(name, nir_model)
    print(name, type(node), node.input_type, node.output_type, "->", [name for name, _ in targets])
    for name, node in targets:
        recurse_layer_shapes(name, node, nir_model)


def model_summary(nir_model):
    inputs = [(name, n) for name, n in nir_model.nodes.items() if isinstance(n, nir.Input)]
    assert len(inputs) == 1
    inp = inputs[0]
    recurse_layer_shapes(inp[0], inp[1], nir_model)


def replace_sumpool2d_by_sumpool2d_if(nir_model):
    nodes = nir_model.nodes
    edges = nir_model.edges
    edges_to_remove = []
    edges_to_add = []
    nodes_to_add = {}
    for name, node in nodes.items():
        if isinstance(node, nir.SumPool2d):
            outgoing_nodes = get_outgoing_nodes(name, nir_model)
            edges = get_outgoing_edges(name, nir_model)
            old_edges = []
            for edge_idx, edge in enumerate(edges):
                if not isinstance(edge, (nir.LIF, nir.IF)):
                    old_edges.append(edge)
                    print("removing edge ", edge)

            if len(old_edges) > 0:
                shape = node.output_type["output"]
                # new_if_node = nir.IF(r=np.ones(shape), v_threshold=np.ones(node.output_type["output"]))
                new_if_node = nir.IF(r=np.ones(shape), v_threshold=np.ones(shape), v_reset=np.zeros(shape))
                new_if_name = f"{name}_if"
                nodes_to_add[new_if_name] = new_if_node
                edges_to_add.append((name, new_if_name))
                for edge in old_edges:
                    print("adding edge", (new_if_name, edge[1]))
                    edges_to_add.append((new_if_name, edge[1]))
                edges_to_remove.extend(old_edges)

    nir_model.nodes.update(nodes_to_add)
    for edge in edges_to_remove:
        print("really removing edge", edge)
        nir_model.edges.remove(edge)
    for edge in edges_to_add:
        print("really adding edge", edge)
        nir_model.edges.append(edge)
    return nir_model


def replace_avgpool2d_by_avgpool2d_if(nir_model):
    nodes = nir_model.nodes
    edges = nir_model.edges
    edges_to_remove = []
    edges_to_add = []
    nodes_to_add = {}
    for name, node in nodes.items():
        if isinstance(node, nir.AvgPool2d):
            outgoing_nodes = get_outgoing_nodes(name, nir_model)
            edges = get_outgoing_edges(name, nir_model)
            old_edges = []
            for edge_idx, edge in enumerate(edges):
                if not isinstance(edge, (nir.LIF, nir.IF)):
                    old_edges.append(edge)
                    print("removing edge ", edge)

            if len(old_edges) > 0:
                shape = node.output_type["output"]
                new_if_node = nir.IF(r=np.ones(shape), v_threshold=4 * np.ones(shape), v_reset=np.zeros(shape))
                new_if_name = f"{name}_if"
                nodes_to_add[new_if_name] = new_if_node
                edges_to_add.append((name, new_if_name))
                for edge in old_edges:
                    print("adding edge", (new_if_name, edge[1]))
                    edges_to_add.append((new_if_name, edge[1]))
                edges_to_remove.extend(old_edges)

    nir_model.nodes.update(nodes_to_add)
    for edge in edges_to_remove:
        print("really removing edge", edge)
        nir_model.edges.remove(edge)
    for edge in edges_to_add:
        print("really adding edge", edge)
        nir_model.edges.append(edge)
    return nir_model


def get_outgoing_edges(node_name, nir_model):
    outgoing_edges = []
    for edge in nir_model.edges:
        if edge[0] == node_name:
            outgoing_edges.append(edge)
    return outgoing_edges


def get_incoming_nodes(node_name, nir_model):
    incoming_nodes = []
    for edge in nir_model.edges:
        # print("edge:", edge, " current node:", node_name)
        if edge[1] == node_name:
            incoming_nodes.append((edge[0], nir_model.nodes[edge[0]]))
    return incoming_nodes


def get_outgoing_nodes(node_name, nir_model):
    """get all outgoing connected nodes of a node."""
    outgoing_nodes = []
    for edge in nir_model.edges:
        if edge[0] == node_name:
            outgoing_nodes.append((edge[1], nir_model.nodes[edge[1]]))
    return outgoing_nodes


def get_connected_nodes(node_name, nir_model):
    connected_nodes = []
    for edge in nir_model.edges:
        if edge[0] == node_name:
            connected_nodes.append((edge[1], nir_model.nodes[edge[1]]))
        if edge[1] == node_name:
            connected_nodes.append((edge[0], nir_model.nodes[edge[0]]))
    return connected_nodes


def fetch_population_by_name(name: str, populations: list):
    for pop in populations:
        if pop.name == name:
            return pop
    raise (NotFoundError(f"Population {name} could not be found!"))

#for this error:  ValueError: non-broadcastable output operand with shape () doesn't match the broadcast shape (11,)
#the shape of the bias array of the LIF node is not compatible with the shape of the bias array of the incomming node -> sol1: tInitialize the bias array with the shape of the incoming bias.
def get_accumulated_bias(node_name: str, nir_model):
    """get the accumulated bias for all units in a node.

    This function is applied to neuron nodes such as `nir.LIF` or `nir.CubaLIF`
    and looks for incoming nodes with bias (currently: `nir.Affine`,
    `nir.Conv1d` and `nir.Conv2d`). From theses incoming nodes the biases are
    accumulated and returned.

    Args:
        node_name: name of node
        nir_model: NIR graph

    Returns:
        np.ndarray: Array with accumulated bias with the same shape as the node
    """
    node = nir_model.nodes[node_name]
    size_bias = node.input_type["input"]
    bias = np.zeros(size_bias)
    for _, in_node in get_incoming_nodes(node_name, nir_model):
        if isinstance(in_node, (nir.Affine, nir.Conv1d, nir.Conv2d)):
            # Initialize bias array with the shape of the incoming bias
            bias = np.zeros_like(in_node.bias)
            bias += in_node.bias
    return bias.flatten()



#def get_accumulated_bias(node_name: str, nir_model):
 #   """get the accumulated bias for all units in a node.

 #    This function is applied to neuron nodes such as `nir.LIF` or `nir.CubaLIF`
  #   and looks for incoming nodes with bias (currently: `nir.Affine`,
   #  `nir.Conv1d` and `nir.Conv2d`). From theses incoming nodes the biases are
   #  accumulated and returned.

   #  Args:
  #       node_name: name of node
   #      nir_model: NIR graph

   #  Returns:
 #        np.ndarray: Array with accumulated bias with the same shape as the node
   #  """
  #   node = nir_model.nodes[node_name]
  #   print("this is the node:", node)
  #   size_bias = node.input_type["input"]
   #  print("size of the bias for LIF node:", size_bias)
   #  bias = np.zeros(node.input_type["input"])
   #  print("this is it's bias:", bias)
   #  for _, in_node in get_incoming_nodes(node_name, nir_model):
 #        if isinstance(in_node, (nir.Affine, nir.Conv1d, nir.Conv2d)):
 #            print("this is the incoming node:", in_node)
 #            print("this is the bias of the incoming node:", in_node.bias )
 #            bias += in_node.bias
 #    return bias.flatten()


def get_max_abs_weight(node_name: str, nir_model):
    """get the maximum absolute weight from all synapses

    This function is applied to neuron nodes such as `nir.LIF` or `nir.CubaLIF`
    and looks for incoming nodes with weights (currently: `nir.Affine`,
    `nir.Linear`, `nir.Conv1d` and `nir.Conv2d`). From theses incoming nodes
    the maximum weight is determined and returned.

    Args:
        node_name: name of node
        nir_model: NIR graph

    Returns:
        float: maximum absolute weight
    """
    max_weight = 0.0
    for _, in_node in get_incoming_nodes(node_name, nir_model):
        if isinstance(in_node, WEIGHT_NODES):
            max_weight_node = np.abs(in_node.weight).max()
            max_weight = np.max((max_weight, max_weight_node))
    return max_weight


def convert_LIF(node: nir.NIRNode, bias: np.ndarray, config: ConversionConfig, w_scale: float):  # noqa: N802
    """convert the LIF parameters to SpiNNaker2.

    Args:
        node: NIR node
        bias: bias array with same shape as node
        config: NIR-to-SpiNNaker2-conversion configuration
        w_scale: factor by which weights are scaled. Will be applied to
            parameters `threshold` and`i_offset`.

    Returns:
        tuple: (neuron_params, v_scale)

        `neuron_params` contains the parameters for the spinnaker2
        `lif_no_delay` population, while `v_scale` is the factor by which the
        threshold was scaled during translation from NIR to SpiNNaker2.
    """
    assert isinstance(node, nir.LIF)
    dt = config.dt
    tau = node.tau.flatten()

    if config.integrator == IntegratorMethod.FORWARD:
        r_factor = (dt / tau) * node.r.flatten()
        v_leak_factor = dt / tau
        alpha_decay = 1 - dt / node.tau
    elif config.integrator == IntegratorMethod.EXPONENTIAL:
        r_factor = (1 - np.exp(-dt / tau)) * node.r.flatten()
        v_leak_factor = 1 - np.exp(-dt / tau)
        alpha_decay = np.exp(-dt / tau)
    else:
        raise Exception("Unsupported IntegratorMethod")

    v_scale = 1.0 / r_factor  # scaling factor from tau_mem+r circuit
    scale = v_scale * w_scale  # overall scaling factor applied to membrane voltage

    neuron_params = {
        "threshold": node.v_threshold.flatten() * scale,
        "alpha_decay": alpha_decay,
        "i_offset": v_leak_factor * node.v_leak.flatten() * scale + bias.flatten() * w_scale,
        "reset": get_s2_reset_method(config.reset),
    }
    return neuron_params, scale


def convert_CubaLIF(node: nir.CubaLIF, bias: np.ndarray, config: ConversionConfig, w_scale: float):  # noqa N802
    """convert the CubaLIF parameters to SpiNNaker2.

    Args:
        node: NIR node
        bias: bias array with same shape as node
        config: NIR-to-SpiNNaker2-conversion configuration
        w_scale: factor by which weights are scaled. Will be applied to
            parameters `threshold` and`i_offset`.

    Returns:
        tuple: (neuron_params, v_scale)

        `neuron_params` contains the parameters for the spinnaker2
        `lif_curr_exp_no_delay` population, while `v_scale` is the factor by
        which the threshold was scaled during translation from NIR to
        SpiNNaker2.
    """
    assert isinstance(node, nir.CubaLIF)
    dt = config.dt

    if config.integrator == IntegratorMethod.FORWARD:
        r_factor = (dt / node.tau_mem) * node.r
        w_in_factor = (dt / node.tau_syn) * node.w_in
        v_leak_factor = dt / node.tau_mem
        syn_decay = 1 - dt / node.tau_syn
        alpha_decay = 1 - dt / node.tau_mem
    elif config.integrator == IntegratorMethod.EXPONENTIAL:
        r_factor = (1 - np.exp(-dt / node.tau_mem)) * node.r
        w_in_factor = (1 - np.exp(-dt / node.tau_syn)) * node.w_in
        v_leak_factor = 1 - np.exp(-dt / node.tau_mem)
        syn_decay = np.exp(-dt / node.tau_syn)
        alpha_decay = np.exp(-dt / node.tau_mem)
    else:
        raise Exception("Unsupported IntegratorMethod")

    v_scale = 1.0 / r_factor  # scaling factor from tau_mem+r circuit
    I_scale = 1.0 / w_in_factor  # scaling factor from the input current circuit
    scale = v_scale * I_scale * w_scale  # overall scaling factor applied to membrane voltage

    neuron_params = {
        "threshold": node.v_threshold * scale,
        "alpha_decay": alpha_decay,
        "exc_decay": syn_decay,
        "inh_decay": syn_decay,
        "i_offset": v_leak_factor * node.v_leak * scale + bias * node.w_in * I_scale * w_scale,
        "reset": get_s2_reset_method(config.reset),
        "t_refrac": 0,
        "v_reset": 0.0,  # will be ignored for `reset_by_subtraction`
    }
    return neuron_params, scale


def convert_IF(node: nir.NIRNode, bias: np.ndarray, config: ConversionConfig, w_scale: float):  # noqa: N802
    """convert the IF parameters to SpiNNaker2.

    Args:
        node: NIR node
        bias: bias array with same shape as node
        config: NIR-to-SpiNNaker2-conversion configuration
        w_scale: factor by which weights are scaled. Will be applied to
            parameters `threshold` and`i_offset`.

    Returns:
        tuple: (neuron_params, v_scale)

        `neuron_params` contains the parameters for the spinnaker2
        `lif_no_delay` population, while `v_scale` is the factor by which the
        threshold was scaled during translation from NIR to SpiNNaker2.
    """
    assert isinstance(node, nir.IF)
    bias_factor = node.r.flatten()
    v_scale = 1.0 / bias_factor * w_scale

    neuron_params = {
        "threshold": node.v_threshold.flatten() * v_scale,
        "alpha_decay": 1.0,
        "i_offset": bias * w_scale,
        "reset": get_s2_reset_method(config.reset),
    }
    return neuron_params, v_scale


def create_populations(nir_model, config: ConversionConfig):
    populations = []
    input_populations = []
    output_populations = []
    for name, node in nir_model.nodes.items():
        print(f"Node '{name}'")
        print(f"Got {type(node)}")
        if isinstance(node, (nir.LIF, nir.IF, nir.CubaLIF)):
            bias = get_accumulated_bias(name, nir_model)
            print("this is the accumulated bias:", bias)
            w_scale = 1.0
            if config.scale_weights:
                max_abs_weight = get_max_abs_weight(name, nir_model)
                w_scale = 127.0 / max_abs_weight  # TODO: 127 should not be hard-coded
        
            if any([isinstance(pre_node, nir.Conv2d) for _, pre_node in get_incoming_nodes(name, nir_model)]):
                is_conv2d = True
            else:
                is_conv2d = False

            if isinstance(node, nir.LIF):
                neuron_model_name = "lif_conv2d" if is_conv2d else "lif_no_delay"
                neuron_params, v_scale = convert_LIF(node, bias, config, w_scale)
            elif isinstance(node, nir.IF):
                neuron_model_name = "lif_conv2d" if is_conv2d else "lif_no_delay"
                neuron_params, v_scale = convert_IF(node, bias, config, w_scale)
            else:  # CubaLIF
                print("Got CubaLIF!")
                assert is_conv2d == False, "Conv2d with CubaLIF is currently not supported!"
                neuron_model_name = "lif_curr_exp_no_delay"
                neuron_params, v_scale = convert_CubaLIF(node, bias, config, w_scale)

            if any([isinstance(out_node, nir.Output) for _, out_node in get_outgoing_nodes(name, nir_model)]):
                record = config.output_record
                print(record)
                has_output = True
            else:
                record = []
                has_output = False

            print("record: ", record)
            input_shape = node.input_type.get("input", [])  # Get input shape or empty list
            if not input_shape:
                # Handle case where input shape is empty
                input_shape = [250]  # Set input shape to 250

            print("for node:", node, "this is the input_type:", input_shape)
            print("this is the input shape:", input_shape)
            print("and this is the size:", np.prod(input_shape))
            print(input_shape, "->", node.output_type.get("output"))

            # Create the population with the adjusted input shape
            pop = snn.Population(
                size=np.prod(input_shape),  # Calculate size based on input shape
                neuron_model=neuron_model_name,
                params=neuron_params,
                name=name,
                record=record,
            )
            print("this is the created population:", pop)
            pop.nir_v_scale = v_scale  # save scaling factors to rescale recorded voltages
            pop.nir_w_scale = w_scale  # weight scale needed later for creation of projections
            populations.append(pop)
            if has_output:
                output_populations.append(pop)

        elif isinstance(node, nir.Input):
            # infer shape
            input_shape = node.input_type["input"]
            print("this is the input shape of your model:", input_shape)
            #assert input_shape.size == 1 or input_shape.size == 3, "only 1d or 3d input allowed"
            pop = snn.Population(size=np.prod(input_shape), neuron_model="spike_list", params={}, name=name)
            populations.append(pop)
            input_populations.append(pop)
        elif isinstance(node, nir.Output):
            pass
        elif isinstance(node, WEIGHT_NODES):
            # TODO: check for any other combinations?
            for _, target_node in get_connected_nodes(name, nir_model):
                if isinstance(target_node, WEIGHT_NODES):
                    raise HardwareConstraintError(
                        f"Two successive layers with weights are not "
                        f"supported! Got: {type(node)} and {type(target_node)}"
                    )
        elif isinstance(node, nir.SumPool2d):
            pass
        elif isinstance(node, nir.AvgPool2d):
            pass
        elif isinstance(node, nir.Flatten):
            pass
        else:
            print("Got", type(node))
            raise NotImplementedError(type(node))
        print("input: ", [(e, type(n)) for e, n in get_incoming_nodes(name, nir_model)])
        print("output: ", [(e, type(n)) for e, n in get_outgoing_nodes(name, nir_model)])
        print("")
    return populations, input_populations, output_populations


def create_projection(origin_name, target_name, affine_node, populations, delay):
    assert isinstance(affine_node, (nir.Affine, nir.Linear))
    pre = fetch_population_by_name(origin_name, populations)
    post = fetch_population_by_name(target_name, populations)
    conns = ann2snn_helpers.connection_list_from_dense_weights(affine_node.weight.T * post.nir_w_scale, delay)
    proj = snn.Projection(pre=pre, post=post, connections=conns)
    return proj


def get_conv2d_params(conv2d_node, post_node):
    input_shape = conv2d_node.input_type["input"]
    conv_weights = conv2d_node.weight.swapaxes(0, 1) * post_node.nir_w_scale
    conv_weights = conv_weights.astype(np.int8)
    conv_params = {
        "in_height": input_shape[1],
        "in_width": input_shape[2],
        "stride_x": nir.ir._index_tuple(conv2d_node.stride, 0),
        "stride_y": nir.ir._index_tuple(conv2d_node.stride, 1),
        "pool_x": 1,
        "pool_y": 1,
        "pad_top": nir.ir._index_tuple(conv2d_node.padding, 0),
        "pad_bottom": nir.ir._index_tuple(conv2d_node.padding, 0),
        "pad_left": nir.ir._index_tuple(conv2d_node.padding, 1),
        "pad_right": nir.ir._index_tuple(conv2d_node.padding, 1),
    }
    return conv_params, conv_weights


def create_conv2d_projection(origin_name, target_name, conv2d_node, populations, delay):
    assert delay == 0, f"Conv2DProjection do not support a delay differnt from 1"
    pre = fetch_population_by_name(origin_name, populations)
    post = fetch_population_by_name(target_name, populations)
    conv_params, conv_weights = get_conv2d_params(conv2d_node, post)
    print(conv_params, "weights: ", conv_weights.shape)

    proj = snn.Conv2DProjection(pre, post, conv_weights, conv_params)

    return proj


def create_sumpool2d_projection(origin_name, target_name, sumpool2d_node, populations, delay):
    # this might be a bit hacky, but should work...
    pre = fetch_population_by_name(origin_name, populations)
    post = fetch_population_by_name(target_name, populations)
    # C_out, C_in, H, W
    sumpool2d_conns, sumpool2d_output_shape = ann2snn_helpers.connection_list_for_sumpool2d(
        input_shape=sumpool2d_node.input_type["input"],
        stride=sumpool2d_node.stride,
        kernel_size=sumpool2d_node.kernel_size,
        padding=sumpool2d_node.padding,
        delay=delay,
        data_order="torch",
    )
    proj = snn.Projection(pre=pre, post=post, connections=sumpool2d_conns)
    return proj


def create_avgpool2d_projection(origin_name, target_name, avgpool2d_node, populations, delay):
    # this might be a bit hacky, but should work...
    pre = fetch_population_by_name(origin_name, populations)
    post = fetch_population_by_name(target_name, populations)
    # C_out, C_in, H, W
    avgpool2d_conns, avgpool2d_output_shape = ann2snn_helpers.connection_list_for_avgpool2d(
        input_shape=avgpool2d_node.input_type["input"],
        stride=avgpool2d_node.stride,
        kernel_size=avgpool2d_node.kernel_size,
        padding=avgpool2d_node.padding,
        delay=delay,
        data_order="torch",
    )

    proj = snn.Projection(pre=pre, post=post, connections=avgpool2d_conns)
    return proj


def create_sumpool2d_conv2d_projection(origin_name, target_name, sumpool2d_node, conv2d_node, populations, delay):
    assert delay == 0, f"Conv2DProjection do not support a delay different from 1"
    # this might be a bit hacky, but should work...
    pre = fetch_population_by_name(origin_name, populations)
    post = fetch_population_by_name(target_name, populations)

    conv_params, conv_weights = get_conv2d_params(conv2d_node, post)
    print(conv_params, "weights: ", conv_weights.shape)
    print(sumpool2d_node)
    assert all(sumpool2d_node.kernel_size == sumpool2d_node.stride)
    assert nir.ir._index_tuple(sumpool2d_node.padding, 0) == 0
    assert nir.ir._index_tuple(sumpool2d_node.padding, 1) == 0
    conv_params["pool_x"] = nir.ir._index_tuple(sumpool2d_node.kernel_size, 0)
    conv_params["pool_y"] = nir.ir._index_tuple(sumpool2d_node.kernel_size, 1)
    # fix input sizes (we need to take the ones from the sumpool2d_node)
    input_shape = sumpool2d_node.input_type["input"]
    conv_params["in_height"] = input_shape[1]
    conv_params["in_width"] = input_shape[2]

    proj = snn.Conv2DProjection(pre, post, conv_weights, conv_params)

    return proj


def create_avgpool2d_conv2d_projection(
    origin_name, target_name, avgpool2d_node, conv2d_node, populations, delay, config
):
    assert delay == 0, f"Conv2DProjection do not support a delay different from 1"
    # this might be a bit hacky, but should work...
    pre = fetch_population_by_name(origin_name, populations)
    post = fetch_population_by_name(target_name, populations)

    conv_params, conv_weights = get_conv2d_params(conv2d_node, post)
    print(conv_params, "weights: ", conv_weights.shape)
    print(avgpool2d_node)
    # Ensure average pooling kernel size and stride are the same
    assert all(avgpool2d_node.kernel_size == avgpool2d_node.stride)
    # Calculate the scaling factor for weights
    kernel_size = avgpool2d_node.kernel_size[0] * avgpool2d_node.kernel_size[1]
    scale_factor = 1 / kernel_size
    # Scale the convolution weights by the factor of 4 (average pooling) and the scale factor
    conv_weights *= scale_factor / 4
    # Check if weight scaling is enabled in the config
    if config.scale_weights:
        max_abs_weight = np.max(np.abs(conv_weights))
        w_scale = 127.0 / max_abs_weight
        conv_weights *= w_scale
    # Set convolution parameters
    conv_params["pool_x"] = nir.ir._index_tuple(avgpool2d_node.kernel_size, 0)
    conv_params["pool_y"] = nir.ir._index_tuple(avgpool2d_node.kernel_size, 1)
    # Set input sizes from the avgpool2d_node
    input_shape = avgpool2d_node.input_type["input"]
    conv_params["in_height"] = input_shape[1]
    conv_params["in_width"] = input_shape[2]

    proj = snn.Conv2DProjection(pre, post, conv_weights, conv_params)

    return proj


def create_sumpool2d_affine_projection(origin_name, target_name, sumpool2d_node, affine_node, populations, delay):
    # this might be a bit hacky, but should work...
    pre = fetch_population_by_name(origin_name, populations)
    post = fetch_population_by_name(target_name, populations)
    # C_out, C_in, H, W
    sumpool2d_conns, sumpool2d_output_shape = ann2snn_helpers.connection_list_for_sumpool2d(
        input_shape=sumpool2d_node.input_type["input"],
        stride=sumpool2d_node.stride,
        kernel_size=sumpool2d_node.kernel_size,
        padding=sumpool2d_node.padding,
        delay=delay,
        data_order="torch",
    )
    affine_conns = ann2snn_helpers.connection_list_from_dense_weights(
        affine_node.weight.T * post.nir_w_scale, delay=delay
    )
    conns = ann2snn_helpers.join_conn_lists(sumpool2d_conns, affine_conns)

    proj = snn.Projection(pre=pre, post=post, connections=conns)
    return proj


def create_avgpool2d_affine_projection(origin_name, target_name, avgpool2d_node, affine_node, populations, delay):
    # this might be a bit hacky, but should work...
    pre = fetch_population_by_name(origin_name, populations)
    post = fetch_population_by_name(target_name, populations)
    # C_out, C_in, H, W
    avgpool2d_conns, avgpool2d_output_shape = ann2snn_helpers.connection_list_for_avgpool2d(
        input_shape=avgpool2d_node.input_type["input"],
        stride=avgpool2d_node.stride,
        kernel_size=avgpool2d_node.kernel_size,
        padding=avgpool2d_node.padding,
        delay=delay,
        data_order="torch",
    )
    affine_conns = ann2snn_helpers.connection_list_from_dense_weights(
        affine_node.weight.T * post.nir_w_scale, delay=delay
    )
    conns = ann2snn_helpers.join_conn_lists(avgpool2d_conns, affine_conns)

    proj = snn.Projection(pre=pre, post=post, connections=conns)
    return proj


def create_projections(nir_model, populations, delay=1):
    logger.debug("creating projections(): start")
    projections = []
    edges = nir_model.edges

    for edge in edges:
        logger.debug(f"checking {edge[0]}->{edge[1]}")
        origin = nir_model.nodes[edge[0]]
        if isinstance(origin, SPIKE_SOURCE_NODES):
            target_name = edge[1]
            target = nir_model.nodes[target_name]
            if isinstance(target, (nir.Affine, nir.Linear)):
                logger.debug(f"  found {type(target)}, next search for neuron node")
                final_targets = get_outgoing_nodes(target_name, nir_model)
                for final_target_name, final_target in final_targets:
                    logger.debug(f"    found final target {final_target_name}")
                    assert isinstance(final_target, NEURON_NODES)
                    logger.debug(f"    create projection between {edge[0]} and {final_target_name}")
                    proj = create_projection(edge[0], final_target_name, target, populations, delay)
                    projections.append(proj)
            elif isinstance(target, nir.SumPool2d):
                logger.debug("  found SumPool2d, next search for path to neurons")
                sumpooltargets = get_outgoing_nodes(target_name, nir_model)
                for sumpooltarget_name, sumpooltarget in sumpooltargets:
                    if isinstance(sumpooltarget, nir.Conv2d):
                        logger.debug(f"    found Conv2d: {sumpooltarget_name}")
                        final_targets = get_outgoing_nodes(sumpooltarget_name, nir_model)
                        for final_target_name, final_target in final_targets:
                            logger.debug(f"    found final target {final_target_name}")
                            assert isinstance(final_target, NEURON_NODES)
                            logger.debug(
                                f"    create sumpool2d_convd_projection between {edge[0]} and {final_target_name}"
                            )
                            proj = create_sumpool2d_conv2d_projection(
                                edge[0], final_target_name, target, sumpooltarget, populations, delay
                            )
                            projections.append(proj)
                    elif isinstance(sumpooltarget, NEURON_NODES):
                        logger.debug(f"    found Neuron: {sumpooltarget_name}")
                        final_targets = get_outgoing_nodes(sumpooltarget_name, nir_model)
                        nodetype = "IF" if isinstance(sumpooltarget, nir.IF) else "LIF"
                        logger.debug(f"    create sumpool2d_projection between {edge[0]} and {sumpooltarget_name}")
                        proj = create_sumpool2d_projection(edge[0], sumpooltarget_name, target, populations, delay)
                        projections.append(proj)
                    elif isinstance(sumpooltarget, nir.Flatten):
                        logger.debug(f"    found Flatten: {sumpooltarget_name}")
                        flatten_targets = get_outgoing_nodes(sumpooltarget_name, nir_model)
                        for flatten_target_name, flatten_target in flatten_targets:
                            if isinstance(flatten_target, nir.Affine):
                                final_targets = get_outgoing_nodes(flatten_target_name, nir_model)
                                for final_target_name, final_target in final_targets:
                                    logger.debug(f"    found final target {final_target_name}")
                                    assert isinstance(final_target, NEURON_NODES)
                                    logger.debug(
                                        f"    create sumpool2d_affine_projection between {edge[0]} and "
                                        f"{final_target_name}"
                                    )
                                    proj = create_sumpool2d_affine_projection(
                                        edge[0],
                                        final_target_name,
                                        target,
                                        flatten_target,
                                        populations,
                                        delay,
                                    )
                                    projections.append(proj)
                            else:
                                raise (
                                    NotImplementedError(
                                        "Currently after SumPool2d->Flatten, Affine has to follow! Other combinations "
                                        "not supported yet!"
                                    )
                                )
                    else:
                        raise (
                            NotImplementedError(
                                "Currently SumPool2d can only be connected to Conv2d or Flatten, others are not "
                                "supported yet!"
                            )
                        )

            elif isinstance(target, nir.AvgPool2d):
                logger.debug("  found AvgPool2d, next search for path to neurons")
                avgpooltargets = get_outgoing_nodes(target_name, nir_model)
                for avgpooltarget_name, avgpooltarget in avgpooltargets:
                    if isinstance(avgpooltarget, nir.Conv2d):
                        logger.debug(f"    found Conv2d: {avgpooltarget_name}")
                        final_targets = get_outgoing_nodes(avgpooltarget_name, nir_model)
                        for final_target_name, final_target in final_targets:
                            logger.debug(f"    found final target {final_target_name}")
                            assert isinstance(final_target, NEURON_NODES)
                            logger.debug(
                                f"    create avgpool2d_convd_projection between {edge[0]} and {final_target_name}"
                            )
                            proj = create_avgpool2d_conv2d_projection(
                                edge[0], final_target_name, target, avgpooltarget, populations, delay
                            )
                            projections.append(proj)
                    elif isinstance(avgpooltarget, NEURON_NODES):
                        logger.debug(f"    found Neuron: {avgpooltarget_name}")
                        final_targets = get_outgoing_nodes(avgpooltarget_name, nir_model)
                        nodetype = "IF" if isinstance(avgpooltarget, nir.IF) else "LIF"
                        logger.debug(f"    create avgpool2d_projection between {edge[0]} and {avgpooltarget_name}")
                        proj = create_avgpool2d_projection(edge[0], avgpooltarget_name, target, populations, delay)
                        projections.append(proj)
                    elif isinstance(avgpooltarget, nir.Flatten):
                        logger.debug(f"    found Flatten: {avgpooltarget_name}")
                        flatten_targets = get_outgoing_nodes(avgpooltarget_name, nir_model)
                        for flatten_target_name, flatten_target in flatten_targets:
                            if isinstance(flatten_target, nir.Affine):
                                final_targets = get_outgoing_nodes(flatten_target_name, nir_model)
                                for final_target_name, final_target in final_targets:
                                    logger.debug(f"    found final target {final_target_name}")
                                    assert isinstance(final_target, NEURON_NODES)
                                    logger.debug(
                                        f"    create avgpool2d_affine_projection between {edge[0]} and "
                                        f"{final_target_name}"
                                    )
                                    proj = create_avgpool2d_affine_projection(
                                        edge[0],
                                        final_target_name,
                                        target,
                                        flatten_target,
                                        populations,
                                        delay,
                                    )
                                    projections.append(proj)
                            else:
                                raise (
                                    NotImplementedError(
                                        "Currently after AvgPool2d->Flatten, Affine has to follow! Other combinations "
                                        "not supported yet!"
                                    )
                                )
                    else:
                        raise (
                            NotImplementedError(
                                "Currently AvgPool2d can only be connected to Conv2d or Flatten, others are not "
                                "supported yet!"
                            )
                        )

            elif isinstance(target, nir.Conv2d):
                logger.debug("  found Conv2d, next search for path to neurons")
                final_targets = get_outgoing_nodes(target_name, nir_model)
                for final_target_name, final_target in final_targets:
                    logger.debug(f"    found final target {final_target_name}")
                    print("NEURON_NODES are: (nir.LIF, nir.IF, nir.CubaLIF)")
                    print("this is the final_target:", final_target)
                    assert isinstance(final_target, NEURON_NODES)
                    logger.debug(f"    create conv2d_projection between {edge[0]} and {final_target_name}")
                    proj = create_conv2d_projection(edge[0], final_target_name, target, populations, delay)
                    projections.append(proj)
            elif isinstance(target, NEURON_NODES):
                raise (
                    NotImplementedError(
                        f"Direct connections from spike source nodes ({SPIKE_SOURCE_NODES}) to neuron nodes "
                        f"({NEURON_NODES}) not supported yet!"
                    )
                )
            else:
                logger.debug("  discard edge as target does not represent a real connection.")
        else:
            logger.debug("  discard edge as source is not spike source node")
    return projections


def from_nir(nir_model: nir.NIRGraph, config: ConversionConfig = None):
    """create SpiNNaker2 network from NIR graph.

    Args:
        nir_model: NIR graph
        config: NIR-to-SpiNNaker2-conversion configuration
    Returns:
        tuple of length 3: (net, input_pops, output_pops)

        Details:
            net(snn.Network): SpiNNaker2 Network
            input_pops(list[snn.Population]): list of input populations
            output_pops(list[snn.Population]): list of output populations
    """
    assert isinstance(nir_model, nir.NIRGraph)
    if config == None:
        config = ConversionConfig()
    logger.info("from_nir(): create spinnaker2.Network from NIR graph")
    populations, input_populations, output_populations = create_populations(nir_model, config)
    print("these are the populations:", populations)
    # Print the names of the populations
    print("Population Names:", [pop.name for pop in populations])
    logger.info(f"Created {len(populations)} populations: {[(_.name) for _ in populations]}")
    projections = create_projections(nir_model, populations, delay=config.conn_delay)
    logger.info(f"Created {len(projections)} projections: {[(_.name) for _ in projections]}")

    net = snn.Network()
    net.add(*populations, *projections)
    net.validate()
    return net, input_populations, output_populations