import snntorch as snn
import numpy as np
import torch
import nir
import typing


# TODO: implement this?
class ImportedNetwork(torch.nn.Module):
    """Wrapper for a snnTorch network. NOTE: not working atm."""
    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x


def create_snntorch_network(module_list):
    return torch.nn.Sequential(*module_list)


def _lif_to_snntorch_module(
        lif: typing.Union[nir.LIF, nir.CubaLIF]
) -> torch.nn.Module:
    """Parse a LIF node into snnTorch."""
    if isinstance(lif, nir.LIF):
        assert np.alltrue(lif.v_leak == 0), 'v_leak not supported'
        assert np.alltrue(lif.r == 1. - 1. / lif.tau), 'r not supported'
        assert np.unique(lif.v_threshold).size == 1, 'v_threshold must be same for all neurons'
        threshold = lif.v_threshold[0]
        mod = snn.RLeaky(
            beta=1. - 1. / lif.tau,
            threshold=threshold,
            all_to_all=True,
            reset_mechanism='zero',
            linear_features=lif.tau.shape[0] if len(lif.tau.shape) == 1 else None,
            init_hidden=True,
        )
        return mod

    elif isinstance(lif, nir.CubaLIF):
        assert np.alltrue(lif.v_leak == 0), 'v_leak not supported'
        assert np.alltrue(lif.r == 1. - 1. / lif.tau_mem), 'r not supported'  # NOTE: is this right?
        assert np.unique(lif.v_threshold).size == 1, 'v_threshold must be same for all neurons'
        threshold = lif.v_threshold[0]
        mod = snn.RSynaptic(
            alpha=1. - 1. / lif.tau_syn,
            beta=1. - 1. / lif.tau_mem,
            threshold=threshold,
            all_to_all=True,
            reset_mechanism='zero',
            linear_features=lif.tau_mem.shape[0] if len(lif.tau_mem.shape) == 1 else None,
            init_hidden=True,
        )
        return mod

    elif isinstance(lif, nir.LI):
        assert np.alltrue(lif.v_leak == 0), 'v_leak not supported'
        assert np.allclose(lif.r , 1. - 1. / lif.tau), 'r not supported'
        mod = snn.Leaky(
            beta=1. - 1. / lif.tau,
            reset_mechanism='none',
            init_hidden=True,
            output=True,
        )
        return mod

    else:
        raise ValueError('called _lif_to_snntorch_module on non-LIF node')


def _to_snntorch_module(node: nir.NIRNode) -> torch.nn.Module:
    """Convert a NIR node to a snnTorch module.

    Supported NIR nodes: Affine.
    """
    if isinstance(node, (nir.LIF, nir.CubaLIF)):
        return _lif_to_snntorch_module(node)

    elif isinstance(node, nir.Affine):
        if len(node.weight.shape) != 2:
            raise NotImplementedError('only 2D weight matrices are supported')
        has_bias = node.bias is not None and not np.alltrue(node.bias == 0)
        linear = torch.nn.Linear(node.weight.shape[1], node.weight.shape[0], bias=has_bias)
        linear.weight.data = torch.Tensor(node.weight)
        if has_bias:
            linear.bias.data = torch.Tensor(node.bias)
        return linear

    else:
        raise NotImplementedError(f'node type {type(node).__name__} not supported')


def _rnn_subgraph_to_snntorch_module(
        lif: typing.Union[nir.LIF, nir.CubaLIF], w_rec: typing.Union[nir.Affine, nir.Linear]
) -> torch.nn.Module:
    """Parse an RNN subgraph consisting of a LIF node and a recurrent weight matrix into snnTorch.

    NOTE: for now always set it as a recurrent linear layer (not RecurrentOneToOne)
    """
    assert isinstance(lif, (nir.LIF, nir.CubaLIF)), 'only LIF or CubaLIF nodes supported as RNNs'
    mod = _lif_to_snntorch_module(lif)
    mod.recurrent.weight.data = torch.Tensor(w_rec.weight)
    if isinstance(w_rec, nir.Linear):
        mod.recurrent.register_parameter('bias', None)
        mod.recurrent.reset_parameters()
    else:
        mod.recurrent.bias.data = torch.Tensor(w_rec.bias)
    return mod


def _get_next_node_key(node_key: str, graph: nir.ir.NIRGraph):
    """Get the next node key in the NIR graph."""
    possible_next_node_keys = [edge[1] for edge in graph.edges if edge[0] == node_key]
    # possible_next_node_keys += [edge[1] + '.input' for edge in graph.edges if edge[0] == node_key]
    assert len(possible_next_node_keys) <= 1, 'branching networks are not supported'
    if len(possible_next_node_keys) == 0:
        return None
    else:
        return possible_next_node_keys[0]


def from_nir(graph: nir.ir.NIRGraph) -> torch.nn.Module:
    """Convert NIR graph to snnTorch module.

    :param graph: a saved snnTorch model as a parameter dictionary
    :type graph: nir.ir.NIRGraph

    :return: snnTorch module
    :rtype: torch.nn.Module
    """
    node_key = 'input'
    visited_node_keys = [node_key]
    module_list = []

    while _get_next_node_key(node_key, graph) is not None:
        node_key = _get_next_node_key(node_key, graph)

        assert node_key not in visited_node_keys, 'cyclic NIR graphs not supported'

        if node_key == 'output':
            visited_node_keys.append(node_key)
            continue

        if node_key in graph.nodes:
            visited_node_keys.append(node_key)
            node = graph.nodes[node_key]
            print(f'simple node {node_key}: {type(node).__name__}')
            module = _to_snntorch_module(node)
        else:
            # check if it's a nested node
            print(f'potential subgraph node: {node_key}')
            sub_node_keys = [n for n in graph.nodes if n.startswith(f'{node_key}.')]
            assert len(sub_node_keys) > 0, f'no nodes found for subgraph {node_key}'

            # parse subgraph
            # NOTE: for now only looking for RNN subgraphs
            rnn_sub_node_keys = [f'{node_key}.{n}' for n in ['input', 'output', 'lif', 'w_rec']]
            if set(sub_node_keys) != set(rnn_sub_node_keys):
                raise NotImplementedError('only RNN subgraphs are supported')
            print('found RNN subgraph')
            module = _rnn_subgraph_to_snntorch_module(
                graph.nodes[f'{node_key}.lif'], graph.nodes[f'{node_key}.w_rec']
            )
            for nk in sub_node_keys:
                visited_node_keys.append(nk)

        module_list.append(module)

    if len(visited_node_keys) != len(graph.nodes):
        print(graph.nodes.keys(), visited_node_keys)
        raise ValueError('not all nodes visited')

    return create_snntorch_network(module_list)
