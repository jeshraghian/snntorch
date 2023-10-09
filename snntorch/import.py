import snntorch as snn
import numpy as np
import torch
import nir


class ImportedNetwork(torch.nn.Module):
    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list

    def forward(self, x):
        for module in self.module_list:
            # TODO: this must be implemented in snnTorch (timestep)
            x = module(x)
        return x


def _to_snntorch_module(node: nir.NIRNode) -> torch.nn.Module:
    """Convert a NIR node to a snnTorch module.

    Supported NIR nodes: Affine.
    """
    if isinstance(node, nir.LIF):
        return snn.Leaky()

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


def _get_next_node_key(node_key: str, graph: nir.ir.NIRGraph):
    """Get the next node key in the NIR graph."""
    possible_next_node_keys = [edge[1] for edge in graph.edges if edge[0] == node_key]
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

    while _get_next_node_key(node_key, graph.edges) is not None:
        node_key = _get_next_node_key(node_key, graph.edges)
        node = graph.nodes[node_key]

        if node_key in visited_node_keys:
            raise NotImplementedError('cyclic NIR graphs are not supported')

        visited_node_keys.append(node_key)
        print(f'node {node_key}: {type(node).__name__}')
        if node_key == 'output':
            continue
        module = _to_snntorch_module(node)
        module_list.append(module)

    if len(visited_node_keys) != len(graph.nodes):
        raise ValueError('not all nodes visited')

    return ImportedNetwork(module_list)
