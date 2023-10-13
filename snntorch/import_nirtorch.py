import numpy as np
import nir
import nirtorch
import torch
import snntorch as snn


def _replace_rnn_subgraph_with_nirgraph(graph: nir.NIRGraph) -> nir.NIRGraph:
    """Take a NIRGraph and replace any RNN subgraphs with a single NIRGraph node."""
    if len([e for e in graph.nodes.values() if isinstance(e, nir.Input)]) > 1:
        cand_sg_nk = [e[1] for e in graph.edges if e[1] not in graph.nodes]
        print('detected subgraph! candidates:', cand_sg_nk)
        assert len(cand_sg_nk) == 1, 'only one subgraph allowed'
        nk = cand_sg_nk[0]
        nodes = {k: v for k, v in graph.nodes.items() if k.startswith(f'{nk}.')}
        edges = [e for e in graph.edges if e[0].startswith(f'{nk}.') or e[1].startswith(f'{nk}.')]
        valid_edges = all([e[0].startswith(f'{nk}.') for e in edges])
        valid_edges = valid_edges and all([e[1].startswith(f'{nk}.') for e in edges])
        assert valid_edges, 'subgraph edges must start with subgraph key'
        sg_graph = nir.NIRGraph(nodes=nodes, edges=edges)
        for k in nodes.keys():
            graph.nodes.pop(k)
        for e in edges:
            graph.edges.remove(e)
        graph.nodes[nk] = sg_graph
    return graph


def _parse_rnn_subgraph(graph: nir.NIRGraph) -> (nir.NIRNode, nir.NIRNode, int):
    """Try parsing the graph as a RNN subgraph.

    Assumes four nodes: Input, Output, LIF | CubaLIF, Affine | Linear
    Checks that all nodes have consistent shapes.
    Will throw an error if either not all nodes are found or consistent shapes are found.

    Returns:
        lif_node: LIF | CubaLIF node
        wrec_node: Affine | Linear node
        lif_size: int, number of neurons in the RNN
    """
    sub_nodes = graph.nodes.values()
    assert len(sub_nodes) == 4, 'only 4-node RNN allowed in subgraph'
    try:
        input_node = [n for n in sub_nodes if isinstance(n, nir.Input)][0]
        output_node = [n for n in sub_nodes if isinstance(n, nir.Output)][0]
        lif_node = [n for n in sub_nodes if isinstance(n, (nir.LIF, nir.CubaLIF))][0]
        wrec_node = [n for n in sub_nodes if isinstance(n, (nir.Affine, nir.Linear))][0]
    except IndexError:
        raise ValueError('invalid RNN subgraph - could not find all required nodes')
    lif_size = list(input_node.input_type.values())[0].size
    assert lif_size == list(output_node.output_type.values())[0].size, 'output size mismatch'
    assert lif_size == lif_node.v_threshold.size, 'lif size mismatch (v_threshold)'
    assert lif_size == wrec_node.weight.shape[0], 'w_rec shape mismatch'
    assert lif_size == wrec_node.weight.shape[1], 'w_rec shape mismatch'

    return lif_node, wrec_node, lif_size


def _nir_to_snntorch_module(node: nir.NIRNode) -> torch.nn.Module:
    if isinstance(node, nir.Input) or isinstance(node, nir.Output):
        return None

    elif isinstance(node, nir.Affine):
        assert node.bias is not None, 'bias must be specified for Affine layer'
        mod = torch.nn.Linear(node.weight.shape[1], node.weight.shape[0])
        mod.weight.data = torch.Tensor(node.weight)
        mod.bias.data = torch.Tensor(node.bias)
        return mod

    elif isinstance(node, nir.Linear):
        mod = torch.nn.Linear(node.weight.shape[1], node.weight.shape[0], bias=False)
        mod.weight.data = torch.Tensor(node.weight)
        return mod

    elif isinstance(node, nir.LIF):
        # NOTE: assuming that parameters are arrays of correct size
        assert np.unique(node.v_threshold).size == 1, 'v_threshold must be same for all neurons'
        dt = 1e-4
        vthr = node.v_threshold
        beta = 1 - (dt / node.tau)
        w_scale = node.r * dt / node.tau
        breakpoint()
        if np.alltrue(w_scale == 1.) or np.unique(w_scale).size == 1:
            # HACK to avoid scaling the inputs
            print('[warning] scaling weights to avoid scaling inputs')
            vthr = vthr / np.unique(w_scale)[0]
        else:
            raise NotImplementedError('w_scale must be 1, or the same for all neurons')
        return snn.Leaky(
            beta=beta,
            threshold=vthr,
            reset_mechanism='zero',
            init_hidden=True,
        )

    elif isinstance(node, nir.NIRGraph):
        lif_node, wrec_node, lif_size = _parse_rnn_subgraph(node)

        if isinstance(lif_node, nir.LIF):
            # TODO: fix neuron parameters
            rleaky = snn.RLeaky(
                beta=1 - (1 / lif_node.tau),
                threshold=lif_node.v_threshold,
                reset_mechanism='zero',
                init_hidden=True,
                all_to_all=True,
                linear_features=lif_size,
            )
            rleaky.recurrent.weight.data = torch.Tensor(wrec_node.weight)
            if isinstance(wrec_node, nir.Affine):
                rleaky.recurrent.bias.data = torch.Tensor(wrec_node.bias)
            return rleaky

        elif isinstance(lif_node, nir.CubaLIF):
            # TODO: fix neuron parameters
            rsynaptic = snn.RSynaptic(
                alpha=1 - (1 / lif_node.tau_syn),
                beta=1 - (1 / lif_node.tau_mem),
                init_hidden=True,
                reset_mechanism='zero',
                all_to_all=True,
                linear_features=lif_size,
            )
            rsynaptic.recurrent.weight.data = torch.Tensor(wrec_node.weight)
            if isinstance(wrec_node, nir.Affine):
                rsynaptic.recurrent.bias.data = torch.Tensor(wrec_node.bias)
            return rsynaptic

    else:
        print('[WARNING] could not parse node of type:', node.__class__.__name__)

    return None


def from_nir(graph: nir.NIRGraph) -> torch.nn.Module:
    # find valid RNN subgraphs, and replace them with a single NIRGraph node
    graph = _replace_rnn_subgraph_with_nirgraph(graph)
    # TODO: right now, the subgraph edges seem to not be parsed correctly - fix this
    return nirtorch.load(graph, _nir_to_snntorch_module)
