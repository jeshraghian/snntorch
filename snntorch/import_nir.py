import numpy as np
import nir
import nirtorch
import torch
import snntorch as snn


def _create_rnn_subgraph(graph: nir.NIRGraph, lif_nk: str, w_nk: str) -> nir.NIRGraph:
    """Take a NIRGraph plus the node keys for a LIF and a W_rec, and return a new NIRGraph
    which has the RNN subgraph replaced with a subgraph (i.e., a single NIRGraph node).
    """
    # NOTE: assuming that the LIF and W_rec have keys of form xyz.abc
    sg_key = lif_nk.split('.')[0]  # TODO: make this more general?

    # create subgraph for RNN
    sg_edges = [
        (lif_nk, w_nk), (w_nk, lif_nk), (lif_nk, f'{sg_key}.output'), (f'{sg_key}.input', w_nk)
    ]
    sg_nodes = {
        lif_nk: graph.nodes[lif_nk],
        w_nk: graph.nodes[w_nk],
        f'{sg_key}.input': nir.Input(graph.nodes[lif_nk].input_type),
        f'{sg_key}.output': nir.Output(graph.nodes[lif_nk].output_type),
    }
    sg = nir.NIRGraph(nodes=sg_nodes, edges=sg_edges)

    # remove subgraph edges from graph
    graph.edges = [e for e in graph.edges if e not in [(lif_nk, w_nk), (w_nk, lif_nk)]]
    # remove subgraph nodes from graph
    graph.nodes = {k: v for k, v in graph.nodes.items() if k not in [lif_nk, w_nk]}

    # change edges of type (x, lif_nk) to (x, sg_key)
    graph.edges = [(e[0], sg_key) if e[1] == lif_nk else e for e in graph.edges]
    # change edges of type (lif_nk, x) to (sg_key, x)
    graph.edges = [(sg_key, e[1]) if e[0] == lif_nk else e for e in graph.edges]

    # insert subgraph into graph and return
    graph.nodes[sg_key] = sg
    return graph


def _replace_rnn_subgraph_with_nirgraph(graph: nir.NIRGraph) -> nir.NIRGraph:
    """Take a NIRGraph and replace any RNN subgraphs with a single NIRGraph node."""
    print('replace rnn subgraph with nirgraph')

    if len(set(graph.edges)) != len(graph.edges):
        print('[WARNING] duplicate edges found, removing')
        graph.edges = list(set(graph.edges))

    # find cycle of LIF <> Dense nodes
    for edge1 in graph.edges:
        for edge2 in graph.edges:
            if not edge1 == edge2:
                if edge1[0] == edge2[1] and edge1[1] == edge2[0]:
                    lif_nk = edge1[0]
                    lif_n = graph.nodes[lif_nk]
                    w_nk = edge1[1]
                    w_n = graph.nodes[w_nk]
                    is_lif = isinstance(lif_n, (nir.LIF, nir.CubaLIF))
                    is_dense = isinstance(w_n, (nir.Affine, nir.Linear))
                    # check if the dense only connects to the LIF
                    w_out_nk = [e[1] for e in graph.edges if e[0] == w_nk]
                    w_in_nk = [e[0] for e in graph.edges if e[1] == w_nk]
                    is_rnn = len(w_out_nk) == 1 and len(w_in_nk) == 1
                    # check if we found an RNN - if so, then parse it
                    if is_rnn and is_lif and is_dense:
                        graph = _create_rnn_subgraph(graph, edge1[0], edge1[1])
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
    lif_size = list(input_node.input_type.values())[0][0]
    assert lif_size == list(output_node.output_type.values())[0][0], 'output size mismatch'
    assert lif_size == lif_node.v_threshold.size, 'lif size mismatch (v_threshold)'
    assert lif_size == wrec_node.weight.shape[0], 'w_rec shape mismatch'
    assert lif_size == wrec_node.weight.shape[1], 'w_rec shape mismatch'

    return lif_node, wrec_node, lif_size


def _nir_to_snntorch_module(
        node: nir.NIRNode, hack_w_scale=True, init_hidden=False
) -> torch.nn.Module:
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

    elif isinstance(node, nir.Conv2d):
        mod = torch.nn.Conv2d(
            node.weight.shape[1],
            node.weight.shape[0],
            kernel_size=[*node.weight.shape[-2:]],
            stride=node.stride,
            padding=node.padding,
            dilation=node.dilation,
            groups=node.groups,
        )
        mod.bias.data = torch.Tensor(node.bias)
        mod.weight.data = torch.Tensor(node.weight)
        return mod

    if isinstance(node, nir.Flatten):
        return torch.nn.Flatten(node.start_dim, node.end_dim)

    if isinstance(node, nir.SumPool2d):
        return torch.nn.AvgPool2d(
            kernel_size=tuple(node.kernel_size),
            stride=tuple(node.stride),
            padding=tuple(node.padding),
            divisor_override=1,  # turn AvgPool into SumPool
        )

    elif isinstance(node, nir.IF):
        assert np.unique(node.v_threshold).size == 1, 'v_threshold must be same for all neurons'
        assert np.unique(node.r).size == 1, 'r must be same for all neurons'
        vthr = np.unique(node.v_threshold)[0]
        r = np.unique(node.r)[0]
        assert r == 1, 'r != 1 not supported'
        mod = snn.Leaky(
            beta=0.9,
            threshold=vthr * r,
            init_hidden=False,
            reset_delay=False,
        )
        return mod

    elif isinstance(node, nir.LIF):
        dt = 1e-4

        assert np.allclose(node.v_leak, 0.), 'v_leak not supported'
        assert np.unique(node.v_threshold).size == 1, 'v_threshold must be same for all neurons'

        beta = 1 - (dt / node.tau)
        vthr = node.v_threshold
        w_scale = node.r * dt / node.tau

        if not np.allclose(w_scale, 1.):
            if hack_w_scale:
                vthr = vthr / np.unique(w_scale)[0]
                print('[warning] scaling weights to avoid scaling inputs')
                print(f'w_scale: {w_scale}, r: {node.r}, dt: {dt}, tau: {node.tau}')
            else:
                raise NotImplementedError('w_scale must be 1, or the same for all neurons')

        assert np.unique(vthr).size == 1, 'LIF v_thr must be same for all neurons'

        return snn.Leaky(
            beta=beta,
            threshold=np.unique(vthr)[0],
            reset_mechanism='zero',
            init_hidden=init_hidden,
            reset_delay=False,
        )

    elif isinstance(node, nir.CubaLIF):
        dt = 1e-4

        assert np.allclose(node.v_leak, 0), 'v_leak not supported'
        assert np.allclose(node.r, node.tau_mem / dt), 'r not supported in CubaLIF'

        alpha = 1 - (dt / node.tau_syn)
        beta = 1 - (dt / node.tau_mem)
        vthr = node.v_threshold
        w_scale = node.w_in * (dt / node.tau_syn)

        if not np.allclose(w_scale, 1.):
            if hack_w_scale:
                vthr = vthr / w_scale
                print('[warning] scaling weights to avoid scaling inputs')
                print(f'w_scale: {w_scale}, w_in: {node.w_in}, dt: {dt}, tau_syn: {node.tau_syn}')
            else:
                raise NotImplementedError('w_scale must be 1, or the same for all neurons')

        assert np.unique(vthr).size == 1, 'CubaLIF v_thr must be same for all neurons'

        if np.unique(alpha).size == 1:
            alpha = float(np.unique(alpha)[0])
        if np.unique(beta).size == 1:
            beta = float(np.unique(beta)[0])

        return snn.Synaptic(
            alpha=alpha,
            beta=beta,
            threshold=float(np.unique(vthr)[0]),
            reset_mechanism='zero',
            init_hidden=init_hidden,
            reset_delay=False,
        )

    elif isinstance(node, nir.NIRGraph):
        lif_node, wrec_node, lif_size = _parse_rnn_subgraph(node)

        if isinstance(lif_node, nir.LIF):
            raise NotImplementedError('LIF in subgraph not supported')

        elif isinstance(lif_node, nir.CubaLIF):
            dt = 1e-4

            assert np.allclose(lif_node.v_leak, 0), 'v_leak not supported'
            assert np.allclose(lif_node.r, lif_node.tau_mem / dt), 'r not supported in CubaLIF'

            alpha = 1 - (dt / lif_node.tau_syn)
            beta = 1 - (dt / lif_node.tau_mem)
            vthr = lif_node.v_threshold
            w_scale = lif_node.w_in * (dt / lif_node.tau_syn)

            if not np.allclose(w_scale, 1.):
                if hack_w_scale:
                    vthr = vthr / w_scale
                    print(f'[warning] scaling weights to avoid scaling inputs. w_scale: {w_scale}')
                    print(f'w_in: {lif_node.w_in}, dt: {dt}, tau_syn: {lif_node.tau_syn}')
                else:
                    raise NotImplementedError('w_scale must be 1, or the same for all neurons')

            assert np.unique(vthr).size == 1, 'CubaLIF v_thr must be same for all neurons'

            diagonal = np.array_equal(wrec_node.weight, np.diag(np.diag(wrec_node.weight)))

            if np.unique(alpha).size == 1:
                alpha = float(np.unique(alpha)[0])
            if np.unique(beta).size == 1:
                beta = float(np.unique(beta)[0])

            if diagonal:
                V = torch.from_numpy(np.diag(wrec_node.weight)).to(dtype=torch.float32)
            else:
                V = None

            rsynaptic = snn.RSynaptic(
                alpha=alpha,
                beta=beta,
                threshold=float(np.unique(vthr)[0]),
                reset_mechanism='zero',
                init_hidden=init_hidden,
                all_to_all=not diagonal,
                linear_features=lif_size,
                V=V,
                reset_delay=False,
            )

            rsynaptic.recurrent.weight.data = torch.Tensor(wrec_node.weight)
            if isinstance(wrec_node, nir.Affine):
                rsynaptic.recurrent.bias.data = torch.Tensor(wrec_node.bias)
            else:
                rsynaptic.recurrent.bias.data = torch.zeros_like(rsynaptic.recurrent.bias)
            return rsynaptic

    else:
        print('[WARNING] could not parse node of type:', node.__class__.__name__)

    return None


def from_nir(graph: nir.NIRGraph) -> torch.nn.Module:
    # find valid RNN subgraphs, and replace them with a single NIRGraph node
    graph = _replace_rnn_subgraph_with_nirgraph(graph)
    # TODO: right now, the subgraph edges seem to not be parsed correctly - fix this
    return nirtorch.load(graph, _nir_to_snntorch_module)
