import numpy as np
import nir
import nirtorch
import torch
import snntorch as snn


def _nir_to_snntorch_module(node: nir.NIRNode) -> torch.nn.Module:
    if isinstance(node, nir.Input) or isinstance(node, nir.Output):
        return None

    elif isinstance(node, nir.Affine):
        mod = torch.nn.Linear(node.weight.shape[1], node.weight.shape[0])
        mod.weight.data = torch.Tensor(node.weight)
        if node.bias is not None:
            mod.bias.data = torch.Tensor(node.bias)
        return mod

    elif isinstance(node, nir.Linear):
        mod = torch.nn.Linear(node.weight.shape[1], node.weight.shape[0], bias=False)
        mod.weight.data = torch.Tensor(node.weight)
        return mod

    elif isinstance(node, nir.LIF):
        # NOTE: assuming that parameters are arrays of correct size
        dt = 1e-4
        assert np.unique(node.v_threshold).size == 1, 'v_threshold must be same for all neurons'
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
            # init_hidden=False,
        )

    else:
        print(node.__class__.__name__, node)

    return None


def from_nir(graph: nir.NIRGraph) -> torch.nn.Module:
    return nirtorch.load(graph, _nir_to_snntorch_module)
