from typing import Optional
import torch
import nir
import numpy as np
import nirtorch
import snntorch as snn


def _extract_snntorch_module(module: torch.nn.Module) -> Optional[nir.NIRNode]:
    if isinstance(module, snn.Leaky):
        # TODO
        raise NotImplementedError('Leaky not supported')
        beta = module.beta.detach().numpy()
        vthr = module.threshold.detach().numpy()

        tau = 1 / (1 - module.beta.detach().numpy())
        r = module.beta.detach().numpy()
        threshold = module.threshold.detach().numpy()
        return nir.LIF(
            tau=tau,
            v_threshold=threshold,
            v_leak=torch.zeros_like(tau),
            r=r,
        )

    elif isinstance(module, torch.nn.Linear):
        if module.bias is None:
            return nir.Linear(
                weight=module.weight.data.detach().numpy()
            )
        else:
            return nir.Affine(
                weight=module.weight.data.detach().numpy(),
                bias=module.bias.data.detach().numpy()
            )

    elif isinstance(module, snn.Synaptic):
        dt = 1e-4

        alpha = module.alpha.detach().numpy()
        beta = module.beta.detach().numpy()
        vthr = module.threshold.detach().numpy()

        # TODO: make sure alpha, beta, vthr are tensors of same size
        alpha = np.ones(7) * alpha
        beta = np.ones(7) * beta
        vthr = np.ones(7) * vthr

        tau_syn = dt / (1 - alpha)
        tau_mem = dt / (1 - beta)
        r = tau_mem / dt
        v_leak = np.zeros_like(beta)
        w_in = tau_syn / dt

        return nir.CubaLIF(
            tau_syn=tau_syn,
            tau_mem=tau_mem,
            v_threshold=vthr,
            v_leak=v_leak,
            r=r,
            w_in=w_in,
        )

    elif isinstance(module, snn.RLeaky):
        raise NotImplementedError('RLeaky not supported')
        if module.all_to_all:
            w_rec = _extract_snntorch_module(module.recurrent)
            n_neurons = w_rec.weight.shape[0]
        else:
            if len(module.recurrent.V.shape) == 0:
                # TODO: handle this better - if V is a scalar, then the weight has wrong shape
                raise ValueError('V must be a vector, cannot infer layer size for scalar V')
            n_neurons = module.recurrent.V.shape[0]
            w = np.diag(module.recurrent.V.data.detach().numpy())
            w_rec = nir.Linear(weight=w)

        # TODO: set the parameters correctly
        v_thr = np.ones(n_neurons) * module.threshold.detach().numpy()
        beta = np.ones(n_neurons) * module.beta.detach().numpy()
        tau = 1 / (1 - beta)
        r = beta
        v_leak = beta

        return nir.NIRGraph(nodes={
            'input': nir.Input(input_type=[n_neurons]),
            'lif': nir.LIF(v_threshold=v_thr, tau=tau, r=r, v_leak=v_leak),
            'w_rec': w_rec,
            'output': nir.Output(output_type=[n_neurons])
        }, edges=[
            ('input', 'lif'), ('lif', 'w_rec'), ('w_rec', 'lif'), ('lif', 'output')
        ])

    elif isinstance(module, snn.RSynaptic):
        if module.all_to_all:
            w_rec = _extract_snntorch_module(module.recurrent)
            n_neurons = w_rec.weight.shape[0]
        else:
            if len(module.recurrent.V.shape) == 0:
                # TODO: handle this better - if V is a scalar, then the weight has wrong shape
                raise ValueError('V must be a vector, cannot infer layer size for scalar V')
            n_neurons = module.recurrent.V.shape[0]
            w = np.diag(module.recurrent.V.data.detach().numpy())
            w_rec = nir.Linear(weight=w)

        dt = 1e-4

        alpha = module.alpha.detach().numpy()
        beta = module.beta.detach().numpy()
        vthr = module.threshold.detach().numpy()
        alpha = np.ones(n_neurons) * alpha
        beta = np.ones(n_neurons) * beta
        vthr = np.ones(n_neurons) * vthr

        tau_syn = dt / (1 - alpha)
        tau_mem = dt / (1 - beta)
        r = tau_mem / dt
        v_leak = np.zeros_like(beta)
        w_in = tau_syn / dt

        return nir.NIRGraph(nodes={
            'input': nir.Input(input_type=[n_neurons]),
            'lif': nir.CubaLIF(
                v_threshold=vthr,
                tau_mem=tau_mem,
                tau_syn=tau_syn,
                r=r,
                v_leak=v_leak,
                w_in=w_in,
            ),
            'w_rec': w_rec,
            'output': nir.Output(output_type=[n_neurons])
        }, edges=[
            ('input', 'lif'), ('lif', 'w_rec'), ('w_rec', 'lif'), ('lif', 'output')
        ])

    else:
        print(f'[WARNING] module not implemented: {module.__class__.__name__}')
        return None


def to_nir(
    module: torch.nn.Module, sample_data: torch.Tensor, model_name: str = "snntorch",
    model_fwd_args=[], ignore_dims=[]
) -> nir.NIRNode:
    """Convert an snnTorch model to the Neuromorphic Intermediate Representation (NIR).
    """
    nir_graph = nirtorch.extract_nir_graph(
        module, _extract_snntorch_module, sample_data, model_name=model_name,
        ignore_submodules_of=[snn.RLeaky, snn.RSynaptic],
        model_fwd_args=model_fwd_args, ignore_dims=ignore_dims
    )
    return nir_graph
