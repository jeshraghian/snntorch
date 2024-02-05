from typing import Optional
import torch
import nir
import numpy as np
import nirtorch
import snntorch as snn


def _extract_snntorch_module(module: torch.nn.Module) -> Optional[nir.NIRNode]:
    """Convert a single snnTorch module to the equivalent object in the Neuromorphic
    Intermediate Representation (NIR). This function is used internally by the export_to_nir
    function to convert each submodule/layer of the network to the NIR.

    Currently supported snnTorch modules: Leaky, Linear, Synaptic, RLeaky, RSynaptic.

    Note that recurrent layers such as RLeaky and RSynaptic will be converted to a NIR graph,
    which will then be embedded as a subgraph into the main NIR graph.

    :param module: snnTorch module
    :type module: torch.nn.Module

    :return: return the NIR node
    :rtype: Optional[nir.NIRNode]
    """
    if isinstance(module, snn.Leaky):
        dt = 1e-4

        beta = module.beta.detach().numpy()
        vthr = module.threshold.detach().numpy()
        tau_mem = dt / (1 - beta)
        r = tau_mem / dt
        v_leak = np.zeros_like(beta)

        return nir.LIF(
            tau=tau_mem,
            v_threshold=vthr,
            v_leak=v_leak,
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

        # TODO: assert that size of the current layer is correct
        alpha = module.alpha.detach().numpy()
        beta = module.beta.detach().numpy()
        vthr = module.threshold.detach().numpy()

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
        # TODO(stevenabreu7): implement RLeaky
        raise NotImplementedError('RLeaky not supported')

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


def export_to_nir(
    module: torch.nn.Module, sample_data: torch.Tensor, model_name: str = "snntorch",
    model_fwd_args=[], ignore_dims=[]
) -> nir.NIRNode:
    """Convert an snnTorch module to the Neuromorphic Intermediate Representation (NIR).
    This function uses nirtorch to extract the computational graph of the torch module,
    and the _extract_snntorch_module method is used to convert each module in the graph
    to the corresponding NIR module.

    The NIR is a graph-based representation of a spiking neural network, which can be used to
    port the network to different neuromorphic hardware and software platforms.

    Missing features:
    - RLeaky

    Example::

        import snntorch as snn
        import torch
        from snntorch import export_to_nir

        lif1 = snn.Leaky(beta=0.9, init_hidden=True)
        lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

        net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(784, 500),
            lif1,
            torch.nn.Linear(500, 10),
            lif2
        )

        sample_data = torch.randn(1, 784)
        nir_graph = export_to_nir(net, sample_data)
    
    :param module: Network model (either wrapped in Sequential container or as a class)
    :type module: torch.nn.Module

    :param sample_data: Sample input data to the network
    :type sample_data: torch.Tensor

    :param model_name: Name of the model
    :type model_name: str, optional

    :param model_fwd_args: Arguments to pass to the forward function of the model
    :type model_fwd_args: list, optional

    :param ignore_dims: List of dimensions to ignore when extracting the NIR
    :type ignore_dims: list, optional

    :return: return the NIR graph
    :rtype: nir.NIRNode
    """
    nir_graph = nirtorch.extract_nir_graph(
        module, _extract_snntorch_module, sample_data, model_name=model_name,
        ignore_submodules_of=[snn.RLeaky, snn.RSynaptic],
        model_fwd_args=model_fwd_args, ignore_dims=ignore_dims
    )
    return nir_graph
