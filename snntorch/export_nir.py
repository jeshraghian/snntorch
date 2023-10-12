from typing import Union, Optional

import torch
import nir
import numpy as np
from nirtorch import extract_nir_graph

from snntorch import Leaky, Synaptic, RLeaky, RSynaptic


def _create_rnn_subgraph(
        module: torch.nn.Module, lif: Union[nir.LIF, nir.CubaLIF], n_neurons=-1
) -> nir.NIRGraph:
    """Create NIR Graph for RNN, from the snnTorch module and the extracted LIF/CubaLIF node."""
    b = None
    if module.all_to_all:
        lif_shape = module.recurrent.weight.shape[0]
        w_rec = module.recurrent.weight.data.detach().numpy()
        if module.recurrent.bias is not None:
            b = module.recurrent.bias.data.detach().numpy()
    else:
        if len(module.recurrent.V.shape) == 0 and n_neurons == -1:
            lif_shape = None
            w_rec = np.eye(1) * module.recurrent.V.data.detach().numpy()
        elif n_neurons != -1:
            lif_shape = n_neurons
            w_rec = np.eye(n_neurons) * module.recurrent.V.data.detach().numpy()
        else:
            lif_shape = module.recurrent.V.shape[0]
            w_rec = np.diag(module.recurrent.V.data.detach().numpy())

    return nir.NIRGraph(
        nodes={
            'input': nir.Input(input_type=[lif_shape]),
            'lif': lif,
            'w_rec': nir.Linear(weight=w_rec) if b is None else nir.Affine(weight=w_rec, bias=b),
            'output': nir.Output(output_type=[lif_shape])
        },
        edges=[('input', 'lif'), ('lif', 'w_rec'), ('w_rec', 'lif'), ('lif', 'output')]
    )


def _get_neuron_count(module: torch.nn.Module) -> int:
    if isinstance(module, RLeaky) or isinstance(module, RSynaptic):
        if module.all_to_all:
            return module.linear_features
        elif isinstance(module.recurrent.V, torch.Tensor) and len(module.recurrent.V.shape) > 0:
            return module.recurrent.V.shape[0]
        elif module.init_hidden is True:
            return module.mem.shape[0]
        else:
            # not implemented
            return -1
    else:
        # not implemented
        return -1


# eqn is assumed to be: v_t+1 = (1-1/tau)*v_t + 1/tau * v_leak + I_in / C
def _extract_snntorch_module(module: torch.nn.Module) -> Optional[nir.NIRNode]:
    """
    NOTE: it might leave the NIR node of neurons with an incompatible shape. This must be fixed.
    """
    if isinstance(module, Leaky):
        # TODO
        tau = 1 / (1 - module.beta.detach().numpy())
        r = module.beta.detach().numpy()
        threshold = module.threshold.detach().numpy()
        return nir.LIF(
            tau=tau,
            v_threshold=threshold,
            v_leak=torch.zeros_like(tau),
            r=r,
        )

    elif isinstance(module, RSynaptic):
        alpha = module.alpha.detach().numpy()
        beta = module.beta.detach().numpy()
        threshold = module.threshold.detach().numpy()
        n_neurons = _get_neuron_count(module)
        if len(alpha.shape) == 0 and n_neurons != -1:
            alpha = np.ones(n_neurons) * alpha
        if len(beta.shape) == 0 and n_neurons != -1:
            beta = np.ones(n_neurons) * beta
        if len(threshold.shape) == 0 and n_neurons != -1:
            threshold = np.ones(n_neurons) * threshold
        lif = nir.CubaLIF(
            tau_syn=1 / (1 - beta),
            tau_mem=1 / (1 - alpha),
            v_threshold=threshold,
            v_leak=np.zeros_like(beta),
            r=beta,
        )
        return _create_rnn_subgraph(module, lif, n_neurons=n_neurons)

    elif isinstance(module, RLeaky):
        beta = module.beta.detach().numpy()
        threshold = module.threshold.detach().numpy()
        n_neurons = _get_neuron_count(module)
        if len(beta.shape) == 0 and n_neurons != -1:
            beta = np.ones(n_neurons) * beta
        if len(threshold.shape) == 0 and n_neurons != -1:
            threshold = np.ones(n_neurons) * threshold
        lif = nir.LIF(
            tau=1 / (1 - beta),
            v_threshold=threshold,
            v_leak=np.zeros_like(beta),
            r=beta,
        )
        return _create_rnn_subgraph(module, lif, n_neurons=n_neurons)

    elif isinstance(module, Synaptic):
        # TODO
        return nir.CubaLIF(
            tau_syn=1 / (1 - module.alpha).detach().numpy(),
            tau_mem=1 / (1 - module.beta).detach().numpy(),
            v_threshold=module.threshold.detach().numpy(),
            v_leak=np.zeros_like(module.beta),
            r=module.beta.detach().numpy(),  # NOTE: is this right?
        )

    elif isinstance(module, torch.nn.Linear):
        if module.bias is None:  # Add zero bias if none is present
            return nir.Affine(
                module.weight.detach().numpy(), np.zeros(*module.weight.shape[:-1])
            )
        else:
            return nir.Affine(module.weight.detach().numpy(), module.bias.detach().numpy())

    else:
        print(f'[WARNING] unknown module type: {type(module).__name__} (ignored)')
        return None


def to_nir(
    module: torch.nn.Module, sample_data: torch.Tensor, model_name: str = "snntorch"
) -> nir.NIRNode:
    """Convert an snnTorch model to the Neuromorphic Intermediate Representation (NIR).

    Example::

        import torch, torch.nn as nn
        import snntorch as snn
        from snntorch import export

        data_path = "untrained-snntorch.pt"

        net = nn.Sequential(nn.Linear(784, 128),
                            snn.Leaky(beta=0.8, init_hidden=True),
                            nn.Linear(128, 10),
                            snn.Leaky(beta=0.8, init_hidden=True, output=True))

        # save model in pt format
        torch.save(net.state_dict(), data_path)

        # load model (does nothing here, but shown for completeness)
        net.load_state_dict(torch.load(data_path))

        # generate input tensor to dynamically construct graph
        x = torch.zeros(784)

        # generate NIR graph
        nir_net = export.to_nir(net, x)


    :param module: a saved snnTorch model as a parameter dictionary
    :type module: torch.nn.Module

    :param sample_data: sample input data to the model
    :type sample_data: torch.Tensor

    :param model_name: name of library used to train model, default: "snntorch"
    :type model_name: str, optional

    :return: NIR computational graph where torch modules are represented as NIR nodes
    :rtype: NIRGraph

    """
    nir_graph = extract_nir_graph(
        module, _extract_snntorch_module, sample_data, model_name=model_name,
        ignore_submodules_of=[RLeaky, RSynaptic]
    )

    # NOTE: hack to define subgraph I/O types
    for node_key, node in nir_graph.nodes.items():
        inp_type = node.input_type.get('input', [None])
        input_undef = len(inp_type) == 0 or inp_type[0] is None
        if isinstance(node, nir.Input) and input_undef and '.' in node_key:
            print('WARNING: subgraph input type not set, inferring from previous node')
            key = '.'.join(node_key.split('.')[:-1])
            prev_keys = [edge[0] for edge in nir_graph.edges if edge[1] == key]
            assert len(prev_keys) == 1, 'multiple previous nodes not supported'
            prev_node = nir_graph.nodes[prev_keys[0]]
            cur_type = prev_node.output_type['output']
            node.input_type['input'] = cur_type
            nir_graph.nodes[f'{key}.output'].output_type['output'] = cur_type

    # NOTE: hack to remove recurrent connections of subgraph to itself
    for edge in nir_graph.edges:
        if edge[0] not in nir_graph.nodes and edge[1] not in nir_graph.nodes:
            nir_graph.edges.remove(edge)

    return nir_graph
