from typing import Union, Optional
from numbers import Number

import torch
import nir
from nirtorch import extract_nir_graph

from snntorch import Leaky, Synaptic

# eqn is assumed to be: v_t+1 = (1-1/tau)*v_t + 1/tau * v_leak + I_in / C
def _extract_snntorch_module(module:torch.nn.Module) -> Optional[nir.NIRNode]:
    if isinstance(module, Leaky):
        return nir.LIF(
            tau = 1 / (1 - module.beta).detach(),
            v_threshold = module.threshold.detach(),
            v_leak = torch.zeros_like(module.beta),
            r = module.beta.detach(),
        )

    if isinstance(module, Synaptic):
        return nir.CubaLIF(
            tau_syn = 1 / (1 - module.beta).detach(),
            tau_mem = 1 / (1 - module.alpha).detach(),
            v_threshold = module.threshold.detach(),
            v_leak = torch.zeros_like(module.beta),
            r = module.beta.detach(),
        )

    elif isinstance(module, torch.nn.Linear):
        if module.bias is None: # Add zero bias if none is present
            return nir.Affine(
                module.weight.detach(), torch.zeros(*module.weight.shape[:-1])
            )
        else:
            return nir.Affine(module.weight.detach(), module.bias.detach())

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
    return extract_nir_graph(
        module, _extract_snntorch_module, sample_data, model_name=model_name
    )
