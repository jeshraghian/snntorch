from typing import Union, Optional
from numbers import Number

import torch
import nir
from nirtorch import extract_nir_graph

from snntorch import Leaky

# eqn is assumed to be: v_t+1 = (1-1/tau)*v_t + 1/tau * v_leak + I_in / C 
def _extract_snntorch_module(module:torch.nn.Module) -> Optional[nir.NIRNode]:
    if isinstance(module, Leaky):
        return nir.LIF(
            tau = -1 / (module.beta + 1).detach(),
            v_threshold = module.threshold.detach(),
            v_leak = torch.zeros_like(module.beta),
            r = -1 / (module.beta + 1).detach(),
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
    return extract_nir_graph(
        module, _extract_snntorch_module, sample_data, model_name=model_name
    )