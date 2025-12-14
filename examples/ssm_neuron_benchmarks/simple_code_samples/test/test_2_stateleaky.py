import os
import sys
import importlib.util
import torch


# Ensure project root on sys.path when running directly
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from snntorch._neurons.stateleaky import StateLeaky


def _load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stateleaky_zero_input():
    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(os.path.join(this_dir, "..", "2_stateleaky.py"))
    mod = _load_module_from_path("stateleaky_module", mod_path)

    B, T, C = 2, 3, 10
    x = torch.zeros(B, T, C)

    layer = StateLeaky(beta=0.9, channels=C, output=True)
    spk_seq, mem_seq = mod.run_stateleaky(layer, x)

    assert isinstance(spk_seq, torch.Tensor)
    assert isinstance(mem_seq, torch.Tensor)
    assert spk_seq.shape == (B, T, C)
    assert mem_seq.shape == (B, T, C)
    assert torch.all(spk_seq == 0), "Spikes should be zero for zero input"
    assert torch.allclose(
        mem_seq, torch.zeros_like(mem_seq)
    ), "Membrane should remain zero for zero input"


def test_stateleaky_single_pulse_spikes_first_step():
    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(os.path.join(this_dir, "..", "2_stateleaky.py"))
    mod = _load_module_from_path("stateleaky_module", mod_path)

    B, T, C = 2, 4, 10
    x = torch.zeros(B, T, C)
    x[:, 0, :] = 2.0  # strong pulse at t=0

    layer = StateLeaky(beta=0.9, channels=C, output=True)
    spk_seq, mem_seq = mod.run_stateleaky(layer, x)

    assert spk_seq.shape == (B, T, C)
    assert mem_seq.shape == (B, T, C)
    assert torch.all(
        spk_seq[:, 0, :] == 1
    ), "All units should spike at t=0 for strong pulse"
    if T > 1:
        assert torch.all(
            spk_seq[:, 1:, :] == 0
        ), "No spikes after t=0 without further input"


if __name__ == "__main__":
    test_stateleaky_zero_input()
    print("test_stateleaky_zero_input passed")
    test_stateleaky_single_pulse_spikes_first_step()
    print("test_stateleaky_single_pulse_spikes_first_step passed")
