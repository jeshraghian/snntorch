import os
import importlib.util
import torch
import sys

# Ensure project root on sys.path so that 'snntorch' can be imported when running directly
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import snntorch as snn


def _load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_leaky_zero_input():
    this_dir = os.path.dirname(__file__)
    leaky_path = os.path.abspath(os.path.join(this_dir, "..", "1_leaky.py"))
    leaky_mod = _load_module_from_path("leaky_module", leaky_path)

    # Instantiate the layer locally
    layer = snn.Leaky(beta=0.9, threshold=1.0)
    run_leaky = leaky_mod.run_leaky

    # Zero input should yield zero spikes and zero membrane across time
    time_steps, batch, channels = 3, 2, 4
    x = torch.zeros(time_steps, batch, channels)

    spk_seq, mem_seq = run_leaky(layer, x)

    assert isinstance(spk_seq, torch.Tensor)
    assert isinstance(mem_seq, torch.Tensor)
    assert spk_seq.shape == (time_steps, batch, channels)
    assert mem_seq.shape == (time_steps, batch, channels)
    assert torch.all(spk_seq == 0), "Spikes should be all zeros for zero input"
    assert torch.allclose(
        mem_seq, torch.zeros_like(mem_seq)
    ), "Membrane should remain zero for zero input"


def test_run_leaky_single_pulse_spikes_first_step():
    this_dir = os.path.dirname(__file__)
    leaky_path = os.path.abspath(os.path.join(this_dir, "..", "1_leaky.py"))
    leaky_mod = _load_module_from_path("leaky_module", leaky_path)

    layer = snn.Leaky(beta=0.9, threshold=1.0)
    run_leaky = leaky_mod.run_leaky

    time_steps, batch, channels = 4, 2, 3
    x = torch.zeros(time_steps, batch, channels)
    # Single strong pulse at t=0 that exceeds threshold (threshold=1.0 in leaky.py)
    x[0, :, :] = 2.0

    spk_seq, mem_seq = run_leaky(layer, x)

    assert spk_seq.shape == (time_steps, batch, channels)
    assert mem_seq.shape == (time_steps, batch, channels)

    # Expect spikes on the first step for all units
    assert torch.all(
        spk_seq[0, :, :] == 1
    ), "All units should spike at t=0 for strong pulse"
    # No further spikes after the initial pulse (residual should decay below threshold)
    if time_steps > 1:
        assert torch.all(
            spk_seq[1:, :, :] == 0
        ), "No spikes expected after t=0 without additional input"


if __name__ == "__main__":
    test_run_leaky_zero_input()
    print("test_run_leaky_zero_input passed")

    test_run_leaky_single_pulse_spikes_first_step()
    print("test_run_leaky_single_pulse_spikes_first_step passed")
