"""Test for SpikingNeuron.zeros in-place fix (Issue #423)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import snntorch as snn
from snntorch._neurons.neurons import SpikingNeuron


def test_zeros_clears_hidden_state():
    """SpikingNeuron.zeros should clear hidden states to zero in-place."""
    lif = snn.Leaky(beta=0.5, init_hidden=True)
    _ = lif(torch.ones(2, 4))

    # Verify state is non-zero before clearing
    assert (lif.mem != 0).sum().item() > 0, "State should be non-zero before zeros()"

    mem_ptr_before = lif.mem.data_ptr()
    SpikingNeuron.zeros(lif.mem)

    # Verify state is all zeros after clearing
    assert torch.equal(lif.mem, torch.zeros_like(lif.mem)), \
        "SpikingNeuron.zeros() should clear the tensor to zeros"

    # Verify it's the same tensor object (in-place modification)
    assert lif.mem.data_ptr() == mem_ptr_before, \
        "zeros() should modify the tensor in-place, not create a new one"


def test_zeros_multiple_args():
    """SpikingNeuron.zeros should handle multiple tensor arguments."""
    t1 = torch.ones(3, 4)
    t2 = torch.ones(5, 6) * 2

    SpikingNeuron.zeros(t1, t2)

    assert torch.equal(t1, torch.zeros_like(t1)), "First tensor should be zeroed"
    assert torch.equal(t2, torch.zeros_like(t2)), "Second tensor should be zeroed"


if __name__ == "__main__":
    test_zeros_clears_hidden_state()
    test_zeros_multiple_args()
    print("All tests passed!")
