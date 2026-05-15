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


def _load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assoc_lm_shapes_and_forward():
    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(
        os.path.join(this_dir, "..", "6_associative_llm.py")
    )
    mod = _load_module_from_path("assoc_llm_module", mod_path)

    vocab_size = 101
    T, B = 8, 2
    tokens = torch.randint(0, vocab_size, (T, B))

    model = mod.SNNAssociativeLanguageModel(
        vocab_size=vocab_size, hidden_dim=64, n_layers=2
    )
    logits = model(tokens)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (T, B, vocab_size)
    assert torch.isfinite(logits).all()


def test_spiking_assoc_block_respects_input_shape():
    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(
        os.path.join(this_dir, "..", "6_associative_llm.py")
    )
    mod = _load_module_from_path("assoc_llm_module", mod_path)

    H = 36
    block = mod.SpikingAssocBlock(hidden_dim=H)
    T, B = 5, 3
    x = torch.randn(T, B, H)
    y = block(x)
    assert y.shape == (T, B, H)
    assert torch.isfinite(y).all()


def test_position_embedding_broadcasting_assoc():
    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(
        os.path.join(this_dir, "..", "6_associative_llm.py")
    )
    mod = _load_module_from_path("assoc_llm_module", mod_path)

    vocab_size = 50
    model = mod.SNNAssociativeLanguageModel(
        vocab_size=vocab_size, hidden_dim=16, n_layers=1
    )
    T, B1, B2 = 6, 1, 4
    tokens_b1 = torch.randint(0, vocab_size, (T, B1))
    tokens_b2 = torch.randint(0, vocab_size, (T, B2))

    logits_b1 = model(tokens_b1)
    logits_b2 = model(tokens_b2)

    assert logits_b1.shape == (T, B1, vocab_size)
    assert logits_b2.shape == (T, B2, vocab_size)


if __name__ == "__main__":
    test_assoc_lm_shapes_and_forward()
    print("test_assoc_lm_shapes_and_forward passed")
    test_spiking_assoc_block_respects_input_shape()
    print("test_spiking_assoc_block_respects_input_shape passed")
    test_position_embedding_broadcasting_assoc()
    print("test_position_embedding_broadcasting_assoc passed")
