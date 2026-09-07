import torch
import snntorch as snn


def test_leaky_torch_save_roundtrip(tmp_path):
    net = snn.Leaky(beta=0.9)
    path = tmp_path / "model.pt"
    torch.save(net, path)
    net2 = torch.load(path, weights_only=False)
    assert isinstance(net2, snn.Leaky)
