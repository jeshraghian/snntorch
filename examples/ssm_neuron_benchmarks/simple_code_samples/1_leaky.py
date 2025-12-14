import torch
import snntorch as snn


# x: (batch, time, features)
def run_leaky(layer, x):
    mem = torch.zeros(x.shape[0], x.shape[-1], device=x.device)
    spk_seq, mem_seq = [], []

    for t in range(x.shape[1]):
        spk, mem = layer(x[:, t], mem)
        spk_seq.append(spk)
        mem_seq.append(mem)

    spk_seq = torch.stack(spk_seq, dim=1)  # (B, T, C)
    mem_seq = torch.stack(mem_seq, dim=1)  # (B, T, C)
    return spk_seq, mem_seq


if __name__ == "__main__":
    lif = snn.Leaky(beta=0.9, threshold=1.0)
    B, T, C = 2, 4, 3
    x = torch.zeros(B, T, C)
    spk, mem = run_leaky(lif, x)
    print("leaky demo:", spk.shape, mem.shape)
