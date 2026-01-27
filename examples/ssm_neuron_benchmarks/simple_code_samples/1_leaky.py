import torch
import snntorch as snn

# x: (T, B, C)
def run_leaky(layer, x):
    T, B, C = x.shape
    mem = torch.zeros(B, C, device=x.device)

    spk_seq, mem_seq = [], []
    for t in range(T):
        spk, mem = layer(x[t], mem)   # x[t]: (B, C)
        spk_seq.append(spk)
        mem_seq.append(mem)

    spk_seq = torch.stack(spk_seq, dim=0)  # (T, B, C)
    mem_seq = torch.stack(mem_seq, dim=0)  # (T, B, C)
    return spk_seq, mem_seq

if __name__ == "__main__":
    lif = snn.Leaky(beta=0.9, threshold=1.0)
    T, B, C = 4, 2, 3
    x = torch.zeros(T, B, C)
    spk, mem = run_leaky(lif, x)
    print("leaky demo:", spk.shape, mem.shape)