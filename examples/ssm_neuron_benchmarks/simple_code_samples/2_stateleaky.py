import torch
from snntorch._neurons.stateleaky import StateLeaky


def run_stateleaky(layer, x):
    spk, mem = layer(x)  # full sequence in one go
    return spk, mem


if __name__ == "__main__":
    B, T, C = 2, 5, 10
    x = torch.zeros(B, T, C)
    stateleaky = StateLeaky(beta=0.9, channels=C, output=True)
    spk, mem = run_stateleaky(stateleaky, x)
    print("stateleaky demo:", spk.shape, mem.shape)
