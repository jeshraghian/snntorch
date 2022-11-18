import torch.nn as nn

"""
    Batch Normalisation Through Time (BNTT) as presented in:
    'Revisiting Batch Normalization for Training Low-Latency Deep Spiking Neural Networks From Scratch'
    By Youngeun Kim & Priyadarshini Panda
    arXiv preprint arXiv:2010.01729
"""
def bntt1d(input_features, time_steps):
    bntt = nn.ModuleList([nn.BatchNorm1d(input_features, eps=1e-4, momentum=0.1, affine=True) for _ in range(time_steps)])

    # Disable bias/beta of Batch Norm
    for bn in bntt:
        bn.bias = None

    return bntt

def bntt2d(input_features, time_steps):
    bntt = nn.ModuleList([nn.BatchNorm2d(input_features, eps=1e-4, momentum=0.1, affine=True) for _ in range(time_steps)])

    # Disable bias/beta of Batch Norm
    for bn in bntt:
        bn.bias = None

    return bntt