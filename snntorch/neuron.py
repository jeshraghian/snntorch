from collections import namedtuple
import torch.nn as nn
import torch

# The following code snippet was by Jacques Kaiser, Hesham Mostafa and Emre Neftci
# "Synaptic Plasticity Dynamics for Deep Continuous Local Learning (DECOLLE)"


# class LIF(nn.Module):
#     NeuronState = namedtuple('NeuronState', ['U', 'I', 'S'])
#
#     def __init__(self, in_features, out_features, bias=True, alpha=.9, beta=.85):
#         super(LIF, self).__init__()
#         self.fc_layer = nn.Linear(in_features, out_features)
#         self.in_channels = in_features
#         self.out_channels = out_features
#         self.alpha = alpha
#         self.beta = beta
#         self.state = state = self.NeuronState(U=torch.zeros(1, out_features),
#                                               I=torch.zeros(1, out_features),
#                                               S=torch.zeros(1, out_features))
#         self.fc_layer.weight.data.uniform_(-.3, .3)
#         self.fc_layer.bias.data.uniform_(-.01, .01)
#
#     def forward(self, Sin_t):
#         state = self.state
#         U = alpha * state.U + state.I - state.S
#         I = beta * state.I + self.fc_layer(Sin_t)
#         # update the neuronal state
#         S = (U > 0).float()
#         self.state = NeuronState(U=U, I=I, S=S)
#         return self.state
