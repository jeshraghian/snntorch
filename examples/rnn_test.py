import torch
import snntorch as snn

device = 'cuda'
num_outputs = 5
beta1 = torch.rand((num_outputs), dtype = torch.float) # independent decay rate for each leaky neuron in layer 2: [0, 1)

lif1 = snn.LeakyParallel(2, 4, beta=0.9, learn_beta=True, device=device)
# lif2 = snn.LeakyParallel(20, 30, device=device)
# lif3 = snn.LeakyParallel(30, num_outputs, beta=beta1, learn_beta=False, device=device)

x = torch.rand(5, 1, 2, device=device)

x = lif1(x)
print(x)
# x = lif2(x)
# x = lif3(x)

print(x.size())