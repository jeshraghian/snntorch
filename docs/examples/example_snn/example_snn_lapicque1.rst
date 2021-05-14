==================================================================
Building Networks: Lapicque's Neuron 
==================================================================

Building a fully-connected network using Lapicque's neuron model.

Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        beta = 0.5

        R = 1
        C = 1.44

        batch_size = 128
      
        num_inputs = 784
        num_hidden = 1000
        num_outputs = 10

        num_steps = 100

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                # initialize layers
                self.fc1 = nn.Linear(num_inputs, num_hidden)
                self.lif1 = snn.Lapicque(beta=beta)
                self.fc2 = nn.Linear(num_hidden, num_outputs)
                self.lif2 = snn.Lapicque(R=R, C=C)  # lif1 and lif2 are approximately equivalent

            def forward(self, x, mem1, spk1, mem2):
                for step in range(num_steps):
                    cur1 = self.fc1(x)
                    spk1, mem1 = self.lif1(cur1, mem1)
                    cur2 = self.fc2(spk1)
                    spk2, mem2 = self.lif2(cur2, mem2)

                    spk2_rec.append(spk2)
                    mem2_rec.append(mem2)

                return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

        net = Net().to(device)
        output, mem_rec = net(data.view(batch_size, -1))
