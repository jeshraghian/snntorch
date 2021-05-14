==================================================================
Building Networks: 0th Order Spike Response Model
==================================================================

Building a fully-connected network using the 0th Order Spike Response Model.

Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        alpha = 0.9
        beta = 0.8

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
                self.lif1 = snn.SRM0(alpha=alpha, beta=beta)
                self.fc2 = nn.Linear(num_hidden, num_outputs)
                self.lif2 = snn.SRM0(alpha=alpha, beta=beta)

            def forward(self, x):
                for step in range(num_steps):
                    cur1 = self.fc1(x)
                    spk1, presyn1, postsyn1, mem1 = self.lif1(cur1, presyn1, postsyn1, mem1)
                    cur2 = self.fc2(spk1)
                    spk2, presyn2, postsyn2, mem2 = self.lif2(cur2, presyn2, postsyn2, mem2)
                    
                    spk2_rec.append(spk2)
                    mem2_rec.append(mem2)

                return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

        net = Net().to(device)
        output, mem_rec = net(data.view(batch_size, -1))
