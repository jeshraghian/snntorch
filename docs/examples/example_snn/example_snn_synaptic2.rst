=========================================================================
Building Networks with Instance Variables: Synaptic Conductance-based LIF Neuron
=========================================================================

Building a fully-connected network using a Synaptic Conductance-based neuron model.
Using instance variables are only required when calling the built-in backprop methods in `snntorch.backprop`.

Example::

      import torch
      import torch.nn as nn
      import snntorch as snn

      alpha = 0.9
      beta = 0.85

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
            snn.LIF.clear_instances() # boilerplate
            self.fc1 = nn.Linear(num_inputs, num_hidden)
            self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, num_inputs=num_hidden, batch_size=batch_size, hidden_init=True)
            self.fc2 = nn.Linear(num_hidden, num_outputs)
            self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, num_inputs=num_outputs, batch_size=batch_size, hidden_init=True)

      # move the time-loop into the training-loop
      def forward(self, x):
            cur1 = self.fc1(x)
            self.lif1.spk1, self.lif1.syn1, self.lif1.mem1 = self.lif1(cur1, self.lif1.syn, self.lif1.mem)
            cur2 = self.fc2(self.lif1.spk)
            self.lif2.spk, self.lif2.syn, self.lif2.mem = self.lif2(cur2, self.lif2.syn, self.lif2.mem)

            return self.lif2.spk, self.lif2.mem


      net = Net().to(device)

      for step in range(num_steps):
            spk_out, mem_out = net(data.view(batch_size, -1))
