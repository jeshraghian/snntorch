import torch
import torch.nn as nn
import snntorch as snn
import time

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.nn.functional as F


batch_size = 128
num_steps = 20
# data_path='/tmp/data/mnist'
data_path='C:\\Users\\jeshr\\Data\\mnist'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
# x = torch.rand(5, 1, 2, device=device)

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        num_inputs = 784 # number of inputs
        num_hidden = 1000 # number of hidden neurons
        num_outputs = 10 # number of classes (i.e., output neurons)

        beta1 = 0.95 # global decay rate for all leaky neurons in layer 1
        beta2 = torch.rand((num_outputs), dtype = torch.float) # independent decay rate for each leaky neuron in layer 2: [0, 1)

        # Initialize layers
        self.lif1 = snn.LeakyParallel(num_inputs, num_hidden, beta=beta1, learn_beta=True, threshold=1) # not a learnable decay rate
        self.lif2 = snn.LeakyParallel(num_hidden, num_outputs, beta=beta1, learn_beta=False, threshold=1) # learnable decay rate

    def forward(self, x):

      spk1 = self.lif1(x.flatten(2))
      spk2 = self.lif2(spk1)

      return spk2

# Load the network onto CUDA if available
net = Net().to(device)

import snntorch.functional as SF

optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0)

num_epochs = 1 # run for 1 epoch - each data sample is seen only once
loss_hist = [] # record loss over iterations
acc_hist = [] # record accuracy over iterations

start_time = time.time()

# training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(iter(train_loader)):
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        spk_rec = net(data.unsqueeze(0).repeat(num_steps, 1, 1, 1, 1)) # forward-pass
        loss_val = loss_fn(spk_rec, targets) # loss calculation
        optimizer.zero_grad() # null gradients
        loss_val.backward() # calculate gradients
        optimizer.step() # update weights
        loss_hist.append(loss_val.item()) # store loss

        # print every 25 iterations
        if i % 25 == 0:
          net.eval()
          print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

          # check accuracy on a single batch
          acc = SF.accuracy_rate(spk_rec, targets)
          acc_hist.append(acc)
          print(f"Accuracy: {acc * 100:.2f}%\n")

        # uncomment for faster termination
        # if i == 150:
        #     break

end_time = time.time()
print(f"Total Time: {end_time - start_time}")