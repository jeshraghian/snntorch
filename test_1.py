import snntorch as snn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import itertools
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Network Architecture
num_inputs = 28 * 28
num_hidden = 1000
num_outputs = 10

# Training Parameters
batch_size = 128
data_path = "/data/mnist"

# Temporal Dynamics
num_steps = 25
time_step = 1e-3
tau_mem = 3e-3
tau_syn = 2.2e-3
alpha = float(np.exp(-time_step / tau_syn))
beta = float(np.exp(-time_step / tau_mem))

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define a transform
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]
)

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Create DataLoader
train_loader = DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, drop_last=True
)
test_loader = DataLoader(
    mnist_test, batch_size=batch_size, shuffle=True, drop_last=True
)

# ============== Define Network ===================


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # initialize layers ---> now we've got num_inputs and hidden_init included.
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Stein(
            alpha=alpha,
            beta=beta,
            num_inputs=num_hidden,
            batch_size=batch_size,
            hidden_init=True,
        )
        # self.lif1 = Stein(alpha=alpha, beta=beta, num_inputs=(5,10,69), hidden_init=True) # this was just to show how to initialize convolutional layers
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Stein(
            alpha=alpha,
            beta=beta,
            num_inputs=num_outputs,
            batch_size=batch_size,
            hidden_init=True,
        )

    def forward(self, x):
        # self.lif1.detach_hidden() # It would be good to combine these two into one single line tho
        # self.lif2.detach_hidden()
        snn.Stein.detach_hidden()

        cur1 = self.fc1(x)
        self.lif1.spk1, self.lif1.syn1, self.lif1.mem1 = self.lif1(
            cur1, self.lif1.syn, self.lif1.mem
        )
        cur2 = self.fc2(self.lif1.spk)
        self.lif2.spk, self.lif2.syn, self.lif2.mem = self.lif2(
            cur2, self.lif2.syn, self.lif2.mem
        )

        return self.lif2.spk, self.lif2.mem


net = Net().to(device)
# print(net.lif1.__dict__)
# ============== Print Accuracy ===================


def print_batch_accuracy(data, targets, train=False):
    spk2_rec = []
    snn.Stein.zeros_hidden()

    for step in range(num_steps):
        spk2, mem2 = net(data.view(batch_size, -1))
        spk2_rec.append(spk2)
    spk2_rec = torch.stack(spk2_rec, dim=0)
    _, idx = spk2_rec.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train Set Accuracy: {acc}")
    else:
        print(f"Test Set Accuracy: {acc}")


def train_printer():
    print(f"Epoch {epoch}, Minibatch {minibatch_counter}")
    print(f"Train Set Loss: {loss_hist[counter]}")
    print(f"Test Set Loss: {test_loss_hist[counter]}")
    print_batch_accuracy(data_it, targets_it, train=True)
    print_batch_accuracy(testdata_it, testtargets_it, train=False)
    print("\n")


# ============== Optimizer/Loss ===================

optimizer = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0.9, 0.999))
log_softmax_fn = nn.LogSoftmax(dim=-1)
loss_fn = nn.NLLLoss()

# ============== Training Loop ===================

test_data = itertools.cycle(test_loader)
loss_hist = []
test_loss_hist = []
counter = 0

# Outer training loop
for epoch in range(1):
    train_batch = iter(train_loader)

    # Minibatch training loop
    minibatch_counter = 0
    for data_it, targets_it in train_batch:
        data_it = data_it.to(device)
        targets_it = targets_it.to(device)

        # Test set iterator
        testdata_it, testtargets_it = next(test_data)
        testdata_it = testdata_it.to(device)
        testtargets_it = testtargets_it.to(device)

        # training loop
        snn.Stein.zeros_hidden()  # reset hidden state to 0's
        for step in range(num_steps):
            spk2, mem2 = net(data_it.view(batch_size, -1))

            # loss p/timestep --- can try truncated approach too
            log_p_y = log_softmax_fn(mem2)  # mem2 = 128 x 10
            loss_val = loss_fn(log_p_y, targets_it)  # targets_it = 128
            # loss_hist.append(loss_val.item()) #optional, only hang onto the loss at the end of the 24 steps

            # Gradient calculation - detach states so gradient can flow
            optimizer.zero_grad()
            loss_val.backward()

            # Weight Update
            optimizer.step()

        # only keep the final loss value
        loss_hist.append(loss_val.item())

        #### Test set forward pass ####

        snn.Stein.zeros_hidden()
        for step in range(num_steps):
            spk2, mem2 = net(testdata_it.view(batch_size, -1))

        # Test set loss
        log_p_ytest = log_softmax_fn(mem2)
        loss_val_test = loss_fn(log_p_ytest, testtargets_it)
        test_loss_hist.append(loss_val_test.item())

        # Print test/train loss/accuracy
        if minibatch_counter % 10 == 0:
            train_printer()
        counter += 1
        minibatch_counter += 1

loss_hist_true_grad = loss_hist
test_loss_hist_true_grad = test_loss_hist
