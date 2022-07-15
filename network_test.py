import snntorch as snn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

alpha = 0.745
beta = 0.9
num_inputs=28*28
num_output=10
num_hidden=1000 # original is 1000
num_steps=25
batch_size = 128
data_path='/Users/vincent/Desktop/python-venv/dataset/mnist'

mnist_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
cifar10_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=mnist_transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=mnist_transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Izhikevich(learn_threshold=False)
        self.fc2 = nn.Linear(num_hidden, num_output)
        self.lif2 = snn.Izhikevich(learn_threshold=False)

    def forward(self, x, labels=None):
        v1, u1 = self.lif1.init_izhikevich(self.lif1.v_rest, self.lif1.u_rest)
        v2, u2 = self.lif2.init_izhikevich(self.lif2.v_rest, self.lif2.u_rest)

        spk2_rec = []
        v2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x.view(batch_size,-1))
            spk1, v1, u1 = self.lif1(cur1)
            cur2 = self.fc2(spk1)
            spk2, v2, u2 = self.lif2(cur2)

            spk2_rec.append(spk2)
            v2_rec.append(v2)

        return torch.stack(spk2_rec, dim=0), torch.stack(v2_rec, dim=0)
        
net = Model()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
log_softmax_fn = nn.LogSoftmax(dim=-1)
loss_fn = nn.NLLLoss()


for epoch in tqdm(range(2)):
    train_batch = iter(train_loader)

    counter = 0

    for data_it, target_it in train_batch:
        output, v_rec = net(data_it.view(batch_size, 1, 28, 28))

        log_p_y = log_softmax_fn(v_rec)
        loss_val = torch.zeros((1))

        for step in range(num_steps):
            loss_val += loss_fn(log_p_y[step], target_it)

        optimizer.zero_grad()
        loss_val.backward(retain_graph=True)

        nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        if (counter+1) % 50 == 0:
            _, idx = output.sum(dim=0).max(1)
            acc = (target_it == idx).sum().item()/len(target_it)

            print(acc)
    
        counter += 1