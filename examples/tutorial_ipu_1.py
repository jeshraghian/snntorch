# Script by Vincent Sun

import argparse
import ctypes
import os

import numpy as np
import popart
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import poptorch
import snntorch as snn
from snntorch.functional import loss as SF
from snntorch import surrogate
from snntorch import spikegen

import csv
import time
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score


alpha = 0.745
beta = 0.9
num_inputs = 784
num_output = 10
num_hidden = 1000
num_steps = 25
batch_size = 128
data_path = "/data/mnist"

transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]
)

mnist_train = datasets.MNIST(
    data_path, train=True, download=True, transform=transform
)
mnist_test = datasets.MNIST(
    data_path, train=False, download=True, transform=transform
)
opts = poptorch.Options()
opts.Precision.halfFloatCasting(
    poptorch.HalfFloatCastingBehavior.HalfUpcastToFloat
)
# Create DataLoaders
train_loader = poptorch.DataLoader(
    options=opts,
    dataset=mnist_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=20,
)
spike_grad = surrogate.straight_through_estimator()
snn.slope = 50


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_output)
        self.lif2 = snn.Leaky(beta=beta)
        self.loss_fn = SF.ce_count_loss()

    def forward(self, x, labels=None):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x.view(batch_size, -1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        spk2_rec = torch.stack(spk2_rec)
        mem2_rec = torch.stack(mem2_rec)
        if self.training:
            return spk2_rec, poptorch.identity_loss(
                self.loss_fn(mem2_rec, labels), "none"
            )
        return spk2_rec


if __name__ == "__main__":
    net = Model()
    # test_net = Model()
    # net.half()
    # net.train()
    # test_net.eval()
    optimizer = poptorch.optim.Adam(
        net.parameters(), lr=0.001, betas=(0.9, 0.999)
    )

    poptorch_model = poptorch.trainingModel(
        net, options=opts, optimizer=optimizer
    )

    # Time
    date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    csv_file = (
        "Batch_Size_"
        + str(batch_size)
        + "_Graphcore_Throughput_"
        + date
        + ".csv"
    )
    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Steps", "Accuracy", "Throughput"])

    epochs = 2
    for epoch in tqdm(range(epochs), desc="epochs"):
        correct = 0.0
        total_loss = 0.0
        total = 0.0

        for i, (data, labels) in enumerate(train_loader):
            data = data.half()
            start_time = time.time()
            output, loss = poptorch_model(data, labels)
            end_time = time.time()

            if i % 250 == 0:
                _, pred = output.sum(dim=0).max(1)
                correct = (labels == pred).sum().item() / len(labels)

                throughput = len(data) / (end_time - start_time)
                print("accuracy: ", correct)
                with open(csv_file, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, i, correct, throughput])
