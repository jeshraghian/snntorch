# Truncated BPTT, k=1

import snntorch as snn
from snntorch.utils import data_subset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from Net import Net


def train(net, device, train_loader, optimizer, criterion, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        t = 0
        loss_trunc = 0
        snn.Stein.zeros_hidden()  # reset hidden state to 0's
        snn.Stein.detach_hidden()

        for step in range(num_steps):
            spk2, mem2 = net(data.view(batch_size, -1))
            log_p_y = log_softmax_fn(mem2)
            loss = criterion(log_p_y, target)
            loss_trunc += loss
            t += 1
            if t == K:
                optimizer.zero_grad()
                loss_trunc.backward()
                optimizer.step()
                snn.Stein.detach_hidden()
                t = 0
                loss_trunc = 0
        if (step == num_steps-1) and (num_steps % K):
            optimizer.zero_grad()
            loss_trunc.backward()
            optimizer.step()
            snn.Stein.detach_hidden()

        if batch_idx % 10 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}], "
                  f"Loss: {loss.item()}")
    loss_hist.append(loss.item())  # only recording at the end of each epoch


def test(net, device, test_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            spk2_rec = []
            snn.Stein.zeros_hidden()  # reset hidden states to 0
            if data.size()[0] == batch_size:
                for step in range(num_steps):
                    spk2, mem2 = net(data.view(batch_size, -1))
                    spk2_rec.append(spk2)

                # Test Loss where batch=128; only calc on final time step
                log_p_ytest = log_softmax_fn(mem2)
                test_loss += criterion(log_p_ytest, target)
                # Test Acc where batch=128
                _, idx = torch.stack(spk2_rec, dim=0).sum(dim=0).max(1)  # predicted indexes
                correct += sum((target == idx).numpy())
                # print(correct)

            else:  # Handle drop_last = False
                temp_data = torch.zeros((batch_size, *(data[0].size())))  # pad out temp_data now
                temp_data[:(data.size()[0])] = data

                for step in range(num_steps):
                    spk2, mem2 = net(temp_data.view(batch_size, -1))
                    spk2_rec.append(spk2)

                # Test set loss - only calc on the final time-step
                log_p_ytest = log_softmax_fn(mem2[:data.size()[0]])
                test_loss += criterion(log_p_ytest, target)
                # Test Acc where batch=128
                _, idx = torch.stack(spk2_rec, dim=0).sum(dim=0).max(1)  # predicted indexes
                correct += sum((target == idx[:data.size()[0]]).numpy())

        test_loss_hist.append(test_loss.item())
        print(f"\nTest set: Average loss: {(test_loss/(len(test_loader.dataset)/batch_size))}, Accuracy: [{correct}/{len(test_loader.dataset)}] ({(correct/len(test_loader.dataset))})\n"
              f"=====================\n")


if __name__ == "__main__":
    no_trials = 1
    lr_values = [2e-4]
    batch_size = 128
    data_path = '/data/mnist'
    subset = 10  # can remove this line in Colab
    num_steps = 25
    epochs = 5
    betas = (0.9, 0.999)
    K = 10  # number of time steps to accumulate over
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
    ])
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    mnist_train = data_subset(mnist_train, subset)  # reduce dataset by x100 - can remove this line in Colab
    mnist_test = data_subset(mnist_test, subset)
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

    for i in range(no_trials):
        for lr in lr_values:
            net = Net().to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=betas)
            log_softmax_fn = nn.LogSoftmax(dim=-1)
            criterion = nn.NLLLoss()

            loss_hist = []
            test_loss_hist = []
            for epoch in range(epochs):
                train(net, device, train_loader, optimizer, criterion, epoch)
                test(net, device, test_loader, criterion)

                # df = df.append(
                #     {'trial': i, 'lr': lr, 'no_stoch_samples': 0, 'epoch': epoch, 'test_set_loss': test_set_loss,
                #      'test_set_accuracy': test_set_accuracy}, ignore_index=True)
                # df.to_csv('SGD.csv', index=False)
                # if SAVE_GOOGLE_COLAB:
                #     shutil.copy("SGD.csv", "/content/gdrive/My Drive/SGD.csv")


loss_hist_true_grad = loss_hist
test_loss_hist_true_grad = test_loss_hist
