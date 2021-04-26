# import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import itertools
import matplotlib.pyplot as plt

from torchneuromorphic.dvs_gestures.create_hdf5 import *
from torchneuromorphic.dvs_gestures.dvsgestures_dataloaders import *
from torchneuromorphic.utils import plot_frames_imshow
import torchneuromorphic.transforms as transforms
    
# from torchneuromorphic.dvs_gestures.dvsgestures_dataloaders import *
# from torchneuromorphic.utils import plot_frames_imshow
# import torchneuromorphic.transforms as transforms

# if __name__ == "__main__":
#     train_dl, test_dl = create_dataloader(
#             root='data/dvsgesture/dvs_gestures_build19.hdf5',
#             batch_size=64,
#             ds=4,
#             num_workers=0)
#     ho = iter(train_dl)
#     frames, labels = next(ho)
# if __name__ == "__main__":
#     out = create_events_hdf5('./data/DvsGesture/', './data/dvs_gestures_build19.hdf5')
    
    # ho = iter(train_dl)
    # frames, labels = next(ho)

# Network Architecture
num_inputs = 32*32*2
num_outputs = 11
num_hidden = 192
# Training Parameters
batch_size=2
data_path='/data/mnist'

# Temporal Dynamics
num_steps = 192
time_step = 1e-3
# tau_mem = 6.5e-4
# tau_syn = 5.5e-4
# alpha = float(np.exp(-time_step/tau_syn))
# beta = float(np.exp(-time_step/tau_mem))
alpha = 1
beta = 1
n_iters = 50
dt = 1000 #us
in_channels = 2
ds = 2 
im_dims = im_width, im_height = (128//ds, 128//ds) 
n_iters_test = 192
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_loader, test_loader = create_dataloader(
            root='./data/dvsgesture/dvs_gestures_build19.hdf5',
            batch_size=batch_size,
            chunk_size_train=num_steps,
            chunk_size_test=n_iters_test,
            ds=ds,
            num_workers=0)

# Binarized Layer Modules
import pdb
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np


def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
        # tmp = tensor.clone()
        # tmp[tensor>0] = 1
        # tmp[tensor==0] = 0
        # tmp[tensor<0] = -1
        # return tmp
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

def Quantize(tensor,quant_mode='det',  params=None, numBits=8):
    tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

# import torch.nn._functions as tnnf


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

# neuron model
class LIF(nn.Module):
    """Parent class for leaky integrate and fire neuron models."""
    instances = []
    def __init__(self, alpha, beta, threshold=1.0, spike_grad=None):
        super(LIF, self).__init__()
        LIF.instances.append(self)

        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

        if spike_grad is None:
            self.spike_grad = self.Heaviside.apply
        else:
            self.spike_grad = spike_grad

    def fire(self, mem):
        """Generates spike if mem > threshold.
        Returns spk and reset."""
        mem_shift = mem - self.threshold
        spk = self.spike_grad(mem_shift).to(device)
        reset = torch.zeros_like(mem)
        spk_idx = (mem_shift > 0)
        reset[spk_idx] = torch.ones_like(mem)[spk_idx]
        return spk, reset

    def fire_single(self, mem):
        """Generates spike if mem > threshold.
        Returns spk and reset."""
        mem_shift = mem - self.threshold
        
        index = torch.argmax(mem_shift, dim=-1)
        
        spk_tmp = self.spike_grad(mem_shift)

        mask_spk1 = torch.zeros_like(spk_tmp)
        # print(mem.size())
        # print(index.size())
        mask_spk1[torch.arange(mem_shift.size()[0]), index] = 1
        spk = (spk_tmp * mask_spk1).to(device)
        # print(spk[0])
        reset = torch.zeros_like(mem)
        spk_idx = (mem_shift > 0)
        reset[spk_idx] = torch.ones_like(mem)[spk_idx]
        return spk, reset

    @classmethod
    def clear_instances(cls):
      cls.instances = []

    @staticmethod
    def init_stein(batch_size, *args):
        """Used to initialize syn, mem and spk.
        *args are the input feature dimensions.
        E.g., batch_size=128 and input feature of size=1x28x28 would require init_hidden(128, 1, 28, 28)."""
        syn = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        mem = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        spk = torch.zeros((batch_size, *args), device=device, dtype=dtype)

        return spk, syn, mem

    @staticmethod
    def init_srm0(batch_size, *args):
        """Used to initialize syn_pre, syn_post, mem and spk.
        *args are the input feature dimensions.
        E.g., batch_size=128 and input feature of size=1x28x28 would require init_hidden(128, 1, 28, 28)."""
        syn_pre = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        syn_post = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        mem = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        spk = torch.zeros((batch_size, *args), device=device, dtype=dtype)

        return spk, syn_pre, syn_post, mem

    @staticmethod
    def detach(*args):
        """Used to detach input arguments from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are global variables."""
        for state in args:
            state.detach_()

    @staticmethod
    def zeros(*args):
        """Used to clear hidden state variables to zero.
            Intended for use where hidden state variables are global variables."""
        for state in args:
            state = torch.zeros_like(state)

    @staticmethod
    class Heaviside(torch.autograd.Function):
        """Default and non-approximate spiking function for neuron.
        Forward pass: Heaviside step function.
        Backward pass: Dirac Delta clipped to 1 at x>0 instead of inf at x=1.
        This assumption holds true on the basis that a spike occurs as long as x>0 and the following time step incurs a reset."""

        @staticmethod
        def forward(ctx, input_):
            ctx.save_for_backward(input_)
            out = torch.zeros_like(input_)
            out[input_ > 0] = 1.0
            return out

        @staticmethod
        def backward(ctx, grad_output):
            input_, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input_ < 0] = 0.0
            grad = grad_input
            return grad

class Stein_single(LIF):
    """
    Stein's model of the leaky integrate and fire neuron.
    The synaptic current jumps upon spike arrival, which causes a jump in membrane potential.
    Synaptic current and membrane potential decay exponentially with rates of alpha and beta, respectively.
    For mem[T] > threshold, spk[T+1] = 0 to account for axonal delay.

    For further reading, see:
    R. B. Stein (1965) A theoretical analysis of neuron variability. Biophys. J. 5, pp. 173-194.
    R. B. Stein (1967) Some models of neuronal variability. Biophys. J. 7. pp. 37-68."""

    def __init__(self, alpha, beta, threshold=1.0, num_inputs=False, spike_grad=None, batch_size=False, hidden_init=False):
        super(Stein_single, self).__init__(alpha, beta, threshold, spike_grad)

        self.num_inputs = num_inputs
        self.batch_size = batch_size
        self.hidden_init = hidden_init

        if self.hidden_init:
            if not self.num_inputs:
                raise ValueError("num_inputs must be specified to initialize hidden states as instance variables.")
            elif not self.batch_size:
                raise ValueError("batch_size must be specified to initialize hidden states as instance variables.")
            elif hasattr(self.num_inputs, '__iter__'):
                self.spk, self.syn, self.mem = self.init_stein(self.batch_size, *(self.num_inputs)) # need to automatically call batch_size
            else:
                self.spk, self.syn, self.mem = self.init_stein(self.batch_size, self.num_inputs)

    def forward(self, input_, syn, mem):
        if not self.hidden_init:
            spk, reset = self.fire(mem)
            # input_[input_>1] = 1
            # input_[input_<-1] = -1
            syn = self.alpha * syn + input_
            mem = self.beta * mem + syn - reset

            return spk, syn, mem

        # intended for truncated-BPTT where instance variables are hidden states
        if self.hidden_init:
            self.spk, self.reset = self.fire(self.mem)
            self.syn = self.alpha * self.syn + input_
            self.mem = self.beta * self.mem + self.syn - self.reset

            return self.spk, self.syn, self.mem

    @classmethod
    def detach_hidden(cls):
        """Used to detach hidden states from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            cls.instances[layer].spk.detach_()
            cls.instances[layer].syn.detach_()
            cls.instances[layer].mem.detach_()

    @classmethod
    def zeros_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            cls.instances[layer].spk = torch.zeros_like(cls.instances[layer].spk)
            cls.instances[layer].syn = torch.zeros_like(cls.instances[layer].syn)
            cls.instances[layer].mem = torch.zeros_like(cls.instances[layer].mem)

# network structure

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()

#     # initialize layers
#         self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.lif1 = Stein_single(alpha=alpha, beta=beta)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.lif2 = Stein_single(alpha=alpha, beta=beta)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.lif3 = Stein_single(alpha=alpha, beta=beta)
#         self.fc4 = nn.Linear(32*4*4, 11, bias= False)
#         self.lif4 = Stein_single(alpha=alpha, beta=beta)
#         self.dropout = nn.Dropout(p=0.1)
#         self.softmax = nn.Softmax(dim=-1)
#     def forward(self, x):
#         # Initialize LIF state variables and spike output tensors
#         spk1, syn1, mem1 = self.lif1.init_stein(x.shape[1], 16, 16, 16)
#         spk2, syn2, mem2 = self.lif2.init_stein(x.shape[1], 32, 8, 8)
#         spk3, syn3, mem3 = self.lif3.init_stein(x.shape[1], 32, 4, 4)
#         spk4, syn4, mem4 = self.lif4.init_stein(x.shape[1], 11)

#         spk4_rec = []
#         mem4_rec = []

#         for step in range(x.shape[0]):
#             cur1 = F.max_pool2d(self.conv1(x[step]), 2)
#             spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)

#             cur2 = F.max_pool2d(self.conv2(spk1), 2)
#             spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

#             cur3 = F.max_pool2d(self.conv3(spk2), 2)
#             spk3, syn3, mem3 = self.lif3(cur3, syn3, mem3)

#             cur4 = self.fc4(spk3.view(x.shape[1], -1))
#             spk4, syn4, mem4 = self.lif4(cur4, syn4, mem4)
#             # spk4 = self.softmax(spk4)
#             spk4_rec.append(spk4)
#             mem4_rec.append(mem4)

#         return torch.stack(spk4_rec, dim=0), torch.stack(mem4_rec, dim=0)

class Net(nn.Module):
    def __init__(self):
        super().__init__()

    # initialize layers
        # self.bn0 = nn.BatchNorm1d(784)
        # self.fc1 = nn.Linear(32*32*2, num_hidden)
        # self.lif1 = Stein_single(alpha=alpha, beta=beta)
        self.conv0 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=2, padding=0, bias=False)
        self.lif0 = Stein_single(alpha=alpha, beta=beta)
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=256, kernel_size=4, stride=2, padding=0, bias=False)
        self.lif1 = Stein_single(alpha=alpha, beta=beta)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.lif2 = Stein_single(alpha=alpha, beta=beta)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False)
        self.lif3 = Stein_single(alpha=alpha, beta=beta)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.lif4 = Stein_single(alpha=alpha, beta=beta)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.lif5 = Stein_single(alpha=alpha, beta=beta)

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.lif6 = Stein_single(alpha=alpha, beta=beta)

        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.lif7 = Stein_single(alpha=alpha, beta=beta)

        # self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, bias=False)
        # self.lif8 = Stein_single(alpha=alpha, beta=beta)

        # self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        # self.lif9 = Stein_single(alpha=alpha, beta=beta)

        # self.conv10 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False)
        # self.lif10 = Stein_single(alpha=alpha, beta=beta)

        # self.conv11 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False)
        # self.lif11 = Stein_single(alpha=alpha, beta=beta)

        # self.conv12 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2, padding=0, bias=False)
        # self.lif12 = Stein_single(alpha=alpha, beta=beta)

        # self.conv13 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False)
        # self.lif13 = Stein_single(alpha=alpha, beta=beta)

        # self.conv14 = nn.Conv2d(in_channels=1024, out_channels=968, kernel_size=1, stride=1, padding=0, bias=False)
        # self.lif14 = Stein_single(alpha=alpha, beta=beta)

        # self.conv15 = nn.Conv2d(in_channels=968, out_channels=2640, kernel_size=1, stride=1, padding=0, bias=False)
        # self.lif15 = Stein_single(alpha=alpha, beta=beta)

        self.fc7 = nn.Linear(512*7*7, num_outputs)
        self.lif7 = Stein_single(alpha=alpha, beta=beta)

        # self.batch_size = batch_size
        # self.num_steps = num_steps

    def forward(self, x):
        # spk1, syn1, mem1 = self.lif1.init_stein(x.shape[1], num_hidden)
        spk0, syn0, mem0 = self.lif0.init_stein(x.shape[1], 12, 31, 31)
        spk1, syn1, mem1 = self.lif1.init_stein(x.shape[1], 256, 14, 14)
        spk2, syn2, mem2 = self.lif2.init_stein(x.shape[1], 256, 14, 14)
        spk3, syn3, mem3 = self.lif3.init_stein(x.shape[1], 256, 7, 7)
        spk4, syn4, mem4 = self.lif4.init_stein(x.shape[1], 512, 7, 7)
        spk5, syn5, mem5 = self.lif5.init_stein(x.shape[1], 512, 7, 7)
        spk6, syn6, mem6 = self.lif6.init_stein(x.shape[1], 512, 7, 7)
        # spk7, syn7, mem7 = self.lif7.init_stein(x.shape[1], 512, 7, 7)
        # spk8, syn8, mem8 = self.lif8.init_stein(x.shape[1], 512, 3, 3)
        # spk9, syn9, mem9 = self.lif9.init_stein(x.shape[1], 1024, 3, 3)
        # spk10, syn10, mem10 = self.lif10.init_stein(x.shape[1], 1024, 3, 3)
        # spk11, syn11, mem11 = self.lif11.init_stein(x.shape[1], 1024, 3, 3)
        # spk12, syn12, mem12 = self.lif12.init_stein(x.shape[1], 1024, 1, 1)
        # spk13, syn13, mem13 = self.lif13.init_stein(x.shape[1], 1024, 1, 1)
        # spk14, syn14, mem14 = self.lif14.init_stein(x.shape[1], 968, 1, 1)
        # spk15, syn15, mem15 = self.lif13.init_stein(x.shape[1], 2640, 1, 1)
        spk7, syn7, mem7 = self.lif7.init_stein(x.shape[1], num_outputs)

        # spk15_rec = []
        # mem15_rec = []

        spk7_rec = []
        mem7_rec = []

        for step in range(x.shape[0]):
            # mask_x = torch.zeros_like(x[step])
            # index = torch.argmax(x[step], dim=-1)
            # mask_x[torch.arange(x[step].size()[0]), index] = 1
            # x_in = (x[step] * mask_x)
            cur0 = (self.conv0(x[step]))
            spk0, syn0, mem0 = self.lif0(cur0, syn0, mem0)
            cur1 = (self.conv1(spk0))
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            cur2 = (self.conv2(spk1))
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            cur3 = (self.conv3(spk2))
            spk3, syn3, mem3 = self.lif3(cur3, syn3, mem3)
            cur4 = (self.conv4(spk3))
            spk4, syn4, mem4 = self.lif4(cur4, syn4, mem4)
            cur5 = (self.conv5(spk4))
            spk5, syn5, mem5 = self.lif5(cur5, syn5, mem5)
            cur6 = (self.conv6(spk5))
            spk6, syn6, mem6 = self.lif6(cur6, syn6, mem6)
            # cur7 = (self.conv7(spk6))
            # spk7, syn7, mem7 = self.lif7(cur7, syn7, mem7)
            # cur8 = (self.conv8(spk7))
            # spk8, syn8, mem8 = self.lif8(cur8, syn8, mem8)
            # cur9 = (self.conv9(spk8))
            # spk9, syn9, mem9 = self.lif9(cur9, syn9, mem9)
            # cur10 = (self.conv10(spk9))
            # spk10, syn10, mem10 = self.lif10(cur10, syn10, mem10)
            # cur11 = (self.conv11(spk10))
            # spk11, syn11, mem11 = self.lif11(cur11, syn11, mem11)
            # cur12 = (self.conv12(spk11))
            # spk12, syn12, mem12 = self.lif12(cur12, syn12, mem12)
            # cur13 = (self.conv13(spk12))
            # spk13, syn13, mem13 = self.lif13(cur13, syn13, mem13)
            # cur14 = (self.conv14(spk13))
            # spk14, syn14, mem14 = self.lif14(cur14, syn14, mem14)
            # cur15 = (self.conv15(spk14))
            # spk15, syn15, mem15 = self.lif15(cur15, syn15, mem15)
            cur7 = self.fc7(spk6.view(-1, 512*7*7))
            spk7, syn7, mem7 = self.lif7(cur7, syn7, mem7)

            # spk15_rec.append(spk15.view(-1, 2640))
            # mem15_rec.append(mem15.view(-1, 2640))
            
            spk7_rec.append(spk7)
            mem7_rec.append(mem7)

        return torch.stack(spk7_rec, dim=0), torch.stack(mem7_rec, dim=0)


print(batch_size)

net = nn.DataParallel(Net(), device_ids=[0,1,2,3])

def print_batch_accuracy(data, targets, train=False):
    # output, _ = net(data.view(batch_size, -1))
    # output, _ = net(data.transpose(1, 0).view(-1, batch_size, 2, 32, 32))
    output, _ = net(data.view(batch_size, -1, 6, 64, 64).transpose(1, 0))
    _, am = output.sum(dim=0).max(1)
    acc = np.mean((torch.argmax(targets[:,0,:],dim=-1) == am). detach().cpu().numpy())
    if train is True:
        print(f"Train Set Accuracy: {acc}")
    else:
        print(f"Test Set Accuracy: {acc}")

def train_printer():
    print(f"Epoch {epoch}, Minibatch {minibatch_counter}")
    print(f"Train Set Loss: {loss_hist[counter-1]}")
    print(f"Test Set Loss: {test_loss_hist[counter-1]}")
    print_batch_accuracy(data_it, targets_it, train=True)
    print_batch_accuracy(testdata_it, testtargets_it, train=False)
    print("\n")
i=0
lr = 0.02/4
optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=1e-7, momentum=0.9)#, betas=(0.9, 0.999)
log_softmax_fn = nn.LogSoftmax(dim=-1)
loss_fn = nn.NLLLoss()


import math
loss_hist = []
test_loss_hist = []
counter = 0
print(f"initial_lr: {lr}")
def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # if epoch< 5:
    new_lr = lr * (0.7 ** epoch)
    # new_lr = lr * math.exp (-0.85*epoch)
    # new_lr = lr * 1 / ( 1 + 0.1*epoch)
    # else: new_lr = lr * (1**4) * (0.7**(epoch-4))
    # else: new_lr = lr * (0.95 ** epoch) * (0.9 ** (epoch-4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

# Outer training loop
for epoch in range(2):

    minibatch_counter = 0
    train_batch = iter(train_loader)
    

    # Minibatch training loop
    for data_it, targets_it in train_batch:

        data_it = data_it.to(device)
        targets_it = targets_it.to(device)

        output, mem_rec = net(data_it.view(-1, int(num_steps/3), 6, 64, 64).transpose(1, 0))
        # output, mem_rec = net(data_it.transpose(1, 0))
        log_p_y = log_softmax_fn(mem_rec)
        loss_val = torch.zeros(1, dtype=dtype, device=device)

        # Sum loss over time steps to perform BPTT
        for step in range(int(num_steps/3)):
          loss_val += loss_fn(log_p_y[step], torch.argmax(targets_it[:,0,:], dim=-1))
        
        # adjust_learning_rate(lr, optimizer, epoch)

        # BNN OPTimization
        optimizer.zero_grad()
        loss_val.backward()
        # for p in list(net.parameters()):
        #         if hasattr(p,'org'):
        #             p.data.copy_(p.org)
        nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        # for p in list(net.parameters()):
        #         if hasattr(p,'org'):
        #             p.org.copy_(p.data.clamp_(-1,1))

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        test_data = itertools.cycle(test_loader)
        testdata_it, testtargets_it = next(test_data)

        testdata_it = testdata_it.to(device)
        testtargets_it = testtargets_it.to(device)

        # Test set forward pass
        test_output, test_mem_rec = net(testdata_it.view(-1, int(n_iters_test/3), 6, 64, 64).transpose(1,0))
        # test_output, test_mem_rec = net(testdata_it.transpose(1,0))

        # Test set loss
        log_p_ytest = log_softmax_fn(test_mem_rec)
        log_p_ytest = log_p_ytest.sum(dim=0)
        loss_val_test = loss_fn(log_p_ytest, torch.argmax(testtargets_it[:,0,:], dim=-1))
        test_loss_hist.append(loss_val_test.item())

        # Print test/train loss/accuracy
        if counter % 10 == 0:
          train_printer()
        minibatch_counter += 1
        counter += 1

    total = 0
    correct = 0

    with torch.no_grad():
      net.eval()
      for data in test_loader:
        input_tests, labels = data
        
        input_tests = input_tests.to(device)
        labels = labels.to(device)

        outputs, _ = net(input_tests.view(-1, int(n_iters_test/3), 6, 64, 64).transpose(1,0))

        # outputs, _ = net(input_tests.transpose(1,0))
        
        _, predicted = outputs.sum(dim=0).max(1)
        total += labels.size(0)
        correct += (predicted == torch.argmax(labels[:,0,:],dim=-1)).sum().item()

    print(f"Total correctly classified test set geatures: {correct}/{total}")
    print(f"Test Set Accuracy: {100 * correct / total}%")   

loss_hist_true_grad = loss_hist
test_loss_hist_true_grad = test_loss_hist