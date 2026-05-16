import unittest

import torch
import torch.nn as nn

import snntorch as snn
from snntorch import utils


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.neuron = snn.Leaky(beta=0.5, init_hidden=True)
        self.neuron2 = snn.Leaky(beta=0.5, init_hidden=True)

    def forward(self, x):
        return self.neuron(x) + self.neuron2(x)


class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(snn.Leaky(beta=0.5, init_hidden=True))

    def forward(self, x):
        return self.net(x)


class ListModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.list = nn.ModuleList([snn.Leaky(beta=0.5, init_hidden=True)])

    def forward(self, x):
        return self.list[0](x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.neuron = snn.Leaky(beta=0.5, init_hidden=True)

    def forward(self, x):
        return self.neuron(x)


class NestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()

    def forward(self, x):
        return self.block(x)


class MultiNeuronBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.neuron1 = snn.Leaky(beta=0.5, init_hidden=True)
        self.neuron2 = snn.Leaky(beta=0.5, init_hidden=True)

    def forward(self, x):
        x = self.neuron1(x)
        return self.neuron2(x)


class NestedMultiNeuronModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = MultiNeuronBlock()

    def forward(self, x):
        return self.block(x)


class TestResetMechanism(unittest.TestCase):

    def _check_reset(self, model_class):
        model = model_class()
        x = torch.randn(1, 10)
        model(x)

        neurons = [m for m in model.modules() if isinstance(m, snn.Leaky)]
        self.assertTrue(len(neurons) > 0)

        for neuron in neurons:
            self.assertNotEqual(neuron.mem.abs().sum().item(), 0)

        utils.reset(model)

        for neuron in neurons:
            self.assertEqual(neuron.mem.abs().sum().item(), 0)

    def test_flat_model(self):
        self._check_reset(Model)

    def test_sequential_model(self):
        self._check_reset(SequentialModel)

    def test_list_model(self):
        self._check_reset(ListModel)

    def test_nested_custom_model(self):
        self._check_reset(NestedModel)

    def test_nested_multi_neuron_model(self):
        self._check_reset(NestedMultiNeuronModel)
