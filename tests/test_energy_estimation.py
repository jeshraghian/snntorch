import pytest
import snntorch as snn
import snntorch.backprop as bp
from snntorch.energy_estimation.estimate_energy import *
from snntorch.energy_estimation.device_profile_registry import *
from tests.conftest import EnergyEfficiencyNetTest1, EnergyEfficiencyNetTest2
from unittest import mock
import torch

"""
def test_network():
    v = EnergyEfficiencyNetTest1(beta=0.5, num_timesteps=1)
    inp_test = torch.Tensor([[1]])

    print("")
    print(inp_test.size())
    print(v(inp_test))
"""


def test_estimate_energy_network1():
    num_values = 1
    percentage_of_ones = 0.

    ones_num = round(percentage_of_ones * num_values)
    zeros_num = num_values - ones_num

    inp_test = torch.Tensor([[[1]]] * ones_num + [[[0]]] * zeros_num)
    # create the network (the "device under test")
    v = EnergyEfficiencyNetTest1(beta=0.9, num_timesteps=inp_test.size(0))

    # setup the weights so it can be analyzed on paper
    v.fc1.weight.data = torch.Tensor([[0.25]])
    v.fc1.bias.data = torch.Tensor([0.5])
    v.fc2.weight.data = torch.Tensor([[0.5]])
    v.fc2.bias.data = torch.Tensor([0.25])

    DeviceProfileRegistry.add_device("cpu-test1", 8.6e-9, 8.6e-9, False)
    out = v(inp_test)
    summary_list = estimate_energy(model=v, input_data=inp_test, devices="cpu-test1",
                                   include_bias_term_in_events=False)

    expected_value = 3.44e-8
    energies = summary_list.total_energies
    # calculate it for only one device
    assert len(energies) == 1
    assert pytest.approx(expected_value, rel=0.01) == energies[0], f"The network consisting of single synapses/neurons " \
                                                                   f"was not within tolerance (was {expected_value}, " \
                                                                   f"is {energies[0]})"


def test_estimate_energy_network2():
    batch_size = 1
    prob_of_one = 1

    prob_matrix = prob_of_one * torch.ones((batch_size, 32))
    inp_test = torch.bernoulli(prob_matrix, p=prob_of_one)

    # create the network (the "device under test")
    DeviceProfileRegistry.add_device("cpu-test2", 8.6e-9, 8.6e-9, False)
    v = EnergyEfficiencyNetTest2(beta=0.9)
    out = v(inp_test)

    summary_list = estimate_energy(model=v, input_data=inp_test, devices="cpu-test2",
                                   include_bias_term_in_events=False)

    expected_value = 2.38e-5
    energies = summary_list.total_energies
    # calculate it for only one device
    assert len(energies) == 1
    assert pytest.approx(expected_value, rel=0.01) == energies[0], f"The energy estimate for simple network 2 " \
                                                                   f"was not within tolerance (was {expected_value}, " \
                                                                   f"is {energies[0]})"


def test_estimate_energy_network2_energies_for_two_new_devices():
    batch_size = 1
    prob_of_one = 1

    prob_matrix = prob_of_one * torch.ones((batch_size, 32))
    inp_test = torch.bernoulli(prob_matrix, p=prob_of_one)

    # create the network (the "device under test")
    DeviceProfileRegistry.add_device("cpu-test3", 10e-9, 1e-9, False)
    DeviceProfileRegistry.add_device("cpu-test4", 1e-9, 10e-9, False)

    v = EnergyEfficiencyNetTest2(beta=0.9)
    out = v(inp_test)

    summary_list = estimate_energy(model=v, input_data=inp_test, devices=["cpu-test3", "cpu-test4"],
                                   include_bias_term_in_events=False)
    energies = summary_list.total_energies

    # calculate it for two devices
    expected_values = [2.70e-5, 3.43e-6]

    assert len(energies) == 2
    assert pytest.approx(expected_values[0], rel=0.01) == energies[0]
    assert pytest.approx(expected_values[1], rel=0.01) == energies[1]
