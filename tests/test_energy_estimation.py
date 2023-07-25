import pytest
import snntorch as snn
import snntorch.backprop as bp
from snntorch.energy_estimation.estimate_energy import *
from snntorch.energy_estimation.device_profile_registry import *
from tests.conftest import *
from snntorch.utils import reset
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

    DeviceProfileRegistry.add_device("cpu-test1", 8.6e-9, 8.6e-9, False)
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
    summary_list = estimate_energy(model=v, input_data=inp_test, devices=["cpu-test3", "cpu-test4"],
                                   include_bias_term_in_events=False)
    energies = summary_list.total_energies

    # calculate it for two devices
    expected_values = [2.70e-5, 3.43e-6]

    assert len(energies) == 2
    assert pytest.approx(expected_values[0], rel=0.01) == energies[0]
    assert pytest.approx(expected_values[1], rel=0.01) == energies[1]


def test_estimate_energy_network3_for_neuromorphic():
    inp_test = torch.Tensor([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]).T
    # create the network (the "device under test")
    v = EnergyEfficiencyNetTest3(beta=0.5)
    v.fc1.weight.data = torch.Tensor([[1]])
    v.fc1.bias.data = torch.Tensor([[0]])
    v.fc2.weight.data = torch.Tensor([[0.5]])
    v.fc2.bias.data = torch.Tensor([[0]])
    v.reset()

    # Debug, see whether the number checks out
    # for t in inp_test:
    #    spk = torch.stack([t])
    #    dat = v.forward_full(spk)
    #    print([dat[idx].item() for idx in range(4)])

    DeviceProfileRegistry.add_device("neuromorphic-test1", 1e-9, 1e-9, True)
    DeviceProfileRegistry.add_device("vn-against-neuromorphic-test1", 1e-9, 1e-9, False)

    summary_list = estimate_energy(model=v, input_data=inp_test, devices=["neuromorphic-test1",
                                                                          "vn-against-neuromorphic-test1"],
                                   include_bias_term_in_events=False)

    # see that the numbers differ a lot for devices that are truely spiking or not (by
    # writing on paper, or checking on code, this set of weights will generate only one spike at time t=6)
    # therefore expected energy is 4e-9 (1e-9 for each synapse and 1e-9 for each neuron)
    # in comparison (in non spiking / von neunmann computing), although we have mostly zeros, energy will constantly
    # be needed to perform operation
    neuromorphic_expected = 4e-9
    von_neunmann_expected = 4e-8
    energies = summary_list.total_energies

    assert len(energies) == 2
    assert pytest.approx(neuromorphic_expected, rel=0.01) == energies[
        0], f"The network consisting of single synapses/neurons " \
            f"was not within tolerance (was {neuromorphic_expected}, " \
            f"is {energies[0]})"
    assert pytest.approx(von_neunmann_expected, rel=0.01) == energies[
        1], f"The network consisting of single synapses/neurons " \
            f"was not within tolerance (was {von_neunmann_expected}, " \
            f"is {energies[0]})"


def test_estimate_energy_network3_for_neuromorphic2():
    inp_test = torch.Tensor([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]).T
    # create the network (the "device under test")
    v = EnergyEfficiencyNetTest3(beta=0.5)
    v.fc1.weight.data = torch.Tensor([[1]])
    v.fc1.bias.data = torch.Tensor([[0]])
    v.fc2.weight.data = torch.Tensor([[0.5]])
    v.fc2.bias.data = torch.Tensor([[0]])
    v.reset()

    # Debug, see whether the number checks out
    # for t in inp_test:
    #    spk = torch.stack([t])
    #    dat = v.forward_full(spk)
    #    print([dat[idx].item() for idx in range(4)])

    DeviceProfileRegistry.add_device("neuromorphic-test2", 1e-9, 0, True)
    DeviceProfileRegistry.add_device("vn-against-neuromorphic-test2", 1e-9, 0, False)

    summary_list = estimate_energy(model=v, input_data=inp_test, devices=["neuromorphic-test2",
                                                                          "vn-against-neuromorphic-test2"],
                                   include_bias_term_in_events=False)

    # see that the numbers differ a lot for devices that are truely spiking or not (by
    # writing on paper, or checking on code, this set of weights will generate only one spike at time t=6)
    # therefore expected energy is 2e-9 (1e-9 for each synapse and 0 for each neuron)
    # in comparison (in non spiking / von neunmann computing), although we have mostly zeros, energy will constantly
    # be needed to perform operation
    neuromorphic_expected = 2e-9
    von_neunmann_expected = 2e-8
    energies = summary_list.total_energies

    assert len(energies) == 2
    assert pytest.approx(neuromorphic_expected, rel=0.01) == energies[
        0], f"The network consisting of single synapses/neurons " \
            f"was not within tolerance (was {neuromorphic_expected}, " \
            f"is {energies[0]})"
    assert pytest.approx(von_neunmann_expected, rel=0.01) == energies[
        1], f"The network consisting of single synapses/neurons " \
            f"was not within tolerance (was {von_neunmann_expected}, " \
            f"is {energies[0]})"


def test_estimate_energy_network3_for_neuromorphic3():
    """
            Input spikes         :     1  0  0  0  1  0  0  0  0  0
            Spiking activation 1 :     1  0  0  0  1  0  0  0  0  0
            Spiking activation 2 :     1  0  0  0  1  0  0  0  0  0

    """
    inp_test = torch.Tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]]).T
    # create the network (the "device under test")
    v = EnergyEfficiencyNetTest3(beta=0.75)
    v.fc1.weight.data = torch.Tensor([[1]])
    v.fc1.bias.data = torch.Tensor([[0]])
    v.fc2.weight.data = torch.Tensor([[0.5]])
    v.fc2.bias.data = torch.Tensor([[0]])
    v.reset()

    # Debug, see whether the number checks out
    print()
    for t in inp_test:
        spk = torch.stack([t])
        dat = v.forward_full(spk)
        print([dat[idx].item() for idx in range(4)])
    v.reset()

    # on input 3 spikes, in layer 1 we have 4 spikes, in final layer 5 spikes
    # - 3 synapse events (on layer 1)
    # - 4 neuron events in layer 1 activation
    # - 4 synapse events in layer 2
    # - 5 neuron events in layer 2 activation
    DeviceProfileRegistry.add_device("neuromorphic-test3", 1e-9, 1e-9, True)
    DeviceProfileRegistry.add_device("vn-against-neuromorphic-test3", 1e-9, 1e-9, False)

    summary_list = estimate_energy(model=v, input_data=inp_test, devices=["neuromorphic-test3",
                                                                          "vn-against-neuromorphic-test3"],
                                   include_bias_term_in_events=False)

    # see that the numbers differ a lot for devices that are truely spiking or not (by
    # writing on paper, or checking on code, this set of weights will generate only one spike at time t=6)
    # therefore expected energy is 2e-9 (1e-9 for each synapse and 0 for each neuron)
    # in comparison (in non spiking / von neunmann computing), although we have mostly zeros, energy will constantly
    # be needed to perform operation
    neuromorphic_expected = (2 + 2 + 2 + 2) * 1e-9
    von_neunmann_expected = 4 * 10 * 1e-9
    energies = summary_list.total_energies
    print(summary_list)

    assert len(energies) == 2
    assert pytest.approx(neuromorphic_expected, rel=0.01) == energies[
        0], f"The network consisting of single synapses/neurons " \
            f"was not within tolerance (was {neuromorphic_expected}, " \
            f"is {energies[0]})"
    assert pytest.approx(von_neunmann_expected, rel=0.01) == energies[
        1], f"The network consisting of single synapses/neurons " \
            f"was not within tolerance (was {von_neunmann_expected}, " \
            f"is {energies[0]})"


def test_estimate_energy_network3_for_neuromorphic4():
    """
            Input spikes         :     1  0  0  0  1  0  0  0  0  0
            Spiking activation 1 :     1  0  0  0  1  0  0  0  0  0
            Spiking activation 2 :     1  0  0  0  1  0  0  0  0  0

    """
    inp_test = torch.Tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]]).T
    # create the network (the "device under test")
    v = EnergyEfficiencyNetTest3(beta=0.75)
    v.fc1.weight.data = torch.Tensor([[1]])
    v.fc1.bias.data = torch.Tensor([[0]])
    v.fc2.weight.data = torch.Tensor([[0.5]])
    v.fc2.bias.data = torch.Tensor([[0]])
    v.reset()

    # Debug, see whether the number checks out
    print()
    for t in inp_test:
        spk = torch.stack([t])
        dat = v.forward_full(spk)
        print([dat[idx].item() for idx in range(4)])
    v.reset()

    # on input 3 spikes, in layer 1 we have 4 spikes, in final layer 5 spikes
    # - 3 synapse events (on layer 1)
    # - 4 neuron events in layer 1 activation
    # - 4 synapse events in layer 2
    # - 5 neuron events in layer 2 activation
    DeviceProfileRegistry.add_device("neuromorphic-test4", 1e-9, 0, True)
    DeviceProfileRegistry.add_device("vn-against-neuromorphic-test4", 1e-9, 0, False)

    summary_list = estimate_energy(model=v, input_data=inp_test, devices=["neuromorphic-test4",
                                                                          "vn-against-neuromorphic-test4"],
                                   include_bias_term_in_events=False)

    # see that the numbers differ a lot for devices that are truely spiking or not (by
    # writing on paper, or checking on code, this set of weights will generate only one spike at time t=6)
    # therefore expected energy is 2e-9 (1e-9 for each synapse and 0 for each neuron)
    # in comparison (in non spiking / von neunmann computing), although we have mostly zeros, energy will constantly
    # be needed to perform operation
    neuromorphic_expected = (2 + 2) * 1e-9
    von_neunmann_expected = 2 * 10 * 1e-9
    energies = summary_list.total_energies
    print(summary_list)

    assert len(energies) == 2
    assert pytest.approx(neuromorphic_expected, rel=0.01) == energies[
        0], f"The network consisting of single synapses/neurons " \
            f"was not within tolerance (was {neuromorphic_expected}, " \
            f"is {energies[0]})"
    assert pytest.approx(von_neunmann_expected, rel=0.01) == energies[
        1], f"The network consisting of single synapses/neurons " \
            f"was not within tolerance (was {von_neunmann_expected}, " \
            f"is {energies[0]})"


def test_estimate_energy_network3_for_neuromorphic5():
    """
            Input spikes         :     1  0  1  0  1  0  0  0  0  0
            Spiking activation 1 :     1  0  1  1  1  0  0  0  0  0
            Spiking activation 2 :     1  0  1  1  1  1  0  0  0  0
    """
    inp_test = torch.Tensor([[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]]).T
    # create the network (the "device under test")
    v = EnergyEfficiencyNetTest3(beta=0.75)
    v.fc1.weight.data = torch.Tensor([[1]])
    v.fc1.bias.data = torch.Tensor([[0]])
    v.fc2.weight.data = torch.Tensor([[0.5]])
    v.fc2.bias.data = torch.Tensor([[0]])
    v.reset()

    # Debug, see whether the number checks out
    print()
    for t in inp_test:
        spk = torch.stack([t])
        dat = v.forward_full(spk)
        print([dat[idx].item() for idx in range(4)])
    v.reset()

    # on input 3 spikes, in layer 1 we have 4 spikes, in final layer 5 spikes
    # - 3 synapse events (on layer 1)
    # - 4 neuron events in layer 1 activation
    # - 4 synapse events in layer 2
    # - 5 neuron events in layer 2 activation
    DeviceProfileRegistry.add_device("neuromorphic-test5", 1e-9, 1e-9, True)
    DeviceProfileRegistry.add_device("vn-against-neuromorphic-test5", 1e-9, 1e-9, False)

    summary_list = estimate_energy(model=v, input_data=inp_test, devices=["neuromorphic-test5",
                                                                          "vn-against-neuromorphic-test5"],
                                   include_bias_term_in_events=False)

    # see that the numbers differ a lot for devices that are truely spiking or not (by
    # writing on paper, or checking on code, this set of weights will generate only one spike at time t=6)
    # therefore expected energy is 2e-9 (1e-9 for each synapse and 0 for each neuron)
    # in comparison (in non spiking / von neunmann computing), although we have mostly zeros, energy will constantly
    # be needed to perform operation
    neuromorphic_expected = (3 + 4 + 4 + 5) * 1e-9
    von_neunmann_expected = 4 * 10 * 1e-9
    energies = summary_list.total_energies
    print(summary_list)

    assert len(energies) == 2
    assert pytest.approx(neuromorphic_expected, rel=0.01) == energies[
        0], f"The network consisting of single synapses/neurons " \
            f"was not within tolerance (was {neuromorphic_expected}, " \
            f"is {energies[0]})"
    assert pytest.approx(von_neunmann_expected, rel=0.01) == energies[
        1], f"The network consisting of single synapses/neurons " \
            f"was not within tolerance (was {von_neunmann_expected}, " \
            f"is {energies[0]})"


def test_estimate_energy_network3_for_neuromorphic6():
    """
            Input spikes         :     1  0  1  0  1  0  0  0  0  0
            Spiking activation 1 :     1  0  1  1  1  0  0  0  0  0
            Spiking activation 2 :     1  0  1  1  1  1  0  0  0  0
    """
    inp_test = torch.Tensor([[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]]).T
    # create the network (the "device under test")
    v = EnergyEfficiencyNetTest3(beta=0.75)
    v.fc1.weight.data = torch.Tensor([[1]])
    v.fc1.bias.data = torch.Tensor([[0]])
    v.fc2.weight.data = torch.Tensor([[0.5]])
    v.fc2.bias.data = torch.Tensor([[0]])
    v.reset()

    # Debug, see whether the number checks out
    print()
    for t in inp_test:
        spk = torch.stack([t])
        dat = v.forward_full(spk)
        print([dat[idx].item() for idx in range(4)])
    v.reset()

    # on input 3 spikes, in layer 1 we have 4 spikes, in final layer 5 spikes
    # - 3 synapse events (on layer 1)
    # - 4 neuron events in layer 1 activation
    # - 4 synapse events in layer 2
    # - 5 neuron events in layer 2 activation
    DeviceProfileRegistry.add_device("neuromorphic-test6", 1e-9, 0e-9, True)
    DeviceProfileRegistry.add_device("vn-against-neuromorphic-test6", 1e-9, 0e-9, False)

    summary_list = estimate_energy(model=v, input_data=inp_test, devices=["neuromorphic-test6",
                                                                          "vn-against-neuromorphic-test6"],
                                   include_bias_term_in_events=False)

    # see that the numbers differ a lot for devices that are truely spiking or not (by
    # writing on paper, or checking on code, this set of weights will generate only one spike at time t=6)
    # therefore expected energy is 2e-9 (1e-9 for each synapse and 0 for each neuron)
    # in comparison (in non spiking / von neunmann computing), although we have mostly zeros, energy will constantly
    # be needed to perform operation
    neuromorphic_expected = (3 + 4) * 1e-9
    von_neunmann_expected = 2 * 10 * 1e-9
    energies = summary_list.total_energies
    print(summary_list)

    assert len(energies) == 2
    assert pytest.approx(neuromorphic_expected, rel=0.01) == energies[
        0], f"The network consisting of single synapses/neurons " \
            f"was not within tolerance (was {neuromorphic_expected}, " \
            f"is {energies[0]})"
    assert pytest.approx(von_neunmann_expected, rel=0.01) == energies[
        1], f"The network consisting of single synapses/neurons " \
            f"was not within tolerance (was {von_neunmann_expected}, " \
            f"is {energies[0]})"


def test_estimate_energy_network4_for_neuromorphic7():
    """
            CNN test

            pattern 1 (p1) = [[1 0 1]
                              [0 1 0]
                              [1 0 1]]

            pattern 2 (p2) = [[0 1 0]
                             [1 0 1]
                             [0 1 0]]

            Input spikes         :     p1  p2
    """

    p1 = [[1, 0, 0, 0, 1],
          [0, 1, 0, 1, 0],
          [0, 0, 1, 0, 0],
          [0, 1, 0, 1, 0],
          [1, 0, 0, 0, 1]]
    p2 = [[0, 1, 1, 1, 0],
          [1, 0, 0, 0, 1],
          [1, 0, 0, 0, 1],
          [1, 0, 0, 0, 1],
          [0, 1, 1, 1, 0]]

    inp_test = torch.Tensor([p1, p2])
    # inp_test = inp_test.permute(1, 2, 0)

    # create the network (the "device under test")
    v = EnergyEfficiencyNetTest4(beta=0.75)
    v.reset()

    # Debug, see whether the number checks out
    for t in inp_test:
        spk = torch.stack([t])
        dat = v.forward_full(spk)
        # print([dat[idx] for idx in range(4)])
    v.reset()

    DeviceProfileRegistry.add_device("neuromorphic-test7", 1e-9, 0e-9, True)
    DeviceProfileRegistry.add_device("vn-against-neuromorphic-test7", 1e-9, 0e-9, False)
    summary_list = estimate_energy(model=v, input_data=inp_test, devices=["neuromorphic-test7",
                                                                          "vn-against-neuromorphic-test7"],
                                   include_bias_term_in_events=False)

    print(summary_list)
