import numpy as np
import pytest
from snntorch.energy_estimation.estimate_energy import *
from snntorch.energy_estimation.device_profile_registry import *
from tests.conftest import *
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
    print()
    print(summary_list)
    print()

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

    print()
    print(summary_list)
    print()

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

    print()
    print(summary_list)
    print()

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

    print()
    print(summary_list)
    print()

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
    print()
    print(summary_list)
    print()
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
    # print()
    # for t in inp_test:
    #    spk = torch.stack([t])
    #    dat = v.forward_full(spk)
    #    print([dat[idx].item() for idx in range(4)])
    # v.reset()

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
    print()
    print(summary_list)
    print()

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
    # print()
    # for t in inp_test:
    #    spk = torch.stack([t])
    #    dat = v.forward_full(spk)
    #    print([dat[idx].item() for idx in range(4)])
    # v.reset()

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
    print()
    print(summary_list)
    print()

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
    # print()
    # for t in inp_test:
    #    spk = torch.stack([t])
    #    dat = v.forward_full(spk)
    #    print([dat[idx].item() for idx in range(4)])
    # v.reset()

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
    print()
    print(summary_list)
    print()

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
    # print()
    # for t in inp_test:
    #    spk = torch.stack([t])
    #    dat = v.forward_full(spk)
    #    print([dat[idx].item() for idx in range(4)])
    # v.reset()

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

    print()
    print(summary_list)
    print()

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
            Input spikes         :     p1  p2

            pattern 1 (p1) = [[1 0 0 0 1]
                              [0 1 0 1 0]
                              [0 0 1 0 0]
                              [0 1 0 1 0]
                              [1 0 0 0 1]]

            pattern 2 (p2) = [[0 1 1 1 0]
                              [1 0 0 0 1]
                              [1 0 0 0 1]
                              [1 0 0 0 1]
                              [0 1 1 1 0]]


            CNN spike counts for the CNN layer:
            p1 : [[3 3 3]
                  [3 5 3]
                  [3 3 3]]

            total spiking events (sum ) : 29

            p2 :  [[4 3 4]
                   [3 0 3]
                   [4 3 4]]

            total spiking events (sum) : 28

            expected total spiking events in 2D CNN = 29 + 28 = 57


            Total Events (excluding the bias event) :
            - map produces 3 by 3 image (sliding a kernel of size 3x3) : 9 * 9 = 81 events
            - neuron layer : 3 x 3 image, each pixel one event = 9 events
            - dense layer : flattens the input map to size [1, 9] = 9 events
            - neuron layer : 1 event

            2 timesteps, therefore = 2 * (81 + 9 + 9 + 1) = 200 events
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
    v = EnergyEfficiencyNetTest4()
    v.reset()
    v.cnn1.weight.data = torch.Tensor([[[[0.75, 0, 0],
                                         [0, 0.75, 0],
                                         [0, 0, 0.75]]]])
    v.cnn1.bias.data = torch.Tensor([0])
    v.fc2.weight.data = torch.Tensor([[0, 0, 0, 0, 2, 0, 0, 0, 0]])
    v.fc2.bias.data = torch.Tensor([0])

    DeviceProfileRegistry.add_device("neuromorphic-test8", 1e-9, 1e-9, True)
    DeviceProfileRegistry.add_device("vn-against-neuromorphic-test8", 1e-9, 1e-9, False)
    summary_list = estimate_energy(model=v, input_data=inp_test, include_bias_term_in_events=False)

    # check whether the number of total/spiking events is correct for root layer
    assert summary_list.summary_list[0].spiking_events == 68
    assert summary_list.summary_list[0].total_events == 200

    # check whether the number of total/spiking events is correct for first CNN layer
    assert summary_list.summary_list[1].spiking_events == (29 + 28)
    assert summary_list.summary_list[1].total_events == 2 * (9 * 9)

    # check whether the number of total/spiking events is correct for first neuron layer
    assert summary_list.summary_list[2].spiking_events == 5
    assert summary_list.summary_list[2].total_events == 2 * 9

    # check whether the number of total/spiking events is correct for dense layer
    assert summary_list.summary_list[3].spiking_events == 5
    assert summary_list.summary_list[3].total_events == 2 * 9

    # check whether the total/spiking events is correct for classification layer
    assert summary_list.summary_list[4].spiking_events == 1
    assert summary_list.summary_list[4].total_events == 2


def test_network_large():
    """
        Compare a large network to equivalent nengo keras_spiking network.
        Below is the keras equivalent of pytorch network. make sure that the energy estimates and total
        event counts are the same in comparison to nengo

            # build the equivalent model
            inp = x = tf.keras.Input((1, 32, 32, 1))
            x = tf.keras.layers.Conv2D(16, kernel_size=3, padding="same")(x)
            x = keras_spiking.SpikingActivation("relu")(x)
            x = tf.keras.layers.Conv2D(16, kernel_size=5, dilation_rate=2, padding="valid")(x)
            x = keras_spiking.SpikingActivation("relu")(x)
            x = tf.keras.layers.Conv2D(16, kernel_size=7, dilation_rate=3, padding="valid")(x)
            x = keras_spiking.SpikingActivation("relu")(x)
            dense = tf.keras.layers.Reshape((1, 6 * 6 * 16))(x)
            dense = tf.keras.layers.Dense(32)(dense)
            dense = keras_spiking.SpikingActivation("relu")(dense)
            dense = tf.keras.layers.Dense(10)(dense)
            dense = keras_spiking.SpikingActivation("relu")(dense)
            model = tf.keras.Model(inp, dense)

            # estimate model energy
            energy = keras_spiking.ModelEnergy(model)
            energy.summary(print_warnings=False, columns=("name", "output_shape", "params", "connections", "neurons", "energy cpu"))

        Output:
            Layer (type)                           |Output shape         |Param #|Conn # |Neuron #|J/inf (cpu)
            ---------------------------------------|---------------------|-------|-------|--------|-----------
            input_1 (InputLayer)                   |[(None, 1, 32, 32, 1)|      0|      0|       0|          0
            conv2d (Conv2D)                        |(None, 1, 32, 32, 16)|    160| 147456|       0|     0.0013
            spiking_activation (SpikingActivation) |(None, 1, 32, 32, 16)|      0|      0|   16384|    0.00014
            conv2d_1 (Conv2D)                      |(None, 1, 24, 24, 16)|   6416|3686400|       0|      0.032
            spiking_activation_1 (SpikingActivation|(None, 1, 24, 24, 16)|      0|      0|    9216|    7.9e-05
            conv2d_2 (Conv2D)                      |  (None, 1, 6, 6, 16)|  12560| 451584|       0|     0.0039
            spiking_activation_2 (SpikingActivation|  (None, 1, 6, 6, 16)|      0|      0|     576|      5e-06
            reshape (Reshape)                      |       (None, 1, 576)|      0|      0|       0|          0
            dense (Dense)                          |        (None, 1, 32)|  18464|  18432|       0|    0.00016
            spiking_activation_3 (SpikingActivation|        (None, 1, 32)|      0|      0|      32|    2.8e-07
            dense_1 (Dense)                        |        (None, 1, 10)|    330|    320|       0|    2.8e-06
            spiking_activation_4 (SpikingActivation|        (None, 1, 10)|      0|      0|      10|    8.6e-08
            ==================================================================================================
            Total energy per inference [Joules/inf] (cpu): 3.72e-02

    """
    p1 = np.ones((1, 32, 32))

    inp_test = torch.Tensor([p1])
    v = EnergyEfficiencyNetTest5Large()
    v.reset()

    summary_list = estimate_energy(model=v, input_data=inp_test, include_bias_term_in_events=False)

    # assert total counts
    assert summary_list.summary_list[0].total_events == (147456 + 16384 + 3686400 + 9216 + 451584 + 576 + 18432 + 32 +
                                                         320 + 10)

    # assert the total count for first convolutional layer
    assert summary_list.summary_list[1].total_events == 147456

    # assert the total count for first spiking layer
    assert summary_list.summary_list[2].total_events == 16384

    # assert the total count for second convolutional layer
    assert summary_list.summary_list[3].total_events == 3686400

    # assert the total count for second spiking layer
    assert summary_list.summary_list[4].total_events == 9216

    # assert the total count for third convolutional layer
    assert summary_list.summary_list[5].total_events == 451584

    # assert the total count for third spiking layer
    assert summary_list.summary_list[6].total_events == 576

    # assert the total count for first dense layer
    assert summary_list.summary_list[7].total_events == 18432

    # assert the total count for fourth neuron spiking layer
    assert summary_list.summary_list[8].total_events == 32

    # assert the total count for second dense layer (classification head)
    assert summary_list.summary_list[9].total_events == 320

    # assert the total count for fifth neuron spiking layer ( classification head)
    assert summary_list.summary_list[10].total_events == 10

    # assert that the total energy calculated by this new module is roughly the same as calculated in nengo
    energies = summary_list.total_energies
    assert len(energies) == 1

    assert pytest.approx(3.72e-02, rel=0.01) == energies[0]
