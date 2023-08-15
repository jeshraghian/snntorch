import numpy as np
import pytest
from snntorch.energy_estimation.estimate_energy import *
from snntorch.energy_estimation.device_profile_registry import *
from snntorch.energy_estimation.layer_parameter_event_calculator import (LayerParameterEventCalculator,
                                                                         synapse_neuron_count_for_linear,
                                                                         count_events_for_linear)
from tests.conftest import *
import torch


def test_estimate_energy_network1():
    """
        Simplest network test :
            - non-neuromorphic architecture (cpu) with energy 8.6e-9 per synapse/neuron event

            - 2 Layers, simplest architecture : ( not including bias in calculations )

                    self.fc1 = nn.Linear(1, 1)              # 1 synaptic event (2 with bias)
                    self.lif1 = snn.Leaky(beta=beta)        # 1 neuron event
                    self.fc2 = nn.Linear(1, 1)              # 1 synaptic event (2 with bias)
                    self.lif2 = snn.Leaky(beta=beta)        # 1 neuron event

        Tests :
            - assert that energy returned is to expected value (without bias 4 * 8.6e-9 = 3.44e-8, with bias 5.16e-8)
            - assert that the synapse/neuron/total counts are as expected
    """

    # Prepare the input data
    # here
    num_values = 1
    percentage_of_ones = 0.
    ones_num = round(percentage_of_ones * num_values)
    zeros_num = num_values - ones_num
    inp_test = torch.Tensor([[[1]]] * ones_num + [[[0]]] * zeros_num)

    # create the network (the "device under test")
    v = EnergyEfficiencyNetTest1(beta=0.9, num_timesteps=inp_test.size(0))

    # register the device
    energy_per_synapse_event = 8.6e-9
    energy_per_neuron_event = 8.6e-9
    DeviceProfileRegistry.add_device("cpu-test1", energy_per_synapse_event,
                                     energy_per_neuron_event,
                                     False)

    # get the summary
    summary_list = estimate_energy(model=v, input_data=inp_test, devices="cpu-test1",
                                   include_bias_term_in_events=False)
    # Expected energy values
    expected_value_without_bias = 2 * energy_per_synapse_event + 2 * energy_per_neuron_event  # 3.44e-8
    expected_value_with_bias = 4 * energy_per_synapse_event + 2 * energy_per_neuron_event  # 5.16e-8

    # get the energies
    energies = summary_list.total_energies

    # only one device specified, therefore length of energies should be 1, and equal to `expected_value_without_bias`,
    # assert that with 1% tolerance
    assert len(energies) == 1
    assert pytest.approx(expected_value_without_bias, rel=0.01) == energies[0]

    # assert the expected numbers of synapses / neurons / total events
    # topmost layer, the EnergyEfficiencyNetTest1
    assert summary_list.summary_list[0].total_events == 4  # 4 total events
    assert summary_list.summary_list[0].synapse_count == 4  # 4 total synapses (2 for weights, 2 for biases)
    assert summary_list.summary_list[0].neuron_count == 2  # 2 neuron events

    assert summary_list.summary_list[1].total_events == 1  # 1 synapse event (bias is not included)
    assert summary_list.summary_list[1].synapse_count == 2  # 2 synapses (single weight + bias)
    assert summary_list.summary_list[1].neuron_count == 0  # 0 neurons (layer is Linear (synaptic), no neurons)

    assert summary_list.summary_list[2].total_events == 1  # 1 synapse event (bias is not included)
    assert summary_list.summary_list[2].synapse_count == 0  # 0 synapses (layer is neuron, no synapses)
    assert summary_list.summary_list[2].neuron_count == 1  # 1 neurons (layer is a neuron, output shape is 1, one event)

    assert summary_list.summary_list[3].total_events == 1  # 1 synapse event (bias is not included)
    assert summary_list.summary_list[3].synapse_count == 2  # 2 synapses (single weight + bias)
    assert summary_list.summary_list[3].neuron_count == 0  # 0 neurons (layer is Linear (synaptic), no neurons)

    assert summary_list.summary_list[4].total_events == 1  # 1 synapse event (bias is not included)
    assert summary_list.summary_list[4].synapse_count == 0  # 0 synapses (layer is neuron, no synapses)
    assert summary_list.summary_list[4].neuron_count == 1  # 1 neurons (layer is a neuron, output shape is 1, one event)

    # now get the summary with bias included in it's calculations
    summary_list_bias = estimate_energy(model=v, input_data=inp_test, devices="cpu-test1",
                                        include_bias_term_in_events=True)
    energies = summary_list_bias.total_energies

    # assert now the same thing, but when bias is included
    assert summary_list_bias.summary_list[0].total_events == 6  # 4 total events
    assert summary_list_bias.summary_list[0].synapse_count == 4  # 4 total synapses (2 for weights, 2 for biases)
    assert summary_list_bias.summary_list[0].neuron_count == 2  # 2 neuron events

    assert summary_list_bias.summary_list[1].total_events == 2  # 1 synapse event (bias is not included)
    assert summary_list_bias.summary_list[1].synapse_count == 2  # 2 synapses (single weight + bias)
    assert summary_list_bias.summary_list[1].neuron_count == 0  # 0 neurons (layer is Linear (synaptic), no neurons)

    assert summary_list_bias.summary_list[2].total_events == 1  # 1 synapse event (bias is not included)
    assert summary_list_bias.summary_list[2].synapse_count == 0  # 0 synapses (layer is neuron, no synapses)
    assert summary_list_bias.summary_list[
               2].neuron_count == 1  # 1 neurons (layer is a neuron, output shape is 1, one event)

    assert summary_list_bias.summary_list[3].total_events == 2  # 1 synapse event (bias is not included)
    assert summary_list_bias.summary_list[3].synapse_count == 2  # 2 synapses (single weight + bias)
    assert summary_list_bias.summary_list[3].neuron_count == 0  # 0 neurons (layer is Linear (synaptic), no neurons)

    assert summary_list_bias.summary_list[4].total_events == 1  # 1 synapse event (bias is not included)
    assert summary_list_bias.summary_list[4].synapse_count == 0  # 0 synapses (layer is neuron, no synapses)
    assert summary_list_bias.summary_list[
               4].neuron_count == 1  # 1 neurons (layer is a neuron, output shape is 1, one event)

    # only one device specified, therefore length of energies should be 1, and equal to `expected_value_with_bias`,
    # assert that with 1% tolerance
    assert len(energies) == 1
    assert pytest.approx(expected_value_with_bias, rel=0.01) == energies[0]


def test_estimate_energy_network2():
    """
        More complicated architecture, testing the same things as test1 but

            - non-neuromorphic architecture (cpu) with energy 8.6e-9 per synapse/neuron event

            - 2 Layers architecture, much more events in comparison to test 1: ( not including bias in calculations )

            self.fc1 = nn.Linear(32, 64)
            self.lif1 = snn.Leaky(beta=beta, threshold=0.2)
            self.fc2 = nn.Linear(64, 10)
            self.lif2 = snn.Leaky(beta=beta, threshold=0.4)

        Same code for keras (and nengo estimations)

            # build an example model
            inp = x = tf.keras.Input((1, 32))
            x = tf.keras.layers.Dense(64)(x)
            x = keras_spiking.SpikingActivation("relu")(x)
            x = tf.keras.layers.Dense(10)(x)
            x = keras_spiking.SpikingActivation("relu")(x)

            model = tf.keras.Model(inp, x)
            # estimate model energy
            energy = keras_spiking.ModelEnergy(model)
            energy.summary(print_warnings=False, columns=("name", "output_shape", "params", "connections", "neurons",
                            "energy cpu-test3", "energy cpu"))

            Layer (type)                     |Output shape |Param #|Conn #|Neuron #|J/inf (cpu-tes|J/inf (cpu)
            ---------------------------------|-------------|-------|------|--------|--------------|-----------
            input_3 (InputLayer)             |[(None, 1, 32|      0|     0|       0|             0|          0
            dense_4 (Dense)                  |(None, 1, 64)|   2112|  2048|       0|         2e-05|    1.8e-05
            spiking_activation_4 (SpikingActi|(None, 1, 64)|      0|     0|      64|       6.4e-08|    5.5e-07
            dense_5 (Dense)                  |(None, 1, 10)|    650|   640|       0|       6.4e-06|    5.5e-06
            spiking_activation_5 (SpikingActi|(None, 1, 10)|      0|     0|      10|         1e-08|    8.6e-08
            ==================================================================================================
            Total energy per inference [Joules/inf] (cpu-test3): 2.70e-05
            Total energy per inference [Joules/inf] (cpu): 2.38e-05
    """

    # prepare the inputs
    batch_size = 1
    prob_of_one = 1
    prob_matrix = prob_of_one * torch.ones((1, batch_size, 32))
    inp_test = torch.bernoulli(prob_matrix)

    # create the network (the "device under test")
    energy_per_synapse_event = 8.6e-9
    energy_per_neuron_event = 8.6e-9
    DeviceProfileRegistry.add_device("cpu-test2", energy_per_synapse_event,
                                     energy_per_neuron_event,
                                     False)
    v = EnergyEfficiencyNetTest2(beta=0.9)

    # get the layer summaries
    summary_list = estimate_energy(model=v, input_data=inp_test, devices="cpu-test2",
                                   include_bias_term_in_events=False)

    # nengo calculated value + as expected from simple calculations
    expected_value = (2048 + 640) * energy_per_synapse_event + (64 + 10) * energy_per_neuron_event

    energies = summary_list.total_energies
    # calculate it for only one device
    assert len(energies) == 1
    assert pytest.approx(expected_value, rel=0.01) == energies[0], f"The energy estimate for simple network 2 " \
                                                                   f"was not within tolerance (was {expected_value}, " \
                                                                   f"is {energies[0]})"

    #                                                   all of these numbers can be simplified, but written explicitly
    #                                                   to show that they make sense (I think)
    assert summary_list.summary_list[0].synapse_count == 2112 + 650
    assert summary_list.summary_list[0].neuron_count == 64 + 10
    assert summary_list.summary_list[0].total_events == (2112 - 64) + (650 - 10) + 64 + 10  # with bias removed

    assert summary_list.summary_list[1].synapse_count == 2112
    assert summary_list.summary_list[1].neuron_count == 0
    assert summary_list.summary_list[1].total_events == 2112 - 64  # synapse count - bias events

    assert summary_list.summary_list[2].synapse_count == 0
    assert summary_list.summary_list[2].neuron_count == 64
    assert summary_list.summary_list[2].total_events == 64

    assert summary_list.summary_list[3].synapse_count == 650
    assert summary_list.summary_list[3].neuron_count == 0
    assert summary_list.summary_list[3].total_events == 650 - 10  # synapse count - bias events

    assert summary_list.summary_list[4].synapse_count == 0
    assert summary_list.summary_list[4].neuron_count == 10
    assert summary_list.summary_list[4].total_events == 10

    # now assert the same for when bias is included
    summary_list = estimate_energy(model=v, input_data=inp_test, devices="cpu-test2",
                                   include_bias_term_in_events=True)

    # nengo calculated value + as expected from simple calculations (but now bias term is also included)
    expected_value = (2112 + 650) * energy_per_synapse_event + (64 + 10) * energy_per_neuron_event

    energies = summary_list.total_energies
    # calculate it for only one device
    assert len(energies) == 1
    assert pytest.approx(expected_value, rel=0.01) == energies[0], f"The energy estimate for simple network 2 " \
                                                                   f"was not within tolerance (was {expected_value}, " \
                                                                   f"is {energies[0]})"

    #                                                   all of these numbers can be simplified, but written explicitly
    #                                                   to show that they make sense (I think)
    assert summary_list.summary_list[0].synapse_count == 2112 + 650
    assert summary_list.summary_list[0].neuron_count == 64 + 10
    assert summary_list.summary_list[0].total_events == 2112 + 650 + 64 + 10

    assert summary_list.summary_list[1].synapse_count == 2112
    assert summary_list.summary_list[1].neuron_count == 0
    assert summary_list.summary_list[1].total_events == 2112

    assert summary_list.summary_list[2].synapse_count == 0
    assert summary_list.summary_list[2].neuron_count == 64
    assert summary_list.summary_list[2].total_events == 64

    assert summary_list.summary_list[3].synapse_count == 650
    assert summary_list.summary_list[3].neuron_count == 0
    assert summary_list.summary_list[3].total_events == 650

    assert summary_list.summary_list[4].synapse_count == 0
    assert summary_list.summary_list[4].neuron_count == 10
    assert summary_list.summary_list[4].total_events == 10


def test_estimate_energy_network2_energies_for_two_new_devices():
    """
        Same architecture as above (EnergyEfficiencyNetTest2), but we create two different devices with two energy
        values. We check whether the numbers reported for these devices are as expected

            - non-neuromorphic architecture (cpu) with energy 8.6e-9 per synapse/neuron event
            self.fc1 = nn.Linear(32, 64)
            self.lif1 = snn.Leaky(beta=beta, threshold=0.2)
            self.fc2 = nn.Linear(64, 10)
            self.lif2 = snn.Leaky(beta=beta, threshold=0.4)

        Using information from previous test / paper calculations :
            Total synaptic events (without bias)  : 2048 + 640 = 2688
            Total neuron events (without bias) : 64 + 10 = 74

            Total synaptic events (with bias)  : 2112 + 650 = 2762
            Total neuron events (with bias) : 64 + 10
    """

    # prepare the input
    batch_size = 1
    prob_of_one = 1
    prob_matrix = prob_of_one * torch.ones((1, batch_size, 32))
    inp_test = torch.bernoulli(prob_matrix)

    # create the devices
    device1_en_per_synapse, device1_en_per_neuron = 10e-9, 1e-9
    device2_en_per_synapse, device2_en_per_neuron = 1e-9, 10e-9
    DeviceProfileRegistry.add_device("cpu-test3", device1_en_per_synapse,
                                     device1_en_per_neuron,
                                     False)
    DeviceProfileRegistry.add_device("cpu-test4", device2_en_per_synapse
                                     , device2_en_per_neuron,
                                     False)

    # create the network (the "device under test")
    v = EnergyEfficiencyNetTest2(beta=0.9)

    # create the summary list
    summary_list = estimate_energy(model=v, input_data=inp_test, devices=["cpu-test3", "cpu-test4"],
                                   include_bias_term_in_events=False)
    energies = summary_list.total_energies

    # calculate it for two devices
    expected_values = [None, None]
    expected_values[0] = 2688 * device1_en_per_synapse + 74 * device1_en_per_neuron
    expected_values[1] = 2688 * device2_en_per_synapse + 74 * device2_en_per_neuron

    # check whether the returned energies are as expected
    assert len(energies) == 2
    assert pytest.approx(expected_values[0], rel=0.01) == energies[0]
    assert pytest.approx(expected_values[1], rel=0.01) == energies[1]

    # now does it but include bias

    # create the summary list
    summary_list = estimate_energy(model=v, input_data=inp_test, devices=["cpu-test3", "cpu-test4"],
                                   include_bias_term_in_events=True)
    energies = summary_list.total_energies

    # calculate it for two devices
    expected_values = [None, None]
    expected_values[0] = 2762 * device1_en_per_synapse + 74 * device1_en_per_neuron
    expected_values[1] = 2762 * device2_en_per_synapse + 74 * device2_en_per_neuron

    # check whether the returned energies are as expected
    assert len(energies) == 2
    assert pytest.approx(expected_values[0], rel=0.01) == energies[0]
    assert pytest.approx(expected_values[1], rel=0.01) == energies[1]


def test_estimate_energy_network2_energies_for_batch_size_and_timesteps():
    """
        Check whether the energy/synapses/neuron counts scales linearly with batch_size
        and check if the energy scales linearly with number of timesteps (for non-neuromorphic)
    """

    # create the devices
    device1_en_per_synapse, device1_en_per_neuron = 1e-9, 1e-9
    DeviceProfileRegistry.add_device("cpu-test-bs", device1_en_per_synapse,
                                     device1_en_per_neuron,
                                     False)

    en = 2.762e-06
    ts, bs = 1, 1
    inp_test = torch.zeros((ts, bs, 32))

    # create the network (the "device under test")
    v = EnergyEfficiencyNetTest2(beta=0.9)

    # create the summary list
    summary_list = estimate_energy(model=v, input_data=inp_test, devices=["cpu-test-bs"],
                                   include_bias_term_in_events=False)
    data = summary_list.summary_list

    # get the energy
    base_counts = [(data[i].synapse_count, data[i].neuron_count, data[i].total_events) for i in range(0, 5)]
    base_energy = summary_list.total_energies[0]

    numbers_to_test = [1, 3, 5, 7, 11, 13]

    for ts in numbers_to_test:
        for bs in numbers_to_test:
            inp_test = torch.zeros((ts, bs, 32))
            net = EnergyEfficiencyNetTest2(beta=0.9)
            summary_list = estimate_energy(model=net, input_data=inp_test, devices=["cpu-test-bs"],
                                           include_bias_term_in_events=False)

            # energy should scale linearly with timesteps (for non-neuromorphic) and batch_size (assert that)
            assert pytest.approx(ts * bs * base_energy, rel=0.01) == summary_list.total_energies[0]

            # total events should scale linearly with timesteps (for non-neuromorphic) and batch_size (assert that)
            # synapse / neuron count should scale linearly with batch_size
            for i in range(5):
                # assert the synapse count
                assert summary_list.summary_list[i].synapse_count == bs * base_counts[i][0]

                # assert the neuron count
                assert summary_list.summary_list[i].neuron_count == bs * base_counts[i][1]

                # assert the total events count
                assert summary_list.summary_list[i].total_events == ts * bs * base_counts[i][2]


def test_estimate_energy_network3_for_neuromorphic():
    """
        First neuromorphic architecture test

        Network is s modified EnergyEfficiencyNetTest1. It implements EnergyEstimationNetworkInterface, which makes it
        easier to reset the network. It also implements "forward_full", which returns all the spikes, which makes it
        slightly easier to test.

        Input            : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        1st Membrane     : [0, 0, 0, 0, 0, 1, 0.25, 0.0625, 0.03125]
        1st spikes       : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        2nd Membrane     : [0, 0, 0, 0, 0, 0.5, 0.125, 0.0625, 0.03125, 0.015625]
        2nd spikes       : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

        There should be only 4 events : 2 synapses and 2 neuron activations
    """

    # prepare the input spike train
    inp_test = torch.Tensor([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]).T
    # create the network (the "device under test")
    v = EnergyEfficiencyNetTest3(beta=0.5)
    v.fc1.weight.data = torch.Tensor([[1]])
    v.fc1.bias.data = torch.Tensor([[0]])
    v.fc2.weight.data = torch.Tensor([[0.5]])
    v.fc2.bias.data = torch.Tensor([[0]])
    v.reset()

    # Debug, see whether the number checks out
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

    # assert that the energies are as expected
    assert len(energies) == 2
    assert pytest.approx(neuromorphic_expected, rel=0.01) == energies[0]
    assert pytest.approx(von_neunmann_expected, rel=0.01) == energies[1]

    data = summary_list.summary_list

    # assert the numbers for synapses/neurons/total events/spiking events
    assert data[0].synapse_count == 4
    assert data[0].neuron_count == 2
    assert data[0].spiking_events == 4
    assert data[0].total_events == 40

    assert data[1].synapse_count == 2
    assert data[1].neuron_count == 0
    assert data[1].spiking_events == 1
    assert data[1].total_events == 10

    assert data[2].synapse_count == 0
    assert data[2].neuron_count == 1
    assert data[2].spiking_events == 1
    assert data[2].total_events == 10

    assert data[3].synapse_count == 2
    assert data[3].neuron_count == 0
    assert data[3].spiking_events == 1
    assert data[3].total_events == 10


def test_estimate_energy_network3_for_neuromorphic2():
    """
        Network is s modified EnergyEfficiencyNetTest1 (Simplest possible). It implements EnergyEstimationNetworkInterface,
        which makes it easier to reset the network. It also implements "forward_full", which returns all the spikes,
        which makes it slightly easier to test.

        Input            : [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        1st spikes       : [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        2nd spikes       : [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]

        There should be only 4 events : 2 synapses and 2 neuron activations
    """

    """
            Input spikes         :     1  0  0  0  1  0  0  0  0  0
            Spiking activation 1 :     1  0  0  0  1  0  0  0  0  0
            Spiking activation 2 :     1  0  0  0  1  0  0  0  0  0

    """
    # prepare the input
    inp_test = torch.Tensor([[[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]]]).T

    # create the network (the "device under test")
    v = EnergyEfficiencyNetTest3(beta=0)
    v.fc1.weight.data = torch.Tensor([[1]])
    v.fc1.bias.data = torch.Tensor([[0]])
    v.fc2.weight.data = torch.Tensor([[1]])
    v.fc2.bias.data = torch.Tensor([[0]])
    v.reset()

    DeviceProfileRegistry.add_device("neuromorphic-test2", 1e-9, 1e-9, True)
    DeviceProfileRegistry.add_device("vn-against-neuromorphic-test2", 1e-9, 1e-9, False)

    summary_list = estimate_energy(model=v, input_data=inp_test, devices=["neuromorphic-test2",
                                                                          "vn-against-neuromorphic-test2"],
                                   include_bias_term_in_events=False)

    # see that the numbers differ a lot for devices that are truely spiking or not (by
    # writing on paper, or checking on code, this set of weights will generate spikes at time t = 0 and t = 5)
    # therefore expected energy is 8e-9 (1e-9 for each synapse and 1e-9 for each neuron)
    neuromorphic_expected = (2 + 2 + 2 + 2) * 1e-9
    von_neunmann_expected = 4 * 10 * 1e-9
    energies = summary_list.total_energies

    assert len(energies) == 2
    assert pytest.approx(neuromorphic_expected, rel=0.01) == energies[0]
    assert pytest.approx(von_neunmann_expected, rel=0.01) == energies[1]

    # go through all the layers and assert the expected values
    # assert the numbers for synapses/neurons/total events/spiking events
    data = summary_list.summary_list
    assert data[0].synapse_count == 4
    assert data[0].neuron_count == 2
    assert data[0].spiking_events == 8
    assert data[0].total_events == 40

    assert data[1].synapse_count == 2
    assert data[1].neuron_count == 0
    assert data[1].spiking_events == 2
    assert data[1].total_events == 10

    assert data[2].synapse_count == 0
    assert data[2].neuron_count == 1
    assert data[2].spiking_events == 2
    assert data[2].total_events == 10

    assert data[3].synapse_count == 2
    assert data[3].neuron_count == 0
    assert data[3].spiking_events == 2
    assert data[3].total_events == 10


def test_estimate_energy_network3_for_neuromorphic3():
    """
        Network is s modified EnergyEfficiencyNetTest1 (Simplest possible). It implements EnergyEstimationNetworkInterface,
        which makes it easier to reset the network. It also implements "forward_full", which returns all the spikes,
        which makes it slightly easier to test.

            Input spikes         :     1  0  1  0  1  0  0  0  0  0
            Spiking activation 1 :     1  0  1  1  1  0  0  0  0  0
            Spiking activation 2 :     1  0  1  1  1  1  0  0  0  0
    """

    # prepare the input
    inp_test = torch.Tensor([[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]]).T

    # create the network (the "device under test")
    v = EnergyEfficiencyNetTest3(beta=0.75)
    v.fc1.weight.data = torch.Tensor([[1]])
    v.fc1.bias.data = torch.Tensor([[0]])
    v.fc2.weight.data = torch.Tensor([[0.5]])
    v.fc2.bias.data = torch.Tensor([[0]])
    v.reset()

    DeviceProfileRegistry.add_device("neuromorphic-test3", 1e-9, 1e-9, True)
    DeviceProfileRegistry.add_device("vn-against-neuromorphic-test3", 1e-9, 1e-9, False)

    summary_list = estimate_energy(model=v, input_data=inp_test, devices=["neuromorphic-test3",
                                                                          "vn-against-neuromorphic-test3"],
                                   include_bias_term_in_events=False)

    # on input 3 spikes, in layer 1 we have 4 spikes, in final layer 5 spikes
    # - 3 synapse events (on layer 1)
    # - 4 neuron events in layer 1 activation
    # - 4 synapse events in layer 2
    # - 5 neuron events in layer 2 activation
    neuromorphic_expected = (3 + 4 + 4 + 5) * 1e-9
    von_neunmann_expected = 4 * 10 * 1e-9
    energies = summary_list.total_energies

    # check that the calculated energies are the same as expected
    assert len(energies) == 2
    assert pytest.approx(neuromorphic_expected, rel=0.01) == energies[0]
    assert pytest.approx(von_neunmann_expected, rel=0.01) == energies[1]

    # go through all the layers and assert the expected values
    # assert the numbers for synapses/neurons/total events/spiking events
    data = summary_list.summary_list
    assert data[0].spiking_events == 3 + 4 + 4 + 5
    assert data[0].total_events == 40

    assert data[1].spiking_events == 3
    assert data[1].total_events == 10

    assert data[2].spiking_events == 4
    assert data[2].total_events == 10

    assert data[3].spiking_events == 4
    assert data[3].total_events == 10

    assert data[4].spiking_events == 5
    assert data[4].total_events == 10


def test_estimate_energy_network4_for_neuromorphic4():
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

    inp_test = torch.Tensor([[p1], [p2]])
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
    summary_list = estimate_energy(model=v, input_data=inp_test, include_bias_term_in_events=False,
                                   network_requires_first_dim_as_time=False)

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

    inp_test = torch.Tensor(np.array([[p1]]))
    v = EnergyEfficiencyNetTest5Large()
    v.reset()

    # it's important to set the network_requires_first_dim_as_time to False : the 2D convolutions does not accept the
    # shape of (time, batch, W, H, C), therefore we want to use the input as (batch, W, H, C), which is done by setting
    # network_requires_first_dim_as_time = False
    summary_list = estimate_energy(model=v, input_data=inp_test, include_bias_term_in_events=False,
                                   network_requires_first_dim_as_time=False)

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


def test_network_1D_conv():
    """
        Compare a large network to equivalent nengo keras_spiking network, do it now for 1D convolutions.
        Below is the keras equivalent of pytorch network. make sure that the energy estimates and total
        event counts are the same in comparison to nengo

            # build an example model
            inp = x = tf.keras.Input((1, 32, 1))
            x = tf.keras.layers.Conv1D(16, kernel_size=3, padding="same")(x)
            x = keras_spiking.SpikingActivation("relu")(x)
            x = tf.keras.layers.Conv1D(16, kernel_size=5, padding="valid")(x)
            x = keras_spiking.SpikingActivation("relu")(x)
            dense = tf.keras.layers.Reshape((1, 16 * 28))(x)
            dense = tf.keras.layers.Dense(64)(dense)
            dense = keras_spiking.SpikingActivation("relu")(dense)
            dense = tf.keras.layers.Dense(1)(dense)
            dense = keras_spiking.SpikingActivation("relu")(dense)

            model = tf.keras.Model(inp, dense)

            # estimate model energy
            energy = keras_spiking.ModelEnergy(model)
            energy.summary(print_warnings=False, columns=("name", "output_shape", "params", "connections", "neurons", "energy cpu"))

        Output:
             Layer (type)                            |Output shape      |Param #|Conn #|Neuron #|J/inf (cpu)
            ----------------------------------------|------------------|-------|------|--------|-----------
            input_3 (InputLayer)                    |[(None, 1, 32, 1)]|      0|     0|       0|          0
            conv1d_1 (Conv1D)                       | (None, 1, 32, 16)|     64|  1536|       0|    1.3e-05
            spiking_activation_6 (SpikingActivation)| (None, 1, 32, 16)|      0|     0|     512|    4.4e-06
            conv1d_2 (Conv1D)                       | (None, 1, 28, 16)|   1296| 35840|       0|    0.00031
            spiking_activation_7 (SpikingActivation)| (None, 1, 28, 16)|      0|     0|     448|    3.9e-06
            reshape_1 (Reshape)                     |    (None, 1, 448)|      0|     0|       0|          0
            dense_2 (Dense)                         |     (None, 1, 64)|  28736| 28672|       0|    0.00025
            spiking_activation_8 (SpikingActivation)|     (None, 1, 64)|      0|     0|      64|    5.5e-07
            dense_3 (Dense)                         |      (None, 1, 1)|     65|    64|       0|    5.5e-07
            spiking_activation_9 (SpikingActivation)|      (None, 1, 1)|      0|     0|       1|    8.6e-09
            ===============================================================================================
            Total energy per inference [Joules/inf] (cpu): 5.77e-04
    """
    p1 = np.ones((1, 32))

    inp_test = torch.Tensor(np.array([p1]))
    v = EnergyEfficiencyNet1DConv()
    v.reset()

    summary_list = estimate_energy(model=v, input_data=inp_test, include_bias_term_in_events=False,
                                   network_requires_first_dim_as_time=False)

    # assert total counts
    assert summary_list.summary_list[0].total_events == (1536 + 512 + 35840 + 448 + 28672 + 64 + 64 + 1)

    # assert the total count for first convolutional layer
    assert summary_list.summary_list[1].total_events == 1536

    # assert the total count for first spiking layer
    assert summary_list.summary_list[2].total_events == 512

    # assert the total count for second convolutional layer
    assert summary_list.summary_list[3].total_events == 35840

    # assert the total count for second spiking layer
    assert summary_list.summary_list[4].total_events == 448

    # assert the total count for first dense layer
    assert summary_list.summary_list[5].total_events == 28672

    # assert the total count for third spiking layer
    assert summary_list.summary_list[6].total_events == 64

    # assert the total count for second dense layer (classification)
    assert summary_list.summary_list[7].total_events == 64

    # assert the total count for fourth neuron spiking layer ( classification)
    assert summary_list.summary_list[8].total_events == 1

    # assert that the total energy calculated by this new module is roughly the same as calculated in nengo
    energies = summary_list.total_energies
    assert len(energies) == 1

    assert pytest.approx(5.77e-04, rel=0.01) == energies[0]


def test_network_unrecognized_leaf_layers():
    """
        Test how the network handles the unrecognized layers : for these, it should set the value to None, which will
        be displayed as '?' in the print command.

        Later test registering this new layer (which is a Linear layer underneath)
    """
    p1 = np.ones((1, 128))

    inp_test = torch.Tensor(np.array([p1]))
    v = EnergyEfficiencyNetUnrecognizedLayers()
    v.reset()

    # do a forward pass, where we have a "UnrecognizedLeafTestLayer" (which under the hood is linear layer)
    # the values for these layers should be unrecognized (None). Assert that these values are None
    summary_list = estimate_energy(model=v, input_data=inp_test, include_bias_term_in_events=False)

    assert summary_list.summary_list[0].neuron_count == 32 + 16
    # TODO : should None here be propagated or not ?
    assert summary_list.summary_list[0].synapse_count == 0

    # First layer (the unrecognized).Doesn't have a callbacks, therefore synapse, neuron, total, spiking values are None
    assert summary_list.summary_list[1].neuron_count is None
    assert summary_list.summary_list[1].synapse_count is None
    assert summary_list.summary_list[1].total_events is None
    assert summary_list.summary_list[1].spiking_events is None

    # second layer (SNN spiking neurons). Here expect the neuron count of 32
    assert summary_list.summary_list[2].neuron_count == 32
    assert summary_list.summary_list[2].synapse_count == 0
    assert summary_list.summary_list[2].total_events == 32

    # third layer (the unrecognized).Doesn't have a callbacks, therefore synapse, neuron, total, spiking values
    # are None
    assert summary_list.summary_list[3].neuron_count is None
    assert summary_list.summary_list[3].synapse_count is None
    assert summary_list.summary_list[3].total_events is None
    assert summary_list.summary_list[3].spiking_events is None

    # fourth layer (SNN spiking neurons). Here expect the neuron count of 16
    assert summary_list.summary_list[4].neuron_count == 16
    assert summary_list.summary_list[4].synapse_count == 0
    assert summary_list.summary_list[4].total_events == 16

    # now add the device to registry with the callback to the same function which handles the linear, and assert
    # as expected
    LayerParameterEventCalculator.register_new_layer(UnrecognizedLeafTestLayer,
                                                     synapse_neuron_count_for_linear,
                                                     count_events_for_linear)
    # do a forward pass, where we have a "UnrecognizedLeafTestLayer" (which under the hood is linear layer)
    # now the layers should be recognized, and no unknown values (None) should be there
    summary_list = estimate_energy(model=v, input_data=inp_test, include_bias_term_in_events=False)

    assert summary_list.summary_list[0].neuron_count == 32 + 16
    # TODO : should None here be propagated or not ?
    assert summary_list.summary_list[0].synapse_count == 4128 + 528

    # First layer (the unrecognized).Doesn't have a callbacks, therefore synapse, neuron, total, spiking values are None
    assert summary_list.summary_list[1].neuron_count == 0
    assert summary_list.summary_list[1].synapse_count == 4128
    assert summary_list.summary_list[1].total_events == 4096
    assert summary_list.summary_list[1].spiking_events is not None

    # second layer (SNN spiking neurons). Here expect the neuron count of 32
    assert summary_list.summary_list[2].neuron_count == 32
    assert summary_list.summary_list[2].synapse_count == 0
    assert summary_list.summary_list[2].total_events == 32

    # third layer (the unrecognized).Doesn't have a callbacks, therefore synapse, neuron, total, spiking values
    # are None
    assert summary_list.summary_list[3].neuron_count == 0
    assert summary_list.summary_list[3].synapse_count == 528
    assert summary_list.summary_list[3].total_events == 512
    assert summary_list.summary_list[3].spiking_events is not None

    # fourth layer (SNN spiking neurons). Here expect the neuron count of 16
    assert summary_list.summary_list[4].neuron_count == 16
    assert summary_list.summary_list[4].synapse_count == 0
    assert summary_list.summary_list[4].total_events == 16


def test_nested_network():
    """
        test whether the numbers are as expected for a pytorch network consisting of few, nested nn.modules.
        We set the depth to 3, and get all the information about all nn.modules (from the topmost layer to basic Linear
        layer). We set the synapse/neuron/total event count to expected values (as in unfolded network)

        # unfolded pytorch model
        class EnergyEfficiencyNestedNetworkTestUnfolded(EnergyEstimationNetworkInterface):
            def __init__(self, Beta=0.9):
                super().__init__()
                self.fc1 = nn.Linear(128, 64)
                self.lif1 = snn.Leaky(Beta)

                self.fc2 = nn.Linear(64, 32)
                self.lif2 = snn.Leaky(Beta)

                self.fc3 = nn.Linear(32, 16)
                self.lif3 = snn.Leaky(Beta)

                self.fc4 = nn.Linear(16, 32)
                self.lif4 = snn.Leaky(Beta)

                self.fc5 = nn.Linear(32, 64)
                self.lif5 = snn.Leaky(Beta)

                self.fc6 = nn.Linear(64, 128)
                self.lif6 = snn.Leaky(Beta)
                self.reset()

            def forward(self, x: torch.Tensor):
                spk, self.mem1 = self.lif1(self.fc1(x), self.mem1)
                spk, self.mem2 = self.lif2(self.fc2(spk), self.mem2)
                spk, self.mem3 = self.lif3(self.fc3(spk), self.mem3)
                spk, self.mem4 = self.lif4(self.fc4(spk), self.mem4)
                spk, self.mem5 = self.lif5(self.fc5(spk), self.mem5)
                return self.lif6(self.fc6(spk), self.mem6)

    -------------------------------------------------------------------------------------
        # Keras/nengo unfolded equivalent model
            # build an example model
            inp = x = tf.keras.Input((1, 128))
            x = tf.keras.layers.Dense(64)(x)
            x = keras_spiking.SpikingActivation("relu")(x)

            x = tf.keras.layers.Dense(32)(x)
            x = keras_spiking.SpikingActivation("relu")(x)

            x = tf.keras.layers.Dense(16)(x)
            x = keras_spiking.SpikingActivation("relu")(x)

            x = tf.keras.layers.Dense(32)(x)
            x = keras_spiking.SpikingActivation("relu")(x)

            x = tf.keras.layers.Dense(64)(x)
            x = keras_spiking.SpikingActivation("relu")(x)

            x = tf.keras.layers.Dense(128)(x)
            x = keras_spiking.SpikingActivation("relu")(x)


            model = tf.keras.Model(inp, x)

            # estimate model energy
            energy = keras_spiking.ModelEnergy(model)
            energy.summary(print_warnings=False, columns=("name", "output_shape", "params", "connections", "neurons", "energy cpu"))


        Layers are as follows :
        0 Model
        1     Block
        2         Module
        3             Linear
        4             Leaky
        5         Module
        6             Linear
        7             Leaky
        8         Module
        9             Linear
        10            Leaky
        11    Block
        12        Module
        13            Linear
        14            Leaky
        15        Module
        16            Linear
        17            Leaky
        18        Module
        19            Linear
        20            Leaky

    """


    p1 = np.ones((1, 128))

    inp_test = torch.Tensor(np.array([p1]))
    v = EnergyEfficiencyNestedNetworkTest()
    v.reset()

    n = EnergyEfficiencyNestedNetworkTestUnfolded()
    synapses = [8256, 2080, 528, 544, 2112, 8320]
    neurons = [64, 32, 16, 32, 64, 128]

    energy_per_neuron, energy_per_synapse = 1e-9, 1e-9
    predicted_energy = (sum(synapses) - sum(neurons)) * energy_per_synapse + sum(neurons) * energy_per_neuron
    DeviceProfileRegistry.add_device("nested-network-test", energy_per_synapse,
                                     energy_per_neuron, False)

    # do a forward pass, where we have a "UnrecognizedLeafTestLayer" (which under the hood is linear layer)
    # the values for these layers should be unrecognized (None). Assert that these values are None
    summary_list = estimate_energy(model=v, input_data=inp_test, include_bias_term_in_events=False,
                                   depth=3, devices="nested-network-test")

    data = summary_list.summary_list

    # assert that the energy is as predicted
    assert pytest.approx(predicted_energy, rel=0.01) == summary_list.total_energies[0]

    assert data[0].synapse_count == sum(synapses)
    assert data[0].neuron_count == sum(neurons)
    assert data[0].total_events == sum(synapses) # this is valid when bias is not included, however when bias is included
                                                 # it should be sum(synapses) + sum(neurons)

    # first block (module -> module -> module)
    assert data[1].synapse_count == sum(synapses[:3])
    assert data[1].neuron_count == sum(neurons[:3])
    assert data[1].total_events == sum(synapses[:3])

    # second block (module -> module -> module)
    assert data[11].synapse_count == sum(synapses[3:])
    assert data[11].neuron_count == sum(neurons[3:])
    assert data[11].total_events == sum(synapses[3:])

    for offset, counts_offset in zip([1, 11], [0, 3]):
        for i in range(3):
            # check the modules (Linear -> Leaky)
            count_idx = counts_offset + i
            assert data[1 + offset + 3*i].synapse_count == synapses[count_idx]
            assert data[1 + offset + 3*i].neuron_count == neurons[count_idx]
            assert data[1 + offset + 3*i].total_events == synapses[count_idx]

            # check the individual layers
            assert data[1 + offset + 3 * i + 1].synapse_count == synapses[count_idx]
            assert data[1 + offset + 3 * i + 1].neuron_count == 0
            assert data[1 + offset + 3 * i + 1].total_events == synapses[count_idx] - neurons[count_idx]

            assert data[1 + offset + 3 * i + 2].synapse_count == 0
            assert data[1 + offset + 3 * i + 2].neuron_count == neurons[count_idx]
            assert data[1 + offset + 3 * i + 2].total_events == neurons[count_idx]


def test_nested_network_depth1():
    """
        test whether the numbers are as expected for a pytorch network consisting of few, nested nn.modules.
        We set the depth to 1, and get only information about two blocks (however the number of parameters should be
        the same for these blocks as in test above)

    """


    p1 = np.ones((1, 128))

    inp_test = torch.Tensor(np.array([p1]))
    v = EnergyEfficiencyNestedNetworkTest()
    v.reset()

    n = EnergyEfficiencyNestedNetworkTestUnfolded()
    synapses = [8256, 2080, 528, 544, 2112, 8320]
    neurons = [64, 32, 16, 32, 64, 128]

    energy_per_neuron, energy_per_synapse = 1e-9, 1e-9
    predicted_energy = (sum(synapses) - sum(neurons)) * energy_per_synapse + sum(neurons) * energy_per_neuron
    DeviceProfileRegistry.add_device("nested-network-test", energy_per_synapse,
                                     energy_per_neuron, False)

    # do a forward pass, where we have a "UnrecognizedLeafTestLayer" (which under the hood is linear layer)
    # the values for these layers should be unrecognized (None). Assert that these values are None
    summary_list = estimate_energy(model=v, input_data=inp_test, include_bias_term_in_events=False,
                                   depth=1, devices="nested-network-test")

    data = summary_list.summary_list

    # assert that the energy is as predicted
    assert pytest.approx(predicted_energy, rel=0.01) == summary_list.total_energies[0]

    assert data[0].synapse_count == sum(synapses)
    assert data[0].neuron_count == sum(neurons)
    assert data[0].total_events == sum(synapses) # this is valid when bias is not included, however when bias is included
                                                 # it should be sum(synapses) + sum(neurons)

    # first block (module -> module -> module)
    assert data[1].synapse_count == sum(synapses[:3])
    assert data[1].neuron_count == sum(neurons[:3])
    assert data[1].total_events == sum(synapses[:3])

    # second block (module -> module -> module)
    assert data[11].synapse_count == sum(synapses[3:])
    assert data[11].neuron_count == sum(neurons[3:])
    assert data[11].total_events == sum(synapses[3:])
