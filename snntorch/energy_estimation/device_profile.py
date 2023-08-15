class DeviceProfile(object):
    """
        A class representing the information about the device, which is used when calculating the energy estimates.
        It considers few properties of the device :

        :param name : unique identifier of the device, used in estimate_energy

        :param is_spiking_architecture : is the device neuromorphic. If set to true, energy is added only when there is
        an event (synaptic or neuron). for example, for final output layer with [1, 0, 0, 0, 0, 0] there is only one
        event.

        :param energy_per_synapse_event: energy required to perform a synapse event. This is when a input parameter
        is multiplied by certain weight in Linear or Convolutional layers. Units are J / event

        :param energy_per_neuron_event : energy required to perform a neuron event. This is energy required, when an
        output is generated from a neuron

    """

    def __init__(self, name : str, energy_per_synapse_event: float, energy_per_neuron_event: float, is_spiking_architecture: bool):
        self.name = name
        self.energy_per_synapse_event = energy_per_synapse_event
        self.energy_per_neuron_event = energy_per_neuron_event
        self.is_spiking_architecture = is_spiking_architecture

    def __str__(self):
        return f"name={self.name}, synapse_event_energy={self.energy_per_synapse_event}, " \
               f"neuron_event_energy={self.energy_per_neuron_event}, isSpiking={self.is_spiking_architecture}"
