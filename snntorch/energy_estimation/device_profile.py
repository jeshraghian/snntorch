class DeviceProfile(object):

    def __init__(self, name : str, energy_per_synapse_event: float, energy_per_neuron_event: float, is_spiking_architecture: bool):
        self.name = name
        self.energy_per_synapse_event = energy_per_synapse_event
        self.energy_per_neuron_event = energy_per_neuron_event
        self.is_spiking_architecture = is_spiking_architecture

    def __str__(self):
        return f"name={self.name}, synapse_event_energy={self.energy_per_synapse_event}, " \
               f"neuron_event_energy={self.energy_per_neuron_event}, isSpiking={self.is_spiking_architecture}"
