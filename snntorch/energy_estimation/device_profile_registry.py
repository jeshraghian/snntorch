from .device_profile import DeviceProfile


class DeviceProfileRegistry:
    _devices_reg = {"cpu": DeviceProfile("cpu", 8.6e-9, 8.6e-9, False)
                    }

    @staticmethod
    def get_device(name):
        if name not in DeviceProfileRegistry._devices_reg:
            raise Exception("")

        return DeviceProfileRegistry._devices_reg[name]

    @staticmethod
    def add_device(name: str, energy_per_synapse_event: float, energy_per_neuron_event: float,
                   is_spiking_architecture: bool):
        if name in DeviceProfileRegistry._devices_reg:
            raise Exception("")

        DeviceProfileRegistry._devices_reg[name] = DeviceProfile(name, energy_per_synapse_event,
                                                                 energy_per_neuron_event, is_spiking_architecture)
