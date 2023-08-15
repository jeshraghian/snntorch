from .device_profile import DeviceProfile
from typing import Dict, List
import logging


class DeviceWithNameDoesntExist(Exception):
    """Raised when the get_device was called with name that doesn't map to any DeviceProfile"""


class DeviceProfileRegistry:
    """
        A singleton with a dictionary _devices_reg consisting of all registered/known devices (a map from name to
        DeviceProfile).

        Example::

            # add the device to Registry
            DeviceProfileRegistry.add_device('new-cpu', 1e-9, 1e-9, False)

            # Specify your network
            # ...

            # show the estimates for that model
            estimate_energy(model=my_pytorch_model, devices="new-cpu")

    """

    _devices_reg: Dict[str, DeviceProfile] = {"cpu": DeviceProfile("cpu", 8.6e-9, 8.6e-9, False)
                                              }

    @staticmethod
    def get_device(name) -> DeviceProfile:
        if name not in DeviceProfileRegistry._devices_reg:
            raise DeviceWithNameDoesntExist()

        return DeviceProfileRegistry._devices_reg[name]

    @staticmethod
    def add_device(name: str, energy_per_synapse_event: float, energy_per_neuron_event: float,
                   is_spiking_architecture: bool) -> None:

        if name in DeviceProfileRegistry._devices_reg:
            logging.warning(f"Device with name {name} already exists, overwriting ...")

        DeviceProfileRegistry._devices_reg[name] = DeviceProfile(name, energy_per_synapse_event,
                                                                 energy_per_neuron_event, is_spiking_architecture)

    @staticmethod
    def get_all_registered_devices() -> List[str]:
        return list(DeviceProfileRegistry._devices_reg.keys())
