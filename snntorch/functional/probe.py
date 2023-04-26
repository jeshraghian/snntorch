import torch
from torch import nn
from typing import Callable, Any


def unpack_len1_tuple(x: tuple or torch.Tensor):
    if isinstance(x, tuple) and x.__len__() == 1:
        return x[0]
    else:
        return x


class BaseMonitor:
    def __init__(self):
        self.hooks = []
        self.monitored_layers = []
        self.records = []
        self.name_records_index = {}
        self._enable = True

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        elif isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        else:
            raise ValueError(i)

    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.name_records_index.items():
            v.clear()

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def is_enable(self):
        return self._enable

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove_hooks()


class OutputMonitor(BaseMonitor):
    """
    A monitor to record the output spikes of each specific neuron layer
    (e.g. Leaky) in a network.
    All output data is recorded in ``self.record`` as data type ''list''.
    Call ``self.enable()`` or ``self.disable()`` to enable or disable the
    monitor.
    Call ``self.clear_recorded_data()`` to clear recorded data.

    Example::

        import snntorch as snn
        from snntorch.functional import probe

        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 4)
                self.lif1 = snn.Leaky()
                self.fc2 = nn.Linear(4, 2)
                self.lif2 = snn.Leaky()

            def forward(self, x_seq: torch.Tensor):
                x_seq = self.fc1(x_seq)
                x_seq = self.lif1(x_seq)
                x_seq = self.fc2(x_seq)
                x_seq = self.lif2(x_seq)
                return x_seq

        net = Net()

        monitor = probe.OutputMonitor(net, instance=snntorch.Leaky())

        with torch.no_grad():
            y = net(torch.rand([1, 8]))
            print(f'monitor.records={monitor.records}')
            print(f'monitor[0]={monitor[0]}')
            print(f'monitor.monitored_layers={monitor.monitored_layers}')
            print(f"monitor['lif1']={monitor['lif1']}")

    :param net: Network model (either wrapped in Sequential container or
        as a class)
    :type net: nn.Module

    :param instance: Instance of modules to be monitored. If ``None``,
        defaults to ``type(net)``
    :type instance: Any or tuple

    :param function_on_output: Function that is applied to the monitored
        modules' outputs
    :type function_on_output: Callable, optional

    """

    def __init__(
        self,
        net: nn.Module,
        instance: Any or tuple = None,
        function_on_output: Callable = lambda x: x,
    ):
        super().__init__()
        self.function_on_output = function_on_output
        if instance is None:
            instance = type(net)
        for name, m in net.named_modules():
            if isinstance(m, instance):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(
                    m.register_forward_hook(self.create_hook(name))
                )

    def create_hook(self, name):
        def hook(m, x, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(
                    self.function_on_output(unpack_len1_tuple(y))
                )

        return hook


class InputMonitor(BaseMonitor):
    """
    A monitor to record the input of each neuron layer (e.g. Leaky)
    in a network.
    All input data is recorded in ``self.record`` as data type ''list''.
    Call ``self.enable()`` or ``self.disable()`` to enable or disable
    the monitor.
    Call ``self.clear_recorded_data()`` to clear recorded data.

    Example::

        import snntorch as snn
        from snntorch.functional import probe

        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 4)
                self.lif1 = snn.Leaky()
                self.fc2 = nn.Linear(4, 2)
                self.lif2 = snn.Leaky()

            def forward(self, x_seq: torch.Tensor):
                x_seq = self.fc1(x_seq)
                x_seq = self.lif1(x_seq)
                x_seq = self.fc2(x_seq)
                x_seq = self.lif2(x_seq)
                return x_seq

        net = Net()

        monitor = probe.InputMonitor(net, instance=snn.Leaky())

        with torch.no_grad():
            y = net(torch.rand([1, 8]))
            print(f'monitor.records={monitor.records}')
            print(f'monitor[0]={monitor[0]}')
            print(f'monitor.monitored_layers={monitor.monitored_layers}')
            print(f"monitor['lif1']={monitor['lif1']}")

    :param net: Network model (either wrapped in Sequential container
        or as a class)
    :type net: nn.Module

    :param instance: Instance of modules to be monitored. If ``None``,
        defaults to ``type(net)``
    :type instance: Any or tuple

    :param function_on_input: Function that is applied to the monitored
        modules' input
    :type function_on_input: Callable, optional

    """

    def __init__(
        self,
        net: nn.Module,
        instance: Any or tuple = None,
        function_on_input: Callable = lambda x: x,
    ):
        super().__init__()
        self.function_on_input = function_on_input
        if instance is None:
            instance = type(net)
        for name, m in net.named_modules():
            if isinstance(m, instance):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(
                    m.register_forward_hook(self.create_hook(name))
                )

    def create_hook(self, name):
        def hook(m, x, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(
                    self.function_on_input(unpack_len1_tuple(x))
                )

        return hook


class AttributeMonitor(BaseMonitor):
    """
    A monitor to record the attribute (e.g. membrane potential) of a
    specific neuron layer (e.g. Leaky) in a network.
    The attribute name can be specified as the first argument of this function.
    All attribute data is recorded in ``self.record`` as data type ''list''.
    Call ``self.enable()`` or ``self.disable()`` to enable or disable
    the monitor.
    Call ``self.clear_recorded_data()`` to clear recorded data.

    Example::

        import snntorch as snn
        from snntorch.functional import probe

        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 4)
                self.lif1 = snn.Leaky()
                self.fc2 = nn.Linear(4, 2)
                self.lif2 = snn.Leaky()

            def forward(self, x_seq: torch.Tensor):
                x_seq = self.fc1(x_seq)
                x_seq = self.lif1(x_seq)
                x_seq = self.fc2(x_seq)
                x_seq = self.lif2(x_seq)
                return x_seq

        net = Net()

        monitor = probe.AttributeMonitor('mem', False, net,
        instance=snn.Leaky())

        with torch.no_grad():
            y = net(torch.rand([1, 8]))
            print(f'monitor.records={monitor.records}')
            print(f'monitor[0]={monitor[0]}')
            print(f'monitor.monitored_layers={monitor.monitored_layers}')
            print(f"monitor['lif1']={monitor['lif1']}")

    :param attribute_name: Attribute's name of probed neuron layer
        (e.g., mem, syn, etc.)
    :type net: str

    :param pre_forward: If ``True``, record the attribute value before
        the forward pass, otherwise record the value after forward pass.
    :type pre_forward: bool

    :param net: Network model (either wrapped in Sequential container or
        as a class)
    :type net: nn.Module

    :param instance: Instance of modules to be monitored. If ``None``,
        defaults to ``type(net)``
    :type instance: Any or tuple

    :param function_on_attribute: Function that is applied to the
        monitored modules' attribute
    :type function_on_attribute: Callable, optional

    """

    def __init__(
        self,
        attribute_name: str,
        pre_forward: bool,
        net: nn.Module,
        instance: Any or tuple = None,
        function_on_attribute: Callable = lambda x: x,
    ):
        super().__init__()
        self.attribute_name = attribute_name
        self.function_on_attribute = function_on_attribute

        for name, m in net.named_modules():
            if isinstance(m, instance):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                if pre_forward:
                    self.hooks.append(
                        m.register_forward_pre_hook(self.create_hook(name))
                    )
                else:
                    self.hooks.append(
                        m.register_forward_hook(self.create_hook(name))
                    )

    def create_hook(self, name):
        def hook(m, x, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(
                    self.function_on_attribute(
                        m.__getattr__(self.attribute_name)
                    )
                )

        return hook


class GradInputMonitor(BaseMonitor):
    """
    A monitor to record the input gradient of each neuron layer
    (e.g. Leaky) in a network.
    All input gradient data is recorded in ``self.record`` as data type
    ''list''.
    Call ``self.enable()`` or ``self.disable()`` to enable or disable
    the monitor.
    Call ``self.clear_recorded_data()`` to clear recorded data.

    Example::

        import snntorch as snn
        from snntorch.functional import probe

        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 4)
                self.lif1 = snn.Leaky()
                self.fc2 = nn.Linear(4, 2)
                self.lif2 = snn.Leaky()

            def forward(self, x_seq: torch.Tensor):
                x_seq = self.fc1(x_seq)
                x_seq = self.lif1(x_seq)
                x_seq = self.fc2(x_seq)
                x_seq = self.lif2(x_seq)
                return x_seq

        net = Net()

        monitor = probe.GradInputMonitor(net, instance=snn.Leaky())

        with torch.no_grad():
            y = net(torch.rand([1, 8]))
            print(f'monitor.records={monitor.records}')
            print(f'monitor[0]={monitor[0]}')
            print(f'monitor.monitored_layers={monitor.monitored_layers}')
            print(f"monitor['lif1']={monitor['lif1']}")

    :param net: Network model (either wrapped in Sequential container or
        as a class)
    :type net: nn.Module

    :param instance: Instance of modules to be monitored. If ``None``,
        defaults to ``type(net)``
    :type instance: Any or tuple

    :param function_on_grad_input: Function that is applied to the
        monitored modules' gradients
    :type function_on_grad_input: Callable, optional

    """

    def __init__(
        self,
        net: nn.Module,
        instance: Any or tuple = None,
        function_on_grad_input: Callable = lambda x: x,
    ):
        super().__init__()
        self.function_on_grad_input = function_on_grad_input

        for name, m in net.named_modules():
            if isinstance(m, instance):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                if torch.__version__ >= torch.torch_version.TorchVersion(
                    "1.8.0"
                ):
                    self.hooks.append(
                        m.register_full_backward_hook(self.create_hook(name))
                    )
                else:
                    self.hooks.append(
                        m.register_backward_hook(self.create_hook(name))
                    )

    def create_hook(self, name):
        def hook(m, grad_input, grad_output):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(
                    self.function_on_grad_input(unpack_len1_tuple(grad_input))
                )

        return hook


class GradOutputMonitor(BaseMonitor):
    """
    A monitor to record the output gradient of each specific neuron layer
    (e.g. Leaky) in a network.
    All output gradient data is recorded in ``self.record`` as data type
    ''list''.
    Call ``self.enable()`` or ``self.disable()`` to enable or disable the
    monitor.
    Call ``self.clear_recorded_data()`` to clear recorded data.

    Example::

        import snntorch as snn
        from snntorch.functional import probe

        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 4)
                self.lif1 = snn.Leaky()
                self.fc2 = nn.Linear(4, 2)
                self.lif2 = snn.Leaky()

            def forward(self, x_seq: torch.Tensor):
                x_seq = self.fc1(x_seq)
                x_seq = self.lif1(x_seq)
                x_seq = self.fc2(x_seq)
                x_seq = self.lif2(x_seq)
                return x_seq

        net = Net()

        mtor = probe.GradOutputMonitor(net, instance=snn.Leaky())

        with torch.no_grad():
            y = net(torch.rand([1, 8]))
            print(f'mtor.records={mtor.records}')
            print(f'mtor[0]={mtor[0]}')
            print(f'mtor.monitored_layers={mtor.monitored_layers}')
            print(f"mtor['lif1']={mtor['lif1']}")

    :param net: Network model (either wrapped in Sequential container
        or as a class)
    :type net: nn.Module

    :param instance: Instance of modules to be monitored. If ``None``,
        defaults to ``type(net)``
    :type instance: Any or tuple

    :param function_on_grad_output: Function that is applied to the
        monitored modules' gradients
    :type function_on_grad_output: Callable, optional

    """

    def __init__(
        self,
        net: nn.Module,
        instance: Any or tuple = None,
        function_on_grad_output: Callable = lambda x: x,
    ):
        super().__init__()
        self.function_on_grad_output = function_on_grad_output
        for name, m in net.named_modules():
            if isinstance(m, instance):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                if torch.__version__ >= torch.torch_version.TorchVersion(
                    "1.8.0"
                ):
                    self.hooks.append(
                        m.register_full_backward_hook(self.create_hook(name))
                    )
                else:
                    self.hooks.append(
                        m.register_backward_hook(self.create_hook(name))
                    )

    def create_hook(self, name):
        def hook(m, grad_input, grad_output):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(
                    self.function_on_grad_output(
                        unpack_len1_tuple(grad_output)
                    )
                )

        return hook
