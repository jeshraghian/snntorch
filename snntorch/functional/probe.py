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
    def __init__(self, net: nn.Module, instance: Any or tuple = None, function_on_output: Callable = lambda x: x):
        super().__init__()
        self.function_on_output = function_on_output
        for name, m in net.named_modules():
            if isinstance(m, instance):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook(name)))

    def create_hook(self, name):
        def hook(m, x, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.function_on_output(unpack_len1_tuple(y)))

        return hook

class InputMonitor(BaseMonitor):
    def __init__(self, net: nn.Module, instance: Any or tuple = None, function_on_input: Callable = lambda x: x):
        super().__init__()
        self.function_on_input = function_on_input
        for name, m in net.named_modules():
            if isinstance(m, instance):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook(name)))

    def create_hook(self, name):
        def hook(m, x, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.function_on_input(unpack_len1_tuple(x)))

        return hook

class AttributeMonitor(BaseMonitor):
    def __init__(self, attribute_name: str, pre_forward: bool, net: nn.Module, instance: Any or tuple = None,
                 function_on_attribute: Callable = lambda x: x):
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
                self.records.append(self.function_on_attribute(m.__getattr__(self.attribute_name)))

        return hook

class GradInputMonitor(BaseMonitor):
    def __init__(self, net: nn.Module, instance: Any or tuple = None, function_on_grad_input: Callable = lambda x: x):
        super().__init__()
        self.function_on_grad_input = function_on_grad_input

        for name, m in net.named_modules():
            if isinstance(m, instance):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                if torch.__version__ >= torch.torch_version.TorchVersion('1.8.0'):
                    self.hooks.append(m.register_full_backward_hook(self.create_hook(name)))
                else:
                    self.hooks.append(m.register_backward_hook(self.create_hook(name)))

    def create_hook(self, name):
        def hook(m, grad_input, grad_output):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.function_on_grad_input(unpack_len1_tuple(grad_input)))

        return hook

class GradOutputMonitor(BaseMonitor):
    def __init__(self, net: nn.Module, instance: Any or tuple = None, function_on_grad_output: Callable = lambda x: x):
        super().__init__()
        self.function_on_grad_output = function_on_grad_output
        for name, m in net.named_modules():
            if isinstance(m, instance):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                if torch.__version__ >= torch.torch_version.TorchVersion('1.8.0'):
                    self.hooks.append(m.register_full_backward_hook(self.create_hook(name)))
                else:
                    self.hooks.append(m.register_backward_hook(self.create_hook(name)))

    def create_hook(self, name):
        def hook(m, grad_input, grad_output):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.function_on_grad_output(unpack_len1_tuple(grad_output)))

        return hook
