from typing import Iterable, Union
import torch


def prod(num_list: Union[Iterable[int] , torch.Size]) -> int:
    result = 1
    if isinstance(num_list, Iterable):
        for item in num_list:
            result *= prod(item) if isinstance(item, Iterable) else item
    return result
