# import the | for compatibility with python 3.7.*
from __future__ import annotations
from typing import Iterable
import torch


def prod(num_list: Iterable[int] | torch.Size) -> int:
    result = 1
    if isinstance(num_list, Iterable):
        for item in num_list:
            result *= prod(item) if isinstance(item, Iterable) else item
    return result
