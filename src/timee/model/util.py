from typing import Optional

import torch
import torch.nn as nn


def activation_from_str(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")


def validate_rope(position_ids: Optional[torch.Tensor]) -> None:
    if position_ids is None:
        raise ValueError("position_ids must not be None when using RoPE")
