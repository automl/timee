from typing import Optional

import torch


def validate_rope(position_ids: Optional[torch.Tensor]) -> None:
    if position_ids is None:
        raise ValueError("position_ids must not be None when using RoPE")
