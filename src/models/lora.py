"""LoRA wrappers for Conv2d and Linear layers, plus an in-place injector.

LoRA reparameterization for a layer with weight W and input x:

    y = W x + (alpha / r) * B(A(x))

with A reducing the input rank to r and B expanding back to the original
output dimension. The base layer's weights are frozen; only A and B (and the
freshly replaced classification head, handled by the caller) are trainable.

Conv2d convention used here (the spec offered two options; we picked one):

    A: 1x1 Conv2d, in_channels -> rank          (channel reducer)
    B: kxk Conv2d, rank -> out_channels         (spatial mixer + expander)

with stride / padding / dilation copied from the base conv onto B (so the
LoRA path produces the same spatial shape as the base path). Groups are
forced to 1 inside the LoRA branch even if the base conv is grouped, because
the LoRA path is dense by design.

Initialization follows the original LoRA paper:
- A: Kaiming uniform (acts like a small random projection)
- B: zeros, so the initial LoRA contribution is exactly 0 and training
     starts from the same predictions as the frozen base model
"""

import math
from typing import Iterable, Optional, Sequence, Tuple, Type

import torch
from torch import nn


class LoRAConv2d(nn.Module):
    def __init__(
        self,
        base_conv: nn.Conv2d,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base_conv = base_conv
        for parameter in self.base_conv.parameters():
            parameter.requires_grad = False

        self.lora_A = nn.Conv2d(
            in_channels=base_conv.in_channels,
            out_channels=rank,
            kernel_size=1,
            bias=False,
        )
        self.lora_B = nn.Conv2d(
            in_channels=rank,
            out_channels=base_conv.out_channels,
            kernel_size=base_conv.kernel_size,
            stride=base_conv.stride,
            padding=base_conv.padding,
            dilation=base_conv.dilation,
            groups=1,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_conv(x)
        lora = self.lora_B(self.lora_A(self.dropout(x)))
        return base + self.scale * lora


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base_linear = base_linear
        for parameter in self.base_linear.parameters():
            parameter.requires_grad = False

        self.lora_A = nn.Linear(base_linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_linear.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_linear(x)
        lora = self.lora_B(self.lora_A(self.dropout(x)))
        return base + self.scale * lora


_LORA_PARAM_TAGS: Tuple[str, ...] = ("lora_A", "lora_B")


def is_lora_parameter_name(qualified_name: str) -> bool:
    """True for parameter names that belong to an injected LoRA A/B module."""
    parts = qualified_name.split(".")
    return any(tag in parts for tag in _LORA_PARAM_TAGS)


def _iter_named_submodules(root: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    return list(root.named_modules())


def inject_lora(
    root: nn.Module,
    target_module_types: Sequence[Type[nn.Module]] = (nn.Conv2d, nn.Linear),
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    skip_names: Optional[Iterable[str]] = None,
) -> int:
    """Walk `root` and replace each instance of `target_module_types` in place
    with the matching LoRA wrapper.

    `skip_names` is a set of fully-qualified submodule names (relative to
    `root`) that should NOT be wrapped. Use this to exclude the stem conv
    and the original classifier head when those should be handled separately
    by the caller (e.g. the head is fully replaced with a fresh trainable
    Linear rather than wrapped in LoRA).

    Returns the number of modules that were wrapped.
    """
    skip = set(skip_names or [])
    target_types = tuple(target_module_types)

    # Two-pass to avoid mutating the tree mid-iteration.
    candidates = []
    for name, module in _iter_named_submodules(root):
        if name in skip:
            continue
        if isinstance(module, target_types) and not isinstance(
            module, (LoRAConv2d, LoRALinear)
        ):
            candidates.append(name)

    wrapped = 0
    for name in candidates:
        parent_name, _, attr = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        child = getattr(parent, attr)

        if isinstance(child, nn.Conv2d):
            setattr(parent, attr, LoRAConv2d(child, rank, alpha, dropout))
            wrapped += 1
        elif isinstance(child, nn.Linear):
            setattr(parent, attr, LoRALinear(child, rank, alpha, dropout))
            wrapped += 1

    return wrapped


def freeze_all(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
