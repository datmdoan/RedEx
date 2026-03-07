from __future__ import annotations
from typing import Callable, Dict, Tuple
import torch
from torch import nn
from torch_geometric.nn import (
    GlobalAttention,
    Set2Set,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)

from RedEx.utils import mlp

# ── base & registry ──────────────────────────────────────────────────────────


class ReadoutBase(nn.Module):
    """Every readout must define ``output_dim(in_dim)``."""

    def output_dim(self, in_dim: int) -> int:
        raise NotImplementedError


_READOUT_BUILDERS: Dict[str, Callable] = {}


def register_readout(name: str):
    def _wrap(cls):
        key = name.lower()
        if key in _READOUT_BUILDERS:
            raise ValueError(f"Readout '{name}' already registered.")

        def _builder(in_dim: int, **kwargs):
            return cls(in_dim=in_dim, **kwargs)

        _READOUT_BUILDERS[key] = _builder
        return cls

    return _wrap


def build_readout(name: str, in_dim: int, **kwargs) -> Tuple[nn.Module, int]:
    """Return ``(readout_module, output_dim)``."""
    key = (name or "mean").lower()
    if key not in _READOUT_BUILDERS:
        raise ValueError(f"Unknown readout '{name}'. Available: {list(_READOUT_BUILDERS)}")
    ro: ReadoutBase = _READOUT_BUILDERS[key](in_dim=in_dim, **kwargs)
    return ro, ro.output_dim(in_dim)


# ── implementations ──────────────────────────────────────────────────────────


@register_readout("mean")
class MeanPool(ReadoutBase):
    def __init__(self, in_dim: int, **_):
        super().__init__()
        self.fn = global_mean_pool

    def forward(self, x, batch):
        return self.fn(x, batch)

    def output_dim(self, in_dim: int) -> int:
        return in_dim


@register_readout("sum")
class SumPool(ReadoutBase):
    def __init__(self, in_dim: int, **_):
        super().__init__()
        self.fn = global_add_pool

    def forward(self, x, batch):
        return self.fn(x, batch)

    def output_dim(self, in_dim: int) -> int:
        return in_dim


@register_readout("max")
class MaxPool(ReadoutBase):
    def __init__(self, in_dim: int, **_):
        super().__init__()
        self.fn = global_max_pool

    def forward(self, x, batch):
        return self.fn(x, batch)

    def output_dim(self, in_dim: int) -> int:
        return in_dim


@register_readout("attn")
class AttentionPool(ReadoutBase):
    """``GlobalAttention`` with an MLP gate."""

    def __init__(self, in_dim: int, attn_gate_dims=None, act="relu", dropout=0.0, **_):
        super().__init__()
        gate_dims = attn_gate_dims or [in_dim, max(in_dim // 2, 1), 1]
        if gate_dims[0] != in_dim:
            gate_dims = [in_dim] + list(gate_dims)
        self.pool = GlobalAttention(
            gate_nn=mlp(gate_dims, act=act, dropout=dropout, last_activation=False)
        )

    def forward(self, x, batch):
        return self.pool(x, batch)

    def output_dim(self, in_dim: int) -> int:
        return in_dim


@register_readout("set2set")
class Set2SetPool(ReadoutBase):
    def __init__(self, in_dim: int, set2set_steps: int = 3, **_):
        super().__init__()
        self.pool = Set2Set(in_dim, processing_steps=set2set_steps)

    def forward(self, x, batch):
        return self.pool(x, batch)

    def output_dim(self, in_dim: int) -> int:
        return 2 * in_dim


@register_readout("sigmoid_weighted")
class SigmoidWeightedReadout(ReadoutBase):
    """Learned sigmoid-weighted add-pool concatenated with max-pool.

    Output dim = 2 × in_dim.
    """

    def __init__(self, in_dim: int, **_):
        super().__init__()
        self.weighting = nn.Linear(in_dim, 1)
        self.score = nn.Sigmoid()
        nn.init.xavier_uniform_(self.weighting.weight)
        nn.init.constant_(self.weighting.bias, 0)

    def forward(self, x, batch):
        w = self.score(self.weighting(x))  # [N, 1]
        out1 = global_add_pool(w * x, batch)  # [B, D]
        out2 = global_max_pool(x, batch)  # [B, D]
        return torch.cat([out1, out2], dim=1)  # [B, 2D]

    def output_dim(self, in_dim: int) -> int:
        return 2 * in_dim
