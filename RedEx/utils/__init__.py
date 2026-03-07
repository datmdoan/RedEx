from __future__ import annotations
from typing import List, Sequence
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GraphNorm, LayerNorm


def get_activation(name: str) -> nn.Module:
    """Return an activation module by name."""
    name = (name or "relu").lower()
    return {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "elu": nn.ELU(),
        "leakyrelu": nn.LeakyReLU(0.01),
        "tanh": nn.Tanh(),
        "identity": nn.Identity(),
    }.get(name, nn.ReLU())


def make_norm(norm_type: str, dim: int) -> nn.Module:
    """Return a normalisation layer by name."""
    nt = (norm_type or "batch").lower()
    if nt in ("batch", "batchnorm", "bn"):
        return BatchNorm(dim)
    if nt in ("layer", "layernorm", "ln"):
        return LayerNorm(dim)
    if nt in ("graph", "graphnorm", "gn"):
        return GraphNorm(dim)
    if nt in ("none", "identity", "id"):
        return nn.Identity()
    raise ValueError(f"Unknown norm_type: {norm_type}")


def mlp(
    dims: Sequence[int],
    act: str = "relu",
    dropout: float = 0.0,
    last_activation: bool = False,
) -> nn.Sequential:
    """Build a simple MLP from a list of layer widths."""
    layers: List[nn.Module] = []
    A = lambda: get_activation(act)
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_activation:
            layers.append(A())
        if dropout and (i < len(dims) - 2):
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)
