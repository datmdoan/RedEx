import torch.nn as nn
from torch_geometric.nn import GINConv
from .base import BaseGNN
from RedEx.utils import mlp


class GIN(BaseGNN):
    """GIN using ``GINConv`` layers."""

    def __init__(self, eps: float = 0.0, **kwargs):
        self.eps = eps
        super().__init__(**kwargs)

    def build_conv_layer(self, in_dim: int, out_dim: int, **kwargs) -> nn.Module:
        # Always include Dropout layer for consistent state_dict keys
        inner_dropout = self.dropout_p if self.dropout_p > 0 else 1e-12
        inner_mlp = mlp([in_dim, out_dim, out_dim], act="relu", dropout=inner_dropout)
        return GINConv(inner_mlp, eps=self.eps)
