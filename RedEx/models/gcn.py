import torch.nn as nn
from torch_geometric.nn import GCNConv
from .base import BaseGNN


class GCN(BaseGNN):
    """GCN using ``GCNConv`` layers."""

    def build_conv_layer(self, in_dim: int, out_dim: int, **kwargs) -> nn.Module:
        return GCNConv(in_dim, out_dim)
