import torch.nn as nn
from torch_geometric.nn import SAGEConv
from .base import BaseGNN


class GraphSAGE(BaseGNN):
    """GraphSAGE using ``SAGEConv`` layers."""

    def __init__(self, aggr: str = "mean", **kwargs):
        self.aggr = aggr
        super().__init__(**kwargs)

    def build_conv_layer(self, in_dim: int, out_dim: int, **kwargs) -> nn.Module:
        return SAGEConv(in_dim, out_dim, aggr=self.aggr)
