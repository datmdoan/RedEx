
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from .base import BaseGNN


class GAT(BaseGNN):
    """GAT with multi-head ``GATv2Conv`` layers."""

    def __init__(self, num_heads: int = 8, concat: bool = True,
                 negative_slope: float = 0.2, **kwargs):
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = negative_slope
        super().__init__(**kwargs)

    def build_conv_layer(self, in_dim: int, out_dim: int, **kwargs) -> nn.Module:
        if self.concat:
            head_dim = out_dim // self.num_heads
            expected_output = head_dim * self.num_heads

            conv = GATv2Conv(
                in_channels=in_dim,
                out_channels=head_dim,
                heads=self.num_heads,
                concat=True,
                negative_slope=self.negative_slope,
                dropout=self.dropout_p,
                edge_dim=kwargs.get("edge_dim") if self.use_edge_attr else None,
                add_self_loops=True,
                bias=True,
                residual=self.residual,
                fill_value="mean",
            )
            if expected_output != out_dim:
                return nn.Sequential(conv, nn.Linear(expected_output, out_dim))
            return conv
        else:
            return GATv2Conv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=self.num_heads,
                concat=False,
                negative_slope=self.negative_slope,
                dropout=self.dropout_p,
                edge_dim=kwargs.get("edge_dim") if self.use_edge_attr else None,
                add_self_loops=True,
                bias=True,
                residual=self.residual,
                fill_value="mean",
            )
