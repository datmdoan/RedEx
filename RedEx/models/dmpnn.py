from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool
from .base import BaseGNN


class DMPNNConv(MessagePassing):
    """D-MPNN convolution layer with directed message passing."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: Optional[int] = None,
                 dropout: float = 0.0, bias: bool = True):
        super().__init__(aggr="add", flow="source_to_target")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim or in_dim

        self.message_net = nn.Sequential(
            nn.Linear(in_dim + self.edge_dim, out_dim, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.update_net = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bn = nn.BatchNorm1d(out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = torch.ones(
                (edge_index.size(1), self.edge_dim), dtype=x.dtype, device=x.device
            )
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        combined = torch.cat([x, out], dim=-1)
        updated = self.update_net(combined)
        if updated.size(0) > 1:
            updated = self.bn(updated)
        return updated

    def message(self, x_j, edge_attr):
        return self.message_net(torch.cat([x_j, edge_attr], dim=-1))


class DMPNN(BaseGNN):

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 3, depth: int = None,
                 batch_norm: bool = True, use_edge_attr: bool = False,
                 edge_dim: Optional[int] = None, dropout: float = 0.0, **kwargs):
        if depth is not None:
            num_layers = depth
        self.batch_norm = batch_norm

        super().__init__(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
            num_layers=num_layers, use_edge_attr=use_edge_attr,
            edge_dim=edge_dim, dropout=dropout, **kwargs,
        )

        if batch_norm:
            pooled_dim = self.readout.output_dim(self.hidden_dim)
            self.fingerprint_bn = nn.BatchNorm1d(pooled_dim)
        else:
            self.fingerprint_bn = nn.Identity()

    def build_conv_layer(self, in_dim: int, out_dim: int, **kwargs) -> nn.Module:
        return DMPNNConv(
            in_dim=in_dim, out_dim=out_dim,
            edge_dim=kwargs.get("edge_dim") if self.use_edge_attr else None,
            dropout=self.dropout_p,
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None) if self.use_edge_attr else None
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if hasattr(self, "pre_mlp"):
            x = self.pre_mlp(x)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual_in = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            if self.residual and i > 0 and x.size(-1) == residual_in.size(-1):
                x = x + residual_in

        hg = self.readout(x, batch)
        if hg.size(0) > 1:
            hg = self.fingerprint_bn(hg)
        hg = self.post_mlp(hg)
        out = self.head(hg)
        return out.squeeze(-1) if self.out_dim == 1 else out
