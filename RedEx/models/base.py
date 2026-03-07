from __future__ import annotations
from typing import Optional, Sequence
import inspect
import warnings

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool

from RedEx.utils import get_activation, make_norm, mlp
from .readouts import build_readout


class BaseGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        activation: str = "relu",
        norm_type: str = "batch",
        residual: bool = True,
        use_edge_attr: bool = False,
        edge_dim: Optional[int] = None,
        # pooling
        pooling: str = "mean",
        attn_gate_dims: Optional[Sequence[int]] = None,
        set2set_steps: int = 3,
        pooling_kwargs: Optional[dict] = None,
        # optional components
        pre_mlp_dims: Sequence[int] | None = None,
        post_mlp_dims: Sequence[int] | None = None,
        virtual_node: bool = False,
        virtual_node_mlp_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        if use_edge_attr and edge_dim is None:
            raise ValueError("edge_dim must be provided when use_edge_attr=True")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout_p = float(dropout)
        self.residual = residual
        self.use_edge_attr = use_edge_attr
        self.edge_dim = edge_dim
        self.virtual_node = virtual_node
        self.pooling = pooling

        self.act = get_activation(activation)
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        # ── pre-GNN MLP ──────────────────────────────────────────────────
        if pre_mlp_dims:
            if pre_mlp_dims[0] != in_dim:
                raise ValueError("pre_mlp_dims[0] must equal in_dim")
            self.pre_mlp = mlp(pre_mlp_dims, act=activation, dropout=dropout)
            conv_in = pre_mlp_dims[-1]
        else:
            self.pre_mlp = nn.Identity()
            conv_in = in_dim

        # ── convolutional stack ───────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(self.build_conv_layer(conv_in, hidden_dim, edge_dim=edge_dim))
        self.norms.append(make_norm(norm_type, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(self.build_conv_layer(hidden_dim, hidden_dim, edge_dim=edge_dim))
            self.norms.append(make_norm(norm_type, hidden_dim))

        self.supports_edges = self._supports_edge_attr(self.convs[0])
        if use_edge_attr and not self.supports_edges:
            warnings.warn(
                f"{self.__class__.__name__} convolution does not accept edge_attr; "
                "edge features will be ignored."
            )

        # ── virtual node ──────────────────────────────────────────────────
        if self.virtual_node:
            vn_dims = list(virtual_node_mlp_dims) if virtual_node_mlp_dims else [hidden_dim, hidden_dim]
            if vn_dims[0] != hidden_dim:
                vn_dims = [hidden_dim] + vn_dims
            if vn_dims[-1] != hidden_dim:
                vn_dims = vn_dims + [hidden_dim]
            self.vn_mlp = mlp(vn_dims, act=activation, dropout=dropout, last_activation=True)
        else:
            self.vn_mlp = None

        # ── readout ───────────────────────────────────────────────────────
        self.readout, pooled_dim = build_readout(
            pooling,
            hidden_dim,
            attn_gate_dims=attn_gate_dims,
            set2set_steps=set2set_steps,
            act=activation,
            dropout=dropout,
        )

        # ── post-MLP ─────────────────────────────────────────────────────
        if post_mlp_dims:
            if post_mlp_dims[0] != pooled_dim:
                post_mlp_dims = [pooled_dim] + list(post_mlp_dims)
            self.post_mlp = mlp(post_mlp_dims, act=activation, dropout=dropout)
            head_in = post_mlp_dims[-1]
        else:
            self.post_mlp = nn.Identity()
            head_in = pooled_dim

        # ── prediction head (funnel: 128 → 64 → out_dim) ─────────────────
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    # ── subclass hook ─────────────────────────────────────────────────────
    def build_conv_layer(self, in_dim: int, out_dim: int, **kwargs) -> nn.Module:
        raise NotImplementedError

    # ── helpers ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def _supports_edge_attr(self, conv: nn.Module) -> bool:
        return "edge_attr" in inspect.signature(conv.forward).parameters

    def _apply_conv(self, conv, x, edge_index, edge_attr):
        if self.use_edge_attr and self._supports_edge_attr(conv):
            return conv(x, edge_index, edge_attr=edge_attr)
        return conv(x, edge_index)

    # ── forward ───────────────────────────────────────────────────────────
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.pre_mlp(x)

        if self.virtual_node:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            vn = x.new_zeros((num_graphs, self.hidden_dim))

        for conv, norm in zip(self.convs, self.norms):
            residual_in = x
            if self.virtual_node:
                x = x + vn[batch]
            x = self._apply_conv(conv, x, edge_index, edge_attr)
            x = norm(x)
            x = self.act(x)
            x = self.dropout(x)
            if self.residual and residual_in.shape == x.shape:
                x = x + residual_in
            if self.virtual_node:
                pooled = global_add_pool(x, batch)
                vn = vn + self.vn_mlp(pooled)

        hg = self.readout(x, batch)
        hg = self.post_mlp(hg)
        out = self.head(hg)
        return out.squeeze(-1) if self.out_dim == 1 else out

    # ── utilities ─────────────────────────────────────────────────────────
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only)
        )

    def describe(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"params={self.num_parameters():,}, "
            f"layers={self.num_layers}, "
            f"hidden={self.hidden_dim}, "
            f"pooling={self.pooling})"
        )
