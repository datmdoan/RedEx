from .base import BaseGNN
from .gcn import GCN
from .gat import GAT
from .gin import GIN
from .sage import GraphSAGE
from .dmpnn import DMPNN
from .readouts import build_readout

MODEL_REGISTRY = {
    "gcn": GCN,
    "gat": GAT,
    "gin": GIN,
    "graphsage": GraphSAGE,
    "dmpnn": DMPNN,
}


def create_model(model_type: str, **kwargs):
    """Instantiate a GNN by name.

    >>> model = create_model("gat", in_dim=74, hidden_dim=256,
    ...                      out_dim=1, num_layers=5, num_heads=2)
    """
    key = model_type.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_type}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](**kwargs)


__all__ = [
    "BaseGNN", "GCN", "GAT", "GIN", "GraphSAGE", "DMPNN",
    "build_readout", "create_model", "MODEL_REGISTRY",
]
