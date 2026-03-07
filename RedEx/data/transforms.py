from __future__ import annotations
import pickle
import torch
from pathlib import Path
from typing import List
from torch_geometric.data import Data


class StandardizeAtomScalars:
    """Zero-mean / unit-variance scaling for continuous atom features.

    Fit on training data only, then apply to all splits.
    """

    def __init__(self, idx: List[int]):
        self.idx = idx
        self.mean = None
        self.std = None
        self._fitted = False

    def fit(self, data_list: List[Data]) -> "StandardizeAtomScalars":
        all_feats = []
        for d in data_list:
            if d is not None:
                all_feats.append(d.x[:, self.idx])
        if not all_feats:
            raise ValueError("No valid data for fitting")
        all_feats = torch.cat(all_feats, dim=0)
        self.mean = all_feats.mean(dim=0)
        self.std = all_feats.std(dim=0)
        self.std = torch.where(self.std > 1e-8, self.std, torch.ones_like(self.std))
        self._fitted = True
        return self

    def transform(self, data: Data) -> Data | None:
        if data is None:
            return None
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        dc = data.clone()
        dc.x[:, self.idx] = (dc.x[:, self.idx] - self.mean) / self.std
        return dc

    def fit_transform(self, data_list: List[Data]) -> List[Data]:
        self.fit(data_list)
        return [self.transform(d) for d in data_list]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"idx": self.idx, "mean": self.mean, "std": self.std}, f)

    @classmethod
    def load(cls, path: str | Path) -> "StandardizeAtomScalars":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(idx=state["idx"])
        obj.mean = state["mean"]
        obj.std = state["std"]
        obj._fitted = True
        return obj
