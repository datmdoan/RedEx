from __future__ import annotations
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader

from .featuriser import MoleculeFeaturiser, smiles_to_data
from .transforms import StandardizeAtomScalars

# Default column names matching the shipped datasets
SMILES_COL = "oxidised_smiles"
TARGET_COLS = ["Redox_Potential(V)", "*Redox_Potential(V)"]


def _get_targets(df: pd.DataFrame):
    """Extract target column as float array; return None when absent."""
    for c in TARGET_COLS:
        if c in df.columns:
            return pd.to_numeric(df[c].astype(str).str.strip("*"), errors="coerce").values
    return None


def load_graphs(csv_path, featuriser=None, verbose=True, encoding=None):
    """Read a CSV and convert SMILES → PyG Data list.

    Returns ``(graphs, featuriser)``.
    """
    df = pd.read_csv(csv_path, encoding=encoding) if encoding else pd.read_csv(csv_path)
    if SMILES_COL not in df.columns:
        raise ValueError(f"CSV must contain a '{SMILES_COL}' column")

    y = _get_targets(df)
    smiles = df[SMILES_COL].astype(str).tolist()
    ftr = featuriser or MoleculeFeaturiser()

    graphs = []
    n_fail = 0
    for i, smi in enumerate(smiles):
        target = float(y[i]) if (y is not None and np.isfinite(y[i])) else None
        g = smiles_to_data(smi, ftr, target)
        if g is None:
            n_fail += 1
        graphs.append(g)

    if verbose and n_fail:
        print(f"Warning: {n_fail}/{len(smiles)} SMILES failed to parse")
    return graphs, ftr


def make_loaders(train, val, test=None, batch_size=32, num_workers=0, pin_memory=False):
    """Wrap graph lists in PyG DataLoaders."""
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    loaders = {
        "train": DataLoader(train, shuffle=True, **kw),
        "val": DataLoader(val, shuffle=False, **kw),
    }
    if test is not None:
        loaders["test"] = DataLoader(test, shuffle=False, **kw)
    return loaders


def standardize_graphs(train, val, test=None, featuriser=None, scaler=None):
    """Standardise continuous atom scalars (fit on train only).

    Returns ``(train, val, test, fitted_scaler)``.
    """
    if featuriser is None or featuriser.atom_cont_idx is None:
        raise RuntimeError("Featuriser must process ≥1 molecule before standardisation")
    fitted = scaler or StandardizeAtomScalars(idx=featuriser.atom_cont_idx)
    if scaler is None:
        fitted.fit(train)
    train = [fitted.transform(g) for g in train]
    val = [fitted.transform(g) for g in val]
    test = [fitted.transform(g) for g in test] if test else None
    return train, val, test, fitted
