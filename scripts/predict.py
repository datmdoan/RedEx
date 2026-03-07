#!/usr/bin/env python3
"""
Predict redox potentials for arbitrary SMILES using a fine-tuned ensemble.

Usage
-----
    # Predict from command line
    python scripts/predict.py \\
        --model-dir weights/finetuned/gat_partial \\
        --smiles "c1ccc2c(c1)[nH]c1ccccc12" "c1ccc(-c2ccccn2)cc1"

    # Predict from a CSV (must have an 'oxidised_smiles' column)
    python scripts/predict.py \\
        --model-dir weights/finetuned/gat_partial \\
        --csv data/raw/Experimental_redox_potentials.csv \\
        --output predictions.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from RedEx.data.featuriser import MoleculeFeaturiser, smiles_to_data
from RedEx.models import create_model


def load_ensemble(model_dir: str | Path, device: torch.device):
    """Load all fold models + metadata from *model_dir*.

    Returns ``(models_list, model_type, model_config)``.
    """
    model_dir = Path(model_dir)
    meta = torch.load(model_dir / "ensemble_metadata.pth", map_location=device)
    model_type = meta["model_type"]
    model_config = meta["model_config"]
    n_folds = meta["n_folds"]

    models = []
    for fi in range(1, n_folds + 1):
        ckpt = torch.load(model_dir / f"fold{fi}_model.pth", map_location=device)
        m = create_model(model_type, **model_config)
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(device).eval()
        models.append(m)
    return models, model_type, model_config


def predict_smiles(smiles_list: list[str], models: list, device: torch.device,
                   batch_size: int = 64):
    """Return ``(mean_preds, std_preds)`` arrays for each SMILES."""
    ftr = MoleculeFeaturiser()
    graphs = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        g = smiles_to_data(smi, ftr)
        if g is not None:
            graphs.append(g)
            valid_idx.append(i)

    if not graphs:
        return np.full(len(smiles_list), np.nan), np.full(len(smiles_list), np.nan)

    all_preds = []
    for model in models:
        preds = []
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[start : start + batch_size]).to(device)
            with torch.no_grad():
                out = model(batch).squeeze().cpu().numpy()
            preds.extend(out.tolist() if out.ndim else [out.item()])
        all_preds.append(preds)

    mean = np.mean(all_preds, axis=0)
    std = np.std(all_preds, axis=0)

    # map back
    full_mean = np.full(len(smiles_list), np.nan)
    full_std = np.full(len(smiles_list), np.nan)
    for j, idx in enumerate(valid_idx):
        full_mean[idx] = mean[j]
        full_std[idx] = std[j]
    return full_mean, full_std


def main():
    ap = argparse.ArgumentParser(description="Predict redox potentials with a fine-tuned ensemble")
    ap.add_argument("--model-dir", required=True,
                    help="Directory with fold*.pth + ensemble_metadata.pth")
    ap.add_argument("--smiles", nargs="+", default=None, help="SMILES strings")
    ap.add_argument("--csv", default=None, help="CSV file with oxidised_smiles column")
    ap.add_argument("--smiles-col", default="oxidised_smiles", help="SMILES column name")
    ap.add_argument("--output", default=None, help="Output CSV path (default: stdout)")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if args.smiles is None and args.csv is None:
        ap.error("Provide --smiles or --csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, model_type, _ = load_ensemble(args.model_dir, device)
    print(f"Loaded {len(models)}-model {model_type.upper()} ensemble (device={device})")

    if args.csv:
        df = pd.read_csv(args.csv)
        smiles_list = df[args.smiles_col].astype(str).tolist()
    else:
        smiles_list = args.smiles

    mean, std = predict_smiles(smiles_list, models, device, args.batch_size)

    out = pd.DataFrame({
        "smiles": smiles_list,
        "predicted_potential_V": np.round(mean, 4),
        "uncertainty_V": np.round(std, 4),
    })

    if args.output:
        out.to_csv(args.output, index=False)
        print(f"Saved {len(out)} predictions to {args.output}")
    else:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
