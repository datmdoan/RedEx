#!/usr/bin/env python3
"""
Fine-tune a pre-trained GNN on experimental data with 5-fold group CV.

Freeze strategies:
    none           – full fine-tuning (all layers trainable, same LR)
    head_only      – freeze everything except prediction head
    readout_head   – freeze convolutions, train readout + head
    top_layers     – freeze bottom X% of convs, train rest + readout + head

Usage
-----
    python scripts/finetune.py --model gat \
        --pretrained weights/pretrained/gat_full_pretrain.pth \
        --config configs/finetune_gat.yaml
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

from RedEx.data import load_graphs
from RedEx.models import create_model
from RedEx.utils.training import evaluate_model, EarlyStopping, seed_everything, load_config


# ══════════════════════════════════════════════════════════════════════════════
#  Stratified group splitting
# ══════════════════════════════════════════════════════════════════════════════


def canonicalize(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None


def stratified_group_test_split(df, test_ratio=0.2, seed=42):
    """Per-class molecule allocation: ~test_ratio of molecules → test."""
    rng = np.random.RandomState(seed)
    test_idx, cv_idx = [], []
    for cls in sorted(df["class"].unique()):
        mols = df.loc[df["class"] == cls, "group_id"].unique()
        n_test = max(1, int(np.round(len(mols) * test_ratio)))
        rng.shuffle(mols)
        test_mols = set(mols[:n_test])
        for idx, row in df[df["class"] == cls].iterrows():
            (test_idx if row["group_id"] in test_mols else cv_idx).append(idx)
    return np.array(cv_idx), np.array(test_idx)


def stratified_group_kfold(df_cv, n_folds=5, seed=42):
    """Round-robin per-class molecule assignment to folds."""
    rng = np.random.RandomState(seed)
    assign = np.full(len(df_cv), -1, dtype=int)
    for cls in sorted(df_cv["class"].unique()):
        mask = df_cv["class"] == cls
        mols = df_cv.loc[mask, "group_id"].unique()
        rng.shuffle(mols)
        mol2fold = {m: i % n_folds for i, m in enumerate(mols)}
        for idx, row in df_cv[mask].iterrows():
            assign[idx] = mol2fold[row["group_id"]]
    folds = []
    for f in range(n_folds):
        val = np.where(assign == f)[0]
        train = np.where((assign != f) & (assign >= 0))[0]
        folds.append((train, val))
    return folds


def load_experimental_cv(exp_path, test_ratio=0.2, n_folds=5, seed=42):
    """Load experimental CSV → graphs + stratified group CV/test split."""
    df = pd.read_csv(exp_path, encoding="latin1")
    smiles_col = "oxidised_smiles" if "oxidised_smiles" in df.columns else "SMILES"

    df["canonical_smiles"] = df[smiles_col].apply(canonicalize)
    df = df[df["canonical_smiles"].notna()].reset_index(drop=True)
    unique_smi = df["canonical_smiles"].unique()
    smi2gid = {s: i for i, s in enumerate(unique_smi)}
    df["group_id"] = df["canonical_smiles"].map(smi2gid)

    cv_idx, test_idx = stratified_group_test_split(df, test_ratio, seed)
    graphs, ftr = load_graphs(exp_path, verbose=False, encoding="latin1")
    valid = {i for i, g in enumerate(graphs) if g is not None}
    cv_idx = [i for i in cv_idx if i in valid]
    test_idx = [i for i in test_idx if i in valid]

    cv_data = [graphs[i] for i in cv_idx]
    test_data = [graphs[i] for i in test_idx]
    cv_meta = df.iloc[cv_idx].reset_index(drop=True)
    test_meta = df.iloc[test_idx].reset_index(drop=True)
    cv_groups = df["group_id"].values[cv_idx]

    n_mol = len(unique_smi)
    print(f"Experimental: {len(df)} measurements, {n_mol} unique molecules")
    print(f"  CV: {len(cv_data)}  Test: {len(test_data)}")
    return cv_data, cv_meta, test_data, test_meta, cv_groups


# ══════════════════════════════════════════════════════════════════════════════
#  Training helpers
# ══════════════════════════════════════════════════════════════════════════════


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch).squeeze(), batch.y)
        loss.backward()
        max_norm = 0.5 if "DMPNN" in model.__class__.__name__ else 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


def apply_freeze(model, strategy, freeze_ratio=0.5):
    """Freeze parameters according to *strategy*."""
    if strategy == "head_only":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True

    elif strategy == "readout_head":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.readout.parameters():
            p.requires_grad = True
        for p in model.head.parameters():
            p.requires_grad = True

    elif strategy == "top_layers":
        for p in model.parameters():
            p.requires_grad = False
        n = len(model.convs)
        n_freeze = int(n * freeze_ratio)
        for i in range(n_freeze, n):
            for p in model.convs[i].parameters():
                p.requires_grad = True
            if model.norms[i] is not None:
                for p in model.norms[i].parameters():
                    p.requires_grad = True
        for p in model.readout.parameters():
            p.requires_grad = True
        if not isinstance(model.post_mlp, nn.Identity):
            for p in model.post_mlp.parameters():
                p.requires_grad = True
        for p in model.head.parameters():
            p.requires_grad = True

    elif strategy in ("none",):
        for p in model.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Strategy={strategy}  trainable={trainable:,}/{total:,}")


def build_optimizer(model, lr, weight_decay):
    return torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay
    )


def fine_tune_fold(model_config, model_type, pretrained_state, train_data, val_data,
                   device, epochs, batch_size, lr, weight_decay, patience,
                   strategy, freeze_ratio):
    model = create_model(model_type, **model_config)
    model.load_state_dict(pretrained_state)
    model.to(device)

    apply_freeze(model, strategy, freeze_ratio)
    optimizer = build_optimizer(model, lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    criterion = nn.L1Loss()
    es = EarlyStopping(patience=patience)

    tl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    vl = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    best_mae, best_state = float("inf"), None
    history = {"train_loss": [], "val_mae": []}

    for ep in range(1, epochs + 1):
        loss = train_epoch(model, tl, optimizer, criterion, device)
        val = evaluate_model(model, vl, criterion, "regression", device)
        history["train_loss"].append(loss)
        history["val_mae"].append(val["MAE"])
        scheduler.step(val["MAE"])
        if val["MAE"] < best_mae:
            best_mae = val["MAE"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        es(val["MAE"], model)
        if es.early_stop:
            break

    return best_state, best_mae, history


# ══════════════════════════════════════════════════════════════════════════════
#  Cross-validation & ensemble evaluation
# ══════════════════════════════════════════════════════════════════════════════


def cross_validate(model_config, model_type, pretrained_state, cv_data, cv_meta,
                   cv_groups, device, n_folds, epochs, batch_size, lr, weight_decay,
                   patience, seed, strategy, freeze_ratio):
    folds = stratified_group_kfold(cv_meta, n_folds, seed)
    fold_models, fold_metrics, fold_histories = [], [], []
    cv_preds = np.zeros(len(cv_data))

    for fi, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"\n── Fold {fi}/{n_folds} ──")
        tr = [cv_data[i] for i in tr_idx]
        va = [cv_data[i] for i in va_idx]
        state, mae, hist = fine_tune_fold(
            model_config, model_type, pretrained_state, tr, va, device,
            epochs, batch_size, lr, weight_decay, patience,
            strategy, freeze_ratio,
        )
        fold_models.append(state)
        fold_histories.append(hist)

        # out-of-fold predictions
        m = create_model(model_type, **model_config)
        m.load_state_dict(state)
        m.to(device).eval()
        vl = DataLoader(va, batch_size=batch_size, shuffle=False)
        criterion = nn.L1Loss()
        vm = evaluate_model(m, vl, criterion, "regression", device)
        fold_metrics.append(vm)
        preds = []
        with torch.no_grad():
            for b in vl:
                b = b.to(device)
                p = m(b).squeeze().cpu().numpy()
                preds.extend(p.tolist() if p.ndim else [p.item()])
        cv_preds[va_idx] = preds
        print(f"  MAE={vm['MAE']:.4f}  R²={vm['R2']:.4f}")

    maes = [m["MAE"] for m in fold_metrics]
    r2s = [m["R2"] for m in fold_metrics]
    print(f"\nCV MAE: {np.mean(maes):.4f}±{np.std(maes):.4f}  "
          f"R²: {np.mean(r2s):.4f}±{np.std(r2s):.4f}")
    return fold_models, fold_metrics, cv_preds, fold_histories


def ensemble_predict(fold_models, model_type, model_config, data, device, batch_size=16):
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    all_preds = []
    for state in fold_models:
        m = create_model(model_type, **model_config)
        m.load_state_dict(state)
        m.to(device).eval()
        preds = []
        with torch.no_grad():
            for b in loader:
                b = b.to(device)
                p = m(b).squeeze().cpu().numpy()
                preds.extend(p.tolist() if p.ndim else [p.item()])
        all_preds.append(preds)
    return np.mean(all_preds, axis=0), np.std(all_preds, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(description="Fine-tune pre-trained GNN on experimental data")
    ap.add_argument("--model", required=True,
                    choices=["gcn", "gat", "gin", "graphsage", "dmpnn"])
    ap.add_argument("--pretrained", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--config", default=None, help="YAML config (overrides defaults)")
    ap.add_argument("--exp-data", default="data/Experimental_redox_potentials.csv")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--n-folds", type=int, default=None)
    ap.add_argument("--test-ratio", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--freeze-strategy", default=None,
                    choices=["none", "head_only", "readout_head", "top_layers"])
    ap.add_argument("--freeze-ratio", type=float, default=None)
    args = ap.parse_args()

    # ── merge config file with CLI overrides ──────────────────────────────
    # Defaults used when neither config nor CLI provides a value
    defaults = dict(lr=3e-3, batch_size=16, epochs=500, patience=30,
                    weight_decay=2e-5, n_folds=5, test_ratio=0.2, seed=42,
                    freeze_strategy="top_layers", freeze_ratio=0.5)

    if args.config is not None:
        cfg = load_config(args.config)
        tc = cfg.get("train", {})
        fc = cfg.get("freeze", {})
        cv = cfg.get("cv", {})
        mc = cfg.get("model", {})
        # Flatten config values into defaults
        defaults.update({
            "lr": tc.get("lr", defaults["lr"]),
            "batch_size": tc.get("batch_size", defaults["batch_size"]),
            "epochs": tc.get("num_epochs", defaults["epochs"]),
            "patience": tc.get("patience", defaults["patience"]),
            "weight_decay": tc.get("weight_decay", defaults["weight_decay"]),
            "seed": tc.get("seed", defaults["seed"]),
            "freeze_strategy": fc.get("strategy", defaults["freeze_strategy"]),
            "freeze_ratio": fc.get("freeze_ratio", defaults["freeze_ratio"]),
            "n_folds": cv.get("n_folds", defaults["n_folds"]),
            "test_ratio": cv.get("test_ratio", defaults["test_ratio"]),
        })
        if "dropout" in mc:
            defaults["dropout"] = mc["dropout"]

    # CLI args override config values
    def _resolve(cli_val, key):
        return cli_val if cli_val is not None else defaults.get(key)

    lr = _resolve(args.lr, "lr")
    batch_size = _resolve(args.batch_size, "batch_size")
    epochs = _resolve(args.epochs, "epochs")
    patience = _resolve(args.patience, "patience")
    weight_decay = _resolve(args.weight_decay, "weight_decay")
    n_folds = _resolve(args.n_folds, "n_folds")
    test_ratio = _resolve(args.test_ratio, "test_ratio")
    seed = _resolve(args.seed, "seed")
    freeze_strategy = _resolve(args.freeze_strategy, "freeze_strategy")
    freeze_ratio = _resolve(args.freeze_ratio, "freeze_ratio")
    dropout_override = _resolve(args.dropout, "dropout") if args.dropout is not None else defaults.get("dropout")

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
    model_config = ckpt["model_config"].copy()
    if dropout_override is not None:
        model_config["dropout"] = dropout_override
    pretrained_state = ckpt["model_state_dict"]
    print(f"Loaded checkpoint: {args.pretrained}  (epoch {ckpt.get('epoch', '?')})")

    # ── load experimental data ────────────────────────────────────────────
    cv_data, cv_meta, test_data, test_meta, cv_groups = load_experimental_cv(
        args.exp_data, test_ratio, n_folds, seed
    )

    # ── cross-validate ────────────────────────────────────────────────────
    fold_models, fold_metrics, cv_preds, histories = cross_validate(
        model_config, args.model, pretrained_state, cv_data, cv_meta,
        cv_groups, device, n_folds, epochs, batch_size,
        lr, weight_decay, patience, seed,
        freeze_strategy, freeze_ratio,
    )

    # ── ensemble test evaluation ──────────────────────────────────────────
    y_pred, y_std = ensemble_predict(fold_models, args.model, model_config, test_data, device)
    y_true = np.array([d.y.item() for d in test_data])
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\nTest ensemble:  MAE={mae:.4f}  RMSE={rmse:.4f}  R\u00b2={r2:.4f}")

    # ── save ──────────────────────────────────────────────────────────────
    strat = freeze_strategy
    if strat == "top_layers":
        strat += f"_{int(freeze_ratio*100)}pct"

    out_dir = Path(args.output_dir) if args.output_dir else Path(f"results/{args.model}/{strat}")
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for fi, state in enumerate(fold_models, 1):
        torch.save({
            "model_state_dict": state,
            "model_config": model_config,
            "model_type": args.model,
            "fold": fi,
        }, model_dir / f"fold{fi}_model.pth")

    torch.save({
        "model_type": args.model,
        "model_config": model_config,
        "n_folds": n_folds,
        "test_mae": mae, "test_rmse": rmse, "test_r2": r2,
        "freeze_strategy": freeze_strategy,
        "freeze_ratio": freeze_ratio,
        "timestamp": datetime.now().isoformat(),
    }, model_dir / "ensemble_metadata.pth")

    pred_df = test_meta.copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["y_std"] = y_std
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(histories, f, indent=2, default=float)

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
