#!/usr/bin/env python3
"""
Pre-train a GNN on the computational RedoxDB dataset (90/10 split).

Each model has its own HPO-tuned config (e.g. configs/pretrain_gat.yaml).
Pass --config to override the default per-model config.

Usage
-----
    # Single model
    python scripts/pretrain.py --model gat

    # All five models
    python scripts/pretrain.py --all
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

# ── project imports ───────────────────────────────────────────────────────────
from RedEx.data import load_graphs
from RedEx.models import create_model
from RedEx.utils.training import (
    load_config, evaluate_model, EarlyStopping, get_criterion, seed_everything,
)


def train_single_model(
    model_type: str,
    config: dict,
    train_data: list,
    val_data: list,
    device: torch.device,
    save_dir: Path | None = None,
):
    """Train one GNN on the computational dataset and save a checkpoint."""

    seed_everything(config["train"]["seed"])

    batch_size = config["data"]["batch_size"]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=config["data"]["num_workers"],
                              pin_memory=config["data"]["pin_memory"])
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=config["data"]["num_workers"],
                            pin_memory=config["data"]["pin_memory"])

    # ── model config ──────────────────────────────────────────────────────
    mc = config["model"].copy()
    mc["in_dim"] = mc.pop("input_dim")
    mc["out_dim"] = mc.pop("output_dim")
    if model_type != "gat":
        mc.pop("num_heads", None)
    if model_type == "dmpnn" and "depth" not in mc:
        mc["depth"] = mc.pop("num_layers", 4)

    model = create_model(model_type, **mc).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[{model_type.upper()}]  parameters: {n_params:,}")

    tc = config["train"]
    criterion = get_criterion(tc["task"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tc["lr"]),
                                  weight_decay=float(tc["weight_decay"]))
    es = EarlyStopping(patience=tc["patience"], verbose=False)

    num_epochs = tc["num_epochs"]
    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    best_metrics = {}
    history = {"train_loss": [], "val_mae": [], "val_rmse": [], "val_r2": []}

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out.view(-1), batch.y.view(-1))
            loss.backward()
            max_norm = 0.5 if model_type == "dmpnn" else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            loss_sum += loss.item() * batch.num_graphs
        train_loss = loss_sum / len(train_data)

        val = evaluate_model(model, val_loader, criterion, "regression", device)
        val_mae = val["MAE"]

        history["train_loss"].append(float(train_loss))
        history["val_mae"].append(float(val_mae))
        history["val_rmse"].append(float(val["RMSE"]))
        history["val_r2"].append(float(val["R2"]))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{num_epochs}  loss={train_loss:.4f}  "
                  f"MAE={val_mae:.4f}  R²={val['R2']:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = model.state_dict().copy()
            best_metrics = val.copy()

        es(val_mae, model)
        if es.early_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - t0
    print(f"  Best epoch {best_epoch}  MAE={best_val_mae:.4f}  R²={best_metrics['R2']:.4f}  "
          f"({elapsed:.0f}s)")

    # ── save ──────────────────────────────────────────────────────────────
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model_state_dict": best_state,
            "model_config": mc,
            "model_type": model_type,
            "epoch": best_epoch,
            "val_mae": best_val_mae,
            "val_metrics": best_metrics,
            "training_config": tc,
            "training_time": elapsed,
            "n_parameters": n_params,
            "timestamp": datetime.now().isoformat(),
        }
        torch.save(ckpt, save_dir / f"{model_type}_full_pretrain.pth")
        with open(save_dir / f"{model_type}_training_log.json", "w") as f:
            json.dump(history, f, indent=2)

    return {
        "model_type": model_type,
        "best_epoch": best_epoch,
        "val_mae": best_val_mae,
        "val_rmse": best_metrics.get("RMSE", float("nan")),
        "val_r2": best_metrics.get("R2", float("nan")),
        "time_s": elapsed,
        "n_params": n_params,
    }


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Pre-train GNN on computational dataset")
    ap.add_argument("--model", choices=["gcn", "gat", "gin", "graphsage", "dmpnn"])
    ap.add_argument("--all", action="store_true", help="Train all models")
    ap.add_argument("--config", help="Override config path (default: per-model)")
    ap.add_argument("--data", default="data/RedoxDB_Complete.csv")
    ap.add_argument("--output-dir", default="weights/pretrained")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.all and not args.model:
        ap.error("Specify --model or --all")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load data ─────────────────────────────────────────────────────────
    graphs, _ = load_graphs(args.data, verbose=True)
    valid = [g for g in graphs if g is not None]
    y = np.array([g.y.item() for g in valid])
    try:
        bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    except Exception:
        bins = pd.cut(y, bins=5, labels=False)
    tr_idx, va_idx = train_test_split(range(len(valid)), test_size=args.val_ratio,
                                       stratify=bins, random_state=args.seed)
    train_data = [valid[i] for i in tr_idx]
    val_data = [valid[i] for i in va_idx]
    print(f"Train {len(train_data)} / Val {len(val_data)}  (device={device})")

    models = ["gcn", "gat", "graphsage", "gin", "dmpnn"] if args.all else [args.model]
    results = []
    save_dir = Path(args.output_dir)

    for mt in models:
        # Use a per-model config if one exists, otherwise fall back to --config
        if args.config:
            cfg_path = Path(args.config)
        else:
            cfg_path = Path(f"configs/pretrain_{mt}.yaml")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        cfg = load_config(cfg_path)
        cfg["train"]["seed"] = args.seed
        print(f"\n[{mt.upper()}] using config: {cfg_path}")
        try:
            r = train_single_model(mt, cfg, train_data, val_data, device, save_dir)
            results.append(r)
        except Exception as exc:
            print(f"  ERROR training {mt}: {exc}")

    if results:
        df = pd.DataFrame(results).sort_values("val_mae")
        print("\n" + df.to_string(index=False))
        df.to_csv(save_dir / "pretrain_summary.csv", index=False)


if __name__ == "__main__":
    main()
