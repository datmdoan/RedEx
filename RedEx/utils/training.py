import random
import numpy as np
import torch
from torch import nn
import yaml
from pathlib import Path
from sklearn import metrics as sk_metrics


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 50, verbose: bool = False, min_delta: float = 0.0):
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model)
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, model)
            self.counter = 0

    def _save(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f}).")
        self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.val_loss_min = val_loss


def evaluate_model(model, data_loader, criterion, task="regression", device="cpu"):
    """Evaluate *model* on *data_loader* and return a metrics dict."""
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            outputs = model(batch)
            targets = batch.y.float()
            loss = criterion(outputs, targets)
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(targets.cpu().numpy())
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    y_true = np.array(y_true).squeeze()
    y_pred = np.array(y_pred).squeeze()

    result = {"loss": avg_loss}
    if task == "regression":
        result["RMSE"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        result["MAE"] = float(np.mean(np.abs(y_true - y_pred)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        result["R2"] = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
    return result


def load_config(path):
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_criterion(task: str):
    """Return loss function for the given task type."""
    if task == "regression":
        return nn.MSELoss()
    if task == "binary":
        return nn.BCEWithLogitsLoss()
    if task == "multiclass":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown task type: {task}")


def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
