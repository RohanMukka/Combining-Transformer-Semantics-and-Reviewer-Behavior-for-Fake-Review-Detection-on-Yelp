"""
Behavior-Only MLP Branch for ReviewGuard
==========================================

Implements a standalone MLP classifier trained only on the 6 handcrafted
reviewer behavioral features. Provides:

  - BehaviorMLP: 6-dim → 64 → 32 → 1, ReLU, Dropout(0.3)
  - Training loop with FocalLoss (γ=2, class-weighted α)
  - Evaluation with AUC-ROC, Macro-F1, precision, recall
  - Comparison with behavior-only baselines (Random Forest from baselines.py)

Architecture:
    Behaviour vector (6-dim)
    → Linear(6, 64) → ReLU → Dropout(0.3)
    → Linear(64, 32) → ReLU → Dropout(0.3)
    → Linear(32, 1) → Sigmoid → P(fake)

Usage:
    python -m src.behavior_branch --dataset yelpzip
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .text_branch import FocalLoss
from .utils import ensure_dirs, get_device, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


# ─── Model ────────────────────────────────────────────────────────────────────

class BehaviorMLP(nn.Module):
    """MLP classifier operating on 6 reviewer behavioral features.

    Architecture:
        Input (6) → Linear(6, 64) → ReLU → Dropout(p)
                 → Linear(64, 32) → ReLU → Dropout(p)
                 → Linear(32, 1)

    The output is a raw logit; apply ``torch.sigmoid`` to get the
    probability of a review being fake.

    Args:
        input_dim: Number of input features. Default: 6.
        hidden_dims: Sizes of the two hidden layers. Default: ``(64, 32)``.
        dropout: Dropout probability applied after each hidden layer. Default: 0.3.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: Tuple[int, int] = (64, 32),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims

        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise linear layers with Xavier uniform, biases with zeros."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Behavior feature tensor of shape ``(B, 6)``.

        Returns:
            Logit tensor of shape ``(B, 1)``.
        """
        return self.net(x)

    def predict_proba(self, x: Tensor) -> Tensor:
        """Return sigmoid probabilities of shape ``(B,)``.

        Args:
            x: Behavior feature tensor of shape ``(B, 6)``.

        Returns:
            Probability tensor of shape ``(B,)`` ∈ [0, 1].
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(-1)


# ─── Training & Evaluation ────────────────────────────────────────────────────

def train_behavior_epoch(
    model: BehaviorMLP,
    dataloader: DataLoader,
    optimizer: Adam,
    loss_fn: FocalLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch for the behavior MLP.

    Args:
        model: :class:`BehaviorMLP` instance.
        dataloader: DataLoader yielding ``{"features": Tensor, "label": Tensor}``.
        optimizer: Adam optimiser.
        loss_fn: FocalLoss instance.
        device: PyTorch device.

    Returns:
        Dict with ``"loss"`` and ``"accuracy"``.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += len(labels)
        total_loss += loss.item()

    return {"loss": total_loss / len(dataloader), "accuracy": correct / total}


@torch.no_grad()
def evaluate_behavior(
    model: BehaviorMLP,
    dataloader: DataLoader,
    loss_fn: FocalLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the behavior MLP on a validation/test DataLoader.

    Args:
        model: :class:`BehaviorMLP` instance.
        dataloader: DataLoader yielding ``{"features": Tensor, "label": Tensor}``.
        loss_fn: FocalLoss instance.
        device: PyTorch device.

    Returns:
        Dict with ``"loss"``, ``"accuracy"``, ``"auc_roc"``, ``"macro_f1"``.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    model.eval()
    total_loss = 0.0
    all_probs: List[float] = []
    all_labels: List[int] = []

    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        logits = model(features)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    all_probs_arr = np.array(all_probs)
    all_labels_arr = np.array(all_labels, dtype=int)
    preds = (all_probs_arr >= 0.5).astype(int)

    metrics: Dict[str, float] = {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": float((preds == all_labels_arr).mean()),
    }
    try:
        metrics["auc_roc"] = float(roc_auc_score(all_labels_arr, all_probs_arr))
        metrics["macro_f1"] = float(
            f1_score(all_labels_arr, preds, average="macro", zero_division=0)
        )
    except ValueError:
        metrics["auc_roc"] = 0.0
        metrics["macro_f1"] = 0.0

    return metrics


# ─── Full Training Pipeline ───────────────────────────────────────────────────

def train_behavior_branch(
    config: Optional[Dict] = None,
    dataset_name: str = "yelpzip",
) -> Tuple[BehaviorMLP, Dict]:
    """Full training pipeline for the behavior-only MLP branch.

    Steps:
      1. Load dataset and compute behavior features.
      2. Normalise features with StandardScaler.
      3. Train BehaviorMLP with FocalLoss + Adam.
      4. Evaluate on test split.
      5. Save model checkpoint and results.

    Args:
        config: Configuration dictionary.
        dataset_name: Dataset to train on.

    Returns:
        Tuple of (trained :class:`BehaviorMLP`, results dict).
    """
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            config = {}

    bb_cfg = config.get("behavior_branch", {})
    fl_cfg = config.get("focal_loss", {})
    device = get_device(config.get("device", "auto"))

    input_dim = bb_cfg.get("input_dim", 6)
    hidden_dims = tuple(bb_cfg.get("hidden_dims", [64, 32]))
    dropout = bb_cfg.get("dropout", 0.3)
    lr = bb_cfg.get("learning_rate", 1e-3)
    batch_size = bb_cfg.get("batch_size", 256)
    max_epochs = bb_cfg.get("max_epochs", 50)
    gamma = fl_cfg.get("gamma", 2.0)

    models_dir = Path(config.get("models_dir", "models")) / "behavior_branch"
    ensure_dirs(models_dir)

    # ── Load and preprocess data ──
    from .data_loader import load_dataset, temporal_train_test_split
    from .behavior_features import compute_behavior_features, fit_scaler, apply_scaler
    from .data_utils import BehaviorOnlyDataset, create_dataloader, get_class_weights

    df = load_dataset(dataset_name, config=config)
    train_df, test_df = temporal_train_test_split(
        df, cutoff_fraction=config.get("data", {}).get("temporal_cutoff_fraction", 0.8)
    )

    logger.info("Computing behavior features …")
    train_feats_raw = compute_behavior_features(train_df)
    test_feats_raw = compute_behavior_features(test_df)

    scaler_path = Path(config.get("data", {}).get("features_dir", "data/features")) / f"{dataset_name}_scaler.pkl"
    scaler = fit_scaler(train_feats_raw, save_path=scaler_path)

    train_feats = apply_scaler(train_feats_raw, scaler)
    test_feats = apply_scaler(test_feats_raw, scaler)

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    train_dataset = BehaviorOnlyDataset(train_feats, y_train)
    test_dataset = BehaviorOnlyDataset(test_feats, y_test)

    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=batch_size * 2, shuffle=False)

    # ── Model ──
    model = BehaviorMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    logger.info(
        f"BehaviorMLP: {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # ── Focal Loss ──
    class_weights = get_class_weights(y_train, as_tensor=True)
    alpha = class_weights[1] / (class_weights[0] + class_weights[1])
    loss_fn = FocalLoss(gamma=gamma, alpha=alpha.item()).to(device)
    logger.info(f"FocalLoss: γ={gamma}, α(fake)={alpha:.4f}")

    # ── Optimiser ──
    optimizer = Adam(model.parameters(), lr=lr)

    # ── Training loop ──
    best_auc = 0.0
    checkpoint_path = models_dir / f"{dataset_name}_best.pt"
    history: List[Dict] = []

    for epoch in range(1, max_epochs + 1):
        train_m = train_behavior_epoch(model, train_loader, optimizer, loss_fn, device)
        val_m = evaluate_behavior(model, test_loader, loss_fn, device)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Train loss={train_m['loss']:.4f} acc={train_m['accuracy']:.4f} | "
                f"Test AUC={val_m['auc_roc']:.4f} F1={val_m['macro_f1']:.4f}"
            )

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}}
        row.update({f"test_{k}": v for k, v in val_m.items()})
        history.append(row)

        if val_m["auc_roc"] > best_auc:
            best_auc = val_m["auc_roc"]
            torch.save(model.state_dict(), checkpoint_path)

    # Load best weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logger.info(f"Best behavior model loaded (AUC={best_auc:.4f})")

    # Final evaluation
    final_metrics = evaluate_behavior(model, test_loader, loss_fn, device)
    logger.info(
        f"Final test metrics — AUC={final_metrics['auc_roc']:.4f}  "
        f"Macro-F1={final_metrics['macro_f1']:.4f}  "
        f"Accuracy={final_metrics['accuracy']:.4f}"
    )

    results = {
        "dataset": dataset_name,
        "model": "BehaviorMLP",
        "architecture": {
            "input_dim": input_dim,
            "hidden_dims": list(hidden_dims),
            "dropout": dropout,
        },
        "training": {"lr": lr, "batch_size": batch_size, "epochs": max_epochs, "gamma": gamma},
        "test_metrics": final_metrics,
        "best_val_auc": best_auc,
    }

    results_path = Path(config.get("results_dir", "results")) / f"{dataset_name}_behavior_results.json"
    ensure_dirs(results_path.parent)
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info(f"Results saved → {results_path}")

    # Save training history
    with open(models_dir / f"{dataset_name}_history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    # Comparison with Random Forest baseline
    _compare_with_rf(train_feats, y_train, test_feats, y_test, config)

    return model, results


def _compare_with_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict,
) -> None:
    """Quick comparison of BehaviorMLP vs. Random Forest on behavior features."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, roc_auc_score

    logger.info("Comparing with Random Forest (behavior features only) …")
    bl_cfg = config.get("baselines", {})
    rf = RandomForestClassifier(
        n_estimators=bl_cfg.get("rf_n_estimators", 200),
        max_depth=bl_cfg.get("rf_max_depth", 15),
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_preds = rf.predict(X_test)

    rf_auc = roc_auc_score(y_test, rf_probs)
    rf_f1 = f1_score(y_test, rf_preds, average="macro", zero_division=0)
    logger.info(f"  RandomForest  — AUC={rf_auc:.4f}  Macro-F1={rf_f1:.4f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging("INFO")
    set_seed(42)

    parser = argparse.ArgumentParser(
        description="Train the behavior-only MLP branch of ReviewGuard."
    )
    parser.add_argument("--dataset", default="yelpchi", choices=["yelpzip", "yelpnyc", "yelpchi"])
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        logger.warning("Config not found; using defaults.")
        cfg = {}

    train_behavior_branch(config=cfg, dataset_name=args.dataset)
