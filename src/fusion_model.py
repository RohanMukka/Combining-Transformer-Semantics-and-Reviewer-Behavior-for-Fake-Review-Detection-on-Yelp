"""
ReviewGuard Fusion Model
========================

Combines the 768-dim RoBERTa [CLS] text embedding with the 6-dim reviewer
behavior feature vector via concatenation, then passes the 774-dim fused
representation through a 2-layer MLP classifier.

Architecture:
    Text embedding   (768-dim)  ─────────┐
                                         ├── Concat (774-dim)
    Behavior vector  (6-dim)   ──────────┘
           ↓
    Linear(774, 256) → ReLU → Dropout(0.3)
           ↓
    Linear(256, 64)  → ReLU → Dropout(0.3)
           ↓
    Linear(64, 1) → Sigmoid → P(fake)

Key design decisions:
  - Pre-extracted embeddings: RoBERTa is NOT re-run during fusion training.
    Embeddings are extracted once and cached (see text_branch.extract_and_save_embeddings).
  - Focal loss (γ=2, class-weighted α) for class imbalance handling.
  - FusionDataset loads pre-extracted embedding arrays from disk.

Usage:
    # Extract embeddings first:
    python -m src.text_branch --mode extract --dataset yelpzip

    # Train fusion model:
    python -m src.fusion_model --dataset yelpzip
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
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .text_branch import FocalLoss
from .utils import ensure_dirs, get_device, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


# ─── Fusion Dataset ───────────────────────────────────────────────────────────

class FusionDataset(torch.utils.data.Dataset):
    """Dataset for the ReviewGuard fusion model.

    Loads pre-extracted text embeddings and behavior features, concatenates
    them to form the 774-dim fused input vector.

    Args:
        text_embeddings: Float array of shape ``(N, 768)`` — RoBERTa [CLS] vectors.
        behavior_features: Float array of shape ``(N, 6)`` — normalised behaviour.
        labels: Binary label array of length N.
    """

    def __init__(
        self,
        text_embeddings: np.ndarray,
        behavior_features: np.ndarray,
        labels: Union[List[int], np.ndarray],
    ) -> None:
        assert len(text_embeddings) == len(behavior_features) == len(labels), (
            "text_embeddings, behavior_features, and labels must have the same length."
        )
        self.text_emb = torch.tensor(text_embeddings.astype(np.float32))
        self.behavior = torch.tensor(behavior_features.astype(np.float32))
        self.labels = torch.tensor(list(labels), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "text_embedding": self.text_emb[idx],
            "behavior": self.behavior[idx],
            "label": self.labels[idx],
        }


# ─── ReviewGuard Fusion Model ─────────────────────────────────────────────────

class ReviewGuard(nn.Module):
    """ReviewGuard: Hybrid Fake Review Detector.

    Fuses 768-dim RoBERTa text embeddings with 6-dim reviewer behavior
    vectors through a 2-layer MLP.

    Architecture:
        concat(text_emb, behavior)  →  774-dim
        → Linear(774, 256) → ReLU → Dropout(p)
        → Linear(256, 64)  → ReLU → Dropout(p)
        → Linear(64, 1)

    Args:
        text_dim: Dimension of text embedding. Default: 768 (RoBERTa-base).
        behavior_dim: Dimension of behavior feature vector. Default: 6.
        hidden_dims: Tuple of hidden layer sizes. Default: ``(256, 64)``.
        dropout: Dropout probability. Default: 0.3.
        output_dim: Number of output neurons. Default: 1 (binary classification).
    """

    def __init__(
        self,
        text_dim: int = 768,
        behavior_dim: int = 6,
        hidden_dims: Tuple[int, ...] = (256, 64),
        dropout: float = 0.3,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        fused_dim = text_dim + behavior_dim  # 774

        layers: List[nn.Module] = []
        in_dim = fused_dim

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"ReviewGuard initialised | "
            f"Input: {fused_dim}={text_dim}+{behavior_dim} | "
            f"MLP: {fused_dim}→{'→'.join(str(h) for h in hidden_dims)}→{output_dim} | "
            f"Params: {total_params:,}"
        )

    def _init_weights(self) -> None:
        """Xavier uniform init for all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, text_embedding: Tensor, behavior: Tensor) -> Tensor:
        """Forward pass.

        Args:
            text_embedding: RoBERTa [CLS] tensor of shape ``(B, 768)``.
            behavior: Behavior feature tensor of shape ``(B, 6)``.

        Returns:
            Logit tensor of shape ``(B, 1)``.
        """
        fused = torch.cat([text_embedding, behavior], dim=-1)  # (B, 774)
        return self.mlp(fused)  # (B, 1)

    def predict_proba(
        self, text_embedding: Tensor, behavior: Tensor
    ) -> Tensor:
        """Return sigmoid probabilities of the fake class.

        Args:
            text_embedding: Shape ``(B, 768)``.
            behavior: Shape ``(B, 6)``.

        Returns:
            Probability tensor of shape ``(B,)`` ∈ [0, 1].
        """
        with torch.no_grad():
            logits = self.forward(text_embedding, behavior)
            return torch.sigmoid(logits).squeeze(-1)


# ─── Training / Evaluation ────────────────────────────────────────────────────

def train_fusion_epoch(
    model: ReviewGuard,
    dataloader: DataLoader,
    optimizer: Adam,
    loss_fn: FocalLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch for the fusion model.

    Args:
        model: :class:`ReviewGuard` instance.
        dataloader: DataLoader yielding batches from :class:`FusionDataset`.
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

    pbar = tqdm(dataloader, desc="  [Train]", leave=False)
    for batch in pbar:
        text_emb = batch["text_embedding"].to(device)
        behavior = batch["behavior"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(text_emb, behavior)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += len(labels)
        total_loss += loss.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {"loss": total_loss / len(dataloader), "accuracy": correct / total}


@torch.no_grad()
def evaluate_fusion(
    model: ReviewGuard,
    dataloader: DataLoader,
    loss_fn: FocalLoss,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate the fusion model on a DataLoader.

    Args:
        model: :class:`ReviewGuard` instance.
        dataloader: DataLoader.
        loss_fn: FocalLoss instance.
        device: PyTorch device.

    Returns:
        Tuple of (metrics_dict, probability_array, label_array).
    """
    from sklearn.metrics import f1_score, roc_auc_score

    model.eval()
    total_loss = 0.0
    all_probs: List[float] = []
    all_labels: List[int] = []

    for batch in tqdm(dataloader, desc="  [Eval]", leave=False):
        text_emb = batch["text_embedding"].to(device)
        behavior = batch["behavior"].to(device)
        labels = batch["label"].to(device)

        logits = model(text_emb, behavior)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    all_probs_arr = np.array(all_probs, dtype=np.float32)
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
        from sklearn.metrics import precision_score, recall_score
        metrics["precision_fake"] = float(
            precision_score(all_labels_arr, preds, pos_label=1, zero_division=0)
        )
        metrics["recall_fake"] = float(
            recall_score(all_labels_arr, preds, pos_label=1, zero_division=0)
        )
        metrics["precision_genuine"] = float(
            precision_score(all_labels_arr, preds, pos_label=0, zero_division=0)
        )
        metrics["recall_genuine"] = float(
            recall_score(all_labels_arr, preds, pos_label=0, zero_division=0)
        )
    except ValueError:
        metrics.update({"auc_roc": 0.0, "macro_f1": 0.0})

    return metrics, all_probs_arr, all_labels_arr


# ─── Embedding Loader ─────────────────────────────────────────────────────────

def load_embeddings(
    dataset_name: str,
    split: str,
    config: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-extracted text embeddings, behavior features, and labels.

    Falls back to random embeddings if saved files don't exist (useful for
    testing the pipeline without running the full RoBERTa extraction).

    Args:
        dataset_name: Dataset identifier.
        split: ``"train"`` or ``"test"``.
        config: Configuration dictionary.

    Returns:
        Tuple of (text_embeddings, behavior_features, labels).
    """
    features_dir = Path(config.get("data", {}).get("features_dir", "data/features"))
    processed_dir = Path(config.get("data", {}).get("processed_dir", "data/processed"))

    emb_path = features_dir / f"{dataset_name}_{split}_text_emb.npy"
    behavior_path = features_dir / f"{dataset_name}_{split}_features.npy"
    data_path = processed_dir / f"{dataset_name}_{split}.parquet"

    # Load labels from parquet
    if data_path.exists():
        import pandas as pd
        df = pd.read_parquet(data_path)
        labels = df["label"].values
        n = len(labels)
    else:
        logger.warning(f"Parquet not found: {data_path}. Generating random labels.")
        n = 5000
        labels = np.random.choice([0, 1], size=n, p=[0.868, 0.132])

    # Load text embeddings
    if not emb_path.exists():
        raise FileNotFoundError(
            f"CRITICAL: Text embeddings NOT found at {emb_path}. "
            "Fusion model cannot train on empty data. Run Step 2 (RoBERTa Extraction) first!"
        )
    text_emb = np.load(emb_path)
    logger.info(f"Loaded text embeddings: {emb_path} {text_emb.shape}")

    # Load behavior features
    if not behavior_path.exists():
        raise FileNotFoundError(
            f"CRITICAL: Behavior features NOT found at {behavior_path}. "
            "Run Step 2 (Behavior Extraction) first!"
        )
    behavior = np.load(behavior_path)
    logger.info(f"Loaded behavior features: {behavior_path} {behavior.shape}")

    # Ensure alignment
    min_n = min(len(text_emb), len(behavior), len(labels))
    return text_emb[:min_n], behavior[:min_n], labels[:min_n]


# ─── Full Training Pipeline ───────────────────────────────────────────────────

def train_fusion_model(
    config: Optional[Dict] = None,
    dataset_name: str = "yelpzip",
) -> Tuple[ReviewGuard, Dict]:
    """Full training pipeline for the ReviewGuard fusion model.

    Requires pre-extracted RoBERTa embeddings (from text_branch.py) and
    pre-computed behavior features (from behavior_features.py).

    Args:
        config: Configuration dictionary.
        dataset_name: Dataset identifier.

    Returns:
        Tuple of (trained :class:`ReviewGuard`, results dict).
    """
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            config = {}

    fusion_cfg = config.get("fusion", {})
    fl_cfg = config.get("focal_loss", {})
    device = get_device(config.get("device", "auto"))

    text_dim = fusion_cfg.get("text_dim", 768)
    behavior_dim = fusion_cfg.get("behavior_dim", 6)
    hidden_dims = tuple(fusion_cfg.get("hidden_dims", [256, 64]))
    dropout = fusion_cfg.get("dropout", 0.3)
    lr = fusion_cfg.get("learning_rate", 1e-3)
    batch_size = fusion_cfg.get("batch_size", 128)
    max_epochs = fusion_cfg.get("max_epochs", 30)
    gamma = fl_cfg.get("gamma", 2.0)

    models_dir = Path(config.get("models_dir", "models")) / "fusion"
    ensure_dirs(models_dir)

    # ── Load embeddings and features ──
    logger.info(f"Loading training embeddings for {dataset_name} …")
    train_emb, train_behavior, y_train = load_embeddings(dataset_name, "train", config)

    logger.info(f"Loading test embeddings for {dataset_name} …")
    test_emb, test_behavior, y_test = load_embeddings(dataset_name, "test", config)

    logger.info(
        f"Train: {len(y_train):,} samples (fake={y_train.mean():.1%}) | "
        f"Test: {len(y_test):,} samples (fake={y_test.mean():.1%})"
    )

    # ── Datasets & DataLoaders ──
    from .data_utils import get_class_weights

    train_dataset = FusionDataset(train_emb, train_behavior, y_train)
    test_dataset = FusionDataset(test_emb, test_behavior, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    # ── Model ──
    model = ReviewGuard(
        text_dim=text_dim,
        behavior_dim=behavior_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

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

    logger.info(f"Starting fusion model training: {max_epochs} epochs")

    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        train_m = train_fusion_epoch(model, train_loader, optimizer, loss_fn, device)
        val_m, _, _ = evaluate_fusion(model, test_loader, loss_fn, device)
        elapsed = time.perf_counter() - t0

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Train loss={train_m['loss']:.4f} acc={train_m['accuracy']:.4f} | "
                f"Test  AUC={val_m['auc_roc']:.4f} F1={val_m['macro_f1']:.4f} "
                f"({elapsed:.1f}s)"
            )

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}}
        row.update({f"test_{k}": v for k, v in val_m.items()})
        history.append(row)

        if val_m["auc_roc"] > best_auc:
            best_auc = val_m["auc_roc"]
            torch.save(model.state_dict(), checkpoint_path)

    # Load best weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logger.info(f"Best fusion model loaded (AUC={best_auc:.4f})")

    # Final evaluation
    final_metrics, probs, labels = evaluate_fusion(model, test_loader, loss_fn, device)
    logger.info(
        f"\nFinal Test Results (ReviewGuard Fusion):\n"
        f"  AUC-ROC   : {final_metrics['auc_roc']:.4f}\n"
        f"  Macro-F1  : {final_metrics['macro_f1']:.4f}\n"
        f"  Accuracy  : {final_metrics['accuracy']:.4f}\n"
        f"  Prec(fake): {final_metrics.get('precision_fake', 0):.4f}\n"
        f"  Rec(fake) : {final_metrics.get('recall_fake', 0):.4f}"
    )

    # Save results
    results = {
        "dataset": dataset_name,
        "model": "ReviewGuard (Fusion)",
        "architecture": {
            "text_dim": text_dim,
            "behavior_dim": behavior_dim,
            "fused_dim": text_dim + behavior_dim,
            "hidden_dims": list(hidden_dims),
            "dropout": dropout,
        },
        "training": {
            "lr": lr, "batch_size": batch_size, "epochs": max_epochs, "gamma": gamma
        },
        "test_metrics": final_metrics,
        "best_val_auc": best_auc,
    }

    results_path = Path(config.get("results_dir", "results")) / f"{dataset_name}_fusion_results.json"
    ensure_dirs(results_path.parent)
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # Save history and probabilities
    with open(models_dir / f"{dataset_name}_history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    np.save(models_dir / f"{dataset_name}_test_probs.npy", probs)
    np.save(models_dir / f"{dataset_name}_test_labels.npy", labels)

    logger.info(f"Results saved → {results_path}")
    return model, results


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_fusion_model(
    checkpoint_path: Union[str, Path],
    config: Optional[Dict] = None,
    device: Optional[torch.device] = None,
) -> ReviewGuard:
    """Load a trained ReviewGuard fusion model from a checkpoint.

    Args:
        checkpoint_path: Path to the ``.pt`` state dict file.
        config: Configuration dictionary (used for architecture params).
        device: PyTorch device. If ``None``, auto-detected.

    Returns:
        Loaded :class:`ReviewGuard` model in eval mode.
    """
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            config = {}

    if device is None:
        device = get_device(config.get("device", "auto"))

    fusion_cfg = config.get("fusion", {})
    model = ReviewGuard(
        text_dim=fusion_cfg.get("text_dim", 768),
        behavior_dim=fusion_cfg.get("behavior_dim", 6),
        hidden_dims=tuple(fusion_cfg.get("hidden_dims", [256, 64])),
        dropout=fusion_cfg.get("dropout", 0.3),
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"ReviewGuard loaded from {checkpoint_path}")
    return model


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging("INFO")
    set_seed(42)

    parser = argparse.ArgumentParser(
        description="Train the ReviewGuard fusion model."
    )
    parser.add_argument("--dataset", default="yelpzip", choices=["yelpzip", "yelpnyc"])
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        logger.warning("Config not found; using defaults.")
        cfg = {}

    train_fusion_model(config=cfg, dataset_name=args.dataset)
