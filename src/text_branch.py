"""
RoBERTa Text Branch for ReviewGuard
=====================================

Implements the text processing branch of the ReviewGuard system:

  - RoBERTaTextClassifier: Fine-tuned roberta-base with classification head
    * First 6 encoder layers frozen for efficient fine-tuning
    * [CLS] token embedding (768-dim) extracted for fusion model
  - FocalLoss: Focal loss with γ=2 and class-frequency-weighted α
  - Training pipeline with AdamW + linear warmup scheduler
  - Embedding extraction for the fusion model

Architecture:
    Input text → RoBERTa-base tokenizer → roberta-base encoder
    → [CLS] embedding (768-dim) → Dropout → Linear(768, 1) → Sigmoid

Usage:
    # Fine-tune on YelpZIP:
    python -m src.text_branch --mode train --dataset yelpzip

    # Extract embeddings for fusion model:
    python -m src.text_branch --mode extract --dataset yelpzip
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaModel, get_linear_schedule_with_warmup

from .utils import ensure_dirs, get_device, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss for binary classification (Lin et al., 2017).

    Focal loss addresses class imbalance by down-weighting well-classified
    examples and focusing training on hard, misclassified examples.

    Loss formula:
        FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    where:
        p_t = p if y=1 else (1-p)
        α_t = α if y=1 else (1-α)     [class-frequency weight]
        γ = focusing parameter (γ=2 as in original paper)

    Args:
        gamma: Focusing parameter. γ=0 reduces to cross-entropy. Default: 2.0.
        alpha: Class weight for the positive class (fake=1). If a float,
               genuine class weight = 1 - alpha. If a Tensor of shape [2],
               alpha[0] = genuine weight, alpha[1] = fake weight.
               If ``None``, uses equal weights (0.5 each).
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, Tensor]] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.register_buffer("alpha", torch.tensor([0.5, 0.5]))
        elif isinstance(alpha, float):
            self.register_buffer("alpha", torch.tensor([1.0 - alpha, alpha]))
        elif isinstance(alpha, Tensor):
            self.register_buffer("alpha", alpha.float())
        else:
            raise TypeError(f"alpha must be float, Tensor, or None; got {type(alpha)}")

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model output of shape ``(N,)`` or ``(N, 1)`` (before sigmoid).
            targets: Binary labels of shape ``(N,)`` with values in {0, 1}.

        Returns:
            Scalar loss value (if reduction ≠ "none").
        """
        logits = logits.squeeze(-1)
        targets = targets.float()

        # Sigmoid probabilities
        p = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p, 1.0 - p)

        # Binary cross-entropy (numerically stable via log-sigmoid)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Focal weight: (1 - p_t)^γ
        focal_weight = (1.0 - p_t) ** self.gamma

        # Per-sample alpha: α for fake, (1-α) for genuine
        alpha_t = torch.where(
            targets == 1,
            self.alpha[1].expand_as(targets),
            self.alpha[0].expand_as(targets),
        )

        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ─── RoBERTa Text Classifier ─────────────────────────────────────────────────

class RoBERTaTextClassifier(nn.Module):
    """RoBERTa-based text classifier with frozen lower layers.

    The model loads ``roberta-base`` and freezes the first *freeze_layers*
    transformer encoder layers (embeddings always frozen). Only the upper
    encoder layers and the classification head are updated during fine-tuning.

    The 768-dimensional [CLS] token embedding is exposed via
    :meth:`extract_embeddings` for downstream fusion model training.

    Architecture:
        roberta-base → [CLS] (768-dim) → Dropout(p) → Linear(768, 1)

    Args:
        model_name: HuggingFace model identifier. Default: ``"roberta-base"``.
        freeze_layers: Number of encoder layers to freeze (0–12). Default: 6.
        dropout_prob: Dropout probability on the [CLS] embedding. Default: 0.1.
        num_labels: Number of output logits. Default: 1 (binary classification).
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        freeze_layers: int = 6,
        dropout_prob: float = 0.1,
        num_labels: int = 1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.freeze_layers = freeze_layers

        # Load pre-trained RoBERTa
        self.roberta = RobertaModel.from_pretrained(
            model_name, add_pooling_layer=False
        )

        self._freeze_layers(freeze_layers)

        hidden_size = self.roberta.config.hidden_size  # 768 for roberta-base
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Weight init for classifier head
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _freeze_layers(self, n: int) -> None:
        """Freeze embeddings and the first *n* encoder layers."""
        # Always freeze embeddings
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False

        # Freeze first n encoder layers
        for layer in self.roberta.encoder.layer[:n]:
            for param in layer.parameters():
                param.requires_grad = False

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"RoBERTa layers frozen: {n}/12 | "
            f"Trainable params: {trainable:,} / {total:,} ({trainable/total:.1%})"
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Optional[Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape ``(B, L)``.
            attention_mask: Attention mask of shape ``(B, L)``.
            token_type_ids: Optional token type IDs (unused by RoBERTa).
            return_embeddings: If ``True``, also return the raw [CLS] embedding
                               before the dropout + linear head.

        Returns:
            Logits of shape ``(B, 1)`` (or ``(B, num_labels)``).
            If *return_embeddings* is ``True``, returns ``(logits, embeddings)``.
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # [CLS] token: first position of last hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (B, 768)

        dropped = self.dropout(cls_embedding)
        logits = self.classifier(dropped)  # (B, 1)

        if return_embeddings:
            return logits, cls_embedding
        return logits

    def extract_embeddings(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Extract [CLS] embeddings without the classification head.

        Used to generate 768-dim embeddings for the fusion model.

        Args:
            input_ids: Token IDs of shape ``(B, L)``.
            attention_mask: Attention mask of shape ``(B, L)``.

        Returns:
            [CLS] embeddings of shape ``(B, 768)``.
        """
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_epoch(
    model: RoBERTaTextClassifier,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    loss_fn: FocalLoss,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Run one training epoch.

    Args:
        model: RoBERTaTextClassifier instance.
        dataloader: DataLoader yielding batches with keys
                    ``"input_ids"``, ``"attention_mask"``, ``"label"``.
        optimizer: AdamW optimizer.
        scheduler: Learning rate scheduler.
        loss_fn: FocalLoss instance.
        device: PyTorch device.
        grad_clip: Max gradient norm for gradient clipping.

    Returns:
        Dictionary with ``"loss"`` and ``"accuracy"`` for this epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="  [Train]", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += len(labels)
        total_loss += loss.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {"loss": total_loss / len(dataloader), "accuracy": correct / total}


@torch.no_grad()
def evaluate_epoch(
    model: RoBERTaTextClassifier,
    dataloader: DataLoader,
    loss_fn: FocalLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model on a validation/test DataLoader.

    Args:
        model: RoBERTaTextClassifier instance.
        dataloader: DataLoader.
        loss_fn: FocalLoss instance.
        device: PyTorch device.

    Returns:
        Dictionary with ``"loss"``, ``"accuracy"``, ``"auc_roc"``, ``"macro_f1"``.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    model.eval()
    total_loss = 0.0
    all_probs: List[float] = []
    all_labels: List[int] = []

    for batch in tqdm(dataloader, desc="  [Eval]", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    all_probs_arr = np.array(all_probs)
    all_labels_arr = np.array(all_labels, dtype=int)
    preds = (all_probs_arr >= 0.5).astype(int)

    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": (preds == all_labels_arr).mean(),
    }
    try:
        metrics["auc_roc"] = roc_auc_score(all_labels_arr, all_probs_arr)
        metrics["macro_f1"] = f1_score(all_labels_arr, preds, average="macro", zero_division=0)
    except ValueError:
        metrics["auc_roc"] = 0.0
        metrics["macro_f1"] = 0.0

    return metrics


# ─── Full Training Pipeline ───────────────────────────────────────────────────

def train_text_branch(
    config: Optional[Dict] = None,
    dataset_name: str = "yelpzip",
) -> RoBERTaTextClassifier:
    """Full training pipeline for the RoBERTa text branch.

    Loads data, builds the model, trains with focal loss + AdamW + warmup,
    and saves the best checkpoint.

    Args:
        config: Configuration dictionary.
        dataset_name: Dataset to train on.

    Returns:
        Trained :class:`RoBERTaTextClassifier` model.
    """
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            config = {}

    tb_cfg = config.get("text_branch", {})
    fl_cfg = config.get("focal_loss", {})
    device = get_device(config.get("device", "auto"))

    model_name = tb_cfg.get("model_name", "roberta-base")
    freeze_layers = tb_cfg.get("freeze_layers", 6)
    dropout = tb_cfg.get("hidden_dropout_prob", 0.1)
    lr = tb_cfg.get("learning_rate", 2e-5)
    warmup_ratio = tb_cfg.get("warmup_ratio", 0.1)
    weight_decay = tb_cfg.get("weight_decay", 0.01)
    batch_size = tb_cfg.get("batch_size", 32)
    max_epochs = tb_cfg.get("max_epochs", 5)
    grad_clip = tb_cfg.get("gradient_clip_norm", 1.0)
    max_length = config.get("data", {}).get("max_review_length", 512)
    gamma = fl_cfg.get("gamma", 2.0)

    models_dir = Path(config.get("models_dir", "models")) / "text_branch"
    ensure_dirs(models_dir)

    # ── Load data ──
    from .data_loader import load_dataset, temporal_train_test_split
    from .data_utils import TextOnlyDataset, create_dataloader, get_class_weights, tokenize_batch

    logger.info(f"Loading {dataset_name} dataset …")
    df = load_dataset(dataset_name, config=config)
    train_df, val_df = temporal_train_test_split(
        df, cutoff_fraction=config.get("data", {}).get("temporal_cutoff_fraction", 0.8)
    )

    y_train = train_df["label"].values

    # ── Tokeniser ──
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Tokenising training set …")
    train_enc = tokenize_batch(
        train_df["text"].fillna("").tolist(), tokenizer, max_length=max_length
    )
    val_enc = tokenize_batch(
        val_df["text"].fillna("").tolist(), tokenizer, max_length=max_length
    )

    train_dataset = TextOnlyDataset(train_enc, y_train)
    val_dataset = TextOnlyDataset(val_enc, val_df["label"].values)

    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    # ── Model ──
    model = RoBERTaTextClassifier(
        model_name=model_name,
        freeze_layers=freeze_layers,
        dropout_prob=dropout,
    ).to(device)

    # ── Focal Loss with class-frequency weights ──
    class_weights = get_class_weights(y_train, as_tensor=True)
    alpha = class_weights[1] / (class_weights[0] + class_weights[1])
    loss_fn = FocalLoss(gamma=gamma, alpha=alpha.item()).to(device)
    logger.info(f"FocalLoss: γ={gamma}, α(fake)={alpha:.4f}")

    # ── Optimiser + Scheduler ──
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    total_steps = len(train_loader) * max_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    logger.info(f"Total steps: {total_steps} | Warmup: {warmup_steps}")

    # ── Training loop ──
    best_auc = 0.0
    best_checkpoint_path = models_dir / f"{dataset_name}_best.pt"
    history: List[Dict] = []

    for epoch in range(1, max_epochs + 1):
        logger.info(f"Epoch {epoch}/{max_epochs}")
        t0 = time.perf_counter()

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, grad_clip
        )
        val_metrics = evaluate_epoch(model, val_loader, loss_fn, device)
        elapsed = time.perf_counter() - t0

        logger.info(
            f"  Train — loss={train_metrics['loss']:.4f}  acc={train_metrics['accuracy']:.4f}"
        )
        logger.info(
            f"  Val   — loss={val_metrics['loss']:.4f}  "
            f"AUC={val_metrics['auc_roc']:.4f}  "
            f"F1={val_metrics['macro_f1']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        if val_metrics["auc_roc"] > best_auc:
            best_auc = val_metrics["auc_roc"]
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info(f"  ✓ New best AUC={best_auc:.4f} — checkpoint saved.")

    # Save training history
    with open(models_dir / f"{dataset_name}_history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    # Load best weights
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    logger.info(f"Best model loaded from {best_checkpoint_path} (AUC={best_auc:.4f})")

    return model


# ─── Embedding Extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_cls_embeddings(
    model: RoBERTaTextClassifier,
    texts: List[str],
    tokenizer,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 512,
) -> np.ndarray:
    """Extract [CLS] embeddings for a list of texts.

    Args:
        model: Trained :class:`RoBERTaTextClassifier`.
        texts: List of review texts.
        tokenizer: HuggingFace tokenizer.
        device: PyTorch device.
        batch_size: Batch size for embedding extraction.
        max_length: Max token length.

    Returns:
        Float array of shape ``(N, 768)``.
    """
    model.eval()
    model.to(device)

    all_embeddings: List[np.ndarray] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
        batch_texts = texts[i: i + batch_size]
        encoding = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        emb = model.extract_embeddings(input_ids, attention_mask)
        all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


def extract_and_save_embeddings(
    config: Optional[Dict] = None,
    dataset_name: str = "yelpzip",
) -> None:
    """Load a trained model and extract + save [CLS] embeddings.

    Saves train/test embeddings to ``data/features/{dataset_name}_*_text_emb.npy``.

    Args:
        config: Configuration dictionary.
        dataset_name: Dataset identifier.
    """
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            config = {}

    device = get_device(config.get("device", "auto"))
    models_dir = Path(config.get("models_dir", "models")) / "text_branch"
    features_dir = Path(config.get("data", {}).get("features_dir", "data/features"))
    ensure_dirs(features_dir)

    tb_cfg = config.get("text_branch", {})
    model_name = tb_cfg.get("model_name", "roberta-base")
    batch_size = tb_cfg.get("batch_size", 64)
    max_length = config.get("data", {}).get("max_review_length", 512)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RoBERTaTextClassifier(
        model_name=model_name,
        freeze_layers=tb_cfg.get("freeze_layers", 6),
        dropout_prob=tb_cfg.get("hidden_dropout_prob", 0.1),
    )

    checkpoint = models_dir / f"{dataset_name}_best.pt"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        logger.info(f"Loaded checkpoint from {checkpoint}")
    else:
        logger.warning(
            f"Checkpoint not found at {checkpoint}. Using untrained model. "
            "Run 'python -m src.text_branch --mode train' first."
        )

    model.to(device)

    from .data_loader import load_dataset, temporal_train_test_split

    df = load_dataset(dataset_name, config=config)
    train_df, test_df = temporal_train_test_split(
        df, cutoff_fraction=config.get("data", {}).get("temporal_cutoff_fraction", 0.8)
    )

    logger.info(f"Extracting embeddings for train split ({len(train_df):,} reviews)…")
    train_texts = train_df["text"].fillna("").tolist()
    train_emb = extract_cls_embeddings(model, train_texts, tokenizer, device, batch_size, max_length)

    logger.info(f"Extracting embeddings for test split ({len(test_df):,} reviews)…")
    test_texts = test_df["text"].fillna("").tolist()
    test_emb = extract_cls_embeddings(model, test_texts, tokenizer, device, batch_size, max_length)

    target_dim = tb_cfg.get("embedding_dim", 768)
    if target_dim < 768:
        logger.info(f"Applying PCA to reduce embeddings from 768 to {target_dim} dimensions")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=target_dim, random_state=42)
        train_emb = pca.fit_transform(train_emb).astype(np.float32)
        test_emb = pca.transform(test_emb).astype(np.float32)

    np.save(features_dir / f"{dataset_name}_train_text_emb.npy", train_emb)
    np.save(features_dir / f"{dataset_name}_test_text_emb.npy", test_emb)
    logger.info(f"Embeddings saved to {features_dir} with shape {train_emb.shape}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging("INFO")

    parser = argparse.ArgumentParser(description="ReviewGuard RoBERTa text branch.")
    parser.add_argument(
        "--mode",
        choices=["train", "extract"],
        default="train",
        help="'train' to fine-tune, 'extract' to extract [CLS] embeddings",
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

    if args.mode == "train":
        train_text_branch(config=cfg, dataset_name=args.dataset)
    elif args.mode == "extract":
        extract_and_save_embeddings(config=cfg, dataset_name=args.dataset)
