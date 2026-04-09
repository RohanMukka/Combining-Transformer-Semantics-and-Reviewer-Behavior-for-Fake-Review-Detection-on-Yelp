"""
PyTorch Dataset Classes and Data Utilities for ReviewGuard
==========================================================

Provides:
  - ReviewDataset: map-style PyTorch Dataset wrapping the unified DataFrame
  - TextOnlyDataset: wraps raw text + labels (for RoBERTa fine-tuning)
  - collate_fn: custom collate function for mixed text/behavior batches
  - tokenize_batch: batch tokenisation helper
  - get_class_weights: compute inverse-frequency class weights for focal loss
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ─── Dataset Classes ──────────────────────────────────────────────────────────

class ReviewDataset(Dataset):
    """PyTorch Dataset for ReviewGuard fusion model.

    Stores review text, pre-computed behavior feature vectors, and labels.
    Text is returned as a raw string; tokenisation is handled by the
    DataLoader's collate function.

    Args:
        df: DataFrame with columns ``["text", "label"]``.
        behavior_features: Float array of shape ``(N, 6)`` with pre-computed
                           behaviour features aligned with *df*.
        text_col: Name of the text column in *df*.
        label_col: Name of the label column in *df*.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        behavior_features: np.ndarray,
        text_col: str = "text",
        label_col: str = "label",
    ) -> None:
        if len(df) != len(behavior_features):
            raise ValueError(
                f"DataFrame length ({len(df)}) must match behavior_features "
                f"rows ({len(behavior_features)})."
            )
        self.texts: List[str] = df[text_col].astype(str).tolist()
        self.labels: Tensor = torch.tensor(df[label_col].values, dtype=torch.float32)
        self.behavior: Tensor = torch.tensor(
            behavior_features.astype(np.float32), dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, Tensor]]:
        return {
            "text": self.texts[idx],
            "behavior": self.behavior[idx],
            "label": self.labels[idx],
        }


class TextOnlyDataset(Dataset):
    """Minimal Dataset for text-only RoBERTa fine-tuning.

    Returns pre-tokenised input IDs and attention masks as tensors, plus
    integer labels. Expects that tokenisation has already been applied (e.g.,
    via :func:`tokenize_batch`) and stored in *encodings*.

    Args:
        encodings: Dict returned by a HuggingFace tokenizer (``input_ids``,
                   ``attention_mask``, optionally ``token_type_ids``).
        labels: Integer label array of length N.
    """

    def __init__(
        self,
        encodings: Dict[str, List[List[int]]],
        labels: Union[List[int], np.ndarray, Tensor],
    ) -> None:
        self.encodings = encodings
        if isinstance(labels, Tensor):
            self.labels = labels.float()
        else:
            self.labels = torch.tensor(list(labels), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["label"] = self.labels[idx]
        return item


class BehaviorOnlyDataset(Dataset):
    """Dataset for the behavior-only MLP branch.

    Args:
        features: Float array of shape ``(N, 6)``.
        labels: Binary label array of length N.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: Union[List[int], np.ndarray],
    ) -> None:
        self.features = torch.tensor(features.astype(np.float32), dtype=torch.float32)
        self.labels = torch.tensor(list(labels), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "features": self.features[idx],
            "label": self.labels[idx],
        }


class EmbeddingDataset(Dataset):
    """Dataset for training the fusion model on pre-extracted embeddings.

    Used when RoBERTa embeddings have already been extracted and saved to disk,
    avoiding redundant forward passes during fusion model training.

    Args:
        text_embeddings: Float array of shape ``(N, 768)``.
        behavior_features: Float array of shape ``(N, 6)``.
        labels: Binary label array of length N.
    """

    def __init__(
        self,
        text_embeddings: np.ndarray,
        behavior_features: np.ndarray,
        labels: Union[List[int], np.ndarray],
    ) -> None:
        self.text_emb = torch.tensor(
            text_embeddings.astype(np.float32), dtype=torch.float32
        )
        self.behavior = torch.tensor(
            behavior_features.astype(np.float32), dtype=torch.float32
        )
        self.labels = torch.tensor(list(labels), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "text_embedding": self.text_emb[idx],
            "behavior": self.behavior[idx],
            "label": self.labels[idx],
        }


# ─── Tokenisation Helpers ─────────────────────────────────────────────────────

def tokenize_batch(
    texts: List[str],
    tokenizer,
    max_length: int = 512,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: Optional[str] = None,
) -> Dict[str, Union[List, Tensor]]:
    """Tokenise a list of strings using a HuggingFace tokenizer.

    Args:
        texts: List of raw review text strings.
        tokenizer: A HuggingFace PreTrainedTokenizer instance.
        max_length: Maximum token sequence length.
        padding: Padding strategy (``"max_length"`` or ``"longest"``).
        truncation: Whether to truncate sequences exceeding *max_length*.
        return_tensors: If ``"pt"``, return PyTorch tensors. If ``None``,
                        return Python lists.

    Returns:
        Dictionary with ``input_ids``, ``attention_mask``, and optionally
        ``token_type_ids``.
    """
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )
    return encodings


# ─── Collate Function ─────────────────────────────────────────────────────────

def make_collate_fn(
    tokenizer,
    max_length: int = 512,
) -> Callable[[List[Dict]], Dict[str, Tensor]]:
    """Return a collate function that tokenises text on-the-fly.

    The returned collate function accepts a list of items from
    :class:`ReviewDataset` and produces a batch dictionary containing
    tokenised tensors, stacked behavior tensors, and labels.

    Args:
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum token sequence length.

    Returns:
        A callable suitable for use as the ``collate_fn`` argument to
        :class:`torch.utils.data.DataLoader`.
    """

    def collate_fn(batch: List[Dict]) -> Dict[str, Tensor]:
        texts = [item["text"] for item in batch]
        behavior = torch.stack([item["behavior"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])

        encoding = tokenizer(
            texts,
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "behavior": behavior,
            "label": labels,
        }

    return collate_fn


# ─── Class Weights ────────────────────────────────────────────────────────────

def get_class_weights(
    labels: Union[List[int], np.ndarray, pd.Series],
    n_classes: int = 2,
    as_tensor: bool = True,
) -> Union[np.ndarray, Tensor]:
    """Compute inverse-frequency class weights for imbalanced datasets.

    Used to set the alpha parameter in :class:`~src.text_branch.FocalLoss`.

    Args:
        labels: Binary label array (0 = genuine, 1 = fake).
        n_classes: Number of classes.
        as_tensor: If ``True``, return a PyTorch tensor; otherwise NumPy array.

    Returns:
        Weight array of length *n_classes* where w_c = N / (n_classes * N_c).
    """
    labels = np.asarray(labels)
    weights = np.zeros(n_classes, dtype=np.float32)
    n_total = len(labels)

    for c in range(n_classes):
        n_c = (labels == c).sum()
        weights[c] = n_total / (n_classes * max(n_c, 1))

    # Normalise so weights sum to n_classes
    weights = weights / weights.mean()

    logger.debug(f"Class weights: {dict(enumerate(weights.tolist()))}")

    if as_tensor:
        return torch.tensor(weights, dtype=torch.float32)
    return weights


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    oversample: bool = False,
    labels: Optional[np.ndarray] = None,
) -> DataLoader:
    """Create a DataLoader, optionally with oversampling for class balance.

    Args:
        dataset: PyTorch Dataset instance.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data (ignored if *oversample* is True).
        num_workers: Number of worker processes for data loading.
        oversample: If ``True``, use :class:`WeightedRandomSampler` to
                    oversample the minority class (fake reviews).
        labels: Label array required when *oversample* is ``True``.

    Returns:
        Configured :class:`torch.utils.data.DataLoader`.
    """
    sampler = None
    if oversample:
        if labels is None:
            raise ValueError("'labels' must be provided when oversample=True.")
        class_weights = get_class_weights(labels, as_tensor=False)
        sample_weights = class_weights[labels.astype(int)]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float64),
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
