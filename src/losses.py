"""Focal loss (Lin et al., 2017) for binary classification under imbalance."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), computed from logits."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        logits = logits.reshape(-1)
        targets = targets.reshape(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()
