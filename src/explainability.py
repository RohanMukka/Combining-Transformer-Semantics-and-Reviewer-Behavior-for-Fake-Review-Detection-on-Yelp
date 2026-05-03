"""
SHAP Explainability Analysis for ReviewGuard
=============================================

Provides SHAP-based explainability for the ReviewGuard fusion model:

  - Global feature importance ranking across the 774-dim input
    (text branch contribution vs. behavior branch contribution)
  - Per-sample SHAP waterfall plots for individual predictions
  - Global SHAP summary plots (beeswarm, bar)
  - Branch-level analysis: which branch dominates different reviewer types?

The analysis uses :class:`shap.DeepExplainer` for the PyTorch MLP, applied
to the 774-dim fused embedding (concatenated text + behavior vectors).

Feature naming convention:
  - text_dim_0 … text_dim_767 : RoBERTa [CLS] embedding dimensions
  - behavior_avg_star_rating   : Behavior feature #1
  - behavior_review_count      : Behavior feature #2
  - behavior_burst_ratio       : Behavior feature #3
  - behavior_rating_deviation  : Behavior feature #4
  - behavior_category_diversity: Behavior feature #5
  - behavior_account_age_at_review: Behavior feature #6

Usage:
    python -m src.explainability --dataset yelpzip --n_samples 500
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .behavior_features import FEATURE_NAMES
from .utils import ensure_dirs, get_device, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)

# Full feature names for the 774-dim fused vector
TEXT_DIM = 768
BEHAVIOR_DIM = 6
FUSED_DIM = TEXT_DIM + BEHAVIOR_DIM

BEHAVIOR_FEATURE_NAMES = [f"behavior_{fn}" for fn in FEATURE_NAMES]

# For SHAP we summarise text dimensions as a group to avoid 768-dim noise
TEXT_BRANCH_NAMES = [f"text_dim_{i}" for i in range(TEXT_DIM)]
FUSED_FEATURE_NAMES = TEXT_BRANCH_NAMES + BEHAVIOR_FEATURE_NAMES


# ─── SHAP Wrapper ─────────────────────────────────────────────────────────────

class ReviewGuardSHAPWrapper:
    """Wrapper around ReviewGuard that accepts numpy arrays for SHAP compatibility.

    SHAP's :class:`shap.DeepExplainer` requires a PyTorch model and a
    background dataset. This wrapper handles numpy→tensor conversion.
    """

    def __init__(self, model, device) -> None:
        self.model = model
        self.device = device

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for SHAP: accepts (N, 774) numpy, returns (N,) probabilities."""
        import torch
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            text_emb = x_tensor[:, :TEXT_DIM]
            behavior = x_tensor[:, TEXT_DIM:]
            logits = self.model(text_emb, behavior)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        return probs


# ─── SHAP Analysis ────────────────────────────────────────────────────────────

def run_shap_analysis(
    model,
    train_emb: np.ndarray,
    train_behavior: np.ndarray,
    test_emb: np.ndarray,
    test_behavior: np.ndarray,
    test_labels: np.ndarray,
    config: Optional[Dict] = None,
    n_background: int = 200,
    n_explain: int = 500,
    save_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values for the ReviewGuard fusion model.

    Uses :class:`shap.DeepExplainer` with a background dataset drawn from
    the training set to estimate expected feature contributions.

    Args:
        model: Trained :class:`~src.fusion_model.ReviewGuard` (PyTorch module).
        train_emb: Training text embeddings of shape ``(N_train, 768)``.
        train_behavior: Training behavior features of shape ``(N_train, 6)``.
        test_emb: Test text embeddings of shape ``(N_test, 768)``.
        test_behavior: Test behavior features of shape ``(N_test, 6)``.
        test_labels: Test ground-truth labels of shape ``(N_test,)``.
        config: Configuration dictionary.
        n_background: Number of background samples for DeepExplainer.
        n_explain: Number of test samples to explain.
        save_dir: Directory to save plots and SHAP values.

    Returns:
        Tuple of (shap_values, fused_X_explain) arrays.
    """
    try:
        import shap
    except ImportError:
        logger.error("shap not installed. Run: pip install shap")
        raise

    import torch

    if config is None:
        config = {}

    if save_dir is None:
        save_dir = Path(config.get("explainability", {}).get("plots_dir", "results/shap_plots"))

    save_dir = Path(save_dir)
    ensure_dirs(save_dir)

    device = get_device(config.get("device", "auto"))
    model = model.to(device)
    model.eval()

    # ── Build fused arrays ──
    train_fused = np.concatenate([train_emb, train_behavior], axis=1)  # (N_train, 774)
    test_fused = np.concatenate([test_emb, test_behavior], axis=1)  # (N_test, 774)

    # ── Background dataset ──
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(train_fused), size=min(n_background, len(train_fused)), replace=False)
    background = torch.tensor(train_fused[bg_idx], dtype=torch.float32).to(device)

    # ── Explain samples ──
    explain_idx = rng.choice(len(test_fused), size=min(n_explain, len(test_fused)), replace=False)
    explain_data = torch.tensor(test_fused[explain_idx], dtype=torch.float32).to(device)

    # ── SHAP DeepExplainer ──
    logger.info(f"Initialising SHAP DeepExplainer (background={n_background})…")

    # Create a PyTorch module that takes the fused 774-dim input
    class FusedForward(torch.nn.Module):
        def __init__(self, fusion_model):
            super().__init__()
            self.fusion_model = fusion_model

        def forward(self, x):
            return self.fusion_model(x[:, :TEXT_DIM], x[:, TEXT_DIM:])

    fused_model = FusedForward(model).to(device)

    explainer = shap.DeepExplainer(fused_model, background)

    logger.info(f"Computing SHAP values for {n_explain} samples …")
    shap_values = explainer.shap_values(explain_data)

    if isinstance(shap_values, list):
        # For binary classification, SHAP returns a list of two arrays (one per class).
        # We take the values for the 'fake' class (index 1).
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_values = np.array(shap_values).squeeze()
    explain_np = explain_data.cpu().numpy()

    # Save SHAP values
    np.save(save_dir / "shap_values.npy", shap_values)
    np.save(save_dir / "shap_explain_data.npy", explain_np)
    np.save(save_dir / "shap_explain_labels.npy", test_labels[explain_idx])
    logger.info(f"SHAP values saved → {save_dir}/shap_values.npy")

    return shap_values, explain_np


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_branch_importance(
    shap_values: np.ndarray,
    save_dir: Path,
) -> None:
    """Plot branch-level SHAP importance: text branch vs. behavior branch.

    Aggregates SHAP magnitudes across the 768 text dimensions and 6 behavior
    dimensions separately to quantify each branch's contribution.

    Args:
        shap_values: SHAP values of shape ``(N, 774)``.
        save_dir: Directory to save the plot.
    """
    save_dir = Path(save_dir)
    ensure_dirs(save_dir)

    text_importance = np.abs(shap_values[:, :TEXT_DIM]).mean()
    behavior_importance = np.abs(shap_values[:, TEXT_DIM:]).mean()

    total = text_importance + behavior_importance
    text_pct = text_importance / total * 100
    behavior_pct = behavior_importance / total * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Pie chart: branch contributions
    axes[0].pie(
        [text_pct, behavior_pct],
        labels=["Text Branch\n(RoBERTa)", "Behavior Branch"],
        colors=["#1976D2", "#E53935"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 12},
    )
    axes[0].set_title("Branch Contribution to Predictions\n(Mean |SHAP|)", fontsize=12, fontweight="bold")

    # Per-behavior feature importance
    behavior_shap = shap_values[:, TEXT_DIM:]
    behavior_mean_abs = np.abs(behavior_shap).mean(axis=0)

    y_pos = np.arange(BEHAVIOR_DIM)
    colors_bar = ["#E53935" if s > 0 else "#1976D2" for s in behavior_shap.mean(axis=0)]

    axes[1].barh(
        [fn.replace("_", "\n") for fn in FEATURE_NAMES],
        behavior_mean_abs,
        color="#E53935",
        edgecolor="white",
        height=0.6,
    )
    axes[1].set_xlabel("Mean |SHAP Value|")
    axes[1].set_title("Behavior Feature Importance\n(Mean |SHAP|)", fontsize=12, fontweight="bold")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_dir / "shap_branch_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Branch importance plot → {save_dir}/shap_branch_importance.png")

    logger.info(
        f"Branch contributions: Text={text_pct:.1f}%  Behavior={behavior_pct:.1f}%"
    )


def plot_behavior_shap_summary(
    shap_values: np.ndarray,
    explain_data: np.ndarray,
    save_dir: Path,
) -> None:
    """Plot a SHAP beeswarm summary plot for the 6 behavior features.

    Args:
        shap_values: SHAP values of shape ``(N, 774)``.
        explain_data: Feature values used for explanation, shape ``(N, 774)``.
        save_dir: Directory to save the plot.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed; skipping beeswarm plot.")
        return

    save_dir = Path(save_dir)
    ensure_dirs(save_dir)

    # Extract just the behavior portion
    behavior_shap = shap_values[:, TEXT_DIM:]  # (N, 6)
    behavior_data = explain_data[:, TEXT_DIM:]  # (N, 6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar summary for behavior features
    mean_abs = np.abs(behavior_shap).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]

    axes[0].barh(
        [FEATURE_NAMES[i].replace("_", " ").title() for i in sorted_idx[::-1]],
        mean_abs[sorted_idx[::-1]],
        color="#E53935",
        edgecolor="white",
    )
    axes[0].set_xlabel("Mean |SHAP Value|")
    axes[0].set_title("Behavior Feature Importance", fontsize=12, fontweight="bold")

    # Feature impact direction
    mean_shap = behavior_shap.mean(axis=0)
    colors = ["#E53935" if v > 0 else "#1976D2" for v in mean_shap]
    axes[1].barh(
        [FEATURE_NAMES[i].replace("_", " ").title() for i in sorted_idx[::-1]],
        mean_shap[sorted_idx[::-1]],
        color=[colors[i] for i in sorted_idx[::-1]],
        edgecolor="white",
    )
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Mean SHAP Value (positive = pushes toward fake)")
    axes[1].set_title("Feature Impact Direction", fontsize=12, fontweight="bold")

    plt.suptitle("SHAP Analysis: Reviewer Behavior Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "shap_behavior_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"SHAP behavior summary → {save_dir}/shap_behavior_summary.png")


def plot_reviewer_type_analysis(
    shap_values: np.ndarray,
    explain_data: np.ndarray,
    explain_labels: np.ndarray,
    save_dir: Path,
) -> None:
    """Analyse which branch dominates for different reviewer types.

    Groups reviews by the reviewer's volume stratum (review count) and
    plots branch contributions separately for each group.

    Args:
        shap_values: SHAP values of shape ``(N, 774)``.
        explain_data: Feature values of shape ``(N, 774)``.
        explain_labels: True labels of shape ``(N,)``.
        save_dir: Directory to save the plot.
    """
    save_dir = Path(save_dir)
    ensure_dirs(save_dir)

    # Use review_count feature (index 1 in behavior features) to bucket reviewers
    review_count_feature = explain_data[:, TEXT_DIM + 1]  # normalised review_count

    # Quartiles for bucketing (since feature is normalised)
    quartiles = np.percentile(review_count_feature, [25, 50, 75])
    bins = [-np.inf, quartiles[0], quartiles[1], quartiles[2], np.inf]
    bucket_labels = ["Low-Volume", "Medium-Volume", "High-Volume", "Power-Reviewer"]

    text_contribs: Dict[str, float] = {}
    behavior_contribs: Dict[str, float] = {}

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (review_count_feature > lo) & (review_count_feature <= hi)
        if mask.sum() < 5:
            continue
        subset_shap = shap_values[mask]
        text_contribs[bucket_labels[i]] = np.abs(subset_shap[:, :TEXT_DIM]).mean()
        behavior_contribs[bucket_labels[i]] = np.abs(subset_shap[:, TEXT_DIM:]).mean()

    buckets = list(text_contribs.keys())
    text_vals = np.array([text_contribs[b] for b in buckets])
    behavior_vals = np.array([behavior_contribs[b] for b in buckets])
    total_vals = text_vals + behavior_vals

    text_pct = text_vals / total_vals * 100
    behavior_pct = behavior_vals / total_vals * 100

    x = np.arange(len(buckets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, text_pct, width, label="Text Branch", color="#1976D2", edgecolor="white")
    bars2 = ax.bar(x + width / 2, behavior_pct, width, label="Behavior Branch", color="#E53935", edgecolor="white")

    ax.set_xlabel("Reviewer Volume Category")
    ax.set_ylabel("% Contribution to Predictions")
    ax.set_title(
        "Branch Dominance by Reviewer Volume\n(which branch drives predictions?)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_dir / "shap_reviewer_type_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Reviewer type analysis → {save_dir}/shap_reviewer_type_analysis.png")

    # Print summary
    print("\nBranch Dominance by Reviewer Volume:")
    print(f"{'Reviewer Type':<20} {'Text %':>8} {'Behavior %':>12}")
    print("-" * 42)
    for bucket in buckets:
        tot = text_contribs[bucket] + behavior_contribs[bucket]
        print(
            f"{bucket:<20} "
            f"{text_contribs[bucket]/tot*100:>7.1f}% "
            f"{behavior_contribs[bucket]/tot*100:>11.1f}%"
        )


def plot_sample_waterfall(
    shap_values: np.ndarray,
    explain_data: np.ndarray,
    explain_labels: np.ndarray,
    sample_idx: int,
    save_dir: Path,
    base_value: float = 0.0,
) -> None:
    """Plot a SHAP waterfall chart for a single review prediction.

    Only shows behavior features + text branch aggregate for readability.

    Args:
        shap_values: SHAP values of shape ``(N, 774)``.
        explain_data: Feature values of shape ``(N, 774)``.
        explain_labels: True labels of shape ``(N,)``.
        sample_idx: Index of the sample to explain.
        save_dir: Directory to save the plot.
        base_value: Expected model output (baseline prediction).
    """
    save_dir = Path(save_dir)
    ensure_dirs(save_dir)

    # Aggregate text dimensions into a single value
    text_shap_total = shap_values[sample_idx, :TEXT_DIM].sum()
    behavior_shap = shap_values[sample_idx, TEXT_DIM:]

    # Combined feature names and values for the waterfall
    feature_names_short = ["Text Branch\n(RoBERTa, 768-dim)"] + [
        fn.replace("_", "\n") for fn in FEATURE_NAMES
    ]
    shap_combined = np.concatenate([[text_shap_total], behavior_shap])

    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(shap_combined))[::-1]
    sorted_names = [feature_names_short[i] for i in sorted_idx]
    sorted_shaps = shap_combined[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#E53935" if v > 0 else "#1976D2" for v in sorted_shaps]
    ax.barh(
        range(len(sorted_shaps)),
        sorted_shaps,
        color=colors,
        edgecolor="white",
        height=0.6,
    )
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP Value (positive → more likely fake)")
    true_lbl = "Fake" if explain_labels[sample_idx] == 1 else "Genuine"
    ax.set_title(
        f"SHAP Waterfall — Sample #{sample_idx} (True label: {true_lbl})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_dir / f"shap_waterfall_sample_{sample_idx}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Waterfall plot → {save_dir}/shap_waterfall_sample_{sample_idx}.png")


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def run_explainability_pipeline(
    config: Optional[Dict] = None,
    dataset_name: str = "yelpzip",
) -> None:
    """End-to-end SHAP explainability pipeline.

    Loads the trained fusion model and pre-extracted embeddings, runs SHAP,
    and saves all plots and SHAP values.

    Args:
        config: Configuration dictionary.
        dataset_name: Dataset identifier.
    """
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            config = {}

    exp_cfg = config.get("explainability", {})
    n_background = exp_cfg.get("n_background_samples", 200)
    n_explain = exp_cfg.get("n_explain_samples", 500)
    save_dir = Path(exp_cfg.get("plots_dir", "results/shap_plots"))

    device = get_device(config.get("device", "auto"))

    # Load fusion model
    from .fusion_model import load_fusion_model, load_embeddings

    models_dir = Path(config.get("models_dir", "models")) / "fusion"
    checkpoint = models_dir / f"{dataset_name}_best.pt"

    logger.info(f"Loading ReviewGuard fusion model from {checkpoint} …")
    model = load_fusion_model(checkpoint, config=config, device=device)

    # Load pre-extracted embeddings
    logger.info("Loading embeddings …")
    train_emb, train_behavior, y_train = load_embeddings(dataset_name, "train", config)
    test_emb, test_behavior, y_test = load_embeddings(dataset_name, "test", config)

    # Run SHAP
    logger.info("Running SHAP analysis …")
    try:
        shap_values, explain_np = run_shap_analysis(
            model=model,
            train_emb=train_emb,
            train_behavior=train_behavior,
            test_emb=test_emb,
            test_behavior=test_behavior,
            test_labels=y_test,
            config=config,
            n_background=n_background,
            n_explain=n_explain,
            save_dir=save_dir,
        )

        # Generate plots
        explain_labels = y_test[:n_explain]

        plot_branch_importance(shap_values, save_dir)
        plot_behavior_shap_summary(shap_values, explain_np, save_dir)
        plot_reviewer_type_analysis(shap_values, explain_np, explain_labels, save_dir)

        # Waterfall for a fake and a genuine sample
        fake_idx = np.where(explain_labels == 1)[0]
        genuine_idx = np.where(explain_labels == 0)[0]

        if len(fake_idx) > 0:
            plot_sample_waterfall(shap_values, explain_np, explain_labels,
                                  fake_idx[0], save_dir)
        if len(genuine_idx) > 0:
            plot_sample_waterfall(shap_values, explain_np, explain_labels,
                                  genuine_idx[0], save_dir)

        logger.info("SHAP explainability analysis complete.")

    except Exception as exc:
        logger.error(f"SHAP analysis failed: {exc}")
        logger.info("Generating synthetic SHAP summaries for illustration…")
        _generate_synthetic_shap_plots(save_dir)


def _generate_synthetic_shap_plots(save_dir: Path) -> None:
    """Generate illustrative SHAP plots using synthetic values (offline fallback)."""
    rng = np.random.default_rng(42)
    n = 200

    text_shap = rng.normal(0.15, 0.08, size=n)
    behavior_shap = rng.normal(0, 0.05, size=(n, 6))
    behavior_shap[:, 2] += 0.08   # burst_ratio matters most
    behavior_shap[:, 4] -= 0.03   # category_diversity negative contribution

    # Bar chart of feature importance
    behavior_mean_abs = np.abs(behavior_shap).mean(axis=0)
    total_text = abs(text_shap).mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    feature_labels = ["RoBERTa\nText"] + [fn.replace("_", "\n") for fn in FEATURE_NAMES]
    vals = [total_text] + behavior_mean_abs.tolist()
    colors = ["#1976D2"] + ["#E53935"] * 6

    ax.barh(feature_labels, vals, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Feature Importance (Illustrative)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    ensure_dirs(save_dir)
    plt.savefig(save_dir / "shap_feature_importance_synthetic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Synthetic SHAP plot saved → {save_dir}/shap_feature_importance_synthetic.png")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging("INFO")
    set_seed(42)

    parser = argparse.ArgumentParser(
        description="Run SHAP explainability analysis for ReviewGuard."
    )
    parser.add_argument("--dataset", default="yelpchi", choices=["yelpzip", "yelpnyc", "yelpchi"])
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of samples to explain")
    parser.add_argument("--config", default="configs/default_config.yaml")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        cfg = {}

    cfg.setdefault("explainability", {})["n_explain_samples"] = args.n_samples
    run_explainability_pipeline(config=cfg, dataset_name=args.dataset)
