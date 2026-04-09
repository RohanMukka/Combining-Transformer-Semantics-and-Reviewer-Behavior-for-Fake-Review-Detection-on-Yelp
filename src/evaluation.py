"""
Temporal Cross-Validation and Evaluation Framework for ReviewGuard
===================================================================

Provides:
  - TemporalCrossValidator: 5-fold temporal cross-validation
  - Metrics computation: AUC-ROC, Macro-F1, precision, recall per class
  - Confusion matrix generation and plotting
  - Per-stratum analysis (by reviewer review-count buckets)
  - Statistical significance tests (Wilcoxon signed-rank)
  - Results aggregation across folds (mean ± std)
  - Comprehensive results saving

Usage:
    python -m src.evaluation --dataset yelpzip --n_folds 5
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .utils import ensure_dirs, load_config, setup_logging

logger = logging.getLogger(__name__)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute a comprehensive set of binary classification metrics.

    Args:
        y_true: Ground-truth labels (0/1).
        y_pred: Predicted labels (0/1).
        y_prob: Predicted probability of the positive class. Required for AUC-ROC.

    Returns:
        Dictionary with:
            - ``auc_roc``: Area Under the ROC Curve
            - ``macro_f1``: Macro-averaged F1 score
            - ``f1_fake``: F1 for the fake class (label=1)
            - ``f1_genuine``: F1 for the genuine class (label=0)
            - ``precision_fake``, ``recall_fake``
            - ``precision_genuine``, ``recall_genuine``
            - ``accuracy``
    """
    metrics: Dict[str, float] = {}

    metrics["accuracy"] = float((y_true == y_pred).mean())
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_fake"] = float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0))
    metrics["f1_genuine"] = float(f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0))
    metrics["precision_fake"] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    metrics["recall_fake"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    metrics["precision_genuine"] = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    metrics["recall_genuine"] = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))

    if y_prob is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auc_roc"] = 0.0
    else:
        metrics["auc_roc"] = float("nan")

    return metrics


def aggregate_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-fold metrics to mean ± std.

    Args:
        fold_metrics: List of metric dicts, one per fold.

    Returns:
        Dict with ``<metric>_mean`` and ``<metric>_std`` for each metric key.
    """
    all_keys = sorted(set(k for m in fold_metrics for k in m.keys()))
    result: Dict[str, float] = {}
    for key in all_keys:
        vals = np.array([m.get(key, float("nan")) for m in fold_metrics])
        valid = vals[~np.isnan(vals)]
        result[f"{key}_mean"] = float(np.mean(valid)) if len(valid) else float("nan")
        result[f"{key}_std"] = float(np.std(valid)) if len(valid) else float("nan")
    return result


# ─── Temporal Cross-Validator ─────────────────────────────────────────────────

class TemporalCrossValidator:
    """5-fold temporal cross-validation for time-series review data.

    Unlike standard k-fold CV, temporal CV ensures that training data always
    precedes test data in time, preventing data leakage.

    Fold construction:
        Suppose we have T reviews sorted by date. We create n_folds folds:
        - Fold k: train = reviews[0 : k*T//n_folds], test = reviews[k*T//n_folds : (k+1)*T//n_folds]
        - Minimum training size enforced to avoid cold-start folds.

    Args:
        n_folds: Number of CV folds. Default: 5.
        min_train_fraction: Minimum fraction of data required in training set.
                            Default: 0.4 (skips earliest folds if needed).
    """

    def __init__(self, n_folds: int = 5, min_train_fraction: float = 0.4) -> None:
        self.n_folds = n_folds
        self.min_train_fraction = min_train_fraction

    def split(
        self, df: pd.DataFrame, date_col: str = "date"
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate (train_indices, test_indices) pairs.

        Args:
            df: DataFrame sorted by *date_col* (or will be sorted internally).
            date_col: Column name containing datetime values.

        Yields:
            Tuples of (train_idx_array, test_idx_array) using the DataFrame's
            integer positional index.
        """
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        n = len(df_sorted)
        fold_size = n // self.n_folds
        min_train_size = int(n * self.min_train_fraction)

        for fold in range(1, self.n_folds + 1):
            train_end = fold * fold_size
            test_start = train_end
            test_end = min((fold + 1) * fold_size, n)

            if train_end < min_train_size:
                logger.debug(f"Fold {fold}: skipping (train_end={train_end} < min={min_train_size})")
                continue
            if test_start >= n:
                break

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            # Map back to original DataFrame positions
            train_positions = df_sorted.index.values[:train_end]
            test_positions = df_sorted.index.values[test_start:test_end]

            logger.debug(
                f"Fold {fold}: train={len(train_positions):,} test={len(test_positions):,} "
                f"| dates {df_sorted[date_col].iloc[0].date()} → "
                f"{df_sorted[date_col].iloc[test_end-1].date()}"
            )

            yield train_positions, test_positions


# ─── Per-Stratum Analysis ─────────────────────────────────────────────────────

STRATUM_BINS = [1, 5, 20, 50, 10_000]
STRATUM_LABELS = ["1–4", "5–19", "20–49", "50+"]


def compute_stratum_metrics(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    stratum_bins: List[int] = STRATUM_BINS,
    stratum_labels: List[str] = STRATUM_LABELS,
) -> pd.DataFrame:
    """Compute AUC-ROC and Macro-F1 per reviewer volume stratum.

    Reviewers are bucketed by their total review count; performance is
    reported separately for each bucket. This tests whether the model is
    robust across different reviewer profiles.

    Args:
        df: DataFrame with columns ``["label"]`` and ``"user_id"`` (for
            computing review counts), aligned with *y_pred* and *y_prob*.
        y_pred: Predicted binary labels (same order as *df*).
        y_prob: Predicted probabilities.
        stratum_bins: Bin edges for review-count buckets.
        stratum_labels: Human-readable label for each bucket.

    Returns:
        DataFrame with one row per stratum and columns
        ``["stratum", "n_reviews", "n_fake", "auc_roc", "macro_f1"]``.
    """
    df_work = df.copy().reset_index(drop=True)
    review_counts = df_work.groupby("user_id")["user_id"].transform("count")
    df_work["review_count"] = review_counts.values
    df_work["_pred"] = y_pred
    df_work["_prob"] = y_prob

    rows = []
    for i, (lo, hi) in enumerate(zip(stratum_bins[:-1], stratum_bins[1:])):
        mask = (df_work["review_count"] >= lo) & (df_work["review_count"] < hi)
        subset = df_work[mask]
        if len(subset) < 10:
            continue
        y_t = subset["label"].values
        y_p = subset["_pred"].values
        y_pr = subset["_prob"].values

        row: Dict[str, Any] = {
            "stratum": stratum_labels[i],
            "n_reviews": len(subset),
            "n_fake": int(y_t.sum()),
            "fake_rate": float(y_t.mean()),
        }
        try:
            row["auc_roc"] = float(roc_auc_score(y_t, y_pr))
        except ValueError:
            row["auc_roc"] = float("nan")
        row["macro_f1"] = float(f1_score(y_t, y_p, average="macro", zero_division=0))
        rows.append(row)

    return pd.DataFrame(rows)


# ─── Statistical Significance ─────────────────────────────────────────────────

def wilcoxon_significance_test(
    scores_a: List[float],
    scores_b: List[float],
    metric_name: str = "AUC-ROC",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Perform a Wilcoxon signed-rank test to compare two models across folds.

    Args:
        scores_a: Per-fold metric values for model A.
        scores_b: Per-fold metric values for model B.
        metric_name: Name of the metric being tested (for logging).
        alpha: Significance level. Default: 0.05.

    Returns:
        Dict with ``"statistic"``, ``"p_value"``, and ``"significant"``.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length.")
    if len(scores_a) < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "significant": False}

    stat, p_value = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
    significant = p_value < alpha

    logger.info(
        f"Wilcoxon test ({metric_name}): statistic={stat:.4f}  p={p_value:.4f}  "
        f"{'significant' if significant else 'NOT significant'} at α={alpha}"
    )

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(significant),
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_cv_roc_curves(
    fold_curves: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str,
    save_dir: Path,
) -> None:
    """Plot per-fold ROC curves with mean ± std band.

    Args:
        fold_curves: List of (fpr, tpr) tuples, one per fold.
        model_name: Model name for the plot title.
        save_dir: Directory to save the figure.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs: List[np.ndarray] = []
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fold_curves)))

    for i, (fpr, tpr) in enumerate(fold_curves):
        roc_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
        ax.plot(fpr, tpr, lw=1, color=colors[i], alpha=0.6,
                label=f"Fold {i+1} (AUC={roc_auc:.3f})")

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(interp_tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)

    ax.plot(mean_fpr, mean_tpr, "b-", lw=2.5,
            label=f"Mean ROC (AUC={mean_auc:.3f})")
    ax.fill_between(
        mean_fpr,
        mean_tpr - std_tpr,
        mean_tpr + std_tpr,
        alpha=0.15,
        color="blue",
        label="± 1 std. dev.",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — Temporal CV ROC Curves", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name.lower().replace(' ', '_')}_cv_roc.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: Path,
) -> None:
    """Plot and save a confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        model_name: Model name for title and filename.
        save_dir: Output directory.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Genuine", "Fake"])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"{model_name}\nConfusion Matrix", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        save_dir / f"{model_name.lower().replace(' ', '_')}_confusion.png",
        dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


# ─── Main Evaluation Runner ───────────────────────────────────────────────────

def run_temporal_cv(
    model_fn: Callable,
    df: pd.DataFrame,
    model_name: str,
    config: Optional[Dict] = None,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Run temporal cross-validation for a model.

    Args:
        model_fn: Callable that accepts (X_train, y_train) and returns
                  a fitted model with ``predict`` and ``predict_proba`` methods.
                  For deep learning models, use a wrapper function.
        df: DataFrame with columns ``["label", "date"]`` and feature columns.
        model_name: Human-readable model name.
        config: Configuration dictionary.
        n_folds: Number of CV folds.

    Returns:
        Dict with ``"fold_metrics"``, ``"aggregated"``, and ``"model_name"``.
    """
    if config is None:
        config = {}

    cv = TemporalCrossValidator(n_folds=n_folds)
    fold_metrics_list: List[Dict[str, float]] = []
    fold_curves: List[Tuple[np.ndarray, np.ndarray]] = []

    for fold_num, (train_idx, test_idx) in enumerate(cv.split(df), start=1):
        logger.info(f"[{model_name}] Fold {fold_num}/{n_folds}")

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        model = model_fn(train_df)

        y_true = test_df["label"].values
        y_pred, y_prob = model.evaluate(test_df)

        metrics = compute_full_metrics(y_true, y_pred, y_prob)
        fold_metrics_list.append(metrics)

        if y_prob is not None:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                fold_curves.append((fpr, tpr))
            except ValueError:
                pass

        logger.info(
            f"  Fold {fold_num}: AUC={metrics.get('auc_roc', 'nan'):.4f}  "
            f"F1={metrics['macro_f1']:.4f}"
        )

    aggregated = aggregate_fold_metrics(fold_metrics_list)

    results = {
        "model_name": model_name,
        "n_folds": n_folds,
        "fold_metrics": fold_metrics_list,
        "aggregated": aggregated,
    }

    # Plot ROC curves
    results_dir = Path(config.get("results_dir", "results"))
    if fold_curves:
        plot_cv_roc_curves(fold_curves, model_name, results_dir)

    logger.info(
        f"[{model_name}] CV Summary: "
        f"AUC={aggregated.get('auc_roc_mean', 0):.4f}±{aggregated.get('auc_roc_std', 0):.4f}  "
        f"F1={aggregated.get('macro_f1_mean', 0):.4f}±{aggregated.get('macro_f1_std', 0):.4f}"
    )

    return results


def save_evaluation_results(
    all_results: Dict[str, Any],
    save_path: Union[str, Path],
) -> None:
    """Save all evaluation results to a JSON file.

    Args:
        all_results: Nested dict of model results.
        save_path: Output file path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialisation
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    import json

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            return convert(obj)

    with open(save_path, "w") as fh:
        json.dump(all_results, fh, indent=2, cls=NumpyEncoder)

    logger.info(f"Evaluation results saved → {save_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from .utils import set_seed

    setup_logging("INFO")
    set_seed(42)

    parser = argparse.ArgumentParser(
        description="Run temporal cross-validation evaluation for ReviewGuard models."
    )
    parser.add_argument("--dataset", default="yelpzip", choices=["yelpzip", "yelpnyc"])
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--config", default="configs/default_config.yaml")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        logger.warning("Config not found; using defaults.")
        cfg = {}

    logger.info("Evaluation framework loaded. Run train_all.py for the full comparison.")
