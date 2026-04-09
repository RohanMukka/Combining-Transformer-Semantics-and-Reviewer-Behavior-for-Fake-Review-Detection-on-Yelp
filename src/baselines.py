"""
Baseline Classifiers for ReviewGuard
======================================

Implements three baseline pipelines evaluated under temporal cross-validation:

  1. TF-IDF + SVM (linear kernel) — text-only
  2. TF-IDF + Logistic Regression — text-only
  3. Behavior features only + Random Forest — behavior-only

All models are evaluated with:
  - AUC-ROC
  - Macro-F1
  - Per-class precision & recall
  - Confusion matrix

Results are saved to ``results/baseline_results.json``.

Usage:
    python -m src.baselines
    python -m src.baselines --dataset yelpzip --config configs/default_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from .utils import ensure_dirs, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


# ─── Metrics Helper ───────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute evaluation metrics for a binary classifier.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_pred: Predicted binary labels.
        y_prob: Predicted probability of the positive class (for AUC-ROC).
                If ``None``, AUC-ROC is omitted.

    Returns:
        Dictionary with keys: ``auc_roc``, ``macro_f1``, ``precision_fake``,
        ``recall_fake``, ``precision_genuine``, ``recall_genuine``.
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics: Dict[str, float] = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_fake": report.get("1", {}).get("f1-score", 0.0),
        "f1_genuine": report.get("0", {}).get("f1-score", 0.0),
        "precision_fake": report.get("1", {}).get("precision", 0.0),
        "recall_fake": report.get("1", {}).get("recall", 0.0),
        "precision_genuine": report.get("0", {}).get("precision", 0.0),
        "recall_genuine": report.get("0", {}).get("recall", 0.0),
        "accuracy": report.get("accuracy", 0.0),
    }
    if y_prob is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc_roc"] = 0.0
    return metrics


# ─── TF-IDF + SVM ─────────────────────────────────────────────────────────────

def build_tfidf_svm_pipeline(config: Dict) -> Pipeline:
    """Build a TF-IDF + LinearSVC pipeline (calibrated for probability output).

    The SVM is calibrated via Platt scaling (CalibratedClassifierCV) so that
    ``predict_proba`` is available for AUC-ROC computation.

    Args:
        config: Configuration dictionary containing ``baselines`` settings.

    Returns:
        Scikit-learn :class:`Pipeline`.
    """
    bl_cfg = config.get("baselines", {})
    tfidf = TfidfVectorizer(
        max_features=bl_cfg.get("tfidf_max_features", 50_000),
        ngram_range=tuple(bl_cfg.get("tfidf_ngram_range", [1, 2])),
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        min_df=2,
    )
    svm = CalibratedClassifierCV(
        LinearSVC(
            C=bl_cfg.get("svm_C", 1.0),
            class_weight="balanced",
            max_iter=2000,
        ),
        cv=3,
        method="sigmoid",
    )
    return Pipeline([("tfidf", tfidf), ("svm", svm)])


# ─── TF-IDF + Logistic Regression ────────────────────────────────────────────

def build_tfidf_lr_pipeline(config: Dict) -> Pipeline:
    """Build a TF-IDF + Logistic Regression pipeline.

    Args:
        config: Configuration dictionary containing ``baselines`` settings.

    Returns:
        Scikit-learn :class:`Pipeline`.
    """
    bl_cfg = config.get("baselines", {})
    tfidf = TfidfVectorizer(
        max_features=bl_cfg.get("tfidf_max_features", 50_000),
        ngram_range=tuple(bl_cfg.get("tfidf_ngram_range", [1, 2])),
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        min_df=2,
    )
    lr = LogisticRegression(
        C=bl_cfg.get("lr_C", 1.0),
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
    )
    return Pipeline([("tfidf", tfidf), ("lr", lr)])


# ─── Behavior Features + Random Forest ────────────────────────────────────────

def build_rf_pipeline(config: Dict) -> Pipeline:
    """Build a behavior features + Random Forest pipeline.

    Features are z-score normalised before being passed to the forest.

    Args:
        config: Configuration dictionary containing ``baselines`` settings.

    Returns:
        Scikit-learn :class:`Pipeline`.
    """
    bl_cfg = config.get("baselines", {})
    scaler = StandardScaler()
    rf = RandomForestClassifier(
        n_estimators=bl_cfg.get("rf_n_estimators", 200),
        max_depth=bl_cfg.get("rf_max_depth", 15),
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("scaler", scaler), ("rf", rf)])


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    y_test: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """Fit a pipeline on training data and evaluate on test data.

    Args:
        pipeline: Scikit-learn Pipeline.
        X_train: Training features (text strings or numeric array).
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        model_name: Human-readable model identifier for logging.

    Returns:
        Results dictionary with metrics and model name.
    """
    logger.info(f"Training {model_name} …")
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    logger.info(f"  Training time: {train_time:.1f}s")

    y_pred = pipeline.predict(X_test)

    # Probability estimates
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["train_time_seconds"] = round(train_time, 2)

    logger.info(
        f"  {model_name}: AUC={metrics.get('auc_roc', 'N/A'):.4f}  "
        f"Macro-F1={metrics['macro_f1']:.4f}  "
        f"Recall(fake)={metrics['recall_fake']:.4f}"
    )

    return {
        "model": model_name,
        "metrics": metrics,
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist() if y_prob is not None else None,
    }


# ─── Confusion Matrix Plotting ────────────────────────────────────────────────

def plot_confusion_matrices(
    results: List[Dict],
    y_test: np.ndarray,
    save_dir: Path,
) -> None:
    """Plot side-by-side confusion matrices for all baseline models.

    Args:
        results: List of result dictionaries from :func:`evaluate_pipeline`.
        y_test: Ground-truth test labels.
        save_dir: Directory to save the plot.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        cm = confusion_matrix(y_test, result["y_pred"])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Genuine", "Fake"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(result["model"], fontsize=11, fontweight="bold")

    plt.suptitle("Baseline Confusion Matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "baseline_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix plot saved → {save_dir}/baseline_confusion_matrices.png")


def plot_roc_curves(
    results: List[Dict],
    y_test: np.ndarray,
    save_dir: Path,
) -> None:
    """Plot ROC curves for all baseline models that have probability estimates.

    Args:
        results: List of result dictionaries.
        y_test: Ground-truth test labels.
        save_dir: Directory to save the plot.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#E91E63", "#9C27B0", "#FF9800", "#4CAF50"]

    for result, color in zip(results, colors):
        if result["y_prob"] is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, result["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{result['model']} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Baseline ROC Curves", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "baseline_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curve plot saved → {save_dir}/baseline_roc_curves.png")


# ─── Main Baseline Runner ─────────────────────────────────────────────────────

def run_baselines(
    config: Optional[Dict] = None,
    dataset_name: str = "yelpzip",
) -> Dict[str, Any]:
    """Run all baseline models and save results.

    Loads the preprocessed dataset, computes behavior features, performs a
    temporal train/test split, trains all baselines, evaluates, and saves
    results to ``results/baseline_results.json``.

    Args:
        config: Configuration dictionary.
        dataset_name: Dataset identifier (``"yelpzip"`` or ``"yelpnyc"``).

    Returns:
        Dictionary containing all results.
    """
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            config = {}

    results_dir = Path(config.get("results_dir", "results"))
    ensure_dirs(results_dir)

    # ── Load data ──
    from .data_loader import load_dataset, temporal_train_test_split
    from .behavior_features import compute_behavior_features, fit_scaler, apply_scaler

    df = load_dataset(dataset_name, config=config)
    train_df, test_df = temporal_train_test_split(
        df,
        cutoff_fraction=config.get("data", {}).get("temporal_cutoff_fraction", 0.8),
    )

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    logger.info(
        f"Splits — Train: {len(train_df):,} (fake={y_train.mean():.1%})  "
        f"Test: {len(test_df):,} (fake={y_test.mean():.1%})"
    )

    # ── Text features (raw strings for TF-IDF) ──
    X_train_text = train_df["text"].fillna("").tolist()
    X_test_text = test_df["text"].fillna("").tolist()

    # ── Behavior features ──
    logger.info("Computing behavior features …")
    train_feats = compute_behavior_features(train_df)
    test_feats = compute_behavior_features(test_df)

    scaler = fit_scaler(train_feats)
    X_train_behavior = apply_scaler(train_feats, scaler)
    X_test_behavior = apply_scaler(test_feats, scaler)

    # ── Train & evaluate all baselines ──
    all_results: List[Dict] = []

    # 1. TF-IDF + SVM
    svm_pipe = build_tfidf_svm_pipeline(config)
    svm_result = evaluate_pipeline(
        svm_pipe, X_train_text, y_train, X_test_text, y_test,
        model_name="TF-IDF + SVM",
    )
    all_results.append(svm_result)

    # 2. TF-IDF + Logistic Regression
    lr_pipe = build_tfidf_lr_pipeline(config)
    lr_result = evaluate_pipeline(
        lr_pipe, X_train_text, y_train, X_test_text, y_test,
        model_name="TF-IDF + LogReg",
    )
    all_results.append(lr_result)

    # 3. Behavior features + Random Forest
    rf_pipe = build_rf_pipeline(config)
    rf_result = evaluate_pipeline(
        rf_pipe, X_train_behavior, y_train, X_test_behavior, y_test,
        model_name="Behavior + RandomForest",
    )
    all_results.append(rf_result)

    # ── Plots ──
    plot_confusion_matrices(all_results, y_test, results_dir)
    plot_roc_curves(all_results, y_test, results_dir)

    # ── Save results ──
    output = {
        "dataset": dataset_name,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "models": [
            {"model": r["model"], "metrics": r["metrics"]}
            for r in all_results
        ],
    }

    results_path = results_dir / "baseline_results.json"
    with open(results_path, "w") as fh:
        json.dump(output, fh, indent=2)

    logger.info(f"Baseline results saved → {results_path}")

    # ── Print summary table ──
    print("\n" + "=" * 65)
    print(f"{'Model':<30} {'AUC-ROC':>8} {'Macro-F1':>9} {'Recall(F)':>10}")
    print("-" * 65)
    for r in all_results:
        m = r["metrics"]
        print(
            f"{r['model']:<30} "
            f"{m.get('auc_roc', 0.0):>8.4f} "
            f"{m['macro_f1']:>9.4f} "
            f"{m['recall_fake']:>10.4f}"
        )
    print("=" * 65 + "\n")

    return output


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging("INFO")
    set_seed(42)

    parser = argparse.ArgumentParser(
        description="Train and evaluate baseline classifiers for fake review detection."
    )
    parser.add_argument("--dataset", default="yelpzip", choices=["yelpzip", "yelpnyc"])
    parser.add_argument("--config", default="configs/default_config.yaml")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        logger.warning("Config not found; using defaults.")
        cfg = {}

    run_baselines(config=cfg, dataset_name=args.dataset)
