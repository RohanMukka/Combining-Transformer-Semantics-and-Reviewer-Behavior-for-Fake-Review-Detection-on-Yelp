"""
Reviewer Behavior Feature Engineering for ReviewGuard
======================================================

Computes 6 handcrafted reviewer behavioral features per review:

  1. avg_star_rating      — Mean star rating across all of the reviewer's reviews
  2. review_count         — Total number of reviews written by this reviewer
  3. burst_ratio          — Fraction of reviews posted within 30-day burst windows
  4. rating_deviation     — |review_rating − product_average_rating|
  5. category_diversity   — Unique product_ids / total reviews (proxy for category spread)
  6. account_age_at_review — Days between the reviewer's first review and this review

All features are standardised (zero mean, unit variance) using StandardScaler
fit on the training set only (scaler is saved and re-applied to test/val sets).

Usage:
    python -m src.behavior_features --input data/processed/yelpzip.parquet
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from .utils import ensure_dirs, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)

FEATURE_NAMES: List[str] = [
    "avg_star_rating",
    "review_count",
    "burst_ratio",
    "rating_deviation",
    "category_diversity",
    "account_age_at_review",
]

BURST_WINDOW_DAYS: int = 30  # window size for burst detection


# ─── Individual Feature Computations ─────────────────────────────────────────

def _compute_avg_star_rating(df: pd.DataFrame) -> pd.Series:
    """Compute the mean star rating per reviewer across ALL their reviews.

    This captures whether a reviewer tends to give systematically high or low
    ratings — a potential indicator of paid positive/negative reviewing.

    Args:
        df: DataFrame with columns ``["user_id", "rating"]``.

    Returns:
        Series indexed by the original DataFrame index.
    """
    user_mean = df.groupby("user_id")["rating"].transform("mean")
    return user_mean.rename("avg_star_rating")


def _compute_review_count(df: pd.DataFrame) -> pd.Series:
    """Count the total number of reviews per reviewer.

    Very prolific reviewers posting bursts of reviews are flagged in the
    literature as suspicious (Lim et al., 2010; Rayana & Akoglu, 2015).

    Args:
        df: DataFrame with column ``["user_id"]``.

    Returns:
        Series indexed by the original DataFrame index.
    """
    counts = df.groupby("user_id")["user_id"].transform("count")
    return counts.rename("review_count")


def _compute_burst_ratio(df: pd.DataFrame) -> pd.Series:
    """Compute the fraction of a reviewer's reviews within 30-day burst windows.

    For each reviewer, a burst window starts at their earliest review. Any
    subsequent review within BURST_WINDOW_DAYS of the window's start extends
    coverage of that burst. The ratio is:

        burst_ratio = (# reviews in bursts) / (total reviews)

    Reviewers with burst_ratio > 0.5 are often spam ring members.

    Args:
        df: DataFrame with columns ``["user_id", "date"]``.
            The ``"date"`` column must be datetime.

    Returns:
        Series indexed by the original DataFrame index.
    """
    df_work = df[["user_id", "date"]].copy()
    df_work["date"] = pd.to_datetime(df_work["date"])

    # Sort per-user by date to iterate chronologically
    df_sorted = df_work.sort_values(["user_id", "date"])

    burst_flags: Dict[int, bool] = {}

    for user_id, group in df_sorted.groupby("user_id"):
        dates = group["date"].tolist()
        indices = group.index.tolist()

        if len(dates) == 1:
            # Single review: by convention not a burst
            burst_flags[indices[0]] = False
            continue

        # Greedy burst window detection
        window_start = dates[0]
        in_burst = [False] * len(dates)

        for i, (idx, date) in enumerate(zip(indices, dates)):
            delta = (date - window_start).days
            if delta <= BURST_WINDOW_DAYS:
                in_burst[i] = True
            else:
                # New window starts here
                window_start = date
                in_burst[i] = False

        # A reviewer is in a burst if ≥ 2 reviews share a window
        window_start = dates[0]
        burst_count = 0
        burst_membership = [False] * len(dates)
        current_window_indices: List[int] = [0]

        for i in range(1, len(dates)):
            delta = (dates[i] - dates[current_window_indices[0]]).days
            if delta <= BURST_WINDOW_DAYS:
                current_window_indices.append(i)
            else:
                if len(current_window_indices) >= 2:
                    for j in current_window_indices:
                        burst_membership[j] = True
                current_window_indices = [i]

        if len(current_window_indices) >= 2:
            for j in current_window_indices:
                burst_membership[j] = True

        for idx, is_burst in zip(indices, burst_membership):
            burst_flags[idx] = is_burst

    burst_series = pd.Series(burst_flags, name="_burst_flag")
    df_work["_burst_flag"] = burst_series

    # Burst ratio = fraction of this user's reviews that are in bursts
    user_burst_ratio = (
        df_work.groupby("user_id")["_burst_flag"]
        .transform("mean")
        .rename("burst_ratio")
    )
    return user_burst_ratio.reindex(df.index)


def _compute_rating_deviation(df: pd.DataFrame) -> pd.Series:
    """Compute |review_rating - product_average_rating|.

    Fake reviews often deviate significantly from the product's true average
    (either extremely positive or extremely negative) to manipulate the score.

    Args:
        df: DataFrame with columns ``["product_id", "rating"]``.

    Returns:
        Series indexed by the original DataFrame index.
    """
    product_mean = df.groupby("product_id")["rating"].transform("mean")
    deviation = (df["rating"] - product_mean).abs().rename("rating_deviation")
    return deviation


def _compute_category_diversity(df: pd.DataFrame) -> pd.Series:
    """Estimate category diversity as unique products / total reviews.

    In the absence of explicit business category labels, product_id is used as
    a proxy. Genuine reviewers tend to have higher diversity (reviewing many
    different businesses); spam ring members focus on a narrow set of targets.

    score = n_unique_products / n_reviews  ∈ (0, 1]

    Args:
        df: DataFrame with columns ``["user_id", "product_id"]``.

    Returns:
        Series indexed by the original DataFrame index.
    """
    unique_products = df.groupby("user_id")["product_id"].transform("nunique")
    review_count = df.groupby("user_id")["product_id"].transform("count")
    diversity = (unique_products / review_count.clip(lower=1)).rename("category_diversity")
    return diversity


def _compute_account_age_at_review(df: pd.DataFrame) -> pd.Series:
    """Compute the number of days between a reviewer's first review and each review.

    Very new accounts posting many reviews quickly (low account age + high
    burst ratio) are a strong signal for fake review behaviour.

    Args:
        df: DataFrame with columns ``["user_id", "date"]``.
            The ``"date"`` column must be datetime.

    Returns:
        Series indexed by the original DataFrame index (values in days).
    """
    df_work = df[["user_id", "date"]].copy()
    df_work["date"] = pd.to_datetime(df_work["date"])

    first_review = df_work.groupby("user_id")["date"].transform("min")
    age_days = (df_work["date"] - first_review).dt.days.rename("account_age_at_review")
    return age_days.reindex(df.index)


# ─── Main Feature Builder ─────────────────────────────────────────────────────

def compute_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 6 reviewer behavior features for a dataset.

    Args:
        df: DataFrame with columns ``["user_id", "product_id", "rating", "date"]``.

    Returns:
        DataFrame with 6 feature columns (same index as input *df*).
        The column order matches :data:`FEATURE_NAMES`.
    """
    logger.info(f"Computing behaviour features for {len(df):,} reviews …")

    df = df.copy()

    # Ensure datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        logger.warning("'date' column missing; account_age_at_review will be 0.")
        df["date"] = pd.Timestamp("2010-01-01")

    features = pd.DataFrame(index=df.index)
    features["avg_star_rating"] = _compute_avg_star_rating(df)
    features["review_count"] = _compute_review_count(df)
    features["burst_ratio"] = _compute_burst_ratio(df)
    features["rating_deviation"] = _compute_rating_deviation(df)
    features["category_diversity"] = _compute_category_diversity(df)
    features["account_age_at_review"] = _compute_account_age_at_review(df)

    # Fill any NaN (e.g., single-product reviewers)
    features = features.fillna(0.0)

    logger.info("Feature computation complete.")
    logger.debug(f"Feature stats:\n{features.describe().to_string()}")

    return features[FEATURE_NAMES]


# ─── Normalisation ────────────────────────────────────────────────────────────

def fit_scaler(
    train_features: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> StandardScaler:
    """Fit a StandardScaler on training features.

    Args:
        train_features: DataFrame with columns matching :data:`FEATURE_NAMES`.
        save_path: If provided, pickle the fitted scaler to this path.

    Returns:
        Fitted :class:`sklearn.preprocessing.StandardScaler`.
    """
    scaler = StandardScaler()
    scaler.fit(train_features[FEATURE_NAMES].values)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as fh:
            pickle.dump(scaler, fh)
        logger.info(f"Scaler saved → {save_path}")

    return scaler


def apply_scaler(
    features: pd.DataFrame,
    scaler: StandardScaler,
) -> np.ndarray:
    """Apply a fitted scaler to a feature DataFrame.

    Args:
        features: DataFrame with columns matching :data:`FEATURE_NAMES`.
        scaler: Fitted StandardScaler.

    Returns:
        Normalised feature array of shape ``(N, 6)``.
    """
    return scaler.transform(features[FEATURE_NAMES].values).astype(np.float32)


def load_scaler(path: Union[str, Path]) -> StandardScaler:
    """Load a pickled StandardScaler from disk.

    Args:
        path: Path to the pickled scaler file.

    Returns:
        Loaded :class:`sklearn.preprocessing.StandardScaler`.
    """
    with open(path, "rb") as fh:
        scaler = pickle.load(fh)
    logger.info(f"Scaler loaded from {path}")
    return scaler


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_feature_distributions(
    features: pd.DataFrame,
    labels: pd.Series,
    save_dir: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Plot feature distributions split by fake vs. genuine label.

    Generates one KDE plot per feature, with overlapping density curves for
    fake (label=1) and genuine (label=0) reviews.

    Args:
        features: DataFrame with columns matching :data:`FEATURE_NAMES`.
        labels: Binary label Series aligned with *features*.
        save_dir: Directory to save plots (PNG format). Created if absent.
        show: If ``True``, call ``plt.show()`` after each plot.
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    palette = {0: "#2196F3", 1: "#F44336"}
    label_names = {0: "Genuine", 1: "Fake"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, feat_name in enumerate(FEATURE_NAMES):
        ax = axes[i]
        for lbl, grp in features.assign(_label=labels).groupby("_label"):
            vals = grp[feat_name].dropna()
            sns.kdeplot(
                vals,
                ax=ax,
                label=label_names[int(lbl)],
                color=palette[int(lbl)],
                fill=True,
                alpha=0.35,
                linewidth=2,
            )
        ax.set_title(feat_name.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel(feat_name)
        ax.set_ylabel("Density")
        ax.legend()

    plt.suptitle(
        "Behavior Feature Distributions: Fake vs. Genuine Reviews",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    if save_dir is not None:
        fig_path = save_dir / "behavior_feature_distributions.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved distribution plot → {fig_path}")

    if show:
        plt.show()
    plt.close(fig)

    # Correlation heatmap
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    corr_matrix = features[FEATURE_NAMES].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax2,
        square=True,
        linewidths=0.5,
    )
    ax2.set_title("Behavior Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_dir is not None:
        fig2_path = save_dir / "behavior_feature_correlation.png"
        plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved correlation plot → {fig2_path}")

    if show:
        plt.show()
    plt.close(fig2)


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def run_feature_pipeline(
    df: pd.DataFrame,
    split: str = "train",
    config: Optional[Dict] = None,
    dataset_name: str = "yelpzip",
    scaler: Optional[StandardScaler] = None,
    fit_new_scaler: bool = True,
) -> Tuple[np.ndarray, StandardScaler]:
    """End-to-end behavior feature pipeline.

    Computes raw features, fits/applies a scaler, saves outputs, and
    generates distribution plots.

    Args:
        df: Preprocessed DataFrame with required columns.
        split: ``"train"`` or ``"test"`` — used for file naming.
        config: Configuration dictionary.
        dataset_name: Dataset identifier for file naming.
        scaler: Pre-fitted scaler. If ``None`` and *fit_new_scaler* is
                ``True``, a new scaler is fitted on *df*.
        fit_new_scaler: Whether to fit a new scaler on *df*.

    Returns:
        Tuple of (normalised_features_array, scaler).
    """
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            config = {}

    features_dir = Path(config.get("data", {}).get("features_dir", "data/features"))
    ensure_dirs(features_dir)

    raw_features = compute_behavior_features(df)

    if fit_new_scaler or scaler is None:
        scaler_path = features_dir / f"{dataset_name}_scaler.pkl"
        scaler = fit_scaler(raw_features, save_path=scaler_path)
        logger.info("New scaler fitted on training features.")

    normalised = apply_scaler(raw_features, scaler)

    # Save raw and normalised features
    raw_features.to_parquet(
        features_dir / f"{dataset_name}_{split}_raw_features.parquet"
    )
    np.save(
        features_dir / f"{dataset_name}_{split}_features.npy",
        normalised,
    )
    logger.info(f"Features saved to {features_dir}/")

    # Distribution plots (for training split only)
    if split == "train" and "label" in df.columns:
        plot_dir = Path(config.get("results_dir", "results")) / "feature_plots"
        plot_feature_distributions(
            raw_features,
            df["label"].reindex(raw_features.index),
            save_dir=plot_dir,
            show=False,
        )

    return normalised, scaler


# ─── Import fix ───────────────────────────────────────────────────────────────
from typing import Union  # noqa: E402 (placed after function defs for clarity)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging("INFO")
    set_seed(42)

    parser = argparse.ArgumentParser(
        description="Compute reviewer behavior features from a preprocessed dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/yelpzip.parquet",
        help="Path to preprocessed parquet file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="yelpzip",
        help="Dataset name for output file naming",
    )
    parser.add_argument(
        "--config",
        default="configs/default_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        cfg = {}

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} reviews from {args.input}")

    from .data_loader import temporal_train_test_split
    train_df, test_df = temporal_train_test_split(
        df, cutoff_fraction=cfg.get("data", {}).get("temporal_cutoff_fraction", 0.8)
    )

    logger.info("Computing features for training split …")
    train_feats, scaler = run_feature_pipeline(
        train_df, split="train", config=cfg, dataset_name=args.dataset,
        fit_new_scaler=True,
    )

    logger.info("Computing features for test split …")
    test_feats, _ = run_feature_pipeline(
        test_df, split="test", config=cfg, dataset_name=args.dataset,
        scaler=scaler, fit_new_scaler=False,
    )

    logger.info(
        f"Done. Train features: {train_feats.shape}, Test features: {test_feats.shape}"
    )
