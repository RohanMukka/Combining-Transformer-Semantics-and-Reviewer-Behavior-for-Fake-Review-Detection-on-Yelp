"""
Main Training Orchestrator for ReviewGuard
==========================================

Runs all model conditions in sequence and compares results:
  1. SVM Baseline (TF-IDF + LinearSVC)
  2. Text-only (RoBERTa fine-tuned)
  3. Behavior-only (MLP)
  4. ReviewGuard (Fusion)

Saves a consolidated comparison to results/all_models_comparison.json.

Usage:
    python -m src.train_all --dataset yelpzip
    python -m src.train_all --dataset yelpzip --skip text  # skip RoBERTa training
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from .utils import ensure_dirs, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def run_all(
    config: Optional[Dict] = None,
    dataset_name: str = "yelpzip",
    skip: Optional[List[str]] = None,
) -> Dict:
    """Run all model conditions and produce a comparison table.

    Args:
        config: Configuration dictionary.
        dataset_name: Dataset to train/evaluate on.
        skip: List of conditions to skip: ``["svm", "text", "behavior", "fusion"]``.

    Returns:
        Dict with results for all conditions.
    """
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            config = {}

    skip = skip or []
    results_dir = Path(config.get("results_dir", "results"))
    ensure_dirs(results_dir)

    all_results: Dict[str, Dict] = {}

    # ── 1. SVM Baselines ──────────────────────────────────────────────────────
    if "svm" not in skip:
        logger.info("\n" + "=" * 60)
        logger.info("Step 1/4: Training SVM / Logistic Regression baselines")
        logger.info("=" * 60)
        t0 = time.perf_counter()
        from .baselines import run_baselines
        baseline_results = run_baselines(config=config, dataset_name=dataset_name)
        elapsed = time.perf_counter() - t0
        logger.info(f"Baselines complete in {elapsed:.1f}s")
        all_results["baselines"] = baseline_results
    else:
        logger.info("Skipping SVM baselines.")

    # ── 2. Text-only (RoBERTa) ────────────────────────────────────────────────
    if "text" not in skip:
        logger.info("\n" + "=" * 60)
        logger.info("Step 2/4: Fine-tuning RoBERTa text branch")
        logger.info("=" * 60)
        t0 = time.perf_counter()
        from .text_branch import train_text_branch, extract_and_save_embeddings
        model_text = train_text_branch(config=config, dataset_name=dataset_name)
        logger.info("Extracting [CLS] embeddings for fusion model …")
        extract_and_save_embeddings(config=config, dataset_name=dataset_name)
        elapsed = time.perf_counter() - t0
        logger.info(f"Text branch complete in {elapsed:.1f}s")
        all_results["text_only_roberta"] = {"status": "trained", "elapsed_s": elapsed}
    else:
        logger.info("Skipping RoBERTa text branch training.")

    # ── 3. Behavior-only MLP ─────────────────────────────────────────────────
    if "behavior" not in skip:
        logger.info("\n" + "=" * 60)
        logger.info("Step 3/4: Training behavior-only MLP")
        logger.info("=" * 60)
        t0 = time.perf_counter()
        from .behavior_branch import train_behavior_branch
        _, behavior_results = train_behavior_branch(config=config, dataset_name=dataset_name)
        elapsed = time.perf_counter() - t0
        logger.info(f"Behavior branch complete in {elapsed:.1f}s")
        all_results["behavior_only_mlp"] = behavior_results
    else:
        logger.info("Skipping behavior-only MLP.")

    # ── 4. ReviewGuard Fusion ─────────────────────────────────────────────────
    if "fusion" not in skip:
        logger.info("\n" + "=" * 60)
        logger.info("Step 4/4: Training ReviewGuard fusion model")
        logger.info("=" * 60)
        t0 = time.perf_counter()
        from .fusion_model import train_fusion_model
        _, fusion_results = train_fusion_model(config=config, dataset_name=dataset_name)
        elapsed = time.perf_counter() - t0
        logger.info(f"Fusion model complete in {elapsed:.1f}s")
        all_results["reviewguard_fusion"] = fusion_results
    else:
        logger.info("Skipping ReviewGuard fusion model.")

    # ── Consolidated results ──────────────────────────────────────────────────
    comparison_path = results_dir / "all_models_comparison.json"
    with open(comparison_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    logger.info(f"\nAll results saved → {comparison_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    _print_summary_table(all_results)

    return all_results


def _print_summary_table(all_results: Dict) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 70)
    print(f"{'Model':<35} {'AUC-ROC':>8} {'Macro-F1':>9} {'Recall(F)':>10}")
    print("-" * 70)

    def _row(name: str, auc: float, f1: float, rec: float) -> None:
        print(f"{name:<35} {auc:>8.4f} {f1:>9.4f} {rec:>10.4f}")

    # Baselines
    if "baselines" in all_results:
        for model_result in all_results["baselines"].get("models", []):
            m = model_result.get("metrics", {})
            _row(
                model_result.get("model", "Unknown"),
                m.get("auc_roc", 0.0),
                m.get("macro_f1", 0.0),
                m.get("recall_fake", 0.0),
            )

    # Behavior-only MLP
    if "behavior_only_mlp" in all_results:
        m = all_results["behavior_only_mlp"].get("test_metrics", {})
        _row("Behavior-only MLP", m.get("auc_roc", 0.0), m.get("macro_f1", 0.0),
             m.get("recall_fake", 0.0))

    # ReviewGuard Fusion
    if "reviewguard_fusion" in all_results:
        m = all_results["reviewguard_fusion"].get("test_metrics", {})
        _row("ReviewGuard (Fusion)", m.get("auc_roc", 0.0), m.get("macro_f1", 0.0),
             m.get("recall_fake", 0.0))

    print("=" * 70 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging("INFO")
    set_seed(42)

    parser = argparse.ArgumentParser(
        description="Train all ReviewGuard models and compare results."
    )
    parser.add_argument("--dataset", default="yelpzip", choices=["yelpzip", "yelpnyc"])
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["svm", "text", "behavior", "fusion"],
        help="Conditions to skip (e.g., --skip text to skip RoBERTa training)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        logger.warning("Config not found; using defaults.")
        cfg = {}

    run_all(config=cfg, dataset_name=args.dataset, skip=args.skip)
