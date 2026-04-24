import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve, confusion_matrix, ConfusionMatrixDisplay

from .utils import ensure_dirs, load_config, setup_logging

def plot_model_comparison(all_results: Dict, save_dir: Path):
    """Generates bar charts comparing AUC-ROC and Macro-F1 across all models."""
    models = []
    auc_scores = []
    f1_scores = []
    rec_fake = []

    # Parse results from all_models_comparison.json structure
    if "baselines" in all_results:
        for m in all_results["baselines"].get("models", []):
            models.append(m["model"])
            auc_scores.append(m["metrics"].get("auc_roc", 0))
            f1_scores.append(m["metrics"].get("macro_f1", 0))
            rec_fake.append(m["metrics"].get("recall_fake", 0))

    if "behavior_only_mlp" in all_results:
        m = all_results["behavior_only_mlp"].get("test_metrics", {})
        models.append("Behavior-only MLP")
        auc_scores.append(m.get("auc_roc", 0))
        f1_scores.append(m.get("macro_f1", 0))
        rec_fake.append(m.get("recall_fake", 0))

    if "reviewguard_fusion" in all_results:
        m = all_results["reviewguard_fusion"].get("test_metrics", {})
        models.append("ReviewGuard (Fusion)")
        auc_scores.append(m.get("auc_roc", 0))
        f1_scores.append(m.get("macro_f1", 0))
        rec_fake.append(m.get("recall_fake", 0))

    df = pd.DataFrame({
        "Model": models,
        "AUC-ROC": auc_scores,
        "Macro-F1": f1_scores,
        "Recall(Fake)": rec_fake
    })

    # Plot 1: AUC and F1 Comparison
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.barplot(data=df, x="Model", y="AUC-ROC", ax=ax[0], palette="viridis")
    ax[0].set_title("Model Comparison: AUC-ROC", fontweight='bold')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
    ax[0].set_ylim(0, 1.0)

    sns.barplot(data=df, x="Model", y="Macro-F1", ax=ax[1], palette="magma")
    ax[1].set_title("Model Comparison: Macro-F1", fontweight='bold')
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')
    ax[1].set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(save_dir / "model_comparison_summary.png", dpi=150)
    plt.close()

def plot_combined_roc(dataset_name: str, config: Dict, save_dir: Path):
    """Plots ROC curves for all models on a single chart."""
    models_dir = Path(config.get("models_dir", "models"))
    results_dir = Path(config.get("results_dir", "results"))
    
    plt.figure(figsize=(8, 6))
    
    # 1. Load Fusion Probs
    fusion_probs_path = models_dir / "fusion" / f"{dataset_name}_test_probs.npy"
    fusion_labels_path = models_dir / "fusion" / f"{dataset_name}_test_labels.npy"
    if fusion_probs_path.exists() and fusion_labels_path.exists():
        probs = np.load(fusion_probs_path)
        labels = np.load(fusion_labels_path)
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.plot(fpr, tpr, label=f"ReviewGuard Fusion (AUC={auc(fpr, tpr):.3f})", lw=2, color='blue')

    # 2. Load Baseline Probs
    baseline_models = [
        ("TF-IDF + SVM", "tf-idf_plus_svm"),
        ("TF-IDF + LogReg", "tf-idf_plus_logreg"),
        ("Behavior + RandomForest", "behavior_plus_randomforest")
    ]
    
    baseline_labels_path = results_dir / f"{dataset_name}_baselines_labels.npy"
    if baseline_labels_path.exists():
        labels = np.load(baseline_labels_path)
        for name, slug in baseline_models:
            prob_path = results_dir / f"{dataset_name}_{slug}_probs.npy"
            if prob_path.exists():
                probs = np.load(prob_path)
                fpr, tpr, _ = roc_curve(labels, probs)
                plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})", lw=1.2, alpha=0.8)

    # 3. Load Behavior MLP Probs
    beh_probs_path = models_dir / "behavior_branch" / f"{dataset_name}_test_probs.npy"
    beh_labels_path = models_dir / "behavior_branch" / f"{dataset_name}_test_labels.npy"
    if beh_probs_path.exists() and beh_labels_path.exists():
        probs = np.load(beh_probs_path)
        labels = np.load(beh_labels_path)
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.plot(fpr, tpr, label=f"Behavior-only MLP (AUC={auc(fpr, tpr):.3f})", lw=1.5, color='orange', linestyle='--')

    # 4. Add random baseline
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Random (AUC=0.500)")
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves Comparison — {dataset_name.upper()}", fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "roc_all_models.png", dpi=150)
    plt.close()

def plot_training_curves(dataset_name: str, config: Dict, save_dir: Path):
    """Plots training loss and accuracy from history files."""
    models_dir = Path(config.get("models_dir", "models"))
    
    # Fusion History
    history_path = models_dir / "fusion" / f"{dataset_name}_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        test_auc = [h['test_auc_roc'] for h in history]

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(epochs, train_loss, color=color, label='Train Loss', lw=2)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Test AUC', color=color)
        ax2.plot(epochs, test_auc, color=color, label='Test AUC', lw=2)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f"Training Convergence: {dataset_name.upper()} Fusion", fontweight='bold')
        fig.tight_layout()
        plt.savefig(save_dir / "training_loss_curves.png", dpi=150)
        plt.close()

def generate_report_figures(dataset_name: str = "yelpzip", config: Optional[Dict] = None):
    """Main entry point to generate all figures for the report/paper."""
    if config is None:
        config = load_config()
    
    results_dir = Path(config.get("results_dir", "results"))
    figures_dir = results_dir / "figures"
    ensure_dirs(figures_dir)
    
    comparison_path = results_dir / "all_models_comparison.json"
    if comparison_path.exists():
        with open(comparison_path, 'r') as f:
            all_results = json.load(f)
        plot_model_comparison(all_results, figures_dir)
    
    plot_combined_roc(dataset_name, config, figures_dir)
    plot_training_curves(dataset_name, config, figures_dir)
    
    print(f"Report figures generated successfully in {figures_dir}")

if __name__ == "__main__":
    setup_logging()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yelpzip")
    args = parser.parse_args()
    generate_report_figures(dataset_name=args.dataset)
