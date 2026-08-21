"""Regenerate every figure in paper/figures/ from the recorded metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
METRICS = REPO / "results" / "metrics"
FIG_RESULTS = REPO / "results" / "figures"
FIG_PAPER = REPO / "paper" / "figures"

MODEL_ORDER = [
    "TF-IDF + LinearSVM",
    "TF-IDF + LogReg",
    "Behavior + LogReg",
    "Behavior + RandomForest",
    "Text-only MLP",
    "Behavior-only MLP",
    "ReviewGuard (Fusion)",
]
HIGHLIGHT = "ReviewGuard (Fusion)"
C_MAIN, C_ALT, C_HL = "#4C72B0", "#C44E52", "#DD8452"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
})


def _load(name: str) -> Dict:
    path = METRICS / name
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run src.run_real_experiments first")
    return json.loads(path.read_text())


def _save(fig, stem: str) -> None:
    for d in (FIG_RESULTS, FIG_PAPER):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s.png", stem)


def fig_protocol_comparison(prod: Dict, user: Dict) -> None:
    """The headline figure: same models, two split protocols."""
    models = [m for m in MODEL_ORDER if m in prod["summary"]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    metrics = [("auc_roc", "AUC-ROC"), ("macro_f1", "Macro-F1"), ("recall_fake", "Recall (Fake)")]
    ypos = np.arange(len(models))

    for ax, (key, title) in zip(axes, metrics):
        pv = [prod["summary"][m][key]["mean"] for m in models]
        pe = [prod["summary"][m][key]["std"] for m in models]
        uv = [user["summary"][m][key]["mean"] for m in models]
        ue = [user["summary"][m][key]["std"] for m in models]
        ax.barh(ypos + 0.2, uv, 0.38, xerr=ue, color=C_ALT, alpha=0.85,
                label="Reviewer-disjoint", error_kw={"lw": 0.8})
        ax.barh(ypos - 0.2, pv, 0.38, xerr=pe, color=C_MAIN, alpha=0.9,
                label="Business-disjoint", error_kw={"lw": 0.8})
        ax.set_yticks(ypos)
        ax.set_yticklabels(models if ax is axes[0] else [])
        ax.set_title(title)
        ax.set_xlim(0, 1.0)
        ax.axvline(0.5, color="grey", lw=0.8, ls=":")
    axes[0].legend(loc="lower right", fontsize=7.5, framealpha=0.95)
    fig.suptitle(
        "Business-disjoint evaluation removes most of the apparent performance",
        y=1.02, fontsize=10.5,
    )
    _save(fig, "protocol_comparison")


def fig_model_comparison(prod: Dict) -> None:
    models = [m for m in MODEL_ORDER if m in prod["summary"]]
    metrics = [("auc_roc", "AUC-ROC"), ("average_precision", "Avg. Precision"),
               ("macro_f1", "Macro-F1"), ("recall_fake", "Recall (Fake)")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    ypos = np.arange(len(models))
    for ax, (key, title) in zip(axes, metrics):
        vals = [prod["summary"][m][key]["mean"] for m in models]
        errs = [prod["summary"][m][key]["std"] for m in models]
        colors = [C_HL if m == HIGHLIGHT else C_MAIN for m in models]
        ax.barh(ypos, vals, xerr=errs, color=colors, alpha=0.9,
                error_kw={"lw": 0.8})
        ax.set_yticks(ypos)
        ax.set_yticklabels(models if ax is axes[0] else [], fontsize=7.5)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(0, max(0.85, max(vals) * 1.3))
        for yi, v in zip(ypos, vals):
            ax.text(v + 0.015, yi, f"{v:.3f}", va="center", fontsize=6.5)
    fig.suptitle("Five-fold performance, business-disjoint protocol (mean ± s.d.)",
                 y=1.04, fontsize=10.5)
    _save(fig, "final_model_comparison")


def fig_fold_stability(prod: Dict) -> None:
    models = [m for m in MODEL_ORDER if m in prod["summary"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ypos = np.arange(len(models))
    for i, m in enumerate(models):
        folds = prod["summary"][m]["auc_roc"]["folds"]
        color = C_HL if m == HIGHLIGHT else C_MAIN
        ax.scatter(folds, [i] * len(folds), color=color, alpha=0.7, s=24, zorder=3)
        ax.plot([np.mean(folds)] * 2, [i - 0.24, i + 0.24],
                color="black", lw=1.6, zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xlabel("AUC-ROC")
    ax.set_title("Per-fold AUC-ROC (bar = mean)", fontsize=9.5)
    _save(fig, "fold_stability")


def fig_business_leakage(df: pd.DataFrame) -> None:
    per = df.groupby("product_id")["label"].agg(["mean", "size"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    axes[0].hist(per["mean"], bins=30, color=C_MAIN, alpha=0.9)
    axes[0].set_xlabel("Fake-review rate of a business")
    axes[0].set_ylabel("Number of businesses")
    axes[0].set_title(
        f"{int((per['mean'] == 0).sum())} of {len(per)} businesses have no fake reviews"
    )
    order = per.sort_values("mean")
    axes[1].plot(np.arange(len(order)), order["mean"].values, color=C_ALT, lw=1.6)
    axes[1].fill_between(np.arange(len(order)), order["mean"].values, color=C_ALT, alpha=0.25)
    axes[1].set_xlabel("Businesses, sorted by fake-review rate")
    axes[1].set_ylabel("Fake-review rate")
    axes[1].set_title("Label is near-constant within a business")
    _save(fig, "business_label_concentration")


def fig_timestamp_contamination(df: pd.DataFrame, audit: Dict) -> None:
    d = pd.to_datetime(df["date"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    monthly = df.assign(_m=d.dt.to_period("M").astype(str)).groupby("_m")["label"].mean()
    axes[0].bar(monthly.index, monthly.values, color=C_ALT, alpha=0.9)
    axes[0].set_ylabel("Fake-review rate")
    axes[0].set_title("Fake rate by month of the released timestamp")
    axes[0].tick_params(axis="x", rotation=30)
    hours = d.dt.hour.value_counts().sort_index()
    axes[1].bar(hours.index, hours.values, color=C_MAIN, alpha=0.9)
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Reviews")
    axes[1].set_title("Posting hour is uniform (times are synthetic)")
    fig.suptitle(
        "Released timestamps are contaminated: AUC "
        f"{audit['timestamps']['auc_from_timestamp_alone']:.3f} from the timestamp alone",
        y=1.03, fontsize=10,
    )
    _save(fig, "timestamp_contamination")


def fig_confusion(scores: Dict) -> None:
    from sklearn.metrics import confusion_matrix
    y = scores["y_true"]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    cm = confusion_matrix(y, (scores["fusion"] >= scores["threshold"]).astype(int))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Genuine", "Fake"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Genuine", "Fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("ReviewGuard, pooled over 5 folds", fontsize=9.5)
    ax.grid(False)
    _save(fig, "cm_reviewguard_fusion")


def fig_roc(scores: Dict) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve
    y = scores["y_true"]
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    for name, key, color in [
        ("ReviewGuard (Fusion)", "fusion", C_HL),
        ("Behavior-only MLP", "behavior", C_MAIN),
        ("Text-only MLP", "text", C_ALT),
    ]:
        if key not in scores:
            continue
        fpr, tpr, _ = roc_curve(y, scores[key])
        ax.plot(fpr, tpr, color=color, lw=1.7,
                label=f"{name} (AUC {roc_auc_score(y, scores[key]):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC, pooled over 5 business-disjoint folds", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=7.5)
    _save(fig, "roc_all_models")


def fig_regime(prod_ind: Dict, prod_tra: Dict) -> None:
    """Business-disjoint split under both feature regimes."""
    models = [m for m in MODEL_ORDER
              if m in prod_ind["summary"] and m in prod_tra["summary"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ypos = np.arange(len(models))
    iv = [prod_ind["summary"][m]["auc_roc"]["mean"] for m in models]
    ie = [prod_ind["summary"][m]["auc_roc"]["std"] for m in models]
    tv = [prod_tra["summary"][m]["auc_roc"]["mean"] for m in models]
    te = [prod_tra["summary"][m]["auc_roc"]["std"] for m in models]
    ax.barh(ypos + 0.2, tv, 0.38, xerr=te, color=C_MAIN, alpha=0.9,
            label="Transductive (label-free)", error_kw={"lw": 0.8})
    ax.barh(ypos - 0.2, iv, 0.38, xerr=ie, color=C_ALT, alpha=0.85,
            label="Inductive (training rows only)", error_kw={"lw": 0.8})
    ax.set_yticks(ypos)
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xlabel("AUC-ROC")
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color="grey", lw=0.8, ls=":")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.95)
    ax.set_title("Business-disjoint split, both feature regimes", fontsize=10)
    _save(fig, "regime_comparison")


def fig_degree_control(deg_prod: Dict, deg_user: Dict) -> None:
    """What a graph degree is worth under each split."""
    keys = ["Behavior MLP", "Behavior MLP + graph degree", "R-S-R degree alone"]
    labels = ["Behavior MLP", "+ graph degree", "R-S-R degree alone"]
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ypos = np.arange(len(keys))
    uv = [deg_user["summary"][k]["auc_roc"]["mean"] for k in keys]
    pv = [deg_prod["summary"][k]["auc_roc"]["mean"] for k in keys]
    ax.barh(ypos + 0.2, uv, 0.38, color=C_ALT, alpha=0.85, label="Reviewer-disjoint")
    ax.barh(ypos - 0.2, pv, 0.38, color=C_MAIN, alpha=0.9, label="Business-disjoint")
    for y, (u, q) in enumerate(zip(uv, pv)):
        ax.text(max(u, q) + 0.012, y, f"gap {u - q:+.3f}", va="center", fontsize=7.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUC-ROC")
    ax.set_xlim(0, 1.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              fontsize=7.5, frameon=False)
    ax.set_title("A degree is worth the same under either split", fontsize=10)
    _save(fig, "degree_control")


def fig_shap_behavior() -> None:
    """Mean |SHAP| per behavior feature, from the behavior-branch MLP."""
    path = METRICS / "shap_behavior.npz"
    if not path.exists():
        logger.warning("  %s absent - skipping SHAP figure", path)
        return
    z = np.load(path, allow_pickle=True)
    vals = np.asarray(z["values"], dtype=float)
    if vals.ndim == 3:          # (classes, samples, features) or (samples, features, classes)
        vals = vals[0] if vals.shape[0] <= 2 else vals[..., 0]
    names = [str(n) for n in z["names"]]
    imp = np.abs(vals).mean(axis=0)
    order = np.argsort(imp)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    colors = [C_ALT if names[i].startswith("prod_") else C_MAIN for i in order]
    ax.barh(np.arange(len(order)), imp[order], color=colors, alpha=0.9)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([names[i].replace("_", r"\_") if False else names[i] for i in order],
                       fontsize=7.5)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Behavior-branch attribution (business-level features in red)")
    _save(fig, "shap_behavior_summary")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Generating figures...")

    prod = _load("real_yelpchi_results_by_product.json")
    user = _load("real_yelpchi_results_by_user.json")
    audit = _load("data_integrity_audit.json")
    raw = pd.read_csv(REPO / "data" / "raw" / "yelpchi.csv")

    fig_protocol_comparison(prod, user)

    prod_tra_p = METRICS / "real_yelpchi_results_by_product_transductive.json"
    if prod_tra_p.exists():
        fig_regime(prod, json.loads(prod_tra_p.read_text()))
    else:
        logger.warning("  transductive results absent - skipping regime figure")

    dp, du = METRICS / "degree_control_by_product.json", METRICS / "degree_control_by_user.json"
    if dp.exists() and du.exists():
        fig_degree_control(json.loads(dp.read_text()), json.loads(du.read_text()))
    else:
        logger.warning("  degree control absent - skipping degree figure")
    fig_model_comparison(prod)
    fig_fold_stability(prod)
    fig_business_leakage(raw)
    fig_timestamp_contamination(raw, audit)

    score_path = METRICS / "fold_scores.npz"
    if score_path.exists():
        z = np.load(score_path)
        scores = {k: z[k] for k in z.files}
        scores["threshold"] = float(scores["threshold"])
        fig_confusion(scores)
        fig_roc(scores)
    else:
        logger.warning("  %s absent - skipping ROC/confusion figures", score_path)

    fig_shap_behavior()

    logger.info("Done.")


if __name__ == "__main__":
    main()
