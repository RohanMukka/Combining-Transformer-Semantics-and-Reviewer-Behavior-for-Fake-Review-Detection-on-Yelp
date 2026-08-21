"""
Control for the structural information a transductive graph hands a GNN.

Under a business-disjoint split, no business spans the split and every
per-business *feature* the model sees is a training-set prior -- for an unseen
business, `prod_review_count` is a single constant, worth AUC 0.500. But the
graph is built over the whole corpus, so a node's R-S-R degree still counts how
many reviews that business received at that star rating. On a business-disjoint
fold that degree alone scores AUC 0.696.

So business-disjoint splitting removes the shortcut from the feature matrix but
not from the edge set, and a transductive GNN is not comparable to the
feature-based models in Table III. This experiment quantifies the difference by
giving the same MLP the two full-graph degrees and nothing else new. If that
closes most of the gap to the GNN, the GNN's advantage is structural
bookkeeping rather than relational learning.

Usage:  python -m src.degree_control [--folds 5] [--group-by product|user]
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .behavior_featurizer import BehaviorFeaturizer
from .run_real_experiments import (
    DATA, METRICS_DIR, SEED, aggregate, evaluate, grouped_val_split,
    mlp_scores, train_mlp, tune_threshold,
)

logger = logging.getLogger(__name__)

DEGREE_FEATURES = ["rsr_degree", "rur_degree"]


def graph_degrees(df: pd.DataFrame) -> pd.DataFrame:
    """Degrees read off the full graph, exactly what the GNN's edges encode."""
    key = df["product_id"].astype(str) + "_" + df["rating"].astype(str)
    return pd.DataFrame({
        "rsr_degree": (key.map(key.value_counts()) - 1).astype(float).values,
        "rur_degree": (df["user_id"].map(df["user_id"].value_counts()) - 1)
                      .astype(float).values,
    }, index=df.index)


def run(df: pd.DataFrame, group_col: str, n_folds: int) -> Dict:
    y = df["label"].values
    deg = graph_degrees(df)

    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    per_fold: List[Dict] = []
    for fold, (tr_idx, te_idx) in enumerate(cv.split(df, y, df[group_col].values), 1):
        tr_df = df.iloc[tr_idx]
        inner_tr, inner_val = grouped_val_split(
            tr_df["label"].values, tr_df[group_col].values
        )
        fit_idx, val_idx = tr_idx[inner_tr], tr_idx[inner_val]

        featurizer = BehaviorFeaturizer().fit(df.iloc[fit_idx])
        base = featurizer.transform(df)
        with_deg = pd.concat([base, deg], axis=1)

        metrics = {}
        for name, frame in (("Behavior MLP", base),
                            ("Behavior MLP + graph degree", with_deg)):
            scaler = StandardScaler().fit(frame.iloc[fit_idx].values)
            X = scaler.transform(frame.values).astype(np.float32)
            model, _ = train_mlp(X[fit_idx], y[fit_idx], X[val_idx], y[val_idx],
                                 hidden=(64, 32))
            s_val, s_te = mlp_scores(model, X[val_idx]), mlp_scores(model, X[te_idx])
            metrics[name] = evaluate(y[te_idx], s_te,
                                     tune_threshold(y[val_idx], s_val))

        # The degree signal on its own, for reference.
        d_auc = roc_auc_score(y[te_idx], deg["rsr_degree"].values[te_idx])
        metrics["R-S-R degree alone"] = {
            "auc_roc": float(max(d_auc, 1 - d_auc)),
            "average_precision": float("nan"), "macro_f1": float("nan"),
            "f1_fake": float("nan"), "precision_fake": float("nan"),
            "recall_fake": float("nan"), "accuracy": float("nan"),
        }
        per_fold.append(metrics)
        logger.info(
            "  fold %d: behavior %.4f | +degree %.4f | degree alone %.4f",
            fold, metrics["Behavior MLP"]["auc_roc"],
            metrics["Behavior MLP + graph degree"]["auc_roc"],
            metrics["R-S-R degree alone"]["auc_roc"],
        )
    return per_fold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--group-by", choices=["product", "user"], default="product")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    df = pd.read_parquet(DATA).reset_index(drop=True)
    group_col = "product_id" if args.group_by == "product" else "user_id"
    logger.info("Degree control, folds disjoint on %s", group_col)

    per_fold = run(df, group_col, args.folds)

    class _F:
        def __init__(self, i, m): self.fold, self.metrics = i, m
    folds = [_F(i + 1, m) for i, m in enumerate(per_fold)]
    result = aggregate(folds, reference="Behavior MLP + graph degree")
    result.pop("wilcoxon_vs_reference", None)
    result["config"] = {
        "folds": args.folds,
        "split": f"StratifiedGroupKFold grouped by {group_col}",
        "degree_features": DEGREE_FEATURES,
        "note": ("Degrees are read off the full graph, matching what a "
                 "transductive GNN's edge set encodes. All other features "
                 "remain fitted on training rows only."),
        "seed": SEED,
    }
    result["per_fold"] = [{"fold": f.fold, "metrics": f.metrics} for f in folds]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / f"degree_control_by_{args.group_by}.json"
    out.write_text(json.dumps(result, indent=2))

    s = result["summary"]
    logger.info("=" * 64)
    for m in ("Behavior MLP", "Behavior MLP + graph degree", "R-S-R degree alone"):
        logger.info("%-30s AUC %.3f ± %.3f", m,
                    s[m]["auc_roc"]["mean"], s[m]["auc_roc"]["std"])
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
