"""
Re-run a single fold and save the held-out scores, for the ROC and confusion
matrix figures. Uses exactly the components and the fold definition of
`src.run_real_experiments`, so the curves match the reported table.

With --all-folds it runs every fold and pools the out-of-fold predictions, so
the curves describe the same five-fold experiment the tables report rather than
one arbitrary fold.

Usage:  python -m src.dump_fold_scores [--all-folds] [--group-by product|user]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .behavior_featurizer import BehaviorFeaturizer
from .run_real_experiments import (
    DATA, METRICS_DIR, SEED, grouped_val_split, mlp_scores, train_mlp, tune_threshold
)
from .text_representation import TfidfSvdEncoder

logger = logging.getLogger(__name__)


def run_one_fold(df, tr_idx, te_idx, group_col, svd_components, want_shap=False):
    """Train the three neural models on one fold; return held-out scores."""
    tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
    y_te = te_df["label"].values

    inner_tr, inner_val = grouped_val_split(
        tr_df["label"].values, tr_df[group_col].values
    )
    fit_df, val_df = tr_df.iloc[inner_tr], tr_df.iloc[inner_val]
    y_fit, y_val = fit_df["label"].values, val_df["label"].values
    logger.info("  fit=%d val=%d test=%d", len(fit_df), len(val_df), len(te_df))

    enc = TfidfSvdEncoder(n_components=svd_components, random_state=SEED)
    T_fit = enc.fit_transform(fit_df["text"])
    T_val, T_te = enc.transform(val_df["text"]), enc.transform(te_df["text"])

    feat = BehaviorFeaturizer().fit(fit_df)
    scaler = StandardScaler()
    B_fit = scaler.fit_transform(feat.transform(fit_df).values)
    B_val = scaler.transform(feat.transform(val_df).values)
    B_te = scaler.transform(feat.transform(te_df).values)

    out = {"y_true": y_te}
    m_txt, _ = train_mlp(T_fit, y_fit, T_val, y_val)
    out["text"] = mlp_scores(m_txt, T_te)

    m_beh, _ = train_mlp(B_fit, y_fit, B_val, y_val, hidden=(64, 32))
    out["behavior"] = mlp_scores(m_beh, B_te)

    F_fit, F_val, F_te = (np.hstack([t, b]) for t, b in
                          ((T_fit, B_fit), (T_val, B_val), (T_te, B_te)))
    m_fus, _ = train_mlp(F_fit, y_fit, F_val, y_val)
    out["fusion"] = mlp_scores(m_fus, F_te)
    out["threshold"] = tune_threshold(y_val, mlp_scores(m_fus, F_val))

    if want_shap:
        _dump_shap(m_beh, B_fit, B_te, feat)
    return out


def _dump_shap(model, B_fit, B_te, feat) -> None:
    """Kernel-SHAP attribution for the behaviour branch."""
    try:
        import shap
        rs = np.random.RandomState(SEED)
        bg = B_fit[rs.choice(len(B_fit), 100, replace=False)]
        sample = B_te[rs.choice(len(B_te), 200, replace=False)]
        expl = shap.KernelExplainer(lambda x: mlp_scores(model, x), bg)
        vals = expl.shap_values(sample, nsamples=100, silent=True)
        np.savez(METRICS_DIR / "shap_behavior.npz", values=np.asarray(vals),
                 data=sample, names=np.array(feat.feature_names))
        logger.info("wrote SHAP values for the behaviour branch")
    except Exception as exc:  # SHAP is optional for the figures
        logger.warning("SHAP step skipped: %s", exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--all-folds", action="store_true",
                    help="run every fold and pool out-of-fold predictions")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--group-by", choices=["product", "user"], default="product")
    ap.add_argument("--svd-components", type=int, default=256)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    df = pd.read_parquet(DATA).reset_index(drop=True)
    group_col = "product_id" if args.group_by == "product" else "user_id"
    y, groups = df["label"].values, df[group_col].values

    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    splits = list(cv.split(df, y, groups))

    if args.all_folds:
        pooled = {k: [] for k in ("y_true", "text", "behavior", "fusion")}
        thresholds = []
        for i, (tr_i, te_i) in enumerate(splits, start=1):
            logger.info("=== fold %d/%d ===", i, len(splits))
            out = run_one_fold(df, tr_i, te_i, group_col, args.svd_components,
                               want_shap=False)
            for k in pooled:
                pooled[k].append(out[k])
            thresholds.append(float(out["threshold"]))
        result = {k: np.concatenate(v) for k, v in pooled.items()}
        # Pooled scores come from different models, so re-tune one threshold
        # against the pooled fusion scores for the confusion matrix.
        result["threshold"] = np.array(
            tune_threshold(result["y_true"], result["fusion"])
        )
        result["per_fold_thresholds"] = np.array(thresholds)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        dest = METRICS_DIR / "fold_scores.npz"
        np.savez(dest, **result)
        logger.info("wrote %s (pooled over %d folds, n=%d)",
                    dest, len(splits), len(result["y_true"]))
        return

    tr_idx, te_idx = splits[args.fold - 1]
    logger.info("=== fold %d ===", args.fold)
    out = run_one_fold(df, tr_idx, te_idx, group_col, args.svd_components,
                       want_shap=True)
    out["threshold"] = np.array(out["threshold"])

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    dest = METRICS_DIR / "fold_scores.npz"
    np.savez(dest, **out)
    logger.info("wrote %s", dest)


if __name__ == "__main__":
    main()
