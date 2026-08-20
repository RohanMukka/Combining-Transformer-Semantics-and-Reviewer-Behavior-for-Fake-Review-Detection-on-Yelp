"""
Re-run a single fold and save the held-out scores, for the ROC and confusion
matrix figures. Uses exactly the components and the fold definition of
`src.run_real_experiments`, so the curves match the reported table.

Usage:  python -m src.dump_fold_scores [--fold 1] [--group-by product|user]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
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
    tr_idx, te_idx = list(cv.split(df, y, groups))[args.fold - 1]
    tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
    y_te = te_df["label"].values

    inner_tr, inner_val = grouped_val_split(tr_df["label"].values, tr_df[group_col].values)
    fit_df, val_df = tr_df.iloc[inner_tr], tr_df.iloc[inner_val]
    y_fit, y_val = fit_df["label"].values, val_df["label"].values
    logger.info("fold %d  fit=%d val=%d test=%d", args.fold, len(fit_df), len(val_df), len(te_df))

    enc = TfidfSvdEncoder(n_components=args.svd_components, random_state=SEED)
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
    out["threshold"] = np.array(tune_threshold(y_val, mlp_scores(m_fus, F_val)))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    dest = METRICS_DIR / "fold_scores.npz"
    np.savez(dest, **out)
    logger.info("wrote %s", dest)

    # Behaviour-branch attribution, for the explainability figure.
    try:
        import shap
        bg = B_fit[np.random.RandomState(SEED).choice(len(B_fit), 100, replace=False)]
        sample = B_te[np.random.RandomState(SEED).choice(len(B_te), 200, replace=False)]
        expl = shap.KernelExplainer(lambda x: mlp_scores(m_beh, x), bg)
        vals = expl.shap_values(sample, nsamples=100, silent=True)
        np.savez(METRICS_DIR / "shap_behavior.npz", values=np.asarray(vals),
                 data=sample, names=np.array(feat.feature_names))
        logger.info("wrote SHAP values for the behaviour branch")
    except Exception as exc:  # SHAP is optional for the figures
        logger.warning("SHAP step skipped: %s", exc)


if __name__ == "__main__":
    main()
