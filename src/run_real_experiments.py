"""
ReviewGuard evaluation on YelpCHI, run end-to-end on the real review data.

Protocol
--------
* 5-fold cross-validation, stratified by label and **grouped by business**
  (`--group-by product`, the primary protocol). Fake reviews on YelpCHI are
  concentrated by business - 657 of 1,224 businesses carry no fake reviews at
  all, and the per-business fake rate alone is a 0.95-AUC predictor - so a
  split that lets a business appear on both sides measures memorisation of
  which businesses were spam targets. `--group-by user` reproduces that weaker
  protocol for the leakage comparison reported in the paper.
* Every fitted object - TF-IDF vocabulary, SVD basis, feature scaler - is fit
  on the training portion of the fold only.
* The reviewer/business aggregate tables have their own axis, `--features`:
  - `inductive` builds them from the fold's training rows, so a business absent
    from training collapses to a training-set prior (flagged by the
    `rev_seen_in_training` / `prod_seen_in_training` indicators).
  - `transductive` builds them from every review in the corpus with labels
    untouched, so a held-out business is described by its own reviews.
  Neither regime uses a label. The strict `inductive` reading is the more
  conservative one, but it discards information a deployed detector genuinely
  holds - a new business's own reviews are observable - and understates what
  behavioural features can do. Reporting both separates leakage from
  conservatism.
* Each fold reserves a grouped validation slice of the training data. Decision
  thresholds for *every* model are tuned on that slice to maximise Macro-F1,
  then applied unchanged to the test fold. Comparing a tuned model against
  baselines left at 0.5 is what produced the earlier 0.003 recall figures.
* Reported numbers are mean +/- std across the five folds, with Wilcoxon
  signed-rank tests of ReviewGuard against each baseline.

Usage:  python -m src.run_real_experiments [--folds 5] [--group-by product|user]
                                          [--features inductive|transductive]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .behavior_featurizer import BehaviorFeaturizer, audit_product_leakage
from .losses import FocalLoss
from .text_representation import TfidfSvdEncoder

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "processed" / "yelpchi_real.parquet"
METRICS_DIR = REPO / "results" / "metrics"
SEED = 42

torch.set_num_threads(4)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def tune_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising Macro-F1, searched over score quantiles."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 199)))
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        f1 = f1_score(y, (scores >= t).astype(int), average="macro")
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def evaluate(y: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc_roc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "f1_fake": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision_fake": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }


# ─── Fusion MLP ───────────────────────────────────────────────────────────────

class FusionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden=(256, 64), dropout: float = 0.3):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 40, lr: float = 1e-3, batch_size: int = 256,
    hidden=(256, 64), dropout: float = 0.3, patience: int = 6,
) -> Tuple[FusionMLP, List[float]]:
    """Train with focal loss; keep the weights with the best validation AUC."""
    torch.manual_seed(SEED)
    model = FusionMLP(X_tr.shape[1], hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = FocalLoss(alpha=float(1 - y_tr.mean()), gamma=2.0)

    dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
        batch_size=batch_size, shuffle=True,
    )
    Xv = torch.FloatTensor(X_val)

    best_auc, best_state, since_best = -1.0, None, 0
    losses: List[float] = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        losses.append(total / len(X_tr))

        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(y_val, model(Xv).numpy())
        if auc > best_auc:
            best_auc, since_best = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, losses


def mlp_scores(model: FusionMLP, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.FloatTensor(X)).numpy()


# ─── Fold runner ──────────────────────────────────────────────────────────────

@dataclass
class FoldOutput:
    fold: int
    metrics: Dict[str, Dict[str, float]]
    losses: Dict[str, List[float]]


def grouped_val_split(
    y: np.ndarray, groups: np.ndarray, frac: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve a grouped, stratified validation slice out of a training fold."""
    splitter = StratifiedGroupKFold(
        n_splits=int(round(1 / frac)), shuffle=True, random_state=SEED
    )
    tr_idx, val_idx = next(splitter.split(np.zeros(len(y)), y, groups))
    return tr_idx, val_idx


def run_fold(
    fold: int,
    df: pd.DataFrame,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    svd_components: int,
    group_col: str,
    transductive: bool = False,
) -> FoldOutput:
    t0 = time.time()
    logger.info("─" * 70)
    logger.info("FOLD %d  train=%d  test=%d", fold, len(tr_idx), len(te_idx))

    tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
    y_tr_full, y_te = tr_df["label"].values, te_df["label"].values

    inner_tr, inner_val = grouped_val_split(y_tr_full, tr_df[group_col].values)
    fit_df, val_df = tr_df.iloc[inner_tr], tr_df.iloc[inner_val]
    y_fit, y_val = fit_df["label"].values, val_df["label"].values
    logger.info("  inner fit=%d  val=%d", len(fit_df), len(val_df))

    metrics: Dict[str, Dict[str, float]] = {}
    losses: Dict[str, List[float]] = {}

    def record(name: str, s_val: np.ndarray, s_te: np.ndarray) -> None:
        t = tune_threshold(y_val, s_val)
        metrics[name] = evaluate(y_te, s_te, t)
        logger.info(
            "  %-26s AUC %.4f  MacroF1 %.4f  Rec(fake) %.4f",
            name, metrics[name]["auc_roc"], metrics[name]["macro_f1"],
            metrics[name]["recall_fake"],
        )

    # ---- Text representations (fit on the inner training slice only) --------
    enc = TfidfSvdEncoder(n_components=svd_components, random_state=SEED)
    Xs_fit = enc.fit_sparse(fit_df["text"])
    Xs_val = enc.transform_sparse(val_df["text"])
    Xs_te = enc.transform_sparse(te_df["text"])
    logger.info("  sparse lexical matrix %s", Xs_fit.shape)

    # ---- Text baselines on the sparse lexical matrix -------------------------
    svm = LinearSVC(C=0.5, class_weight="balanced", random_state=SEED, max_iter=5000)
    svm.fit(Xs_fit, y_fit)
    record("TF-IDF + LinearSVM", svm.decision_function(Xs_val), svm.decision_function(Xs_te))

    lr_txt = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED, n_jobs=-1
    )
    lr_txt.fit(Xs_fit, y_fit)
    record(
        "TF-IDF + LogReg",
        lr_txt.predict_proba(Xs_val)[:, 1], lr_txt.predict_proba(Xs_te)[:, 1],
    )

    # ---- Dense text features for the neural branches -------------------------
    T_fit = enc.fit_transform(fit_df["text"])
    T_val = enc.transform(val_df["text"])
    T_te = enc.transform(te_df["text"])

    # ---- Behaviour features (aggregates fit on this fold's training rows) ---
    featurizer = BehaviorFeaturizer(transductive=transductive).fit(
        fit_df, corpus=df if transductive else None
    )
    B_fit_df = featurizer.transform(fit_df)
    B_val_df = featurizer.transform(val_df)
    B_te_df = featurizer.transform(te_df)
    logger.info(
        "  behaviour %d features | test rows with unseen reviewer %.1f%%, "
        "unseen business %.1f%%",
        B_fit_df.shape[1],
        100 * (1 - B_te_df["rev_seen_in_training"].mean()),
        100 * (1 - B_te_df["prod_seen_in_training"].mean()),
    )
    scaler = StandardScaler()
    B_fit = scaler.fit_transform(B_fit_df.values)
    B_val = scaler.transform(B_val_df.values)
    B_te = scaler.transform(B_te_df.values)

    # ---- Behaviour baselines -------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    rf.fit(B_fit, y_fit)
    record(
        "Behavior + RandomForest",
        rf.predict_proba(B_val)[:, 1], rf.predict_proba(B_te)[:, 1],
    )

    lr_beh = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED
    )
    lr_beh.fit(B_fit, y_fit)
    record(
        "Behavior + LogReg",
        lr_beh.predict_proba(B_val)[:, 1], lr_beh.predict_proba(B_te)[:, 1],
    )

    # ---- Single-branch MLPs (the ablations) ---------------------------------
    m_txt, l_txt = train_mlp(T_fit, y_fit, T_val, y_val)
    losses["Text-only MLP"] = l_txt
    record("Text-only MLP", mlp_scores(m_txt, T_val), mlp_scores(m_txt, T_te))

    m_beh, l_beh = train_mlp(B_fit, y_fit, B_val, y_val, hidden=(64, 32))
    losses["Behavior-only MLP"] = l_beh
    record("Behavior-only MLP", mlp_scores(m_beh, B_val), mlp_scores(m_beh, B_te))

    # ---- ReviewGuard fusion --------------------------------------------------
    F_fit = np.hstack([T_fit, B_fit])
    F_val = np.hstack([T_val, B_val])
    F_te = np.hstack([T_te, B_te])
    m_fus, l_fus = train_mlp(F_fit, y_fit, F_val, y_val)
    losses["ReviewGuard (Fusion)"] = l_fus
    record("ReviewGuard (Fusion)", mlp_scores(m_fus, F_val), mlp_scores(m_fus, F_te))

    logger.info("  fold done in %.1f min", (time.time() - t0) / 60)
    return FoldOutput(fold=fold, metrics=metrics, losses=losses)


# ─── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(folds: List[FoldOutput], reference: str) -> Dict:
    models = list(folds[0].metrics.keys())
    keys = ["auc_roc", "average_precision", "macro_f1", "f1_fake",
            "precision_fake", "recall_fake", "accuracy"]

    summary = {}
    for m in models:
        summary[m] = {
            k: {
                "mean": float(np.mean([f.metrics[m][k] for f in folds])),
                "std": float(np.std([f.metrics[m][k] for f in folds])),
                "folds": [float(f.metrics[m][k]) for f in folds],
            }
            for k in keys
        }

    tests = {}
    for m in models:
        if m == reference:
            continue
        entry = {}
        for k in ("auc_roc", "macro_f1"):
            a = [f.metrics[reference][k] for f in folds]
            b = [f.metrics[m][k] for f in folds]
            diff = np.array(a) - np.array(b)
            if np.allclose(diff, 0):
                entry[k] = {"delta": 0.0, "p_value": 1.0}
            else:
                stat, p = wilcoxon(a, b)
                entry[k] = {
                    "delta": float(np.mean(diff)),
                    "p_value": float(p),
                    "statistic": float(stat),
                }
        tests[m] = entry
    return {"summary": summary, "wilcoxon_vs_reference": tests, "reference": reference}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--svd-components", type=int, default=256)
    ap.add_argument("--group-by", choices=["product", "user"], default="product",
                    help="what CV folds are disjoint on (see module docstring)")
    ap.add_argument("--features", choices=["inductive", "transductive"],
                    default="inductive",
                    help="where reviewer/business aggregates come from")
    ap.add_argument("--quick", action="store_true", help="one fold, 64 SVD dims")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df = pd.read_parquet(DATA).reset_index(drop=True)
    group_col = "product_id" if args.group_by == "product" else "user_id"
    y, groups = df["label"].values, df[group_col].values
    logger.info("Loaded %d reviews, fake rate %.4f", len(df), y.mean())
    logger.info("Grouping folds by %s (%d distinct) | feature regime: %s",
                group_col, df[group_col].nunique(), args.features)
    leak = audit_product_leakage(df)
    logger.info("Business-identity check: %d/%d businesses have zero fakes; "
                "per-business fake rate alone scores AUC %.3f",
                leak["products_with_zero_fakes"], leak["n_products"],
                leak["auc_of_product_fake_rate_as_predictor"])

    n_folds = 1 if args.quick else args.folds
    svd_dim = 64 if args.quick else args.svd_components

    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    outs: List[FoldOutput] = []
    for i, (tr, te) in enumerate(cv.split(df, y, groups), start=1):
        outs.append(run_fold(i, df, tr, te, svd_dim, group_col,
                             transductive=(args.features == "transductive")))
        if len(outs) >= n_folds:
            break

    result = aggregate(outs, reference="ReviewGuard (Fusion)")
    result["config"] = {
        "folds": n_folds, "svd_components": svd_dim, "seed": SEED,
        "behavior_features": BehaviorFeaturizer().feature_names,
        "feature_regime": args.features,
        "text_encoder": "TF-IDF(word 1-2gram + char 3-5gram) -> SVD + stylometric",
        "split": f"StratifiedGroupKFold grouped by {group_col}",
        "thresholds": "tuned per-model on a grouped validation slice",
    }
    result["product_leakage_audit"] = leak
    result["per_fold"] = [
        {"fold": o.fold, "metrics": o.metrics} for o in outs
    ]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.features == "inductive" else f"_{args.features}"
    out_path = METRICS_DIR / f"real_yelpchi_results_by_{args.group_by}{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))
    np.save(METRICS_DIR / f"training_losses_by_{args.group_by}{suffix}.npy",
            np.array([o.losses for o in outs], dtype=object), allow_pickle=True)

    logger.info("\n" + "=" * 78)
    logger.info("%-26s %-14s %-14s %-14s", "Model", "AUC-ROC", "Macro-F1", "Rec(fake)")
    logger.info("=" * 78)
    for m, s in result["summary"].items():
        logger.info(
            "%-26s %.3f ± %.3f   %.3f ± %.3f   %.3f ± %.3f",
            m, s["auc_roc"]["mean"], s["auc_roc"]["std"],
            s["macro_f1"]["mean"], s["macro_f1"]["std"],
            s["recall_fake"]["mean"], s["recall_fake"]["std"],
        )
    logger.info("=" * 78)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
