"""
Relation-aware GNN over the YelpCHI review graph, evaluated under both protocols.

Motivation
----------
The rest of this study shows that a model's drop between reviewer-disjoint and
business-disjoint evaluation scales with how directly it can reach business
identity. Graph methods are the extreme case of that argument: the review graph
wires each review to other reviews of the *same business*, so business identity
is not merely available as a feature, it is the edge set. This module measures
the effect rather than leaving it as an inference.

Graph construction
------------------
We build the two relations of the standard YelpCHI graph that survive our data
integrity audit:

  R-U-R  reviews written by the same reviewer          (13,264 edges)
  R-S-R  reviews of the same business with the same star rating (1,939,774)

The third relation of the canonical graph, R-T-R (same business, same month),
is *not* built: it is derived from the timestamps that our audit shows are
label-contaminated (AUC 0.965 from the timestamp alone). Including it would
import that leakage into the graph.

Model
-----
A two-layer relation-aware GNN in the CARE-GNN family: mean aggregation within
each relation, a separate weight matrix per relation plus a self weight, and
the same focal loss and threshold tuning used by every other model in this
study. This is a re-implementation in that family, not CARE-GNN itself -- we do
not reproduce its reinforcement-learned neighbour filtering.

Message passing is transductive, as in that literature: the whole graph is
visible and only *labels* are split. Nothing about an evaluation fold's labels
reaches training.

Usage:  python -m src.graph_gnn [--folds 5] [--group-by product|user]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from .behavior_featurizer import BehaviorFeaturizer
from .losses import FocalLoss
from .run_real_experiments import (
    DATA, METRICS_DIR, SEED, aggregate, evaluate, grouped_val_split, tune_threshold,
)

logger = logging.getLogger(__name__)

torch.set_num_threads(4)
RELATIONS = ("R-U-R", "R-S-R")


# ─── Graph construction ───────────────────────────────────────────────────────

def _pairs_within_groups(codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """All ordered (i, j), i != j, sharing a group code, without materialising
    a Python loop per pair. Groups are small enough here that the blockwise
    expansion below stays well inside memory."""
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    rows, cols = [], []
    for block in np.split(order, boundaries):
        n = len(block)
        if n < 2:
            continue
        a = np.repeat(block, n)
        b = np.tile(block, n)
        keep = a != b
        rows.append(a[keep])
        cols.append(b[keep])
    if not rows:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(rows), np.concatenate(cols)


def build_relations(df: pd.DataFrame) -> Dict[str, sparse.csr_matrix]:
    """Row-normalised adjacency per relation (mean aggregation over neighbours)."""
    n = len(df)
    out: Dict[str, sparse.csr_matrix] = {}

    specs = {
        "R-U-R": df["user_id"].astype("category").cat.codes.to_numpy(),
        "R-S-R": (
            df["product_id"].astype(str) + "_" + df["rating"].astype(str)
        ).astype("category").cat.codes.to_numpy(),
    }
    for name, codes in specs.items():
        rows, cols = _pairs_within_groups(codes)
        A = sparse.coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
        ).tocsr()
        deg = np.asarray(A.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        out[name] = sparse.diags(1.0 / deg).dot(A).tocsr().astype(np.float32)
        logger.info("  %s: %d edges, mean degree %.1f",
                    name, A.nnz, A.nnz / n)
    return out


def to_torch_sparse(A: sparse.csr_matrix) -> Tensor:
    A = A.tocoo()
    idx = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
    val = torch.from_numpy(A.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, A.shape).coalesce()


# ─── Model ────────────────────────────────────────────────────────────────────

class RelationLayer(nn.Module):
    """h' = W_self h + sum_r W_r (A_r h)."""

    def __init__(self, in_dim: int, out_dim: int, n_relations: int) -> None:
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.rel_lins = nn.ModuleList(
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_relations)
        )

    def forward(self, h: Tensor, adjs: List[Tensor]) -> Tensor:
        out = self.self_lin(h)
        for lin, A in zip(self.rel_lins, adjs):
            out = out + lin(torch.sparse.mm(A, h))
        return out


class RelationGNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64,
                 n_relations: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.l1 = RelationLayer(in_dim, hidden, n_relations)
        self.l2 = RelationLayer(hidden, hidden, n_relations)
        self.out = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: Tensor, adjs: List[Tensor]) -> Tensor:
        h = self.drop(self.act(self.l1(x, adjs)))
        h = self.drop(self.act(self.l2(h, adjs)))
        return self.out(h).squeeze(-1)


def train_gnn(
    X: np.ndarray, y: np.ndarray, adjs: List[Tensor],
    fit_idx: np.ndarray, val_idx: np.ndarray,
    epochs: int = 300, lr: float = 5e-3, weight_decay: float = 5e-4,
    patience: int = 40,
) -> RelationGNN:
    """Full-batch transductive training; loss uses training-fold labels only."""
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(SEED)
    model = RelationGNN(X.shape[1], n_relations=len(adjs))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = FocalLoss(alpha=float(1 - y[fit_idx].mean()), gamma=2.0)

    xt = torch.FloatTensor(X)
    yt = torch.FloatTensor(y)
    fit_t = torch.from_numpy(fit_idx)

    best_auc, best_state, since = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(xt, adjs)
        loss = crit(logits[fit_t], yt[fit_t])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            scores = model(xt, adjs).numpy()
        auc = roc_auc_score(y[val_idx], scores[val_idx])
        if auc > best_auc:
            best_auc, since = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ─── Experiment ───────────────────────────────────────────────────────────────

def audit_neighborhoods(df: pd.DataFrame, group_col: str, n_folds: int,
                        sample: int = 5000) -> Dict[str, float]:
    """Where do a test review's same-business neighbours live?

    This is the mechanism behind the protocol gap for graph models. Under a
    reviewer-disjoint split most of a held-out review's R-S-R neighbourhood is
    supervised training data from the same business; under a business-disjoint
    split, none of it is.
    """
    y = df["label"].values
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    tr_idx, te_idx = next(cv.split(df, y, df[group_col].values))
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[tr_idx] = True

    key = df["product_id"].astype(str) + "_" + df["rating"].astype(str)
    groups = df.groupby(key).indices
    keys = key.to_numpy()

    rng = np.random.RandomState(SEED)
    picks = rng.choice(te_idx, min(sample, len(te_idx)), replace=False)
    in_train = total = 0
    for i in picks:
        nb = groups[keys[i]]
        nb = nb[nb != i]
        if nb.size == 0:
            continue
        in_train += int(train_mask[nb].sum())
        total += nb.size
    frac = in_train / total if total else 0.0
    logger.info("  R-S-R neighbours of held-out reviews that sit in training: "
                "%.1f%%", 100 * frac)
    return {
        "grouped_by": group_col,
        "frac_rsr_neighbors_in_training": float(frac),
        "neighbor_pairs_sampled": int(total),
    }


def run(df: pd.DataFrame, group_col: str, n_folds: int) -> Dict:
    y = df["label"].values
    groups = df[group_col].values

    logger.info("Building relations over %d reviews", len(df))
    adj_np = build_relations(df)
    adjs = [to_torch_sparse(adj_np[r]) for r in RELATIONS]

    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    per_fold = []
    for fold, (tr_idx, te_idx) in enumerate(cv.split(df, y, groups), start=1):
        t0 = time.time()
        tr_df = df.iloc[tr_idx]
        inner_tr, inner_val = grouped_val_split(
            tr_df["label"].values, tr_df[group_col].values
        )
        fit_idx = tr_idx[inner_tr]
        val_idx = tr_idx[inner_val]

        # Node features: fitted on this fold's training rows, applied to all
        # nodes. Message passing needs a feature vector for every node, but the
        # aggregate tables behind those features never see held-out rows.
        featurizer = BehaviorFeaturizer().fit(df.iloc[fit_idx])
        scaler = StandardScaler().fit(featurizer.transform(df.iloc[fit_idx]).values)
        X = scaler.transform(featurizer.transform(df).values).astype(np.float32)

        model = train_gnn(X, y, adjs, fit_idx, val_idx)
        with torch.no_grad():
            scores = model(torch.FloatTensor(X), adjs).numpy()

        thr = tune_threshold(y[val_idx], scores[val_idx])
        m = evaluate(y[te_idx], scores[te_idx], thr)
        per_fold.append({"Relation GNN": m})
        logger.info(
            "  fold %d: AUC %.4f  MacroF1 %.4f  Rec(fake) %.4f  [%.1f min]",
            fold, m["auc_roc"], m["macro_f1"], m["recall_fake"],
            (time.time() - t0) / 60,
        )
    return per_fold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--group-by", choices=["product", "user"], default="product")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df = pd.read_parquet(DATA).reset_index(drop=True)
    group_col = "product_id" if args.group_by == "product" else "user_id"
    logger.info("Protocol: folds disjoint on %s", group_col)

    neighborhood = audit_neighborhoods(df, group_col, args.folds)
    per_fold = run(df, group_col, args.folds)

    # Reuse the aggregation used for every other model in the study.
    class _F:
        def __init__(self, i, m): self.fold, self.metrics = i, m
    folds = [_F(i + 1, m) for i, m in enumerate(per_fold)]
    result = aggregate(folds, reference="Relation GNN")
    result.pop("wilcoxon_vs_reference", None)
    result["config"] = {
        "folds": args.folds,
        "split": f"StratifiedGroupKFold grouped by {group_col}",
        "relations": list(RELATIONS),
        "excluded_relation": "R-T-R (built from the contaminated timestamps)",
        "model": "2-layer relation-aware GNN, mean aggregation, focal loss",
        "message_passing": "transductive; only labels are split",
        "seed": SEED,
    }
    result["per_fold"] = [{"fold": f.fold, "metrics": f.metrics} for f in folds]
    result["neighborhood_audit"] = neighborhood

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / f"gnn_results_by_{args.group_by}.json"
    out.write_text(json.dumps(result, indent=2))

    s = result["summary"]["Relation GNN"]
    logger.info("=" * 60)
    logger.info("Relation GNN (%s-disjoint): AUC %.3f ± %.3f | MacroF1 %.3f ± %.3f",
                args.group_by, s["auc_roc"]["mean"], s["auc_roc"]["std"],
                s["macro_f1"]["mean"], s["macro_f1"]["std"])
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
