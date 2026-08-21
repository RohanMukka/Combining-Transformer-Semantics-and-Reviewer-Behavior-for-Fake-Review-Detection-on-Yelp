"""
Check every number asserted in the paper against the recorded metrics.

The tables are generated (src/make_tables.py), so they cannot drift. The prose
can. This module holds each numeric claim made in the body text next to the
metrics file it comes from, and fails loudly when the two disagree -- so a
result that changes cannot quietly leave a stale sentence behind it.

Run after any experiment:  python -m src.verify_paper_numbers
Exits non-zero if any claim is unsupported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
METRICS = REPO / "results" / "metrics"

TOL = 6e-4  # claims are quoted to three decimals


def _load(name: str) -> Optional[Dict]:
    p = METRICS / name
    return json.loads(p.read_text()) if p.exists() else None


def _auc(d: Dict, model: str) -> float:
    return d["summary"][model]["auc_roc"]["mean"]


def build_claims() -> List[Tuple[str, float, Optional[float]]]:
    """(description, value claimed in the paper, value from metrics)."""
    prod = _load("real_yelpchi_results_by_product.json")
    user = _load("real_yelpchi_results_by_user.json")
    audit = _load("data_integrity_audit.json")
    gnn_p = _load("gnn_results_by_product.json")
    gnn_u = _load("gnn_results_by_user.json")
    deg_p = _load("degree_control_by_product.json")
    deg_u = _load("degree_control_by_user.json")
    prod_t = _load("real_yelpchi_results_by_product_transductive.json")
    user_t = _load("real_yelpchi_results_by_user_transductive.json")

    c: List[Tuple[str, float, Optional[float]]] = []
    add = lambda *a: c.append(a)

    # Descriptive statistics quoted in the prose. These sit in sentences rather
    # than tables, which is exactly where a wrong number survives unnoticed --
    # an earlier draft said the median business had 37 reviews, which is the
    # mean.
    corpus = REPO / "data" / "processed" / "yelpchi_real.parquet"
    if corpus.exists():
        import pandas as pd
        df = pd.read_parquet(corpus)
        per_business = df.groupby("product_id").size()
        add("reviews in corpus", 45954, len(df))
        add("fake reviews", 6677, int(df["label"].sum()))
        add("fake rate", 0.1453, float(df["label"].mean()))
        add("distinct businesses", 1224, df["product_id"].nunique())
        add("distinct reviewers", 39623, df["user_id"].nunique())
        add("median reviews per business", 10, float(per_business.median()))
        add("mean reviews per business", 38, round(float(per_business.mean())))
        add("businesses held out per fold", 244, df["product_id"].nunique() // 5)

    if audit:
        t, b = audit["timestamps"], audit.get("business_identity", {})
        add("AUC from timestamp alone", 0.965, t["auc_from_timestamp_alone"])
        add("span of released dates (days)", 111, t["span_days"])
        add("distinct seconds values", 1, t["distinct_seconds_values"])
        if b:
            add("AUC from per-business fake rate", 0.951,
                b["auc_of_product_fake_rate_as_predictor"])
            add("businesses with zero fakes", 657, b["products_with_zero_fakes"])
            add("total businesses", 1224, b["n_products"])
            add("all-fake businesses", 58, b["products_all_fake"])

    if prod and user:
        add("RF, reviewer-disjoint", 0.934, _auc(user, "Behavior + RandomForest"))
        add("RF, business-disjoint", 0.646, _auc(prod, "Behavior + RandomForest"))
        add("RF protocol gap", 0.289,
            _auc(user, "Behavior + RandomForest") - _auc(prod, "Behavior + RandomForest"))
        add("text-only protocol gap", 0.032,
            _auc(user, "Text-only MLP") - _auc(prod, "Text-only MLP"))
        add("fusion, business-disjoint", 0.730, _auc(prod, "ReviewGuard (Fusion)"))
        add("behavior-only, business-disjoint", 0.706, _auc(prod, "Behavior-only MLP"))
        add("fusion Macro-F1, business-disjoint", 0.626,
            prod["summary"]["ReviewGuard (Fusion)"]["macro_f1"]["mean"])

    if gnn_p:
        add("Relation GNN, business-disjoint", 0.816, _auc(gnn_p, "Relation GNN"))
    if gnn_u:
        add("Relation GNN, reviewer-disjoint", 0.938, _auc(gnn_u, "Relation GNN"))
    if gnn_p and gnn_u:
        add("Relation GNN protocol gap", 0.122,
            _auc(gnn_u, "Relation GNN") - _auc(gnn_p, "Relation GNN"))
    if deg_p and deg_u:
        sp, su = deg_p["summary"], deg_u["summary"]
        add("behavior MLP protocol gap", 0.183,
            su["Behavior MLP"]["auc_roc"]["mean"] - sp["Behavior MLP"]["auc_roc"]["mean"])
        add("MLP+degree protocol gap", 0.095,
            su["Behavior MLP + graph degree"]["auc_roc"]["mean"]
            - sp["Behavior MLP + graph degree"]["auc_roc"]["mean"])
        add("R-S-R degree protocol gap", 0.001,
            su["R-S-R degree alone"]["auc_roc"]["mean"]
            - sp["R-S-R degree alone"]["auc_roc"]["mean"])
    if deg_p:
        s = deg_p["summary"]
        add("behavior MLP + graph degree, bus-disj", 0.788,
            s["Behavior MLP + graph degree"]["auc_roc"]["mean"])
        add("R-S-R degree alone, bus-disj", 0.708,
            s["R-S-R degree alone"]["auc_roc"]["mean"])
        add("behavior MLP, bus-disj (degree run)", 0.706,
            s["Behavior MLP"]["auc_roc"]["mean"])
    if deg_u:
        s = deg_u["summary"]
        add("behavior MLP, rev-disj (degree run)", 0.889,
            s["Behavior MLP"]["auc_roc"]["mean"])
        add("R-S-R degree alone, rev-disj", 0.709,
            s["R-S-R degree alone"]["auc_roc"]["mean"])
        add("degree gain, rev-disj", -0.006,
            s["Behavior MLP + graph degree"]["auc_roc"]["mean"]
            - s["Behavior MLP"]["auc_roc"]["mean"])
    if deg_p:
        s = deg_p["summary"]
        add("degree gain, bus-disj", 0.082,
            s["Behavior MLP + graph degree"]["auc_roc"]["mean"]
            - s["Behavior MLP"]["auc_roc"]["mean"])
    if gnn_p and gnn_p.get("neighborhood_audit"):
        add("R-S-R neighbours in training, business-disjoint", 0.000,
            gnn_p["neighborhood_audit"]["frac_rsr_neighbors_in_training"])
    if gnn_u and gnn_u.get("neighborhood_audit"):
        add("R-S-R neighbours in training, reviewer-disjoint", 0.801,
            gnn_u["neighborhood_audit"]["frac_rsr_neighbors_in_training"])

    return c


def main() -> int:
    claims = build_claims()
    if not claims:
        print("no metrics files found; run the experiments first")
        return 1

    bad = 0
    print(f"{'claim':<48} {'paper':>8} {'actual':>8}")
    print("-" * 68)
    for desc, claimed, actual in claims:
        if actual is None:
            print(f"{desc:<48} {claimed:>8} {'MISSING':>8}")
            bad += 1
            continue
        tol = TOL if isinstance(claimed, float) else 0
        ok = abs(claimed - actual) <= tol
        bad += not ok
        flag = " " if ok else " <-- MISMATCH"
        shown = f"{actual:.4f}" if isinstance(claimed, float) else str(int(actual))
        print(f"{desc:<48} {claimed:>8} {shown:>8}{flag}")

    print("-" * 68)
    print("ALL CLAIMS SUPPORTED" if not bad else f"{bad} UNSUPPORTED CLAIM(S)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
