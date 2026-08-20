"""
Build an honest, leakage-audited dataset for ReviewGuard from the raw YelpCHI CSV.

This module replaces the earlier pipeline, which trained on `.mat` metadata
features split arbitrarily in half and labelled the two halves "text" and
"behavior". Here every feature is computed from data we can point at:

  data/raw/yelpchi.csv  ->  review_id, user_id, product_id, rating, text, label

Two columns of the raw file are deliberately NOT used:

  * `date`   - the released timestamps are contaminated. A RandomForest on the
               raw timestamp alone scores AUC 0.965, and the monthly fake rate
               swings from 49.8% (2010-01) to 2.6% (2010-03) with a perfectly
               uniform hour-of-day distribution and a constant seconds field.
               Any burstiness / account-age feature derived from them would
               inherit that leakage. See `audit_timestamps()`.
  * data/raw/behavioral_features.csv - reproduces none of its own column
               definitions (corr ~0.00 against the natural reconstruction of
               each column from the review table) and correlates ~0.00 with the
               label. It is noise. See `audit_supplied_behavior_file()`.

This module produces the audited review table only. Behavioural features are
built per cross-validation fold by `src/behavior_featurizer.py`, because
reviewer and business aggregates computed over the whole corpus leak the label
across the split (see `BehaviorFeaturizer` and `audit_product_leakage`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
RAW_CSV = REPO / "data" / "raw" / "yelpchi.csv"
OUT_DIR = REPO / "data" / "processed"

# Columns of the raw file that survive the integrity audit.
TRUSTED_COLUMNS = ["review_id", "user_id", "product_id", "rating", "text", "label"]

# ─── Integrity audits ─────────────────────────────────────────────────────────

def audit_timestamps(df: pd.DataFrame) -> Dict[str, float]:
    """Quantify the leakage in the released `date` column.

    Returns diagnostics used in the paper's data-integrity section. This is a
    measurement, not a fix: the conclusion is that `date` is unusable.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    d = pd.to_datetime(df["date"])
    ts = (d.astype("int64") // 10**9).values.reshape(-1, 1)
    y = df["label"].values

    ts_tr, ts_te, y_tr, y_te = train_test_split(
        ts, y, test_size=0.2, stratify=y, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(ts_tr, y_tr)
    auc = roc_auc_score(y_te, rf.predict_proba(ts_te)[:, 1])

    monthly = df.assign(_m=d.dt.to_period("M")).groupby("_m")["label"].mean()

    return {
        "auc_from_timestamp_alone": float(auc),
        "monthly_fake_rate_min": float(monthly.min()),
        "monthly_fake_rate_max": float(monthly.max()),
        "distinct_seconds_values": int(d.dt.second.nunique()),
        "hour_of_day_chi2_uniform_ratio": float(
            d.dt.hour.value_counts().max() / d.dt.hour.value_counts().min()
        ),
        "span_days": int((d.max() - d.min()).days),
        "verdict": "contaminated - excluded from all features",
    }


def audit_supplied_behavior_file(df: pd.DataFrame) -> Dict[str, float]:
    """Check data/raw/behavioral_features.csv against the review table."""
    path = REPO / "data" / "raw" / "behavioral_features.csv"
    if not path.exists():
        return {"status": "absent"}
    b = pd.read_csv(path)
    if len(b) != len(df):
        return {"status": f"row count mismatch {len(b)} vs {len(df)}"}

    natural = {
        "avg_star_rating": df.groupby("user_id")["rating"].transform("mean"),
        "review_count": df.groupby("user_id")["review_id"].transform("size"),
        "rating_deviation": (
            df["rating"] - df.groupby("product_id")["rating"].transform("mean")
        ).abs(),
    }
    out = {
        f"corr_{k}_vs_natural": float(np.corrcoef(v, b[k])[0, 1])
        for k, v in natural.items()
    }
    out.update(
        {
            f"corr_{c}_vs_label": float(np.corrcoef(b[c], df["label"])[0, 1])
            for c in b.columns
        }
    )
    out["verdict"] = "uninformative - not used"
    return out


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    df["text"] = df["text"].astype(str)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    logger.info("Loaded %d reviews, fake rate %.4f", len(df), df["label"].mean())

    audit = {
        "timestamps": audit_timestamps(df),
        "supplied_behavior_file": audit_supplied_behavior_file(df),
    }
    audit_path = REPO / "results" / "metrics" / "data_integrity_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2))
    logger.info("Integrity audit -> %s", audit_path)
    logger.info("  AUC from timestamp alone: %.4f",
                audit["timestamps"]["auc_from_timestamp_alone"])

    from .behavior_featurizer import audit_product_leakage

    audit["business_identity"] = audit_product_leakage(df)
    audit_path.write_text(json.dumps(audit, indent=2))
    logger.info("  AUC from per-business fake rate: %.4f",
                audit["business_identity"]["auc_of_product_fake_rate_as_predictor"])

    out = df[TRUSTED_COLUMNS]
    dest = OUT_DIR / "yelpchi_real.parquet"
    out.to_parquet(dest, index=False)
    logger.info("Wrote %s  shape=%s", dest, out.shape)


if __name__ == "__main__":
    main()
