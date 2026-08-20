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

Reviewer and product aggregates are computed leave-one-out, so a review never
contributes to the statistics used to classify it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
RAW_CSV = REPO / "data" / "raw" / "yelpchi.csv"
OUT_DIR = REPO / "data" / "processed"

# Columns of the raw file that survive the integrity audit.
TRUSTED_COLUMNS = ["review_id", "user_id", "product_id", "rating", "text", "label"]

BEHAVIOR_FEATURES = [
    "rating",
    "rating_extremity",
    "rating_dev_from_product",
    "rating_dev_signed",
    "rev_review_count",
    "rev_is_singleton",
    "rev_mean_rating_loo",
    "rev_std_rating",
    "rev_extreme_ratio_loo",
    "rev_positive_ratio_loo",
    "rev_mean_dev_loo",
    "rev_n_products",
    "rev_max_reviews_one_product",
    "rev_mean_review_len",
    "prod_review_count",
    "prod_mean_rating_loo",
    "prod_std_rating",
    "review_word_count",
]


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


# ─── Feature construction ─────────────────────────────────────────────────────

def _loo_mean(values: pd.Series, group: pd.Series, fallback: float) -> pd.Series:
    """Leave-one-out group mean: (sum - x) / (n - 1), fallback when n == 1."""
    g = values.groupby(group)
    s, n = g.transform("sum"), g.transform("size")
    out = (s - values) / (n - 1)
    return out.where(n > 1, fallback)


def _loo_ratio(mask: pd.Series, group: pd.Series, fallback: float) -> pd.Series:
    """Leave-one-out group mean of a boolean indicator."""
    return _loo_mean(mask.astype(float), group, fallback)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute reviewer / product behavioral features. No timestamps used."""
    f = pd.DataFrame(index=df.index)

    rating = df["rating"].astype(float)
    user, prod = df["user_id"], df["product_id"]
    global_mean = float(rating.mean())

    word_count = df["text"].astype(str).str.split().str.len().astype(float)

    # Product context, leave-one-out so the review never sets its own baseline.
    prod_mean_loo = _loo_mean(rating, prod, global_mean)
    dev_signed = rating - prod_mean_loo

    f["rating"] = rating
    f["rating_extremity"] = (rating - 3.0).abs()
    f["rating_dev_from_product"] = dev_signed.abs()
    f["rating_dev_signed"] = dev_signed

    # Reviewer activity.
    rev_count = user.map(user.value_counts()).astype(float)
    f["rev_review_count"] = rev_count
    f["rev_is_singleton"] = (rev_count == 1).astype(float)
    f["rev_mean_rating_loo"] = _loo_mean(rating, user, global_mean)
    f["rev_std_rating"] = rating.groupby(user).transform("std").fillna(0.0)
    f["rev_extreme_ratio_loo"] = _loo_ratio(
        rating.isin([1.0, 5.0]), user, float(rating.isin([1.0, 5.0]).mean())
    )
    f["rev_positive_ratio_loo"] = _loo_ratio(
        rating >= 4.0, user, float((rating >= 4.0).mean())
    )
    f["rev_mean_dev_loo"] = _loo_mean(
        dev_signed.abs(), user, float(dev_signed.abs().mean())
    )
    f["rev_n_products"] = user.map(
        df.groupby("user_id")["product_id"].nunique()
    ).astype(float)
    f["rev_max_reviews_one_product"] = user.map(
        df.groupby(["user_id", "product_id"]).size().groupby("user_id").max()
    ).astype(float)
    f["rev_mean_review_len"] = _loo_mean(word_count, user, float(word_count.mean()))

    # Product popularity.
    f["prod_review_count"] = prod.map(prod.value_counts()).astype(float)
    f["prod_mean_rating_loo"] = prod_mean_loo
    f["prod_std_rating"] = rating.groupby(prod).transform("std").fillna(0.0)

    f["review_word_count"] = word_count

    assert list(f.columns) == BEHAVIOR_FEATURES, (
        f"column drift: {set(f.columns) ^ set(BEHAVIOR_FEATURES)}"
    )
    assert f.notna().all().all(), "NaNs in behavior features"
    return f


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

    feats = build_features(df)
    # `rating` lives in the feature block; keep identity columns only here.
    identity = [c for c in TRUSTED_COLUMNS if c not in feats.columns]
    out = pd.concat([df[identity], feats], axis=1)
    dest = OUT_DIR / "yelpchi_real.parquet"
    out.to_parquet(dest, index=False)
    logger.info("Wrote %s  shape=%s", dest, out.shape)

    corr = feats.corrwith(df["label"]).sort_values(key=np.abs, ascending=False)
    logger.info("\nBehavior feature correlation with label:\n%s", corr.round(4))


if __name__ == "__main__":
    main()
