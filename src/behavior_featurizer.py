"""
Fold-aware behavioural featuriser.

Reviewer and product statistics are *fitted* on the training rows of a fold and
then applied to held-out rows. Nothing about the evaluation fold - not its
labels, not its aggregate counts - reaches the training representation.

Why this matters on YelpCHI: fake reviews are concentrated by business. 657 of
the 1,224 businesses carry zero fake reviews, and the per-business fake rate is
by itself a 0.95-AUC predictor of the label. Product-level aggregates computed
over the whole corpus therefore smuggle business identity - and with it the
label - into the feature vector. `audit_product_leakage()` measures the effect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REVIEW_LEVEL = [
    "rating",
    "rating_extremity",
    "review_word_count",
]
REVIEWER_LEVEL = [
    "rev_review_count",
    "rev_is_singleton",
    "rev_mean_rating",
    "rev_std_rating",
    "rev_extreme_ratio",
    "rev_positive_ratio",
    "rev_n_products",
    "rev_max_reviews_one_product",
    "rev_mean_review_len",
    "rev_seen_in_training",
]
PRODUCT_LEVEL = [
    "prod_review_count",
    "prod_mean_rating",
    "prod_std_rating",
    "prod_seen_in_training",
    "rating_dev_from_product",
    "rating_dev_signed",
]
BEHAVIOR_FEATURES: List[str] = REVIEW_LEVEL + REVIEWER_LEVEL + PRODUCT_LEVEL


@dataclass
class BehaviorFeaturizer:
    """Fit reviewer/product statistics on training rows; apply them anywhere.

    include_product controls whether product-level columns are emitted at all,
    so the same code can produce the leakage-free and leakage-exposed variants.
    """

    include_product: bool = True
    _user: Optional[pd.DataFrame] = field(default=None, repr=False)
    _prod: Optional[pd.DataFrame] = field(default=None, repr=False)
    _prior: Dict[str, float] = field(default_factory=dict, repr=False)

    @property
    def feature_names(self) -> List[str]:
        names = REVIEW_LEVEL + REVIEWER_LEVEL
        if self.include_product:
            names = names + PRODUCT_LEVEL
        return names

    def fit(self, train: pd.DataFrame) -> "BehaviorFeaturizer":
        t = train.copy()
        t["_wc"] = t["text"].astype(str).str.split().str.len().astype(float)
        r = t["rating"].astype(float)

        self._prior = {
            "rating_mean": float(r.mean()),
            "rating_std": float(r.std()),
            "word_count": float(t["_wc"].mean()),
            "extreme_ratio": float(r.isin([1.0, 5.0]).mean()),
            "positive_ratio": float((r >= 4.0).mean()),
            "rev_review_count": 1.0,
            "prod_review_count": float(t.groupby("product_id").size().median()),
        }

        gu = t.groupby("user_id")
        self._user = pd.DataFrame({
            "rev_review_count": gu.size().astype(float),
            "rev_mean_rating": gu["rating"].mean(),
            "rev_std_rating": gu["rating"].std().fillna(0.0),
            "rev_extreme_ratio": gu["rating"].apply(lambda s: s.isin([1, 5]).mean()),
            "rev_positive_ratio": gu["rating"].apply(lambda s: (s >= 4).mean()),
            "rev_n_products": gu["product_id"].nunique().astype(float),
            "rev_max_reviews_one_product": (
                t.groupby(["user_id", "product_id"]).size().groupby("user_id").max().astype(float)
            ),
            "rev_mean_review_len": gu["_wc"].mean(),
        })

        gp = t.groupby("product_id")
        self._prod = pd.DataFrame({
            "prod_review_count": gp.size().astype(float),
            "prod_mean_rating": gp["rating"].mean(),
            "prod_std_rating": gp["rating"].std().fillna(0.0),
        })
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._user is None or self._prod is None:
            raise RuntimeError("fit() must be called before transform()")

        out = pd.DataFrame(index=df.index)
        rating = df["rating"].astype(float)
        wc = df["text"].astype(str).str.split().str.len().astype(float)

        out["rating"] = rating
        out["rating_extremity"] = (rating - 3.0).abs()
        out["review_word_count"] = wc

        u = self._user.reindex(df["user_id"].values)
        u.index = df.index
        out["rev_seen_in_training"] = u["rev_review_count"].notna().astype(float)
        out["rev_review_count"] = u["rev_review_count"].fillna(self._prior["rev_review_count"])
        out["rev_is_singleton"] = (out["rev_review_count"] <= 1).astype(float)
        out["rev_mean_rating"] = u["rev_mean_rating"].fillna(self._prior["rating_mean"])
        out["rev_std_rating"] = u["rev_std_rating"].fillna(0.0)
        out["rev_extreme_ratio"] = u["rev_extreme_ratio"].fillna(self._prior["extreme_ratio"])
        out["rev_positive_ratio"] = u["rev_positive_ratio"].fillna(self._prior["positive_ratio"])
        out["rev_n_products"] = u["rev_n_products"].fillna(1.0)
        out["rev_max_reviews_one_product"] = u["rev_max_reviews_one_product"].fillna(1.0)
        out["rev_mean_review_len"] = u["rev_mean_review_len"].fillna(self._prior["word_count"])

        if self.include_product:
            p = self._prod.reindex(df["product_id"].values)
            p.index = df.index
            out["prod_seen_in_training"] = p["prod_review_count"].notna().astype(float)
            out["prod_review_count"] = p["prod_review_count"].fillna(self._prior["prod_review_count"])
            prod_mean = p["prod_mean_rating"].fillna(self._prior["rating_mean"])
            out["prod_mean_rating"] = prod_mean
            out["prod_std_rating"] = p["prod_std_rating"].fillna(self._prior["rating_std"])
            out["rating_dev_signed"] = rating - prod_mean
            out["rating_dev_from_product"] = (rating - prod_mean).abs()

        out = out[self.feature_names]
        assert out.notna().all().all(), "NaNs in behaviour features"
        return out


def audit_product_leakage(df: pd.DataFrame) -> Dict[str, float]:
    """Quantify how far business identity alone determines the YelpCHI label."""
    from sklearn.metrics import roc_auc_score

    per_product = df.groupby("product_id")["label"].agg(["mean", "size"])
    return {
        "n_products": int(len(per_product)),
        "products_with_zero_fakes": int((per_product["mean"] == 0).sum()),
        "products_all_fake": int((per_product["mean"] == 1).sum()),
        "median_product_fake_rate": float(per_product["mean"].median()),
        "auc_of_product_fake_rate_as_predictor": float(
            roc_auc_score(df["label"], df["product_id"].map(per_product["mean"]))
        ),
        "note": (
            "Business identity is close to a sufficient statistic for the label. "
            "Splits that let a business appear in both train and test overstate "
            "generalisation; the primary protocol groups folds by product_id."
        ),
    }
