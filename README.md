# ReviewGuard: Fake Review Detection on Yelp

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A two-branch fake review detector for YelpCHI, and an evaluation of how much of
the performance usually reported on that benchmark is real.

## Summary

Fake reviews in YelpCHI are concentrated by business. 657 of its 1,224
businesses contain **zero** fake reviews, the median business fake rate is
exactly zero, and mapping each review to its business's fake rate is by itself
a **0.951-AUC** predictor of the label. Any evaluation that lets a business
appear in both training and test folds is therefore partly measuring
memorisation of which businesses were spam targets.

We evaluate the same models under two protocols that differ *only* in how folds
are partitioned:

| Model | Reviewer-disjoint | Business-disjoint | Gap |
|---|---|---|---|
| Text-only MLP | 0.743 | 0.712 | +0.032 |
| TF-IDF + LogReg | 0.762 | 0.700 | +0.062 |
| TF-IDF + LinearSVM | 0.740 | 0.666 | +0.074 |
| **ReviewGuard (Fusion)** | 0.856 | **0.730** | +0.126 |
| Behavior-only MLP | 0.889 | 0.706 | +0.183 |
| Behavior + LogReg | 0.782 | 0.599 | +0.183 |
| Behavior + RandomForest | 0.934 | 0.646 | **+0.289** |

*AUC-ROC, mean over 5 folds.*

The gap scales with how directly a model can reach business identity. The
behavioral random forest goes from best model under the weaker protocol to
worst under the stricter one — the ranking of behavioral and textual features
inverts. Most of what a behavior branch contributes on this benchmark is
knowing which businesses attract spam, not knowing what a spammer does.

Under the business-disjoint protocol ReviewGuard reaches **0.730 AUC-ROC** and
**0.626 Macro-F1**, ahead of the text-only branch on 5/5 folds and the
behavior-only branch on 4/5. With five folds the Wilcoxon signed-rank test
bottoms out at *p* = 0.0625, so none of these margins clears α = 0.05; we read
them as a real but modest fusion benefit, not a decisive win.

## Data integrity audit

Two inputs shipped with the CSV distribution of YelpCHI fail an audit and are
excluded from every experiment here. Run `python -m src.build_real_dataset` to
reproduce the diagnostics (`results/metrics/data_integrity_audit.json`).

**The `date` column is label-contaminated.** A random forest given nothing but
the raw Unix timestamp scores **AUC 0.965**. The monthly fake rate swings from
49.8% (2010-01) to 2.6% (2010-03), the corpus spans only 111 days, posting
hours are uniform across all 24, and the seconds field holds exactly one
distinct value. Real posting times are diurnal; this column's time component
was generated, and generated conditioned on the class. Every temporal feature —
burstiness, inter-review gaps, account age — would inherit the leakage, so we
use no timestamps at all. This costs us the burst features prior work found
useful, and is a genuine limitation.

**`data/raw/behavioral_features.csv` is noise.** Its columns correlate at most
**0.005** with the natural reconstruction of their own definitions from the
review table (e.g. a reviewer's actual mean rating vs. its `avg_star_rating`),
and at most **0.008** with the label. `burst_ratio` is 0 for 97.5% of rows;
`category_diversity` is 1.0 for 99.4%. It is not used.

## Architecture

```
                    ┌──────────────────────────────────────────────┐
  Review text  ───► │ Text branch                                  │
                    │  TF-IDF word 1-2gram  ─┐                     │
                    │  TF-IDF char 3-5gram  ─┴─► truncated SVD 256 │
                    │  + 10 stylometric features                   │
                    └────────────────────┬─────────────────────────┘
                                         │ 266-dim
  user_id      ───► ┌──────────────────────────────────────────────┐
  product_id        │ Behavior branch (19 features)                │
  rating            │  review-level:   rating, extremity, length   │
                    │  reviewer-level: counts, rating stats,       │
                    │                  breadth, mean length        │
                    │  business-level: counts, rating stats,       │
                    │                  rating deviation            │
                    │  + seen-in-training indicators (cold start)  │
                    └────────────────────┬─────────────────────────┘
                                         │ 19-dim
                    ──────────────────────────────────────────────
                            concatenate: 266 + 19 = 285-dim
                    ──────────────────────────────────────────────
                                         ▼
                    ┌──────────────────────────────────────────────┐
                    │ Fusion MLP, focal loss (γ=2)                 │
                    │  Linear(285,256) → ReLU → Dropout(0.3)       │
                    │  Linear(256, 64) → ReLU → Dropout(0.3)       │
                    │  Linear(64, 1)                               │
                    └────────────────────┬─────────────────────────┘
                                         ▼
                               P(fake review) ∈ [0,1]
```

**No transformer branch in the reported results.** `RobertaEncoder` in
`src/text_representation.py` implements the fine-tuned roberta-base variant we
originally intended, but it was not run: the environment used for this study had
no GPU and no network route to the model hub. The reported text branch is
lexical, and should be read as a floor on what a text branch can do. Running the
transformer variant is the single most promising open item.

## Methodology notes

Three choices matter for interpreting the numbers:

1. **Fold-aware features.** Reviewer and business aggregates are fitted on a
   fold's *training* rows and applied to held-out rows, with training-set priors
   for unseen keys. Computing them over the full corpus — the natural
   implementation — leaks the evaluation fold into the features.
2. **Every model's threshold is tuned.** Under 14.53% positive prevalence, a
   classifier left at 0.5 predicts the majority class nearly everywhere.
   Thresholds for all models, baselines included, are tuned on a held-out
   validation slice for Macro-F1 and applied unchanged to the test fold. The
   earlier version of this project reported a 0.003 fake-class recall for an
   untuned SVM, which reflects calibration rather than ranking ability.
3. **Tables are generated, not typed.** `src/make_tables.py` emits
   `paper/tables/*.tex` directly from `results/metrics/*.json`, so a number
   cannot appear in the paper unless it appears in a metrics file.

## Reproducing

```bash
pip install -r requirements.txt

# 1. Audit the raw data and write the trusted review table
python -m src.build_real_dataset

# 2. Both evaluation protocols (~15 min each, 4 CPU cores)
python -m src.run_real_experiments --folds 5 --group-by product   # primary
python -m src.run_real_experiments --folds 5 --group-by user      # comparison

# 3. Per-fold scores for the ROC / confusion-matrix / SHAP figures
python -m src.dump_fold_scores --fold 1 --group-by product

# 4. Figures and LaTeX tables
python -m src.make_figures
python -m src.make_tables
```

Building the paper:

```bash
cd paper && pdflatex main.tex && pdflatex main.tex
```

## Layout

```
src/
  build_real_dataset.py     data integrity audits; writes the trusted table
  behavior_featurizer.py    fold-aware reviewer/business features
  text_representation.py    TF-IDF+SVD encoder; RobertaEncoder (not run here)
  losses.py                 focal loss
  run_real_experiments.py   5-fold CV, both protocols, Wilcoxon tests
  dump_fold_scores.py       single-fold scores for curve figures + SHAP
  make_figures.py           all figures
  make_tables.py            all LaTeX tables, from the metrics JSON
results/metrics/
  real_yelpchi_results_by_product.json   primary protocol
  real_yelpchi_results_by_user.json      comparison protocol
  data_integrity_audit.json              audit diagnostics
  superseded/                            earlier, unsound results — see its README
paper/
  main.tex, tables/, figures/
```

## Status of the earlier version

An earlier version of this project reported AUC-ROC 0.866 / Macro-F1 0.715 with
a RoBERTa text branch. Those numbers should not be used. The run that produced
them read no review text: it loaded the 32 precomputed metadata features from
`YelpChi.mat`, split the vector in half, and labelled the halves "behavior" and
"text". Its baselines, reported as "TF-IDF + SVM" and "TF-IDF + LogReg", were
fit on the same metadata features and left at a 0.5 threshold. Those outputs are
retained under `results/metrics/superseded/` with a full explanation.

## Dataset

YelpCHI (Rayana & Akoglu, KDD 2015): 45,954 reviews, 6,677 fake (14.53%),
39,623 reviewers, 1,224 businesses. This project uses the CSV distribution
carrying review text, reviewer and business identifiers, star ratings, and
labels.

## License

MIT — see `LICENSE`.
