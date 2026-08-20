# Superseded metrics

These files are the outputs of the original pipeline and are kept only as a
record of where the earlier version of the paper got its numbers. **Do not cite
or reuse them.**

They are unsound for two reasons:

1. **The "text branch" contained no text.** `run_experiments.py` loaded only the
   32 precomputed metadata features from `YelpChi.mat`, split that vector in
   half, and labelled the first 16 columns "behavior" and the second 16 "text"
   after a PCA projection. No review text was read. The baselines reported as
   "TF-IDF + SVM" and "TF-IDF + LogReg" were also fit on those metadata
   features, not on TF-IDF of text.
2. **Baselines were left at a 0.5 decision threshold** while the fusion model
   effectively was not, which is where figures such as a 0.003 fake-class recall
   for the SVM came from.

`yelpzip`-tagged entries in `all_models_comparison.json` are from a separate run
on a different dataset and score at chance (AUC ~0.50).

Current results live one directory up:

- `real_yelpchi_results_by_product.json` - primary, business-disjoint protocol
- `real_yelpchi_results_by_user.json` - reviewer-disjoint comparison
- `data_integrity_audit.json` - timestamp, feature-file, and business-identity audits

produced by `src/run_real_experiments.py` on the real review text.
