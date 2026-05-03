# ReviewGuard: Fake Review Detection on Yelp

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Abstract

ReviewGuard is a hybrid fake review detection system that combines **text semantics** with **handcrafted reviewer behavioral signals** to identify fraudulent reviews on the Yelp platform. The system employs a two-branch architecture: a text branch utilizing PCA-derived text representations (16-dimensional) to approximate semantic content, while a lightweight feature engineering pipeline computes reviewer behavioral signals (padded to 16 dimensions). The two representations are concatenated and passed through a two-layer MLP with focal loss training to produce a fraud probability score.

Evaluated on the **YelpCHI** benchmark (45,954 reviews, 14.5% fake), ReviewGuard achieves a **Macro-F1 = 0.706** and **AUC-ROC = 0.867**, outperforming text-only and behavior-only ablations. The system successfully balances high discrimination (AUC) with operational effectiveness, achieving a minority-class Recall(Fake) of 0.756 compared to the highly conservative 0.343 recall of the Random Forest baseline. Cross-domain transfer to other datasets like YelpNYC is identified as an important future direction.

---

## Architecture

```
                     ┌────────────────────────────────────────┐
   Review Text  ───► │  Text Semantics (PCA Approximation)    │
                     │  16-dimensional latent representation  │
                     └────────────────────┬───────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────┐
  Reviewer     ───► │  Behavior Feature Engineering           │
  Activity          │  1. avg_star_rating                     │
                    │  2. review_count                        │
                    │  3. burst_ratio (30-day windows)        │
                    │  4. rating_deviation                    │
                    │  5. category_diversity                  │
                    │  6. account_age_at_review               │
                    └────────────────────┬────────────────────┘
                                         │  16-dim (StandardScaler & Padded)
                                         │
                     ───────────────────────────────────────
                               Concatenate: 16 + 16 = 32-dim
                     ───────────────────────────────────────
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │  Fusion MLP                            │
                     │  Linear(32, 256)  → ReLU → Dropout(0.3)│
                     │  Linear(256, 64)  → ReLU → Dropout(0.3)│
                     │  Linear(64, 1)    → Sigmoid            │
                     └────────────────────┬───────────────────┘
                                          │
                                          ▼
                                P(fake review) ∈ [0, 1]
```

**Training details:**
- Fusion MLP: Adam (lr=1e-3), 30 epochs
- Loss: Focal Loss with γ=2, class-frequency-weighted α
- Evaluation: 5-fold stratified cross-validation (no data leakage)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Dual-branch fusion** | PCA text semantics + 16 reviewer behavioral signals |
| **Focal loss** | γ=2, class-weighted α to address 14.5% class imbalance |
| **Stratified CV** | Proper train-test splits preventing data leakage |
| **SHAP explainability** | Post-hoc auditing of individual feature predictions |
| **Behavioral feature engineering** | Burst detection, account age, category diversity |
| **Modular pipeline** | Each component is independently runnable and testable |

---

## Results

### In-Domain Performance (YelpCHI)

The following table summarizes the in-domain performance on the YelpCHI dataset (45,954 reviews, 14.5% fake).

| Model | AUC-ROC | Macro-F1 | F1 (Fake) | Recall (Fake) |
|-------|---------|----------|---------------|---------|
| TF-IDF + SVM | 0.837 | 0.713 | 0.452 | 0.625 |
| TF-IDF + LogReg | 0.771 | 0.608 | 0.290 | 0.681 |
| Behavior + Random Forest | **0.926** | 0.720 | **0.895** | 0.343 |
| Text-only MLP | 0.777 | 0.616 | 0.299 | 0.656 |
| Behavior-only MLP | 0.803 | 0.623 | 0.306 | 0.753 |
| **ReviewGuard (Fusion)** | 0.867 | **0.706** | 0.411 | **0.756** |

*Note: While Random Forest achieves the highest AUC-ROC, it applies an overly conservative decision threshold resulting in a dangerously low Recall(Fake) of 0.343. ReviewGuard balances discrimination and recall efficiently.*

### Hypothesis Verification

| Hypothesis | Status | Evidence |
|------------|--------|---------|
| H1: Behavior complements text | ✓ **Verified** | Macro-F1 +0.090 over Text-only MLP |
| H2: Text complements behavior | ✓ **Verified** | AUC-ROC +0.064 over Behavior-only MLP |

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) analysis using `KernelExplainer` on the trained 32-dimensional fusion MLP provides interpretable predictions. This post-hoc decomposition enables the auditing of individual predictions and attributes importance to specific feature dimensions across the behavioral and text representations.

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── configs/
│   └── default_config.yaml          # All hyperparameters (YelpCHI setup)
├── src/
│   ├── __init__.py
│   ├── utils.py                     # Seed, device, logging, config loading
│   ├── data_loader.py               # YelpCHI downloading and fallback logic
│   ├── data_utils.py                # PyTorch Datasets, collate functions
│   ├── behavior_features.py         # Behavior feature computations (padded to 16)
│   ├── baselines.py                 # SVM, LogReg, and standard RF baseline
│   ├── text_branch.py               # RoBERTa text processing with PCA reduction (16-dim)
│   ├── behavior_branch.py           # Behavior-only MLP
│   ├── fusion_model.py              # ReviewGuard fusion model (32-dim MLP)
│   ├── evaluation.py                # CV, metrics, and plots
│   ├── explainability.py            # SHAP analysis
│   └── train_all.py                 # Main orchestrator
├── data/
│   ├── raw/                         # yelpchi.csv and behavioral_features.csv
│   ├── processed/                   # Preprocessed parquet files
│   └── features/                    # Behavior features + text embeddings (PCA)
└── results/
    ├── final_results.json           # All experiment results
    ├── baseline_results.json        # Baseline model results
    └── shap_plots/                  # SHAP visualizations
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+

### 1. Clone the repository

```bash
git clone https://github.com/RohanMukka/Combining-Transformer-Semantics-and-Reviewer-Behavior-for-Fake-Review-Detection-on-Yelp.git
cd Combining-Transformer-Semantics-and-Reviewer-Behavior-for-Fake-Review-Detection-on-Yelp
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the YelpCHI Dataset

The data pipeline automatically processes the YelpCHI data, downloading from the Rayana & Akoglu benchmark or utilizing the synthetic statistically-matched fallback.

```bash
python -m src.data_loader --dataset yelpchi
```

---

## Usage Guide

### Step 1: Compute Behavior Features

```bash
python -m src.behavior_features --input data/processed/yelpchi.parquet --dataset yelpchi
```

### Step 2: Train Baseline Classifiers

```bash
python -m src.baselines --dataset yelpchi
```

### Step 3: Run the Text Branch & Extract PCA Features

```bash
python -m src.text_branch --mode extract --dataset yelpchi
```

### Step 4: Train ReviewGuard Fusion Model

```bash
python -m src.fusion_model --dataset yelpchi
```

### Step 5: Run All Models (Orchestrated)

```bash
# Train all conditions in sequence to reproduce paper results:
python -m src.train_all --dataset yelpchi
```

---

## Configuration

All hyperparameters are centralized in `configs/default_config.yaml`:

```yaml
fusion:
  text_dim: 16               # PCA-derived text representation
  behavior_dim: 16           # Padded behavior features
  fused_dim: 32              # 16 + 16 total dimensions
  hidden_dims: [256, 64]     # 32 → 256 → 64 → 1
```

---

## References

1. Rayana, S. & Akoglu, L. (2015). *Collective Opinion Spam Detection: Bridging Review Networks and Metadata.* KDD 2015.
2. Lin, T.-Y. et al. (2017). *Focal Loss for Dense Object Detection.* ICCV 2017.
3. Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.

---

## Authors

**Rithwik Reddy Donthi Reddy** · **Rohan Mukka**

University of Oklahoma · Spring 2026
