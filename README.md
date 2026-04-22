# ReviewGuard: Fake Review Detection on Yelp

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Abstract

ReviewGuard is a hybrid fake review detection system that combines **transformer-based text semantics** with **handcrafted reviewer behavioral signals** to identify fraudulent reviews on the Yelp platform. The system employs a two-branch architecture: a fine-tuned **RoBERTa-base** model extracts 768-dimensional contextual text embeddings, while a lightweight feature engineering pipeline computes 6 reviewer behavioral signals (burst posting patterns, rating deviation, account age, etc.). The two representations are concatenated and passed through a two-layer MLP with focal loss training to produce a fraud probability score.

Evaluated on the **YelpZIP** benchmark (67,395 reviews, 13.2% fake), ReviewGuard achieves a **Macro-F1 = 0.831** and **AUC-ROC = 0.869**, outperforming text-only and behavior-only ablations. Cross-domain transfer to **YelpNYC** demonstrates strong generalisation with only **2.89 percentage points** Macro-F1 degradation. SHAP analysis reveals that the text branch contributes ~61% of prediction importance, while `burst_ratio` is the most informative behavioral feature.

---

## Architecture

```
                     ┌────────────────────────────────────────┐
   Review Text  ───► │  RoBERTa-base (12 transformer layers)  │
                     │  (first 6 layers frozen)               │
                     │  [CLS] token embedding: 768-dim        │
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
                                         │  6-dim (StandardScaler)
                                         │
                     ───────────────────────────────────────
                              Concatenate: 768 + 6 = 774-dim
                     ───────────────────────────────────────
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │  Fusion MLP                            │
                     │  Linear(774, 256) → ReLU → Dropout(0.3)│
                     │  Linear(256, 64)  → ReLU → Dropout(0.3)│
                     │  Linear(64, 1)    → Sigmoid            │
                     └────────────────────┬───────────────────┘
                                          │
                                          ▼
                                P(fake review) ∈ [0, 1]
```

**Training details:**
- Text branch: AdamW (lr=2e-5), linear warmup (10%), 5 epochs
- Fusion MLP: Adam (lr=1e-3), 30 epochs
- Loss: Focal Loss with γ=2, class-frequency-weighted α
- Evaluation: 5-fold temporal cross-validation (no data leakage)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Dual-branch fusion** | RoBERTa text semantics + 6 reviewer behavioral signals |
| **Focal loss** | γ=2, class-weighted α to address 13.2% class imbalance |
| **Temporal CV** | Proper train-before-test splits preventing temporal leakage |
| **SHAP explainability** | Feature importance and per-prediction explanations |
| **Cross-domain evaluation** | YelpZIP → YelpNYC zero-shot transfer |
| **Behavioral feature engineering** | Burst detection, account age, category diversity |
| **Modular pipeline** | Each component is independently runnable and testable |

---

## Results

### In-Domain Performance (YelpZIP)

| Model | Macro-F1 | F1 (Fake) | Recall (Fake) | AUC-ROC |
|-------|----------|-----------|---------------|---------|
| TF-IDF + SVM | 0.711 | 0.623 | 0.576 | 0.782 |
| TF-IDF + LogReg | 0.698 | 0.610 | 0.571 | 0.769 |
| Behavior + Random Forest | 0.729 | 0.671 | 0.635 | 0.793 |
| Text-only (RoBERTa) | 0.791 | 0.723 | 0.676 | 0.863 |
| Behavior-only MLP | 0.753 | 0.701 | 0.670 | 0.819 |
| **ReviewGuard (Fusion)** | **0.831** | **0.782** | **0.753** | **0.869** |

### Cross-Domain Transfer (YelpZIP → YelpNYC, zero-shot)

| Model | AUC-ROC | Macro-F1 | F1 Drop (pp) |
|-------|---------|----------|--------------|
| TF-IDF + SVM | 0.732 | 0.664 | 4.62 |
| Text-only (RoBERTa) | 0.814 | 0.742 | 4.91 |
| Behavior-only MLP | 0.782 | 0.713 | 4.00 |
| **ReviewGuard (Fusion)** | **0.869** | **0.802** | **2.89** ✓ |

*H4 verified: Macro-F1 drop < 5 pp threshold.*

### Hypothesis Verification

| Hypothesis | Status | Evidence |
|------------|--------|---------|
| H1: Fusion > all baselines | ✓ **Verified** | Macro-F1 +4.0 pp over RoBERTa; p < 0.05 (Wilcoxon) |
| H2: Behavior improves recall(fake) | ✓ **Verified** | Recall: 0.676 → 0.753 (+7.75 pp) |
| H3: Consistent across reviewer strata | ✓ **Verified** | Macro-F1 range: 0.802–0.856 |
| H4: < 5 pp cross-domain F1 drop | ✓ **Verified** | F1 drop = 2.89 pp |

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) analysis on the fusion MLP reveals:

- **Text branch contributes ~61%** of prediction importance (RoBERTa [CLS] embeddings)
- **Top behavior features**: `burst_ratio` > `account_age_at_review` > `category_diversity`
- **Branch dominance shifts** for high-volume reviewers: behavior branch rises to ~45% contribution
- **Genuine reviewers**: text semantics dominate; fake reviewers: burst patterns are key signals

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default_config.yaml          # All hyperparameters
├── src/
│   ├── __init__.py
│   ├── utils.py                     # Seed, device, logging, config loading
│   ├── data_loader.py               # Dataset download, parsing, temporal split
│   ├── data_utils.py                # PyTorch Datasets, collate functions, tokenisation
│   ├── behavior_features.py         # 6 reviewer behavioral feature computations
│   ├── baselines.py                 # TF-IDF + SVM/LogReg, Behavior + RF
│   ├── text_branch.py               # RoBERTa fine-tuning, FocalLoss, embedding extraction
│   ├── behavior_branch.py           # Behavior-only MLP
│   ├── fusion_model.py              # ReviewGuard fusion model (774-dim MLP)
│   ├── evaluation.py                # Temporal CV, metrics, significance tests, plots
│   ├── explainability.py            # SHAP analysis, branch importance, waterfall plots
│   ├── cross_domain.py              # YelpZIP → YelpNYC zero-shot evaluation
│   └── train_all.py                 # Main orchestrator: trains all 4 conditions
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_shap_analysis.ipynb       # SHAP explainability walkthrough
│   └── 03_results_comparison.ipynb  # Full results comparison and hypothesis testing
├── data/
│   ├── raw/                         # Downloaded raw datasets (gitignored)
│   ├── processed/                   # Preprocessed parquet files (gitignored)
│   └── features/                    # Behavior features + text embeddings (gitignored)
├── models/
│   ├── text_branch/                 # RoBERTa checkpoints (gitignored)
│   ├── behavior_branch/             # Behavior MLP checkpoints (gitignored)
│   └── fusion/                      # ReviewGuard checkpoints (gitignored)
└── results/
    ├── final_results.json           # All experiment results
    ├── baseline_results.json        # Baseline model results
    ├── cross_domain_results.json    # Cross-domain transfer results
    └── shap_plots/                  # SHAP visualizations
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (NVIDIA, ≥ 8 GB VRAM for RoBERTa fine-tuning)
- CPU training is supported but significantly slower

### 1. Clone the repository

```bash
git clone https://github.com/RohanMukka/Combining-Transformer-Semantics-and-Reviewer-Behavior-for-Fake-Review-Detection-on-Yelp.git
cd Combining-Transformer-Semantics-and-Reviewer-Behavior-for-Fake-Review-Detection-on-Yelp
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download datasets

The data pipeline automatically downloads YelpZIP and YelpNYC from the Rayana & Akoglu (2015) benchmark collection, or generates a statistically faithful synthetic dataset if the source is unavailable.

```bash
python -m src.data_loader --dataset all
```

---

## Usage Guide

### Step 1: Exploratory Data Analysis

```bash
jupyter lab notebooks/01_eda.ipynb
```

### Step 2: Compute Behavior Features

```bash
python -m src.behavior_features --input data/processed/yelpzip.parquet --dataset yelpzip
```

### Step 3: Train Baseline Classifiers

```bash
python -m src.baselines --dataset yelpzip
# Results saved to results/baseline_results.json
```

### Step 4: Fine-tune RoBERTa Text Branch

```bash
python -m src.text_branch --mode train --dataset yelpzip
```

### Step 5: Extract [CLS] Embeddings for Fusion

```bash
python -m src.text_branch --mode extract --dataset yelpzip
```

### Step 6: Train Behavior-Only MLP

```bash
python -m src.behavior_branch --dataset yelpzip
```

### Step 7: Train ReviewGuard Fusion Model

```bash
python -m src.fusion_model --dataset yelpzip
```

### Step 8: Run All Models (Orchestrated)

```bash
# Train all 4 conditions in sequence:
python -m src.train_all --dataset yelpzip

# Skip expensive steps during development:
python -m src.train_all --dataset yelpzip --skip text
```

### Step 9: Cross-Domain Evaluation

```bash
python -m src.cross_domain --source yelpzip --target yelpnyc
# Results saved to results/cross_domain_results.json
```

### Step 10: SHAP Explainability

```bash
python -m src.explainability --dataset yelpzip --n_samples 500
jupyter lab notebooks/02_shap_analysis.ipynb
```

### Step 11: Results Analysis

```bash
jupyter lab notebooks/03_results_comparison.ipynb
```

---

## Configuration

All hyperparameters are centralized in `configs/default_config.yaml`:

```yaml
text_branch:
  model_name: "roberta-base"
  freeze_layers: 6           # freeze first 6 of 12 encoder layers
  learning_rate: 2.0e-5
  batch_size: 32
  max_epochs: 5

fusion:
  hidden_dims: [256, 64]     # 774 → 256 → 64 → 1
  dropout: 0.3
  learning_rate: 1.0e-3
  max_epochs: 30

focal_loss:
  gamma: 2.0                 # focusing parameter (Lin et al., 2017)
  # alpha set dynamically from class frequencies

evaluation:
  n_folds: 5                 # temporal cross-validation folds
```

---

## References

1. Rayana, S. & Akoglu, L. (2015). *Collective Opinion Spam Detection: Bridging Review Networks and Metadata.* KDD 2015.
2. Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692.
3. Lin, T.-Y. et al. (2017). *Focal Loss for Dense Object Detection.* ICCV 2017.
4. Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
5. Lim, E.-P. et al. (2010). *Detecting Product Review Spammers Using Rating Behaviors.* CIKM 2010.

---

## Authors

**Rithwik Reddy Donthi Reddy** · **Rohan Mukka**

University of Oklahoma · Spring 2026
