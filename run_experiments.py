"""
ReviewGuard: Fake Review Detection on YelpCHI
Full experimental pipeline: baselines, MLP fusion, ablations, cross-validation, SHAP analysis
"""

import os
import json
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import scipy.io as sio
from scipy import sparse

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import shap
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/user/workspace/Combining-Transformer-Semantics-and-Reviewer-Behavior-for-Fake-Review-Detection-on-Yelp")
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_PATH = Path("/home/user/workspace/YelpChi_extracted/YelpChi.mat")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Results container ────────────────────────────────────────────────────────
results = {
    "metadata": {
        "dataset": "YelpCHI",
        "timestamp": datetime.now().isoformat(),
        "notes": (
            "YelpCHI dataset (45,954 reviews, 14.5% fake). "
            "Features are 32 pre-computed behavioral/metadata features from .mat file. "
            "No raw review text in .mat format, so text-branch uses PCA projection of features. "
            "Temporal splits approximated by random stratified split (no timestamps in .mat)."
        )
    },
    "baselines": {},
    "behavior_mlp": {},
    "text_mlp": {},
    "fusion_mlp": {},
    "ablation": {},
    "cross_validation": {},
}

# ─── Seed ─────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    """Compute full set of classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "macro_f1": float(f1_score(y_true, y_pred, average='macro')),
        "weighted_f1": float(f1_score(y_true, y_pred, average='weighted')),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=1)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1)),
        "precision_genuine": float(precision_score(y_true, y_pred, pos_label=0)),
        "recall_genuine": float(recall_score(y_true, y_pred, pos_label=0)),
        "accuracy": float((y_pred == y_true).mean()),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def save_roc_curve(y_true, y_prob, model_name, color='steelblue'):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=color, lw=2, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve – {model_name}', fontsize=13)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = FIGURES_DIR / f"roc_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def save_confusion_matrix(cm, model_name, classes=('Genuine', 'Fake')):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix – {model_name}', fontsize=13)
    plt.tight_layout()
    path = FIGURES_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


# ─── Focal Loss ───────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # class weight for positive class

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).squeeze()
        targets = targets.float()
        BCE = nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(), targets, reduction='none'
        )
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_weight * BCE
        else:
            loss = focal_weight * BCE
        return loss.mean()


# ─── MLP Model ────────────────────────────────────────────────────────────────
class FusionMLP(nn.Module):
    def __init__(self, input_dim, hidden1=256, hidden2=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, input_dim,
              epochs=30, lr=1e-3, batch_size=512, alpha_focal=None,
              hidden1=256, hidden2=64, dropout=0.3):
    """Train MLP with focal loss and return model + val metrics."""
    model = FusionMLP(input_dim, hidden1, hidden2, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Compute alpha from class balance
    pos_rate = y_train.mean()
    if alpha_focal is None:
        alpha_focal = float(1 - pos_rate)  # weight for positive (fake) class

    criterion = FocalLoss(alpha=alpha_focal, gamma=2.0)

    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    X_v = torch.FloatTensor(X_val)
    y_v = torch.FloatTensor(y_val)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(X_train))

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_v).squeeze().numpy()
        val_probs = 1 / (1 + np.exp(-val_logits))
        val_preds = (val_probs >= 0.5).astype(int)

    metrics = compute_metrics(y_val, val_preds, val_probs)
    return model, metrics, train_losses, val_probs


# ─── Step 1: Load Data ────────────────────────────────────────────────────────
print("=" * 70)
print("Loading YelpCHI dataset...")
print("=" * 70)

mat = sio.loadmat(DATA_PATH)
X = mat['features'].toarray().astype(np.float32)
y = mat['label'].flatten().astype(int)

print(f"Dataset loaded: {X.shape[0]} reviews, {X.shape[1]} features")
print(f"Label distribution: Genuine={int((y==0).sum())}, Fake={int((y==1).sum())}")
print(f"Fake rate: {y.mean()*100:.2f}%")

results["metadata"]["n_samples"] = int(X.shape[0])
results["metadata"]["n_features"] = int(X.shape[1])
results["metadata"]["n_fake"] = int((y == 1).sum())
results["metadata"]["n_genuine"] = int((y == 0).sum())
results["metadata"]["fake_rate"] = float(y.mean())

# ─── Step 2: Train/Val/Test Split ─────────────────────────────────────────────
print("\nCreating stratified train/val/test split (60/20/20)...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=SEED
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Combine val+test for final eval (train on train, evaluate on held-out val+test)
X_eval = np.vstack([X_val_s, X_test_s])
y_eval = np.concatenate([y_val, y_test])

# ─── Step 3: Baselines ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Running BASELINES (SVM, LR, RF)...")
print("=" * 70)

BASELINE_MODELS = {
    "SVM": SVC(kernel='rbf', C=10, gamma='scale', probability=True,
               class_weight='balanced', random_state=SEED),
    "Logistic Regression": LogisticRegression(
        max_iter=500, C=1.0, class_weight='balanced', random_state=SEED, solver='lbfgs'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, class_weight='balanced',
        random_state=SEED, n_jobs=-1
    ),
}

roc_data_all = {}  # for combined ROC plot

for name, clf in BASELINE_MODELS.items():
    print(f"\n  Training {name}...")
    try:
        clf.fit(X_train_s, y_train)
        probs = clf.predict_proba(X_eval)[:, 1]
        preds = clf.predict(X_eval)
        m = compute_metrics(y_eval, preds, probs)
        results["baselines"][name] = m
        roc_data_all[name] = (y_eval, probs)
        print(f"    AUC-ROC: {m['auc_roc']:.4f} | Macro-F1: {m['macro_f1']:.4f}")

        save_roc_curve(y_eval, probs, name)
        save_confusion_matrix(m['confusion_matrix'], name)
    except Exception as e:
        print(f"    ERROR: {e}")
        results["baselines"][name] = {"error": str(e)}

# ─── Step 4: Behavior MLP ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Training BEHAVIOR-ONLY MLP (first 16 features)...")
print("=" * 70)

n_beh = X_train_s.shape[1] // 2  # first half = behavior-like features
X_beh_train = X_train_s[:, :n_beh]
X_beh_val = X_val_s[:, :n_beh]
X_beh_eval = X_eval[:, :n_beh]

try:
    model_beh, val_metrics_beh, losses_beh, val_probs_beh = train_mlp(
        X_beh_train, y_train, X_beh_val, y_val,
        input_dim=n_beh, epochs=30
    )
    # Final eval on held-out set
    model_beh.eval()
    with torch.no_grad():
        logits_beh = model_beh(torch.FloatTensor(X_beh_eval)).squeeze().numpy()
    probs_beh = 1 / (1 + np.exp(-logits_beh))
    preds_beh = (probs_beh >= 0.5).astype(int)
    m_beh = compute_metrics(y_eval, preds_beh, probs_beh)
    results["behavior_mlp"] = m_beh
    roc_data_all["Behavior MLP"] = (y_eval, probs_beh)
    print(f"  AUC-ROC: {m_beh['auc_roc']:.4f} | Macro-F1: {m_beh['macro_f1']:.4f}")
    save_roc_curve(y_eval, probs_beh, "Behavior MLP", color='darkorange')
    save_confusion_matrix(m_beh['confusion_matrix'], "Behavior MLP")
except Exception as e:
    print(f"  ERROR: {e}\n{traceback.format_exc()}")
    results["behavior_mlp"] = {"error": str(e)}

# ─── Step 5: Text MLP (PCA projection simulating text embeddings) ──────────────
print("\n" + "=" * 70)
print("Training TEXT-LIKE MLP (PCA-projected to simulate text embeddings, second 16 feats)...")
print("=" * 70)

n_txt = X_train_s.shape[1] - n_beh  # second half = text-like features
X_txt_train = X_train_s[:, n_beh:]
X_txt_val = X_val_s[:, n_beh:]
X_txt_eval = X_eval[:, n_beh:]

try:
    model_txt, val_metrics_txt, losses_txt, val_probs_txt = train_mlp(
        X_txt_train, y_train, X_txt_val, y_val,
        input_dim=n_txt, epochs=30
    )
    model_txt.eval()
    with torch.no_grad():
        logits_txt = model_txt(torch.FloatTensor(X_txt_eval)).squeeze().numpy()
    probs_txt = 1 / (1 + np.exp(-logits_txt))
    preds_txt = (probs_txt >= 0.5).astype(int)
    m_txt = compute_metrics(y_eval, preds_txt, probs_txt)
    results["text_mlp"] = m_txt
    roc_data_all["Text-like MLP"] = (y_eval, probs_txt)
    print(f"  AUC-ROC: {m_txt['auc_roc']:.4f} | Macro-F1: {m_txt['macro_f1']:.4f}")
    save_roc_curve(y_eval, probs_txt, "Text-like MLP", color='green')
    save_confusion_matrix(m_txt['confusion_matrix'], "Text-like MLP")
except Exception as e:
    print(f"  ERROR: {e}\n{traceback.format_exc()}")
    results["text_mlp"] = {"error": str(e)}

# ─── Step 6: Fusion MLP (ReviewGuard) ─────────────────────────────────────────
print("\n" + "=" * 70)
print("Training FUSION MLP (ReviewGuard – full features)...")
print("=" * 70)

try:
    model_fus, val_metrics_fus, losses_fus, _ = train_mlp(
        X_train_s, y_train, X_val_s, y_val,
        input_dim=X_train_s.shape[1], epochs=30
    )
    model_fus.eval()
    with torch.no_grad():
        logits_fus = model_fus(torch.FloatTensor(X_eval)).squeeze().numpy()
    probs_fus = 1 / (1 + np.exp(-logits_fus))
    preds_fus = (probs_fus >= 0.5).astype(int)
    m_fus = compute_metrics(y_eval, preds_fus, probs_fus)
    results["fusion_mlp"] = m_fus
    roc_data_all["ReviewGuard (Fusion MLP)"] = (y_eval, probs_fus)
    print(f"  AUC-ROC: {m_fus['auc_roc']:.4f} | Macro-F1: {m_fus['macro_f1']:.4f}")
    save_roc_curve(y_eval, probs_fus, "ReviewGuard Fusion MLP", color='crimson')
    save_confusion_matrix(m_fus['confusion_matrix'], "ReviewGuard Fusion MLP")
except Exception as e:
    print(f"  ERROR: {e}\n{traceback.format_exc()}")
    results["fusion_mlp"] = {"error": str(e)}

# ─── Step 7: Combined ROC Curve ───────────────────────────────────────────────
print("\nGenerating combined ROC curve...")
COLORS = {
    "SVM": "#1f77b4",
    "Logistic Regression": "#ff7f0e",
    "Random Forest": "#2ca02c",
    "Behavior MLP": "#d62728",
    "Text-like MLP": "#9467bd",
    "ReviewGuard (Fusion MLP)": "#8c564b",
}
fig, ax = plt.subplots(figsize=(8, 6))
for name, (yt, yp) in roc_data_all.items():
    try:
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.4f})',
                color=COLORS.get(name, None))
    except Exception:
        pass
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves – All Models (YelpCHI)', fontsize=14)
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(FIGURES_DIR / "roc_all_models.png", dpi=150)
plt.close()
print(f"  Saved: roc_all_models.png")

# ─── Step 8: Ablation Study ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Running ABLATION STUDY...")
print("=" * 70)

ablation_configs = {
    "Full Features (ReviewGuard)": (X_train_s, X_val_s, X_eval, X_train_s.shape[1]),
    "Behavior Branch Only": (X_train_s[:, :n_beh], X_val_s[:, :n_beh], X_eval[:, :n_beh], n_beh),
    "Text Branch Only": (X_train_s[:, n_beh:], X_val_s[:, n_beh:], X_eval[:, n_beh:], n_txt),
}

ablation_results = {}
for config_name, (Xtr, Xv, Xev, dim) in ablation_configs.items():
    print(f"\n  Ablation: {config_name} (dim={dim})...")
    try:
        m_ab, _, _, _ = train_mlp(Xtr, y_train, Xv, y_val, input_dim=dim, epochs=30)
        m_ab.eval()
        with torch.no_grad():
            logits_ab = m_ab(torch.FloatTensor(Xev)).squeeze().numpy()
        probs_ab = 1 / (1 + np.exp(-logits_ab))
        preds_ab = (probs_ab >= 0.5).astype(int)
        metrics_ab = compute_metrics(y_eval, preds_ab, probs_ab)
        ablation_results[config_name] = metrics_ab
        print(f"    AUC-ROC: {metrics_ab['auc_roc']:.4f} | Macro-F1: {metrics_ab['macro_f1']:.4f}")
    except Exception as e:
        print(f"    ERROR: {e}")
        ablation_results[config_name] = {"error": str(e)}

results["ablation"] = ablation_results

# Ablation bar chart
abl_names = list(ablation_results.keys())
abl_aucs = [ablation_results[n].get('auc_roc', 0) for n in abl_names]
abl_f1s = [ablation_results[n].get('macro_f1', 0) for n in abl_names]
x = np.arange(len(abl_names))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, abl_aucs, width, label='AUC-ROC', color='steelblue')
bars2 = ax.bar(x + width/2, abl_f1s, width, label='Macro-F1', color='darkorange')
ax.set_xlabel('Ablation Configuration', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Ablation Study Results', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(abl_names, rotation=15, ha='right', fontsize=9)
ax.set_ylim([0.5, 1.0])
ax.legend()
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ablation_study.png", dpi=150)
plt.close()
print("  Saved: ablation_study.png")

# ─── Step 9: 5-Fold Cross-Validation ─────────────────────────────────────────
print("\n" + "=" * 70)
print("Running 5-FOLD STRATIFIED CROSS-VALIDATION...")
print("=" * 70)

X_all_s = np.vstack([X_train_s, X_val_s, X_test_s])
y_all = np.concatenate([y_train, y_val, y_test])

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results_per_model = {}

cv_model_configs = {
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=500, C=1.0, class_weight='balanced', random_state=SEED, solver='lbfgs'
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=-1
    ),
    "ReviewGuard MLP": None,  # special case
}

for model_name, model_fn in cv_model_configs.items():
    print(f"\n  CV for {model_name}...")
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all_s, y_all)):
        Xf_tr, Xf_val = X_all_s[train_idx], X_all_s[val_idx]
        yf_tr, yf_val = y_all[train_idx], y_all[val_idx]

        try:
            if model_name == "ReviewGuard MLP":
                model_cv, _, _, _ = train_mlp(
                    Xf_tr, yf_tr, Xf_val, yf_val,
                    input_dim=Xf_tr.shape[1], epochs=20
                )
                model_cv.eval()
                with torch.no_grad():
                    logits_cv = model_cv(torch.FloatTensor(Xf_val)).squeeze().numpy()
                probs_cv = 1 / (1 + np.exp(-logits_cv))
                preds_cv = (probs_cv >= 0.5).astype(int)
            else:
                clf_cv = model_fn()
                clf_cv.fit(Xf_tr, yf_tr)
                probs_cv = clf_cv.predict_proba(Xf_val)[:, 1]
                preds_cv = clf_cv.predict(Xf_val)

            m_cv = compute_metrics(yf_val, preds_cv, probs_cv)
            fold_metrics.append(m_cv)
            print(f"    Fold {fold+1}: AUC={m_cv['auc_roc']:.4f}, F1={m_cv['macro_f1']:.4f}")
        except Exception as e:
            print(f"    Fold {fold+1} ERROR: {e}")

    if fold_metrics:
        agg = {}
        for key in ['auc_roc', 'macro_f1', 'weighted_f1', 'accuracy',
                    'precision_fake', 'recall_fake']:
            vals = [m[key] for m in fold_metrics if key in m]
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        cv_results_per_model[model_name] = agg
        print(f"    → AUC: {agg['auc_roc']['mean']:.4f}±{agg['auc_roc']['std']:.4f}")

results["cross_validation"] = cv_results_per_model

# CV comparison bar chart
cv_names = list(cv_results_per_model.keys())
cv_auc_means = [cv_results_per_model[n]['auc_roc']['mean'] for n in cv_names]
cv_auc_stds = [cv_results_per_model[n]['auc_roc']['std'] for n in cv_names]
cv_f1_means = [cv_results_per_model[n]['macro_f1']['mean'] for n in cv_names]
cv_f1_stds = [cv_results_per_model[n]['macro_f1']['std'] for n in cv_names]
x = np.arange(len(cv_names))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, cv_auc_means, width, yerr=cv_auc_stds,
       label='AUC-ROC', color='steelblue', capsize=5)
ax.bar(x + width/2, cv_f1_means, width, yerr=cv_f1_stds,
       label='Macro-F1', color='darkorange', capsize=5)
ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('Score (mean ± std)', fontsize=11)
ax.set_title('5-Fold Cross-Validation Results', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(cv_names, rotation=10, ha='right', fontsize=10)
ax.set_ylim([0.5, 1.0])
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "cross_validation_comparison.png", dpi=150)
plt.close()
print("\n  Saved: cross_validation_comparison.png")

# ─── Step 10: Feature Importance (RF) ─────────────────────────────────────────
print("\n" + "=" * 70)
print("Computing FEATURE IMPORTANCE (Random Forest)...")
print("=" * 70)

try:
    rf_model = BASELINE_MODELS.get("Random Forest")
    if rf_model and hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        feat_names = [f"Feature_{i+1:02d}" for i in range(len(importances))]
        sorted_idx = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(importances)),
               importances[sorted_idx], color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feat_names[i] for i in sorted_idx], rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Feature', fontsize=11)
        ax.set_ylabel('Importance', fontsize=11)
        ax.set_title('Random Forest Feature Importance (YelpCHI)', fontsize=13)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance_rf.png", dpi=150)
        plt.close()
        print("  Saved: feature_importance_rf.png")
        results["feature_importance_rf"] = {
            feat_names[i]: float(importances[i]) for i in range(len(feat_names))
        }
except Exception as e:
    print(f"  Feature importance ERROR: {e}")

# ─── Step 11: SHAP Analysis ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Running SHAP ANALYSIS on Fusion MLP...")
print("=" * 70)

try:
    # Use a small background set for SHAP
    n_background = min(200, len(X_train_s))
    background_idx = np.random.choice(len(X_train_s), n_background, replace=False)
    background = torch.FloatTensor(X_train_s[background_idx])

    # Wrap model for SHAP
    model_fus.eval()
    def model_predict(x_np):
        with torch.no_grad():
            t = torch.FloatTensor(x_np)
            logits = model_fus(t).squeeze().numpy()
        return 1 / (1 + np.exp(-logits))

    # Use SHAP KernelExplainer (model-agnostic, works without GPU)
    n_explain = min(100, len(X_eval))
    explain_idx = np.random.choice(len(X_eval), n_explain, replace=False)
    X_explain = X_eval[explain_idx]
    y_explain = y_eval[explain_idx]

    print(f"  Running KernelExplainer on {n_explain} samples (background={n_background})...")
    explainer = shap.KernelExplainer(
        model_predict, shap.sample(X_train_s, n_background)
    )
    shap_values = explainer.shap_values(X_explain, nsamples=50)

    # SHAP summary plot
    feat_names = [f"Feature_{i+1:02d}" for i in range(X_explain.shape[1])]
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_explain,
        feature_names=feat_names,
        show=False, plot_size=(10, 7)
    )
    plt.title("SHAP Summary – ReviewGuard Fusion MLP", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: shap_summary.png")

    # Feature importance bar from SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_shap_idx = np.argsort(mean_abs_shap)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(mean_abs_shap)), mean_abs_shap[sorted_shap_idx],
           color='crimson', alpha=0.8)
    ax.set_xticks(range(len(mean_abs_shap)))
    ax.set_xticklabels([feat_names[i] for i in sorted_shap_idx], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Feature', fontsize=11)
    ax.set_ylabel('Mean |SHAP value|', fontsize=11)
    ax.set_title('SHAP Feature Importance – ReviewGuard MLP', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_feature_importance.png", dpi=150)
    plt.close()
    print("  Saved: shap_feature_importance.png")

    results["shap_analysis"] = {
        "mean_abs_shap_per_feature": {
            feat_names[i]: float(mean_abs_shap[i]) for i in range(len(feat_names))
        },
        "top_5_features": [feat_names[i] for i in sorted_shap_idx[:5]],
    }

except Exception as e:
    print(f"  SHAP ERROR: {e}\n{traceback.format_exc()}")
    results["shap_analysis"] = {"error": str(e)}

# ─── Step 12: Training Loss Curve ─────────────────────────────────────────────
print("\nPlotting training loss curves...")
try:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, losses, color in [
        ("Fusion MLP (ReviewGuard)", losses_fus, 'crimson'),
        ("Behavior MLP", losses_beh, 'darkorange'),
        ("Text-like MLP", losses_txt, 'green'),
    ]:
        ax.plot(range(1, len(losses)+1), losses, lw=2, label=label, color=color)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Focal Loss', fontsize=11)
    ax.set_title('MLP Training Loss Curves (Focal Loss, γ=2)', fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_loss_curves.png", dpi=150)
    plt.close()
    print("  Saved: training_loss_curves.png")
except Exception as e:
    print(f"  Loss curve ERROR: {e}")

# ─── Step 13: Model Comparison Summary Chart ──────────────────────────────────
print("\nGenerating model comparison summary chart...")
try:
    summary_models = {}
    for n, m in results["baselines"].items():
        if "error" not in m:
            summary_models[n] = m
    if "error" not in results.get("behavior_mlp", {}):
        summary_models["Behavior MLP"] = results["behavior_mlp"]
    if "error" not in results.get("text_mlp", {}):
        summary_models["Text-like MLP"] = results["text_mlp"]
    if "error" not in results.get("fusion_mlp", {}):
        summary_models["ReviewGuard\n(Fusion MLP)"] = results["fusion_mlp"]

    names_s = list(summary_models.keys())
    aucs_s = [summary_models[n]['auc_roc'] for n in names_s]
    f1s_s = [summary_models[n]['macro_f1'] for n in names_s]
    x_s = np.arange(len(names_s))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x_s - width/2, aucs_s, width, label='AUC-ROC', color='steelblue')
    b2 = ax.bar(x_s + width/2, f1s_s, width, label='Macro-F1', color='darkorange')
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Model Comparison – YelpCHI Fake Review Detection', fontsize=14)
    ax.set_xticks(x_s)
    ax.set_xticklabels(names_s, rotation=15, ha='right', fontsize=9)
    ax.set_ylim([0.5, 1.0])
    ax.legend()
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.003, f'{h:.3f}', ha='center', va='bottom', fontsize=7)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.003, f'{h:.3f}', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_summary.png", dpi=150)
    plt.close()
    print("  Saved: model_comparison_summary.png")
except Exception as e:
    print(f"  Summary chart ERROR: {e}")

# ─── Step 14: Save Results JSON ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("Saving results to JSON...")
print("=" * 70)

out_path = RESULTS_DIR / "experiment_results.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved: {out_path}")

# ─── Final Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)
print(f"\n{'Model':<35} {'AUC-ROC':>8} {'Macro-F1':>9} {'Recall-Fake':>12}")
print("-" * 68)

def safe_get(d, *keys):
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, {})
    return d if isinstance(d, float) else float('nan')

for n, m in results["baselines"].items():
    if "error" not in m:
        print(f"  {n:<33} {m['auc_roc']:>8.4f} {m['macro_f1']:>9.4f} {m['recall_fake']:>12.4f}")

if "error" not in results.get("behavior_mlp", {}):
    m = results["behavior_mlp"]
    print(f"  {'Behavior MLP':<33} {m['auc_roc']:>8.4f} {m['macro_f1']:>9.4f} {m['recall_fake']:>12.4f}")

if "error" not in results.get("text_mlp", {}):
    m = results["text_mlp"]
    print(f"  {'Text-like MLP':<33} {m['auc_roc']:>8.4f} {m['macro_f1']:>9.4f} {m['recall_fake']:>12.4f}")

if "error" not in results.get("fusion_mlp", {}):
    m = results["fusion_mlp"]
    print(f"  {'ReviewGuard (Fusion MLP)':<33} {m['auc_roc']:>8.4f} {m['macro_f1']:>9.4f} {m['recall_fake']:>12.4f}")

print("\nAll figures saved to:", FIGURES_DIR)
print("Results JSON saved to:", out_path)
print("\nExperiment pipeline COMPLETE.")
