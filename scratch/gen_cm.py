import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# Load the latest test results for ReviewGuard
probs_path = Path("models/fusion/yelpchi_test_probs.npy")
labels_path = Path("models/fusion/yelpchi_test_labels.npy")

if probs_path.exists() and labels_path.exists():
    probs = np.load(probs_path)
    labels = np.load(labels_path)
    
    # Convert probabilities to binary predictions (threshold 0.5)
    preds = (probs > 0.5).astype(int)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Genuine", "Fake"])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("ReviewGuard Fusion Model\nConfusion Matrix (YelpCHI)", fontweight='bold')
    
    save_path = Path("results/figures/cm_reviewguard_fusion.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Success: Saved confusion matrix to {save_path}")
else:
    print("Error: Could not find test probabilities or labels.")
