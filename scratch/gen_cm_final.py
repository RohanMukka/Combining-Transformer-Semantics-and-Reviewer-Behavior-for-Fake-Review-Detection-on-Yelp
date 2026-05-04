import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Reconstructed counts from the JSON metrics
# Total Test: 9,190 | Total Fake: 1,536 | Total Genuine: 7,654
cm = np.array([
    [6282, 1372],  # Genuine (TN, FP)
    [420, 1116]    # Fake (FN, TP)
])

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Genuine", "Fake"])
disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format='d')

# Put the AUC-ROC score clearly in the title so it's not inside a box
ax.set_title("ReviewGuard Fusion Model (YelpCHI)\nAUC-ROC: 0.867", fontweight='bold', pad=20, fontsize=12)
ax.set_xlabel("Predicted Label", fontweight='bold')
ax.set_ylabel("True Label", fontweight='bold')

save_path = "results/figures/cm_reviewguard_fusion.png"
plt.tight_layout()
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Success: Final confusion matrix saved to {save_path}")
