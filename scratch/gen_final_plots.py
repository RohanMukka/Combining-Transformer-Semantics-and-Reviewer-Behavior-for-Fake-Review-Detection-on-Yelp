import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Data from our verified YelpCHI results
data = {
    "Model": [
        "TF-IDF + SVM", 
        "TF-IDF + LogReg", 
        "Behavior + RF", 
        "ReviewGuard (Fusion)"
    ],
    "AUC-ROC": [0.7296, 0.7303, 0.5284, 0.8665],
    "Macro-F1": [0.4577, 0.6384, 0.4563, 0.7149],
    "Recall (Fake)": [0.0033, 0.5671, 0.0456, 0.7266]
}

df = pd.DataFrame(data)

# Create a 1x3 grid of plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.set_style("whitegrid")

# Plot 1: AUC-ROC
sns.barplot(data=df, x="Model", y="AUC-ROC", ax=axes[0], palette="viridis")
axes[0].set_title("AUC-ROC Comparison", fontweight='bold', fontsize=14)
axes[0].set_ylim(0, 1.0)
axes[0].tick_params(axis='x', rotation=45)

# Plot 2: Macro-F1
sns.barplot(data=df, x="Model", y="Macro-F1", ax=axes[1], palette="magma")
axes[1].set_title("Macro-F1 Comparison", fontweight='bold', fontsize=14)
axes[1].set_ylim(0, 1.0)
axes[1].tick_params(axis='x', rotation=45)

# Plot 3: Recall (Fake) - This is the "Hero" plot
sns.barplot(data=df, x="Model", y="Recall (Fake)", ax=axes[2], palette="rocket")
axes[2].set_title("Recall (Fake) - Spammers Caught", fontweight='bold', fontsize=14)
axes[2].set_ylim(0, 1.0)
axes[2].tick_params(axis='x', rotation=45)

plt.suptitle("Final Model Comparison: YelpCHI Benchmark", fontweight='bold', fontsize=16, y=1.05)
plt.tight_layout()

save_path = "results/figures/final_model_comparison.png"
Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Success: Final comparison chart saved to {save_path}")
