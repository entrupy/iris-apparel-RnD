#!/root/.venv/bin/python
"""Plot auth-positive ROC curves per region, zoomed at low FPR (missed fakes)."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REGIONS = ["care_label", "front", "front_exterior_logo", "brand_tag"]

data = json.load(open(SCRIPT_DIR / "ml_results" / "test_results_auth_positive_all_regions.json"))
partial = json.load(open(SCRIPT_DIR / "ml_results" / "test_partial_auth_positive.json"))
data["care_label"].update(partial)

FPR_TARGETS = [0.005, 0.01, 0.02, 0.05, 0.10]
FPR_NAMES = ["0.5%", "1%", "2%", "5%", "10%"]

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("TPR (true authentics) vs FPR (missed fakes) — Test ROC, auth-positive",
             fontsize=14, fontweight="bold")

for ax, region in zip(axes.flat, REGIONS):
    results = data[region]

    # Pick top 10 by TPR@2%
    top = sorted(results.keys(),
                 key=lambda t: results[t]["tpr_at_fpr"].get("2%", {}).get("tpr", 0),
                 reverse=True)[:10]

    for i, tag in enumerate(top):
        m = results[tag]
        tf = m["tpr_at_fpr"]
        fprs = [tf.get(n, {}).get("fpr", 0) for n in FPR_NAMES]
        tprs = [tf.get(n, {}).get("tpr", 0) for n in FPR_NAMES]

        marker = "o" if "partial" in tag else ("s" if "finetune" in tag else ("D" if "svm" in tag else "^"))
        ax.plot(fprs, tprs, marker=marker, markersize=5, linewidth=1.3, alpha=0.85,
                label=f"{tag} (AUC={m['auc_roc']:.3f})")

    ax.set_xlabel("FPR (% of fakes missed)", fontsize=11)
    ax.set_ylabel("TPR (% of authentic passed)", fontsize=11)
    ax.set_title(f"{region}", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.005, 0.12)
    ax.set_ylim(0, 1.0)
    ax.axvline(0.02, color="red", linestyle="--", alpha=0.3, linewidth=0.8, label="_2% missed")
    ax.axvline(0.05, color="orange", linestyle="--", alpha=0.3, linewidth=0.8, label="_5% missed")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="lower right")

    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = SCRIPT_DIR / "roc_auth_positive_per_region.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved to {out}")
