"""Parse finetune nohup logs and plot epoch-vs-TPR curves.

- Same color for same FPR level
- Different marker for each model
"""

import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

NOHUP_DIR = Path(__file__).parent / "nohup"
OUTPUT_PATH = Path(__file__).parent / "nohup" / "finetune_tpr_at_fpr.png"

FILES = {
    "vitb16@518": "finetune_vitb16_518.out",
    "vitb16@714": "finetune_vitb16_714.out",
    "vitl16@518": "finetune_vitl16_518.out",
    "vitl16@714": "finetune_vitl16_714.out",
}

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\((\w+).*?"
    r"AUC=([\d.]+).*?"
    r"0\.5%=([\d.]+)\s*\|\s*1%=([\d.]+)\s*\|\s*2%=([\d.]+)\s*\|\s*5%=([\d.]+)\s*\|\s*10%=([\d.]+)"
)

FPR_LABELS = ["0.5%", "1%", "2%", "5%", "10%"]
FPR_COLORS = {
    "0.5%": "#e63946",
    "1%":   "#f77f00",
    "2%":   "#2a9d8f",
    "5%":   "#264653",
    "10%":  "#7209b7",
}

MODEL_MARKERS = {
    "vitb16@518": "o",
    "vitb16@714": "s",
    "vitl16@518": "D",
    "vitl16@714": "^",
}


def parse_log(path):
    text = path.read_text()
    rows = []
    for m in EPOCH_RE.finditer(text):
        epoch = int(m.group(1))
        phase = m.group(2)
        auc = float(m.group(3))
        tprs = {
            "0.5%": float(m.group(4)),
            "1%":   float(m.group(5)),
            "2%":   float(m.group(6)),
            "5%":   float(m.group(7)),
            "10%":  float(m.group(8)),
        }
        rows.append({"epoch": epoch, "phase": phase, "auc": auc, **tprs})
    return rows


def main():
    data = {}
    for model, fname in FILES.items():
        data[model] = parse_log(NOHUP_DIR / fname)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.suptitle("Finetune: TPR at Target FPR vs Epoch", fontsize=15, fontweight="bold", y=0.98)

    for ax, fpr_set, title in [
        (axes[0], ["0.5%", "1%", "2%"], "Low FPR (0.5%, 1%, 2%)"),
        (axes[1], ["5%", "10%"], "High FPR (5%, 10%)"),
    ]:
        for model, rows in data.items():
            epochs = [r["epoch"] for r in rows]
            marker = MODEL_MARKERS[model]
            for fpr in fpr_set:
                vals = [r[fpr] for r in rows]
                ax.plot(
                    epochs, vals,
                    color=FPR_COLORS[fpr],
                    marker=marker,
                    markersize=5,
                    linewidth=1.2,
                    alpha=0.85,
                )

        for model, rows in data.items():
            unfreeze_epoch = None
            for r in rows:
                if r["phase"] == "unfrozen":
                    unfreeze_epoch = r["epoch"]
                    break
            if unfreeze_epoch:
                ax.axvline(unfreeze_epoch, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 27)
        ax.set_ylim(-0.02, 0.72)

    axes[0].set_ylabel("TPR (True Positive Rate)", fontsize=12)

    color_handles = [
        mlines.Line2D([], [], color=FPR_COLORS[fpr], linewidth=2, label=f"FPR={fpr}")
        for fpr in FPR_LABELS
    ]
    marker_handles = [
        mlines.Line2D([], [], color="gray", marker=MODEL_MARKERS[m], linestyle="None",
                       markersize=7, label=m)
        for m in MODEL_MARKERS
    ]
    divider = [mlines.Line2D([], [], color="gray", linestyle="--", linewidth=0.8, label="unfreeze")]

    all_handles = color_handles + [mlines.Line2D([], [], alpha=0)] + marker_handles + divider
    fig.legend(
        handles=all_handles,
        loc="lower center",
        ncol=len(all_handles),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
