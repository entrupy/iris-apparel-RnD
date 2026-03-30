#!/root/.venv/bin/python
"""
Recalculate test metrics for ALL regions with positive=authentic convention,
and regenerate archive/02-dinov3-multi-region-summary.md.

TPR = % of authentic items correctly passed (true authentics)
FPR = % of fakes that slipped through (missed fakes)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_VARIANTS

SCRIPT_DIR = Path(__file__).resolve().parent
REGIONS = ["care_label", "front", "front_exterior_logo", "brand_tag"]
TARGET_FPRS = [0.005, 0.01, 0.02, 0.05, 0.10]
FPR_NAMES = ["0.5%", "1%", "2%", "5%", "10%"]

ML_NAMES = ["svm", "catboost", "lgbm"]


class LinearHead(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.head = torch.nn.Linear(in_dim, 1)
    def forward(self, x):
        return self.head(x).squeeze(-1)


def load_test_features(region, feat_key):
    cache_dir = SCRIPT_DIR / "cached_features" / region
    prefix = f"test_{feat_key}"
    features = torch.load(cache_dir / f"{prefix}_features.pt", weights_only=True)
    labels = torch.load(cache_dir / f"{prefix}_labels.pt", weights_only=True)
    nan_mask = torch.isnan(features).any(dim=1) | torch.isinf(features).any(dim=1)
    if nan_mask.any():
        features, labels = features[~nan_mask], labels[~nan_mask]
    return features, labels


def compute_auth_positive_metrics(y_true_orig, y_score_orig):
    """y_true_orig: 0=auth, 1=fake. y_score_orig: higher=more fake.
    Returns metrics with positive=authentic."""
    y_true_new = 1 - np.asarray(y_true_orig)
    y_score_new = 1.0 - np.asarray(y_score_orig)

    if len(np.unique(y_true_new)) < 2:
        return {"auc_roc": 0.0, "tpr_at_fpr": {n: {"tpr": 0, "fpr": 0} for n in FPR_NAMES}}

    auc = float(roc_auc_score(y_true_new, y_score_new))
    fpr_arr, tpr_arr, _ = roc_curve(y_true_new, y_score_new)

    results = {}
    for target, name in zip(TARGET_FPRS, FPR_NAMES):
        dists = np.abs(fpr_arr - target)
        tied = np.where(dists == dists.min())[0]
        idx = tied[np.argmax(tpr_arr[tied])]
        results[name] = {"tpr": float(tpr_arr[idx]), "fpr": float(fpr_arr[idx])}

    return {"auc_roc": auc, "tpr_at_fpr": results}


def get_scores_linear(region, feat_key, device):
    ckpt_dir = SCRIPT_DIR / "checkpoints" / region
    ckpt_path = ckpt_dir / f"{feat_key}_linear_probe_best.pt"
    if not ckpt_path.exists():
        return None, None
    features, labels = load_test_features(region, feat_key)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = LinearHead(features.shape[1]).to(device)
    head.load_state_dict(ckpt["model_state_dict"])
    head.eval()
    with torch.no_grad():
        scores = torch.sigmoid(head(features.to(device))).cpu().numpy()
    return labels.numpy(), scores


def get_scores_ml(region, feat_key, ml_name):
    ml_dir = SCRIPT_DIR / "ml_results" / region
    features, labels = load_test_features(region, feat_key)
    X = features.numpy()

    if ml_name == "svm":
        import joblib
        path = ml_dir / f"{feat_key}_svm.joblib"
        if not path.exists():
            return None, None
        d = joblib.load(path)
        scores = d["model"].predict_proba(d["scaler"].transform(X))[:, 1]
    elif ml_name == "catboost":
        import catboost as cb
        path = ml_dir / f"{feat_key}_catboost.cbm"
        if not path.exists():
            return None, None
        clf = cb.CatBoostClassifier()
        clf.load_model(str(path))
        scores = clf.predict_proba(X)[:, 1]
    elif ml_name == "lgbm":
        import lightgbm as lgb
        path = ml_dir / f"{feat_key}_lgbm.txt"
        if not path.exists():
            return None, None
        scores = lgb.Booster(model_file=str(path)).predict(X)
    else:
        return None, None

    return labels.numpy(), scores


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_region_results = {}

    for region in REGIONS:
        print(f"\n{'='*60}")
        print(f"  Region: {region}")
        print(f"{'='*60}")

        cache_dir = SCRIPT_DIR / "cached_features" / region
        feat_files = sorted(cache_dir.glob("test_*_features.pt"))
        feat_keys = [f.stem.replace("test_", "").replace("_features", "") for f in feat_files]

        region_results = {}

        for fk in feat_keys:
            # Linear probe
            y_true, scores = get_scores_linear(region, fk, device)
            if y_true is not None:
                tag = f"{fk}_linear"
                m = compute_auth_positive_metrics(y_true, scores)
                region_results[tag] = m

            # ML classifiers
            for ml in ML_NAMES:
                try:
                    y_true, scores = get_scores_ml(region, fk, ml)
                except Exception:
                    continue
                if y_true is None:
                    continue
                tag = f"{fk}_{ml}"
                m = compute_auth_positive_metrics(y_true, scores)
                region_results[tag] = m

        # Print top 25 by TPR@2%
        sorted_tags = sorted(region_results.keys(),
                             key=lambda t: region_results[t]["tpr_at_fpr"].get("2%", {}).get("tpr", 0),
                             reverse=True)

        print(f"\n  Top 25 by TPR @ 2% missed fakes:")
        print(f"  {'Model':<36s} {'AUC':>6} {'@0.5%':>7} {'@1%':>7} {'@2%':>7} {'@5%':>7} {'@10%':>7}")
        for tag in sorted_tags[:25]:
            m = region_results[tag]
            tf = m["tpr_at_fpr"]
            def g(n): return tf.get(n, {}).get("tpr", 0)
            print(f"  {tag:<36s} {m['auc_roc']:>6.4f} {g('0.5%'):>7.2%} {g('1%'):>7.2%} {g('2%'):>7.2%} {g('5%'):>7.2%} {g('10%'):>7.2%}")

        all_region_results[region] = region_results

    # Save
    out_path = SCRIPT_DIR / "ml_results" / "test_results_auth_positive_all_regions.json"
    with open(out_path, "w") as f:
        json.dump(all_region_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Generate markdown
    generate_archive(all_region_results)


def generate_archive(all_region_results):
    path = Path(__file__).resolve().parent.parent.parent / "archive" / "02-dinov3-multi-region-summary.md"

    # Read existing file up to "## Per-Region Results"
    with open(path) as f:
        old = f.read()

    # Keep everything before per-region results
    marker = "## Per-Region Results"
    idx = old.index(marker)
    header = old[:idx]

    lines = [header.rstrip() + "\n\n"]

    test_counts = {
        "care_label": (962, 33), "front": (1398, 41),
        "front_exterior_logo": (875, 35), "brand_tag": (1192, 29),
    }

    lines.append("## Per-Region Results (test ROC, sorted by TPR @ 2% missed)\n\n")
    lines.append("**TPR** = % of authentic items correctly passed (true authentics)\n")
    lines.append("**FPR** = % of fakes that slipped through (missed fakes)\n\n")
    lines.append("Computed from the **test ROC curve** directly. Top 25 models per region.\n\n")

    for region in REGIONS:
        n_imgs, n_fake = test_counts[region]
        results = all_region_results[region]
        sorted_tags = sorted(results.keys(),
                             key=lambda t: results[t]["tpr_at_fpr"].get("2%", {}).get("tpr", 0),
                             reverse=True)[:25]

        lines.append(f"### {region} (test: {n_imgs} imgs, {n_fake} fake)\n\n")
        lines.append("| Model | AUC | TPR @ 0.5% missed | TPR @ 1% missed | TPR @ 2% missed | TPR @ 5% missed | TPR @ 10% missed |\n")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |\n")

        for tag in sorted_tags:
            m = results[tag]
            tf = m["tpr_at_fpr"]
            def g(n): return tf.get(n, {}).get("tpr", 0)
            lines.append(
                f"| {tag} | {m['auc_roc']:.4f} "
                f"| {g('0.5%'):.2%} | {g('1%'):.2%} | {g('2%'):.2%} "
                f"| {g('5%'):.2%} | {g('10%'):.2%} |\n"
            )
        lines.append("\n")

    # Best per region summary
    lines.append("## Best per Region\n\n")
    lines.append("| Region | Best Method | AUC | TPR @ 2% missed | TPR @ 5% missed | TPR @ 10% missed |\n")
    lines.append("| --- | --- | --- | --- | --- | --- |\n")
    for region in REGIONS:
        results = all_region_results[region]
        best_tag = max(results.keys(),
                       key=lambda t: results[t]["tpr_at_fpr"].get("2%", {}).get("tpr", 0))
        m = results[best_tag]
        tf = m["tpr_at_fpr"]
        def g(n): return tf.get(n, {}).get("tpr", 0)
        lines.append(
            f"| {region} | {best_tag} | {m['auc_roc']:.4f} "
            f"| {g('2%'):.2%} | {g('5%'):.2%} | {g('10%'):.2%} |\n"
        )
    lines.append("\n")

    with open(path, "w") as f:
        f.writelines(lines)
    print(f"Updated {path}")


if __name__ == "__main__":
    main()
