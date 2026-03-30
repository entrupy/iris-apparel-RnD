#!/root/.venv/bin/python
"""
Recalculate test metrics with positive=authentic convention.

TPR = authentic pass rate (% of authentic items correctly passed)
FPR = fake miss rate (% of fakes wrongly passed as authentic)

At each target FPR (fake miss rate), find the threshold on the TEST ROC curve
and report the TPR (auth pass rate).

Outputs updated tables for the archive.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MODEL_VARIANTS, IMAGENET_MEAN, IMAGENET_STD,
    DINOv3Classifier, build_transform, load_metadata,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REGION = "care_label"
CACHE_DIR = SCRIPT_DIR / "cached_features" / REGION
CKPT_DIR = SCRIPT_DIR / "checkpoints" / REGION
ML_DIR = SCRIPT_DIR / "ml_results" / REGION

TARGET_FPRS_NEW = [0.005, 0.01, 0.02, 0.05, 0.10]
TARGET_FPR_NAMES = ["0.5%", "1%", "2%", "5%", "10%"]

FINETUNE_MODELS = ["vitb16", "vitl16"]
RESOLUTIONS = [518, 714]


class LinearHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.head = nn.Linear(in_dim, 1)
    def forward(self, x):
        return self.head(x).squeeze(-1)


class ImageDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        return self.transform(img), rec["label"], rec["session_uuid"]


def compute_metrics_auth_positive(y_true_orig, y_score_orig):
    """Compute metrics with positive=authentic.

    y_true_orig: 0=authentic, 1=fake (original labels)
    y_score_orig: higher = more likely fake (original scores)

    Flips to: y_true_new=1 for authentic, y_score_new=1-score (higher=more authentic)
    Then TPR = auth pass rate, FPR = fake miss rate.
    """
    y_true_new = 1 - np.asarray(y_true_orig)
    y_score_new = 1.0 - np.asarray(y_score_orig)

    if len(np.unique(y_true_new)) < 2:
        return {"auc_roc": 0.0, "tpr_at_fpr": {}}

    auc_roc = float(roc_auc_score(y_true_new, y_score_new))

    fpr_arr, tpr_arr, thresholds = roc_curve(y_true_new, y_score_new)
    results = {}
    for target_fpr, name in zip(TARGET_FPRS_NEW, TARGET_FPR_NAMES):
        distances = np.abs(fpr_arr - target_fpr)
        min_dist = distances.min()
        tied = np.where(distances == min_dist)[0]
        idx = tied[np.argmax(tpr_arr[tied])]
        results[name] = {
            "tpr": float(tpr_arr[idx]),
            "fpr": float(fpr_arr[idx]),
        }

    return {"auc_roc": auc_roc, "tpr_at_fpr": results}


def load_test_features(feat_key):
    prefix = f"test_{feat_key}"
    features = torch.load(CACHE_DIR / f"{prefix}_features.pt", weights_only=True)
    labels = torch.load(CACHE_DIR / f"{prefix}_labels.pt", weights_only=True)
    uuid_path = CACHE_DIR / f"{prefix}_uuids.pt"
    uuids = torch.load(uuid_path, weights_only=False) if uuid_path.exists() else []
    nan_mask = torch.isnan(features).any(dim=1) | torch.isinf(features).any(dim=1)
    if nan_mask.any():
        keep = ~nan_mask
        features, labels = features[keep], labels[keep]
        uuids = [u for u, k in zip(uuids, keep.tolist()) if k]
    return features, labels, uuids


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    records = load_metadata("care_label", split="test")
    n_images = len(records)
    n_pos_fake = sum(r["label"] for r in records)
    n_auth = n_images - n_pos_fake
    print(f"Test: {n_images} images, {n_auth} authentic, {n_pos_fake} fake")

    all_results = {}

    # Discover all cached test features
    feat_files = sorted(CACHE_DIR.glob("test_*_features.pt"))
    all_feature_keys = []
    for f in feat_files:
        key = f.stem.replace("test_", "").replace("_features", "")
        all_feature_keys.append(key)
    print(f"Found {len(all_feature_keys)} cached test feature sets: {all_feature_keys}")

    # --- Linear probes ---
    print("\n=== Linear Probes ===")
    for model_key in MODEL_VARIANTS:
        for res in RESOLUTIONS:
            ckpt_path = CKPT_DIR / f"{model_key}_{res}_linear_probe_best.pt"
            feat_key = f"{model_key}_{res}"
            if not ckpt_path.exists() or feat_key not in all_feature_keys:
                continue

            features, labels, _ = load_test_features(feat_key)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            head = LinearHead(features.shape[1]).to(device)
            head.load_state_dict(ckpt["model_state_dict"])
            head.eval()

            with torch.no_grad():
                scores = torch.sigmoid(head(features.to(device))).cpu().numpy()
            y_true = labels.numpy()

            metrics = compute_metrics_auth_positive(y_true, scores)
            tag = f"{model_key}_{res}_linear"
            all_results[tag] = metrics
            fpr2 = metrics["tpr_at_fpr"].get("2%", {})
            print(f"  {tag:<30s} AUC={metrics['auc_roc']:.4f}  TPR@2%FPR={fpr2.get('tpr',0):.2%} (actual FPR={fpr2.get('fpr',0):.2%})")

    # --- Fine-tuned (use cached finetuned features + linear head) ---
    print("\n=== Fine-tuned (checkpoint inference) ===")
    for model_key in FINETUNE_MODELS:
        model_id = MODEL_VARIANTS[model_key]
        for res in RESOLUTIONS:
            ckpt_path = CKPT_DIR / f"{model_key}_{res}_finetune_best.pt"
            if not ckpt_path.exists():
                continue

            print(f"  Loading {model_key}_{res}_finetune ...")
            model = DINOv3Classifier(model_id, freeze_backbone=False)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device).eval()

            transform = build_transform(res)
            ds = ImageDataset(records, transform)
            dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

            all_scores, all_labels = [], []
            with torch.no_grad():
                for pixels, labs, _ in dl:
                    logits = model(pixels.to(device))
                    all_scores.append(torch.sigmoid(logits).cpu().numpy())
                    all_labels.extend(labs)

            scores = np.concatenate(all_scores)
            y_true = np.array(all_labels)

            metrics = compute_metrics_auth_positive(y_true, scores)
            tag = f"{model_key}_{res}_finetune"
            all_results[tag] = metrics
            fpr2 = metrics["tpr_at_fpr"].get("2%", {})
            print(f"  {tag:<30s} AUC={metrics['auc_roc']:.4f}  TPR@2%FPR={fpr2.get('tpr',0):.2%} (actual FPR={fpr2.get('fpr',0):.2%})")

            del model
            torch.cuda.empty_cache()

    # --- ML classifiers (on all feature keys including finetuned) ---
    print("\n=== ML Classifiers ===")
    for feat_key in all_feature_keys:
        features, labels, _ = load_test_features(feat_key)
        X = features.numpy()
        y_true = labels.numpy()

        for ml_name, load_fn in [("svm", _load_svm), ("catboost", _load_catboost), ("lgbm", _load_lgbm)]:
            try:
                scores = load_fn(feat_key, X)
            except (FileNotFoundError, Exception):
                continue

            metrics = compute_metrics_auth_positive(y_true, scores)
            tag = f"{feat_key}_{ml_name}"
            all_results[tag] = metrics
            fpr2 = metrics["tpr_at_fpr"].get("2%", {})
            print(f"  {tag:<36s} AUC={metrics['auc_roc']:.4f}  TPR@2%FPR={fpr2.get('tpr',0):.2%} (actual FPR={fpr2.get('fpr',0):.2%})")

    # Save
    out_path = SCRIPT_DIR / "ml_results" / "test_results_auth_positive.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print summary table
    print(f"\n{'='*90}")
    print("SUMMARY: Test ROC with positive=authentic (sorted by FPR@2%)")
    print(f"{'='*90}")
    print(f"{'Model':<36s} {'AUC':>6} {'TPR@0.5%':>9} {'TPR@1%':>8} {'TPR@2%':>8} {'TPR@5%':>8} {'TPR@10%':>9}")
    print("-" * 90)

    sorted_tags = sorted(all_results.keys(),
                         key=lambda t: all_results[t]["tpr_at_fpr"].get("2%", {}).get("fpr", 1))

    for tag in sorted_tags:
        m = all_results[tag]
        tf = m["tpr_at_fpr"]
        print(f"  {tag:<34s} {m['auc_roc']:>6.4f}"
              f" {tf.get('0.5%',{}).get('tpr',0):>8.2%}"
              f" {tf.get('1%',{}).get('tpr',0):>8.2%}"
              f" {tf.get('2%',{}).get('tpr',0):>8.2%}"
              f" {tf.get('5%',{}).get('tpr',0):>8.2%}"
              f" {tf.get('10%',{}).get('tpr',0):>8.2%}")


def _load_svm(prefix, X):
    import joblib
    path = ML_DIR / f"{prefix}_svm.joblib"
    if not path.exists():
        raise FileNotFoundError
    d = joblib.load(path)
    return d["model"].predict_proba(d["scaler"].transform(X))[:, 1]

def _load_catboost(prefix, X):
    import catboost as cb
    path = ML_DIR / f"{prefix}_catboost.cbm"
    if not path.exists():
        raise FileNotFoundError
    clf = cb.CatBoostClassifier()
    clf.load_model(str(path))
    return clf.predict_proba(X)[:, 1]

def _load_lgbm(prefix, X):
    import lightgbm as lgb
    path = ML_DIR / f"{prefix}_lgbm.txt"
    if not path.exists():
        raise FileNotFoundError
    return lgb.Booster(model_file=str(path)).predict(X)


if __name__ == "__main__":
    main()
