"""
Evaluate all saved models on the Jan-Feb 2026 care-label test set.

Steps:
  1. Extract test features for all 14 backbone x resolution combos
  2. Evaluate 14 linear probe checkpoints
  3. Evaluate 4 fine-tuned checkpoints (vitb16, vitl16 x 518, 714)
  4. Evaluate ML classifiers (SVM, CatBoost, LightGBM) on test features
  5. Write results JSON and append to archive summary

Usage:
  python evaluate_test.py                       # full run
  python evaluate_test.py --skip-extract        # reuse cached test features
  python evaluate_test.py --models vitb16 vitl16 --resolutions 518
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModel

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cached_features"
CKPT_DIR = SCRIPT_DIR / "checkpoints"
ML_DIR = SCRIPT_DIR / "ml_results"
ARCHIVE_MD = SCRIPT_DIR.parent.parent / "archive" / "01-dinov3-care-label-linear-probe-summary.md"

TEST_ROOT = SCRIPT_DIR.parent.parent / "resources" / "apparel_supreme_jan_to_feb_2026_care_label"
TEST_META = TEST_ROOT / "test" / "metadata.csv"
TEST_IMG_DIR = TEST_ROOT / "test" / "camera" / "care_label" / "0"

MODEL_VARIANTS = {
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "convnext_tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "convnext_small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "convnext_base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "convnext_large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
}
RESOLUTIONS = [518, 714]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TARGET_FPRS = [0.005, 0.01, 0.02, 0.05, 0.10]
TARGET_FPR_NAMES = ["0.5%", "1%", "2%", "5%", "10%"]

FINETUNE_MODELS = ["vitb16", "vitl16"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_test_metadata():
    df = pd.read_csv(TEST_META)
    df = df[df["internal_merged_result_id"].isin([1, 3])].copy()
    df["label"] = (df["internal_merged_result_id"] == 3).astype(int)

    records = []
    for _, row in df.iterrows():
        uuid = row["session_uuid"]
        matches = sorted(TEST_IMG_DIR.glob(f"{uuid}.macro.care_label.*.jpg"))
        if matches:
            records.append({
                "session_uuid": uuid,
                "image_path": str(matches[0]),
                "label": row["label"],
            })

    n_pos = sum(r["label"] for r in records)
    n_neg = len(records) - n_pos
    print(f"Test metadata: {len(df)} rows, {len(records)} with images")
    print(f"  Authentic (0): {n_neg}  |  Not-authentic (1): {n_pos}  |  Ratio: {n_neg / max(n_pos, 1):.1f}:1")
    return records


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


def build_transform(resolution):
    return transforms.Compose([
        transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Metrics (same as train_care_label.py)
# ---------------------------------------------------------------------------

def compute_tpr_at_fprs(y_true, y_score):
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_score)
    results = {}
    for target_fpr, name in zip(TARGET_FPRS, TARGET_FPR_NAMES):
        distances = np.abs(fpr_arr - target_fpr)
        min_dist = distances.min()
        tied = np.where(distances == min_dist)[0]
        idx = tied[np.argmax(tpr_arr[tied])]
        thresh_idx = min(idx, len(thresholds) - 1)
        results[name] = {
            "tpr": float(tpr_arr[idx]),
            "actual_fpr": float(fpr_arr[idx]),
            "threshold": float(thresholds[thresh_idx]),
        }
    return results


def compute_all_metrics(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return {"auc_roc": 0.0, "auc_pr": 0.0, "tpr_at_fpr": {}}
    return {
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "auc_pr": float(average_precision_score(y_true, y_score)),
        "tpr_at_fpr": compute_tpr_at_fprs(y_true, y_score),
    }


def apply_val_thresholds(y_true, y_score, val_thresholds):
    """Apply thresholds chosen on the val set to test scores.
    Returns actual TPR and FPR on the test set at each val threshold."""
    results = {}
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    for fpr_name in TARGET_FPR_NAMES:
        thresh = val_thresholds.get(fpr_name)
        if thresh is None:
            continue
        preds = (y_score >= thresh).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        tpr = float(tp / max(n_pos, 1))
        fpr = float(fp / max(n_neg, 1))
        results[fpr_name] = {"tpr": tpr, "fpr": fpr, "threshold": float(thresh)}
    return results


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.head(x).squeeze(-1)


class DINOv3Classifier(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float32, attn_implementation="sdpa",
        )
        embed_dim = self.backbone.config.hidden_size
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        return self.head(outputs.pooler_output).squeeze(-1)


# ---------------------------------------------------------------------------
# Step 1: Extract test features
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_features_batch(model, dataloader, device):
    all_features, all_labels, all_uuids = [], [], []
    for batch_idx, (pixels, labels, uuids) in enumerate(dataloader):
        pixels = pixels.to(device, dtype=torch.bfloat16)
        outputs = model(pixel_values=pixels)
        feats = outputs.pooler_output.float().cpu()
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        all_features.append(feats)
        all_labels.extend(labels)
        all_uuids.extend(uuids)
        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            print(f"    Batch {batch_idx + 1}/{len(dataloader)}", flush=True)
    return torch.cat(all_features), torch.tensor(all_labels, dtype=torch.long), all_uuids


def extract_test_features(model_keys, resolutions, records, device, batch_size, num_workers):
    print("\n" + "=" * 60)
    print("STEP 1: Extract test features")
    print("=" * 60)

    for model_key in model_keys:
        model_id = MODEL_VARIANTS[model_key]
        for res in resolutions:
            prefix = f"test_{model_key}_{res}"
            feat_path = CACHE_DIR / f"{prefix}_features.pt"
            if feat_path.exists():
                print(f"  [SKIP] {prefix} already cached")
                continue

            print(f"\n  [{model_key} @ {res}] Loading {model_id} ...")
            load_kwargs = {"dtype": torch.bfloat16}
            if not model_key.startswith("convnext"):
                load_kwargs["attn_implementation"] = "sdpa"
            model = AutoModel.from_pretrained(model_id, **load_kwargs).to(device).eval()

            transform = build_transform(res)
            ds = ImageDataset(records, transform)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)

            t0 = time.time()
            features, labels, uuids = extract_features_batch(model, dl, device)
            print(f"  Done: {features.shape} in {time.time() - t0:.1f}s")

            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(features, feat_path)
            torch.save(labels, CACHE_DIR / f"{prefix}_labels.pt")
            torch.save(uuids, CACHE_DIR / f"{prefix}_uuids.pt")

            del model
            torch.cuda.empty_cache()


def load_test_features(model_key, resolution):
    prefix = f"test_{model_key}_{resolution}"
    features = torch.load(CACHE_DIR / f"{prefix}_features.pt", weights_only=True)
    labels = torch.load(CACHE_DIR / f"{prefix}_labels.pt", weights_only=True)
    uuids = torch.load(CACHE_DIR / f"{prefix}_uuids.pt", weights_only=False)
    nan_mask = torch.isnan(features).any(dim=1) | torch.isinf(features).any(dim=1)
    if nan_mask.any():
        keep = ~nan_mask
        features, labels = features[keep], labels[keep]
        uuids = [u for u, k in zip(uuids, keep.tolist()) if k]
    return features, labels, uuids


# ---------------------------------------------------------------------------
# Step 2: Evaluate linear probes
# ---------------------------------------------------------------------------

def _extract_val_thresholds(tpr_at_fpr_dict):
    """Pull threshold values from a val-set tpr_at_fpr dict."""
    return {name: d["threshold"] for name, d in tpr_at_fpr_dict.items() if "threshold" in d}


def eval_linear_probes(model_keys, resolutions, device):
    print("\n" + "=" * 60)
    print("STEP 2: Evaluate linear probes")
    print("=" * 60)

    results = {}
    fixed_thresh_results = {}
    for model_key in model_keys:
        for res in resolutions:
            ckpt_path = CKPT_DIR / f"{model_key}_{res}_linear_probe_best.pt"
            if not ckpt_path.exists():
                continue

            features, labels, _ = load_test_features(model_key, res)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            embed_dim = features.shape[1]
            head = LinearHead(embed_dim).to(device)
            head.load_state_dict(ckpt["model_state_dict"])
            head.eval()

            with torch.no_grad():
                logits = head(features.to(device))
                scores = torch.sigmoid(logits).cpu().numpy()

            y_true = labels.numpy()
            metrics = compute_all_metrics(y_true, scores)
            tag = f"{model_key}_{res}"
            results[tag] = metrics

            val_thresholds = _extract_val_thresholds(ckpt.get("metrics", {}).get("tpr_at_fpr", {}))
            fixed_thresh_results[tag] = apply_val_thresholds(y_true, scores, val_thresholds)

            tpr_2 = metrics["tpr_at_fpr"].get("2%", {}).get("tpr", 0)
            print(f"  {tag:<28s} AUC-ROC={metrics['auc_roc']:.4f}  AUC-PR={metrics['auc_pr']:.4f}  TPR@2%={tpr_2:.4f}")

    return results, fixed_thresh_results


# ---------------------------------------------------------------------------
# Step 3: Evaluate fine-tuned models
# ---------------------------------------------------------------------------

@torch.inference_mode()
def eval_finetune(model_keys, resolutions, records, device, batch_size, num_workers):
    print("\n" + "=" * 60)
    print("STEP 3: Evaluate fine-tuned models")
    print("=" * 60)

    results = {}
    fixed_thresh_results = {}
    ft_keys = [k for k in model_keys if k in FINETUNE_MODELS]
    for model_key in ft_keys:
        model_id = MODEL_VARIANTS[model_key]
        for res in resolutions:
            ckpt_path = CKPT_DIR / f"{model_key}_{res}_finetune_best.pt"
            if not ckpt_path.exists():
                continue

            print(f"  [{model_key} @ {res} finetune] Loading checkpoint ...")
            model = DINOv3Classifier(model_id)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device).eval()

            transform = build_transform(res)
            ds = ImageDataset(records, transform)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)

            all_scores, all_labels = [], []
            for pixels, labs, _ in dl:
                pixels = pixels.to(device)
                logits = model(pixels)
                all_scores.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(labs)

            scores = np.concatenate(all_scores)
            y_true = np.array(all_labels)
            metrics = compute_all_metrics(y_true, scores)
            tag = f"{model_key}_{res}_finetune"
            results[tag] = metrics

            val_thresholds = _extract_val_thresholds(ckpt.get("metrics", {}).get("tpr_at_fpr", {}))
            fixed_thresh_results[tag] = apply_val_thresholds(y_true, scores, val_thresholds)

            tpr_2 = metrics["tpr_at_fpr"].get("2%", {}).get("tpr", 0)
            print(f"  {tag:<28s} AUC-ROC={metrics['auc_roc']:.4f}  AUC-PR={metrics['auc_pr']:.4f}  TPR@2%={tpr_2:.4f}")

            del model
            torch.cuda.empty_cache()

    return results, fixed_thresh_results


# ---------------------------------------------------------------------------
# Step 4: Evaluate ML classifiers (SVM, CatBoost, LightGBM)
# ---------------------------------------------------------------------------

def _load_ml_val_thresholds(prefix, ml_name):
    """Load val-set thresholds from the ML metrics JSON."""
    metrics_path = ML_DIR / f"{prefix}_{ml_name}.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        d = json.load(f)
    return _extract_val_thresholds(d.get("metrics", {}).get("tpr_at_fpr", {}))


def eval_ml_classifiers(model_keys, resolutions):
    print("\n" + "=" * 60)
    print("STEP 4: Evaluate ML classifiers")
    print("=" * 60)

    results = {}
    fixed_thresh_results = {}
    for model_key in model_keys:
        for res in resolutions:
            prefix = f"{model_key}_{res}"
            features, labels, _ = load_test_features(model_key, res)
            X = features.numpy()
            y_true = labels.numpy()

            for ml_name, load_fn in [("svm", _eval_svm), ("catboost", _eval_catboost), ("lgbm", _eval_lgbm)]:
                tag = f"{prefix}_{ml_name}"
                try:
                    scores = load_fn(prefix, X)
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"  [WARN] {tag}: {e}")
                    continue

                metrics = compute_all_metrics(y_true, scores)
                results[tag] = metrics

                val_thresholds = _load_ml_val_thresholds(prefix, ml_name)
                fixed_thresh_results[tag] = apply_val_thresholds(y_true, scores, val_thresholds)

                tpr_2 = metrics["tpr_at_fpr"].get("2%", {}).get("tpr", 0)
                print(f"  {tag:<36s} AUC-ROC={metrics['auc_roc']:.4f}  AUC-PR={metrics['auc_pr']:.4f}  TPR@2%={tpr_2:.4f}")

    return results, fixed_thresh_results


def _eval_svm(prefix, X):
    import joblib
    path = ML_DIR / f"{prefix}_svm.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    d = joblib.load(path)
    X_scaled = d["scaler"].transform(X)
    return d["model"].predict_proba(X_scaled)[:, 1]


def _eval_catboost(prefix, X):
    import catboost as cb
    path = ML_DIR / f"{prefix}_catboost.cbm"
    if not path.exists():
        raise FileNotFoundError(path)
    clf = cb.CatBoostClassifier()
    clf.load_model(str(path))
    return clf.predict_proba(X)[:, 1]


def _eval_lgbm(prefix, X):
    import lightgbm as lgb
    path = ML_DIR / f"{prefix}_lgbm.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    booster = lgb.Booster(model_file=str(path))
    return booster.predict(X)


# ---------------------------------------------------------------------------
# Step 5: Write results
# ---------------------------------------------------------------------------

def write_results(lp_results, ft_results, ml_results,
                   lp_fixed, ft_fixed, ml_fixed, n_images, n_pos):
    print("\n" + "=" * 60)
    print("STEP 5: Writing results")
    print("=" * 60)

    all_results = {
        "test_data": {"n_images": n_images, "n_positive": n_pos, "n_negative": n_images - n_pos},
        "linear_probe": lp_results,
        "finetune": ft_results,
        "ml_classifiers": ml_results,
        "fixed_threshold": {"linear_probe": lp_fixed, "finetune": ft_fixed, "ml_classifiers": ml_fixed},
    }
    out_path = ML_DIR / "test_results.json"
    ML_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved raw results to {out_path}")

    _append_to_archive(lp_results, ft_results, ml_results,
                       lp_fixed, ft_fixed, ml_fixed, n_images, n_pos)


def _tpr(m, fpr_name):
    return m.get("tpr_at_fpr", {}).get(fpr_name, {}).get("tpr", 0)


def _ft_tpr(fixed, fpr_name):
    return fixed.get(fpr_name, {}).get("tpr", 0)

def _ft_fpr(fixed, fpr_name):
    return fixed.get(fpr_name, {}).get("fpr", 0)


def _append_to_archive(lp_results, ft_results, ml_results,
                        lp_fixed, ft_fixed, ml_fixed, n_images, n_pos):
    lines = []
    lines.append("\n---\n")
    lines.append("## 4. Test Set Evaluation — Jan-Feb 2026\n")
    lines.append(f"**Dataset:** `apparel_supreme_jan_to_feb_2026_care_label/test/`\n")
    lines.append(f"| Stat | Value |")
    lines.append(f"|------|-------|")
    lines.append(f"| Images with labels | {n_images} |")
    lines.append(f"| Not-authentic (positive) | {n_pos} |")
    lines.append(f"| Authentic (negative) | {n_images - n_pos} |")
    lines.append(f"| Imbalance ratio | {(n_images - n_pos) / max(n_pos, 1):.1f}:1 |")
    lines.append("")

    # Linear probes
    lines.append("### Linear Probes (sorted by AUC-ROC)\n")
    lines.append("| Model | AUC-ROC | AUC-PR | TPR@0.5% | TPR@1% | TPR@2% | TPR@5% | TPR@10% |")
    lines.append("|-------|---------|--------|----------|--------|--------|--------|---------|")
    for tag in sorted(lp_results, key=lambda t: lp_results[t]["auc_roc"], reverse=True):
        m = lp_results[tag]
        lines.append(
            f"| {tag} | {m['auc_roc']:.4f} | {m['auc_pr']:.4f} "
            f"| {_tpr(m,'0.5%'):.4f} | {_tpr(m,'1%'):.4f} | {_tpr(m,'2%'):.4f} "
            f"| {_tpr(m,'5%'):.4f} | {_tpr(m,'10%'):.4f} |"
        )
    lines.append("")

    # Fine-tuned
    if ft_results:
        lines.append("### Fine-tuned Models\n")
        lines.append("| Model | AUC-ROC | AUC-PR | TPR@0.5% | TPR@1% | TPR@2% | TPR@5% | TPR@10% |")
        lines.append("|-------|---------|--------|----------|--------|--------|--------|---------|")
        for tag in sorted(ft_results, key=lambda t: ft_results[t]["auc_roc"], reverse=True):
            m = ft_results[tag]
            lines.append(
                f"| {tag} | {m['auc_roc']:.4f} | {m['auc_pr']:.4f} "
                f"| {_tpr(m,'0.5%'):.4f} | {_tpr(m,'1%'):.4f} | {_tpr(m,'2%'):.4f} "
                f"| {_tpr(m,'5%'):.4f} | {_tpr(m,'10%'):.4f} |"
            )
        lines.append("")

    # ML classifiers — show best per embedding
    if ml_results:
        lines.append("### ML Classifiers — Best per Embedding (sorted by AUC-ROC)\n")
        lines.append("| Embedding | Best ML | AUC-ROC | AUC-PR | TPR@2% | TPR@5% | TPR@10% |")
        lines.append("|-----------|---------|---------|--------|--------|--------|---------|")

        by_embed = {}
        for tag, m in ml_results.items():
            parts = tag.rsplit("_", 1)
            embed_key, ml_type = parts[0], parts[1]
            if embed_key not in by_embed or m["auc_roc"] > by_embed[embed_key][1]["auc_roc"]:
                by_embed[embed_key] = (ml_type, m)

        for embed_key in sorted(by_embed, key=lambda k: by_embed[k][1]["auc_roc"], reverse=True):
            ml_type, m = by_embed[embed_key]
            lines.append(
                f"| {embed_key} | {ml_type.upper()} | {m['auc_roc']:.4f} | {m['auc_pr']:.4f} "
                f"| {_tpr(m,'2%'):.4f} | {_tpr(m,'5%'):.4f} | {_tpr(m,'10%'):.4f} |"
            )
        lines.append("")

    # ---------------------------------------------------------------
    # Fixed-threshold evaluation (val thresholds applied to test set)
    # ---------------------------------------------------------------
    all_fixed = {}
    for tag, d in lp_fixed.items():
        all_fixed[f"{tag} (linear)"] = d
    for tag, d in ft_fixed.items():
        all_fixed[tag] = d
    for tag, d in ml_fixed.items():
        all_fixed[tag] = d

    if all_fixed:
        lines.append("### Fixed-Threshold Evaluation (val-set thresholds on test data)\n")
        lines.append("Thresholds calibrated on the validation set are applied unchanged to test scores.")
        lines.append("This shows realistic deployment performance including probability shift.\n")
        lines.append("| Model | Val FPR target | Actual test FPR | Actual test TPR |")
        lines.append("|-------|---------------|-----------------|-----------------|")

        # Build rows sorted by tag, then by FPR target
        for tag in sorted(all_fixed.keys()):
            d = all_fixed[tag]
            for fpr_name in TARGET_FPR_NAMES:
                if fpr_name not in d:
                    continue
                lines.append(
                    f"| {tag} | {fpr_name} "
                    f"| {_ft_fpr(d, fpr_name):.2%} "
                    f"| {_ft_tpr(d, fpr_name):.2%} |"
                )
        lines.append("")

        # Summary: best TPR at each FPR target
        lines.append("### Best Fixed-Threshold TPR at Each FPR Target\n")
        lines.append("| Val FPR Target | Best Method | Test FPR | Test TPR |")
        lines.append("|----------------|------------|----------|----------|")
        for fpr_name in TARGET_FPR_NAMES:
            best_tag, best_tpr, best_fpr = None, -1, 0
            for tag, d in all_fixed.items():
                t = _ft_tpr(d, fpr_name)
                if t > best_tpr:
                    best_tpr = t
                    best_fpr = _ft_fpr(d, fpr_name)
                    best_tag = tag
            if best_tag:
                lines.append(f"| {fpr_name} | {best_tag} | {best_fpr:.2%} | {best_tpr:.2%} |")
        lines.append("")

    # Best overall at each FPR (ROC-curve based, for reference)
    all_tagged = {}
    for tag, m in lp_results.items():
        all_tagged[f"{tag} (linear)"] = m
    for tag, m in ft_results.items():
        all_tagged[tag] = m
    for tag, m in ml_results.items():
        all_tagged[tag] = m

    if all_tagged:
        lines.append("### Best Overall at Each FPR (test ROC curve, for reference)\n")
        lines.append("| FPR Budget | Best Method | TPR |")
        lines.append("|------------|------------|-----|")
        for fpr_name in TARGET_FPR_NAMES:
            best_tag, best_tpr = None, -1
            for tag, m in all_tagged.items():
                t = _tpr(m, fpr_name)
                if t > best_tpr:
                    best_tpr = t
                    best_tag = tag
            lines.append(f"| {fpr_name} | {best_tag} | {best_tpr:.1%} |")
        lines.append("")

    text = "\n".join(lines)
    with open(ARCHIVE_MD, "a") as f:
        f.write(text)
    print(f"  Appended results to {ARCHIVE_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on test set")
    parser.add_argument("--models", nargs="+", default=list(MODEL_VARIANTS.keys()),
                        choices=list(MODEL_VARIANTS.keys()))
    parser.add_argument("--resolutions", nargs="+", type=int, default=RESOLUTIONS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-extract", action="store_true", help="Skip feature extraction (use cached)")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    records = load_test_metadata()
    n_images = len(records)
    n_pos = sum(r["label"] for r in records)

    if not args.skip_extract:
        extract_test_features(args.models, args.resolutions, records, args.device,
                              args.batch_size, args.num_workers)

    lp_results, lp_fixed = eval_linear_probes(args.models, args.resolutions, args.device)
    ft_results, ft_fixed = eval_finetune(args.models, args.resolutions, records, args.device,
                                          args.batch_size, args.num_workers)
    ml_results, ml_fixed = eval_ml_classifiers(args.models, args.resolutions)

    write_results(lp_results, ft_results, ml_results,
                  lp_fixed, ft_fixed, ml_fixed, n_images, n_pos)
    print("\nDone.")


if __name__ == "__main__":
    main()
