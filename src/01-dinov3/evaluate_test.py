#!/root/.venv/bin/python
"""
Evaluate all saved models on the Jan-Feb 2026 test set for a given region.

Steps:
  1. Extract test features for all 14 backbone x resolution combos (fp32)
  2. Evaluate 14 linear probe checkpoints
  3. Evaluate 4 fine-tuned checkpoints
  4. Evaluate ML classifiers (XGB, SVM, CatBoost, LGBM) on frozen features
  5. Extract finetuned backbone features + evaluate finetuned-ML
  6. Apply val-set thresholds to test scores (fixed-threshold evaluation)
  7. Write results JSON

Usage:
  python evaluate_test.py --region care_label
  python evaluate_test.py --region front --skip-extract
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from config import (
    MODEL_VARIANTS,
    REGIONS,
    RESOLUTIONS,
    TARGET_FPR_NAMES,
    DINOv3Classifier,
    ImageDatasetWithUUID,
    LinearHead,
    build_transform,
    cache_dir,
    ckpt_dir,
    compute_all_metrics,
    load_metadata,
    results_dir,
)

FINETUNE_MODELS = ["vitb16", "vitl16"]
FINETUNE_RESOLUTIONS = [518, 714]


def _extract_val_thresholds(tpr_at_fpr_dict):
    return {name: d["threshold"] for name, d in tpr_at_fpr_dict.items() if "threshold" in d}


def apply_val_thresholds(y_true, y_score, val_thresholds):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    results = {}
    for fpr_name in TARGET_FPR_NAMES:
        thresh = val_thresholds.get(fpr_name)
        if thresh is None:
            continue
        preds = (y_score >= thresh).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        results[fpr_name] = {
            "tpr": float(tp / max(n_pos, 1)),
            "fpr": float(fp / max(n_neg, 1)),
            "threshold": float(thresh),
        }
    return results


# ---------------------------------------------------------------------------
# Step 1: Extract test features (frozen pretrained backbones)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _extract_features_batch(model, dataloader, device):
    all_features, all_labels, all_uuids = [], [], []
    for batch_idx, (pixels, labels, uuids) in enumerate(dataloader):
        pixels = pixels.to(device, dtype=torch.float32)
        feats = model(pixel_values=pixels).pooler_output.float().cpu()
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        all_features.append(feats)
        all_labels.extend(labels)
        all_uuids.extend(uuids)
        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            print(f"    Batch {batch_idx + 1}/{len(dataloader)}", flush=True)
    return torch.cat(all_features), torch.tensor(all_labels, dtype=torch.long), all_uuids


def extract_test_features(region, model_keys, resolutions, records, device, batch_size, num_workers):
    print("\n" + "=" * 60)
    print("STEP 1: Extract test features (frozen backbones)")
    print("=" * 60)
    cdir = cache_dir(region)

    for model_key in model_keys:
        model_id = MODEL_VARIANTS[model_key]
        for res in resolutions:
            prefix = f"test_{model_key}_{res}"
            feat_path = cdir / f"{prefix}_features.pt"
            if feat_path.exists():
                print(f"  [SKIP] {prefix} already cached")
                continue

            print(f"\n  [{model_key} @ {res}] Loading {model_id} (fp32) ...")
            load_kwargs = {"torch_dtype": torch.float32}
            if not model_key.startswith("convnext"):
                load_kwargs["attn_implementation"] = "sdpa"
            model = AutoModel.from_pretrained(model_id, **load_kwargs).to(device).eval()

            transform = build_transform(res)
            ds = ImageDatasetWithUUID(records, transform)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)

            t0 = time.time()
            features, labels, uuids = _extract_features_batch(model, dl, device)
            print(f"  Done: {features.shape} in {time.time() - t0:.1f}s")

            cdir.mkdir(parents=True, exist_ok=True)
            torch.save(features, feat_path)
            torch.save(labels, cdir / f"{prefix}_labels.pt")
            torch.save(uuids, cdir / f"{prefix}_uuids.pt")

            del model
            torch.cuda.empty_cache()


def _load_test_features(region, model_key, resolution):
    cdir = cache_dir(region)
    prefix = f"test_{model_key}_{resolution}"
    features = torch.load(cdir / f"{prefix}_features.pt", weights_only=True)
    labels = torch.load(cdir / f"{prefix}_labels.pt", weights_only=True)
    nan_mask = torch.isnan(features).any(dim=1) | torch.isinf(features).any(dim=1)
    if nan_mask.any():
        keep = ~nan_mask
        features, labels = features[keep], labels[keep]
    return features, labels


# ---------------------------------------------------------------------------
# Step 2: Evaluate linear probes
# ---------------------------------------------------------------------------

def eval_linear_probes(region, model_keys, resolutions, device):
    print("\n" + "=" * 60)
    print("STEP 2: Evaluate linear probes")
    print("=" * 60)
    cdir = ckpt_dir(region)
    results, fixed = {}, {}

    for model_key in model_keys:
        for res in resolutions:
            ckpt_path = cdir / f"{model_key}_{res}_linear_probe_best.pt"
            if not ckpt_path.exists():
                continue
            features, labels = _load_test_features(region, model_key, res)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            head = LinearHead(features.shape[1]).to(device)
            head.load_state_dict(ckpt["model_state_dict"])
            head.eval()

            with torch.no_grad():
                scores = torch.sigmoid(head(features.to(device))).cpu().numpy()
            y_true = labels.numpy()
            metrics = compute_all_metrics(y_true, scores)
            tag = f"{model_key}_{res}_linear"
            results[tag] = metrics

            val_thresh = _extract_val_thresholds(ckpt.get("metrics", {}).get("tpr_at_fpr", {}))
            fixed[tag] = apply_val_thresholds(y_true, scores, val_thresh)

            t2 = metrics["tpr_at_fpr"].get("2%", {}).get("tpr", 0)
            print(f"  {tag:<35s} AUC={metrics['auc_roc']:.4f}  TPR@2%={t2:.4f}")

    return results, fixed


# ---------------------------------------------------------------------------
# Step 3: Evaluate fine-tuned models
# ---------------------------------------------------------------------------

@torch.inference_mode()
def eval_finetune(region, records, device, batch_size, num_workers):
    print("\n" + "=" * 60)
    print("STEP 3: Evaluate fine-tuned models")
    print("=" * 60)
    cdir = ckpt_dir(region)
    results, fixed = {}, {}

    for model_key in FINETUNE_MODELS:
        model_id = MODEL_VARIANTS[model_key]
        for res in FINETUNE_RESOLUTIONS:
            ckpt_path = cdir / f"{model_key}_{res}_finetune_best.pt"
            if not ckpt_path.exists():
                continue

            print(f"  [{model_key} @ {res} finetune] Loading ...")
            model = DINOv3Classifier(model_id, freeze_backbone=False)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device).eval()

            transform = build_transform(res)
            ds = ImageDatasetWithUUID(records, transform)
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

            val_thresh = _extract_val_thresholds(ckpt.get("metrics", {}).get("tpr_at_fpr", {}))
            fixed[tag] = apply_val_thresholds(y_true, scores, val_thresh)

            t2 = metrics["tpr_at_fpr"].get("2%", {}).get("tpr", 0)
            print(f"  {tag:<35s} AUC={metrics['auc_roc']:.4f}  TPR@2%={t2:.4f}")

            del model
            torch.cuda.empty_cache()

    return results, fixed


# ---------------------------------------------------------------------------
# Step 4: Evaluate ML classifiers on frozen features
# ---------------------------------------------------------------------------

def eval_ml_classifiers(region, model_keys, resolutions):
    print("\n" + "=" * 60)
    print("STEP 4: Evaluate ML classifiers (frozen features)")
    print("=" * 60)
    rdir = results_dir(region)
    results, fixed = {}, {}

    for model_key in model_keys:
        for res in resolutions:
            features, labels = _load_test_features(region, model_key, res)
            X, y_true = features.numpy(), labels.numpy()
            prefix = f"{model_key}_{res}"

            for ml_name, eval_fn in [("xgb", _eval_xgb), ("svm", _eval_svm),
                                      ("catboost", _eval_catboost), ("lgbm", _eval_lgbm)]:
                tag = f"{prefix}_{ml_name}"
                try:
                    scores = eval_fn(rdir, prefix, ml_name, X)
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"  [WARN] {tag}: {e}")
                    continue

                metrics = compute_all_metrics(y_true, scores)
                results[tag] = metrics

                val_thresh = _load_ml_val_thresholds(rdir, prefix, ml_name)
                fixed[tag] = apply_val_thresholds(y_true, scores, val_thresh)

                t2 = metrics["tpr_at_fpr"].get("2%", {}).get("tpr", 0)
                print(f"  {tag:<35s} AUC={metrics['auc_roc']:.4f}  TPR@2%={t2:.4f}")

    return results, fixed


# ---------------------------------------------------------------------------
# Step 5: Evaluate finetuned-backbone ML
# ---------------------------------------------------------------------------

@torch.inference_mode()
def eval_finetuned_ml(region, records, device, batch_size, num_workers):
    print("\n" + "=" * 60)
    print("STEP 5: Evaluate finetuned-backbone ML classifiers")
    print("=" * 60)
    cdir_ck = ckpt_dir(region)
    cdir_cache = cache_dir(region)
    rdir = results_dir(region)
    results, fixed = {}, {}

    for model_key in FINETUNE_MODELS:
        model_id = MODEL_VARIANTS[model_key]
        for res in FINETUNE_RESOLUTIONS:
            ckpt_path = cdir_ck / f"{model_key}_{res}_finetune_best.pt"
            if not ckpt_path.exists():
                continue

            ft_prefix = f"test_finetuned_{model_key}_{res}"
            feat_path = cdir_cache / f"{ft_prefix}_features.pt"

            if feat_path.exists():
                features = torch.load(feat_path, weights_only=True)
                labels = torch.load(cdir_cache / f"{ft_prefix}_labels.pt", weights_only=True)
            else:
                print(f"  Extracting finetuned features: {model_key} @ {res} ...")
                model = DINOv3Classifier(model_id, freeze_backbone=False)
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                model = model.to(device).eval()

                transform = build_transform(res)
                ds = ImageDatasetWithUUID(records, transform)
                dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True,
                                persistent_workers=num_workers > 0)

                all_feats, all_labels = [], []
                for pixels, labs, _ in dl:
                    pixels = pixels.to(device, dtype=torch.float32)
                    feats = model.backbone(pixel_values=pixels).pooler_output.float().cpu()
                    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                    all_feats.append(feats)
                    all_labels.extend(labs)

                features = torch.cat(all_feats)
                labels = torch.tensor(all_labels, dtype=torch.long)

                cdir_cache.mkdir(parents=True, exist_ok=True)
                torch.save(features, feat_path)
                torch.save(labels, cdir_cache / f"{ft_prefix}_labels.pt")

                del model
                torch.cuda.empty_cache()

            X, y_true = features.numpy(), labels.numpy()
            src_prefix = f"finetuned_{model_key}_{res}"

            for ml_name, eval_fn in [("xgb", _eval_xgb), ("svm", _eval_svm),
                                      ("catboost", _eval_catboost), ("lgbm", _eval_lgbm)]:
                tag = f"finetuned_{model_key}_{res}_{ml_name}"
                try:
                    scores = eval_fn(rdir, src_prefix, ml_name, X)
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"  [WARN] {tag}: {e}")
                    continue

                metrics = compute_all_metrics(y_true, scores)
                results[tag] = metrics

                val_thresh = _load_ml_val_thresholds(rdir, src_prefix, ml_name)
                fixed[tag] = apply_val_thresholds(y_true, scores, val_thresh)

                t2 = metrics["tpr_at_fpr"].get("2%", {}).get("tpr", 0)
                print(f"  {tag:<40s} AUC={metrics['auc_roc']:.4f}  TPR@2%={t2:.4f}")

    return results, fixed


# ---------------------------------------------------------------------------
# ML model loaders
# ---------------------------------------------------------------------------

def _eval_xgb(rdir, prefix, ml_name, X):
    import xgboost as xgb
    path = rdir / f"{prefix}_xgb.json"
    if not path.exists():
        raise FileNotFoundError(path)
    clf = xgb.XGBClassifier()
    clf.load_model(str(path))
    return clf.predict_proba(X)[:, 1]


def _eval_svm(rdir, prefix, ml_name, X):
    import joblib
    path = rdir / f"{prefix}_svm.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    d = joblib.load(path)
    return d["model"].predict_proba(d["scaler"].transform(X))[:, 1]


def _eval_catboost(rdir, prefix, ml_name, X):
    import catboost as cb
    path = rdir / f"{prefix}_catboost.cbm"
    if not path.exists():
        raise FileNotFoundError(path)
    clf = cb.CatBoostClassifier()
    clf.load_model(str(path))
    return clf.predict_proba(X)[:, 1]


def _eval_lgbm(rdir, prefix, ml_name, X):
    import lightgbm as lgb
    path = rdir / f"{prefix}_lgbm.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    booster = lgb.Booster(model_file=str(path))
    return booster.predict(X)


def _load_ml_val_thresholds(rdir, prefix, ml_name):
    metrics_path = rdir / f"{prefix}_{ml_name}.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        d = json.load(f)
    return _extract_val_thresholds(d.get("metrics", {}).get("tpr_at_fpr", {}))


# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------

def _tpr(m, fpr_name):
    return m.get("tpr_at_fpr", {}).get(fpr_name, {}).get("tpr", 0)


def write_results(region, lp, ft, ml, ftml, lp_fix, ft_fix, ml_fix, ftml_fix, n_images, n_pos):
    print("\n" + "=" * 60)
    print("Writing results")
    print("=" * 60)

    all_results = {
        "region": region,
        "test_data": {"n_images": n_images, "n_positive": n_pos, "n_negative": n_images - n_pos},
        "linear_probe": lp, "finetune": ft, "ml_classifiers": ml, "finetuned_ml": ftml,
        "fixed_threshold": {
            "linear_probe": lp_fix, "finetune": ft_fix,
            "ml_classifiers": ml_fix, "finetuned_ml": ftml_fix,
        },
    }
    rdir = results_dir(region)
    rdir.mkdir(parents=True, exist_ok=True)
    out_path = rdir / "test_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved to {out_path}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print(f"TEST RESULTS SUMMARY — {region}")
    print(f"{'=' * 80}")
    print(f"  Test set: {n_images} images ({n_pos} pos, {n_images - n_pos} neg)\n")

    all_methods = {}
    all_methods.update(lp)
    all_methods.update(ft)
    all_methods.update(ml)
    all_methods.update(ftml)

    all_fixed = {}
    all_fixed.update({k: v for k, v in lp_fix.items()})
    all_fixed.update({k: v for k, v in ft_fix.items()})
    all_fixed.update({k: v for k, v in ml_fix.items()})
    all_fixed.update({k: v for k, v in ftml_fix.items()})

    # ROC-curve based
    print(f"  {'Method':<40} {'AUC-ROC':>8} {'TPR@0.5%':>9} {'TPR@1%':>8} {'TPR@2%':>8} {'TPR@5%':>8} {'TPR@10%':>8}")
    print(f"  {'-'*89}")
    for tag in sorted(all_methods, key=lambda t: -_tpr(all_methods[t], "2%")):
        m = all_methods[tag]
        print(f"  {tag:<40} {m['auc_roc']:>8.4f} {_tpr(m,'0.5%'):>9.4f} {_tpr(m,'1%'):>8.4f} "
              f"{_tpr(m,'2%'):>8.4f} {_tpr(m,'5%'):>8.4f} {_tpr(m,'10%'):>8.4f}")

    # Fixed threshold best at each FPR
    print(f"\n  Best fixed-threshold (val thresholds on test):")
    print(f"  {'FPR Target':<12} {'Method':<40} {'Test FPR':>9} {'Test TPR':>9}")
    print(f"  {'-'*70}")
    for fpr_name in TARGET_FPR_NAMES:
        best_tag, best_tpr, best_fpr = None, -1, 0
        for tag, d in all_fixed.items():
            if fpr_name in d and d[fpr_name]["tpr"] > best_tpr:
                best_tpr = d[fpr_name]["tpr"]
                best_fpr = d[fpr_name]["fpr"]
                best_tag = tag
        if best_tag:
            print(f"  {fpr_name:<12} {best_tag:<40} {best_fpr:>9.2%} {best_tpr:>9.2%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate all models on test set")
    parser.add_argument("--region", type=str, required=True, choices=REGIONS)
    parser.add_argument("--models", nargs="+", default=list(MODEL_VARIANTS.keys()),
                        choices=list(MODEL_VARIANTS.keys()))
    parser.add_argument("--resolutions", nargs="+", type=int, default=RESOLUTIONS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    region = args.region
    device = torch.device(args.device)
    print(f"Region: {region}  |  Device: {device}")

    records = load_metadata(region, split="test")
    n_images = len(records)
    n_pos = sum(r["label"] for r in records)

    if not args.skip_extract:
        extract_test_features(region, args.models, args.resolutions, records,
                              device, args.batch_size, args.num_workers)

    lp, lp_fix = eval_linear_probes(region, args.models, args.resolutions, device)
    ft, ft_fix = eval_finetune(region, records, device, args.batch_size, args.num_workers)
    ml, ml_fix = eval_ml_classifiers(region, args.models, args.resolutions)
    ftml, ftml_fix = eval_finetuned_ml(region, records, device, args.batch_size, args.num_workers)

    write_results(region, lp, ft, ml, ftml, lp_fix, ft_fix, ml_fix, ftml_fix, n_images, n_pos)
    print("\nDone.")


if __name__ == "__main__":
    main()
