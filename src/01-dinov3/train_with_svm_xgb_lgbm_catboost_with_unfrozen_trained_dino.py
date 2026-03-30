#!/root/.venv/bin/python
"""
Extract embeddings from a fine-tuned DINOv3 backbone, then train ML
classifiers (SVM, XGBoost, CatBoost, LightGBM) on those features.

Requires a checkpoint from train_with_unfreeze_dino.py.

Usage:
  python train_with_svm_xgb_lgbm_catboost_with_unfrozen_trained_dino.py \
      --region care_label --model vitl16 --resolution 518
  python train_with_svm_xgb_lgbm_catboost_with_unfrozen_trained_dino.py \
      --region front --model vitb16 --resolution 714 --classifiers svm xgb
"""

import argparse
import json
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from config import (
    MODEL_VARIANTS,
    REGIONS,
    RESOLUTIONS,
    SEED,
    DINOv3Classifier,
    ImageDatasetWithUUID,
    build_transform,
    cache_dir,
    ckpt_dir,
    compute_all_metrics,
    get_or_create_val_split,
    load_metadata,
    results_dir,
)

AVAILABLE_CLASSIFIERS = ["xgb", "svm", "catboost", "lgbm"]


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Feature extraction from finetuned backbone
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_finetuned_features(model, dataloader, device):
    all_features, all_labels, all_uuids = [], [], []
    for batch_idx, (pixel_values, labels, uuids) in enumerate(dataloader):
        pixel_values = pixel_values.to(device, dtype=torch.float32)
        outputs = model.backbone(pixel_values=pixel_values)
        feats = outputs.pooler_output.float().cpu()
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        all_features.append(feats)
        all_labels.extend(labels)
        all_uuids.extend(uuids)

        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            print(f"    Batch {batch_idx + 1}/{len(dataloader)}", flush=True)

    features = torch.cat(all_features, dim=0)
    labels_t = torch.tensor(all_labels, dtype=torch.long)
    return features, labels_t, all_uuids


# ---------------------------------------------------------------------------
# Classifier training (same as train_svm_xgb_lgbm_catboost.py)
# ---------------------------------------------------------------------------

def train_xgb(X_train, y_train, X_val, y_val, device, seed=SEED):
    import xgboost as xgb
    params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "tree_method": "hist", "seed": seed, "max_depth": 6,
        "learning_rate": 0.05, "n_estimators": 300,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "scale_pos_weight": float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
    }
    if device == "cuda":
        params["device"] = "cuda"
    clf = xgb.XGBClassifier(**params)
    t0 = time.time()
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return clf, clf.predict_proba(X_val)[:, 1], time.time() - t0


def train_svm(X_train, y_train, X_val, y_val, scale_pos_weight, seed=SEED):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
              class_weight={0: 1.0, 1: scale_pos_weight}, random_state=seed)
    t0 = time.time()
    clf.fit(X_train_s, y_train)
    return (clf, scaler), clf.predict_proba(X_val_s)[:, 1], time.time() - t0


def train_catboost(X_train, y_train, X_val, y_val, device, seed=SEED):
    import catboost as cb
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    clf = cb.CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        task_type="GPU" if device == "cuda" else "CPU",
        random_seed=seed, scale_pos_weight=spw, verbose=0,
    )
    t0 = time.time()
    clf.fit(cb.Pool(X_train, label=y_train), eval_set=cb.Pool(X_val, label=y_val))
    return clf, clf.predict_proba(X_val)[:, 1], time.time() - t0


def train_lgbm(X_train, y_train, X_val, y_val, device, seed=SEED):
    import lightgbm as lgb
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    clf = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=spw, random_state=seed, verbosity=-1,
        device="gpu" if device == "cuda" else "cpu",
    )
    t0 = time.time()
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc")
    return clf, clf.predict_proba(X_val)[:, 1], time.time() - t0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(region, model_key, resolution, selected_classifiers, device, batch_size, num_workers,
        ckpt_tag="finetune"):
    # 1. Load finetuned checkpoint
    cdir = ckpt_dir(region)
    ckpt_path = cdir / f"{model_key}_{resolution}_{ckpt_tag}_best.pt"
    if not ckpt_path.exists():
        print(f"  ERROR: No checkpoint at {ckpt_path}")
        return None

    model_id = MODEL_VARIANTS[model_key]
    print(f"  Loading checkpoint from {ckpt_path} ...")
    model = DINOv3Classifier(model_id, freeze_backbone=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # 2. Check for cached finetuned features
    ft_cache = cache_dir(region)
    ft_prefix = f"{ckpt_tag}_{model_key}_{resolution}"
    feat_path = ft_cache / f"{ft_prefix}_features.pt"
    label_path = ft_cache / f"{ft_prefix}_labels.pt"
    uuid_path = ft_cache / f"{ft_prefix}_uuids.pt"

    if feat_path.exists() and label_path.exists() and uuid_path.exists():
        print(f"  Loading cached finetuned features from {feat_path}")
        features = torch.load(feat_path, weights_only=True)
        labels_t = torch.load(label_path, weights_only=True)
        uuids = torch.load(uuid_path, weights_only=False)
    else:
        # 3. Extract features from finetuned backbone
        records = load_metadata(region, split="train")
        transform = build_transform(resolution)
        dataset = ImageDatasetWithUUID(records, transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=num_workers > 0,
        )

        print(f"  Extracting finetuned features at {resolution}x{resolution} ...")
        t0 = time.time()
        features, labels_t, uuids = extract_finetuned_features(model, dataloader, device)
        print(f"  Done: {features.shape} in {time.time() - t0:.1f}s")

        ft_cache.mkdir(parents=True, exist_ok=True)
        torch.save(features, feat_path)
        torch.save(labels_t, label_path)
        torch.save(uuids, uuid_path)

    del model
    torch.cuda.empty_cache()

    # 4. Split using same val split
    records = load_metadata(region, split="train")
    train_idx, val_idx = get_or_create_val_split(region, records)

    uuid_to_idx = {u: i for i, u in enumerate(uuids)}
    feat_train_idx = [uuid_to_idx[records[i]["session_uuid"]] for i in train_idx if records[i]["session_uuid"] in uuid_to_idx]
    feat_val_idx = [uuid_to_idx[records[i]["session_uuid"]] for i in val_idx if records[i]["session_uuid"] in uuid_to_idx]

    X_train = features[feat_train_idx].numpy()
    y_train = labels_t[feat_train_idx].numpy()
    X_val = features[feat_val_idx].numpy()
    y_val = labels_t[feat_val_idx].numpy()

    n_pos_train = int(y_train.sum())
    scale_pos_weight = (len(y_train) - n_pos_train) / max(n_pos_train, 1)
    print(f"  Train: {len(y_train)} ({n_pos_train} pos) | Val: {len(y_val)} ({int(y_val.sum())} pos)")

    # 5. Train classifiers
    rdir = results_dir(region)
    rdir.mkdir(parents=True, exist_ok=True)
    results_summary = {}

    for name in selected_classifiers:
        if name == "xgb":
            clf, y_score, fit_time = train_xgb(X_train, y_train, X_val, y_val, device)
        elif name == "svm":
            (clf, scaler), y_score, fit_time = train_svm(X_train, y_train, X_val, y_val, scale_pos_weight)
        elif name == "catboost":
            clf, y_score, fit_time = train_catboost(X_train, y_train, X_val, y_val, device)
        elif name == "lgbm":
            clf, y_score, fit_time = train_lgbm(X_train, y_train, X_val, y_val, device)
        else:
            continue

        metrics = compute_all_metrics(y_val, y_score)
        auc_roc = metrics.get("auc_roc", 0)
        auc_pr = metrics.get("auc_pr", 0)
        tpr_2 = metrics.get("tpr_at_fpr", {}).get("2%", {}).get("tpr", 0)

        print(f"  [{name.upper():8s}] fit={fit_time:.1f}s | AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f} TPR@2%FPR={tpr_2:.4f}")

        results_summary[name] = {
            "auc_roc": auc_roc, "auc_pr": auc_pr,
            "tpr_at_fpr": metrics.get("tpr_at_fpr", {}),
            "fit_time_s": fit_time,
        }

        save_prefix = f"{ckpt_tag}_{model_key}_{resolution}"
        out_path = rdir / f"{save_prefix}_{name}.json"
        save_obj = {"metrics": results_summary[name], "model_key": model_key,
                     "resolution": resolution, "region": region, "source": ckpt_tag}

        if name == "svm":
            import joblib
            joblib.dump({"model": clf, "scaler": scaler}, rdir / f"{save_prefix}_svm.joblib")
        elif name == "xgb":
            clf.save_model(str(rdir / f"{save_prefix}_xgb.json"))
        elif name == "catboost":
            clf.save_model(str(rdir / f"{save_prefix}_catboost.cbm"))
        elif name == "lgbm":
            clf.booster_.save_model(str(rdir / f"{save_prefix}_lgbm.txt"))

        with open(out_path, "w") as f:
            json.dump(save_obj, f, indent=2)

    return results_summary


def main():
    parser = argparse.ArgumentParser(
        description="Train ML classifiers on finetuned DINOv3 embeddings")
    parser.add_argument("--region", type=str, default="care_label", choices=REGIONS)
    parser.add_argument("--model", type=str, default="vitl16", choices=list(MODEL_VARIANTS.keys()))
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--classifiers", nargs="+", default=AVAILABLE_CLASSIFIERS,
                        choices=AVAILABLE_CLASSIFIERS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt-tag", type=str, default="finetune",
                        help="Checkpoint tag, e.g. 'finetune' or 'partial_last4'")
    args = parser.parse_args()

    np.random.seed(SEED)
    device = _get_device()
    print(f"Device: {device}")

    print(f"\n{'=' * 60}")
    print(f"[{args.region} | {args.model} @ {args.resolution} | {args.ckpt_tag} -> ML]")
    print(f"{'=' * 60}")
    run(args.region, args.model, args.resolution, args.classifiers,
        device, args.batch_size, args.num_workers, ckpt_tag=args.ckpt_tag)


if __name__ == "__main__":
    main()
