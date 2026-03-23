"""
Train ML classifiers on cached DINOv3 features: GPU XGBoost, SVM, CatBoost, LightGBM.

Uses same cached features and evaluation (TPR @ FPR, AUC-ROC, AUC-PR) as
train_care_label.py. Features are scaled for SVM only; tree models use raw features.

Usage:
  python train_care_label_ml.py --model vits16 --resolution 518
  python train_care_label_ml.py --sweep   # all cached variant x resolution combos
  python train_care_label_ml.py --model vitb16 --resolution 518 --models xgb svm  # subset
"""

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cached_features"
RESULTS_DIR = SCRIPT_DIR / "ml_results"
SEED = 42

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

TARGET_FPRS = [0.005, 0.01, 0.02, 0.05, 0.10]
TARGET_FPR_NAMES = ["0.5%", "1%", "2%", "5%", "10%"]


def load_cached_features(model_key, resolution):
    prefix = f"{model_key}_{resolution}"
    features = torch.load(CACHE_DIR / f"{prefix}_features.pt", weights_only=True)
    labels = torch.load(CACHE_DIR / f"{prefix}_labels.pt", weights_only=True)
    uuids = torch.load(CACHE_DIR / f"{prefix}_uuids.pt", weights_only=False)
    nan_mask = torch.isnan(features).any(dim=1) | torch.isinf(features).any(dim=1)
    if nan_mask.any():
        keep = ~nan_mask
        features, labels = features[keep], labels[keep]
        uuids = [u for u, k in zip(uuids, keep.tolist()) if k]
    return features, labels, uuids


def stratified_split(labels, val_ratio=0.2, seed=SEED):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return train_idx, val_idx


def _compute_tpr_at_fprs(y_true, y_score):
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_score)
    results = {}
    for target_fpr, name in zip(TARGET_FPRS, TARGET_FPR_NAMES):
        distances = np.abs(fpr_arr - target_fpr)
        tied = np.where(distances == distances.min())[0]
        idx = tied[np.argmax(tpr_arr[tied])]
        thresh_idx = min(idx, len(thresholds) - 1)
        results[name] = {"tpr": float(tpr_arr[idx]), "actual_fpr": float(fpr_arr[idx]), "threshold": float(thresholds[thresh_idx])}
    return results


def compute_all_metrics(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return {"auc_roc": 0.0, "auc_pr": 0.0, "tpr_at_fpr": {}}
    return {
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "auc_pr": float(average_precision_score(y_true, y_score)),
        "tpr_at_fpr": _compute_tpr_at_fprs(y_true, y_score),
    }


AVAILABLE_MODELS = ["xgb", "svm", "catboost", "lgbm"]


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_xgb(X_train, y_train, X_val, y_val, device, seed=SEED):
    import xgboost as xgb
    use_gpu = device == "cuda"
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "seed": seed,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
    }
    if use_gpu:
        params["tree_method"] = "hist"
        params["device"] = "cuda"
    clf = xgb.XGBClassifier(**params)
    t0 = time.time()
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    fit_time = time.time() - t0
    y_score = clf.predict_proba(X_val)[:, 1]
    return clf, y_score, fit_time


def train_svm(X_train, y_train, X_val, y_val, scale_pos_weight, seed=SEED):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    # class_weight compensates imbalance; scale_pos_weight ~ n_neg/n_pos
    clf = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight={0: 1.0, 1: scale_pos_weight},
        random_state=seed,
    )
    t0 = time.time()
    clf.fit(X_train_s, y_train)
    fit_time = time.time() - t0
    y_score = clf.predict_proba(X_val_s)[:, 1]
    return (clf, scaler), y_score, fit_time


def train_catboost(X_train, y_train, X_val, y_val, device, seed=SEED):
    import catboost as cb
    use_gpu = device == "cuda"
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    pool_train = cb.Pool(X_train, label=y_train)
    pool_val = cb.Pool(X_val, label=y_val)
    clf = cb.CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU" if use_gpu else "CPU",
        random_seed=seed,
        scale_pos_weight=scale_pos_weight,
        verbose=0,
    )
    t0 = time.time()
    clf.fit(pool_train, eval_set=pool_val)
    fit_time = time.time() - t0
    y_score = clf.predict_proba(X_val)[:, 1]
    return clf, y_score, fit_time


def train_lgbm(X_train, y_train, X_val, y_val, device, seed=SEED):
    import lightgbm as lgb
    use_gpu = device == "cuda"
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        verbosity=-1,
        device="gpu" if use_gpu else "cpu",
    )
    t0 = time.time()
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
    )
    fit_time = time.time() - t0
    y_score = clf.predict_proba(X_val)[:, 1]
    return clf, y_score, fit_time


def run_one(model_key, resolution, selected_models, device):
    features, labels, _ = load_cached_features(model_key, resolution)
    X = features.numpy()
    y = labels.numpy()

    train_idx, val_idx = stratified_split(y, val_ratio=0.2)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    n_pos_train = int(y_train.sum())
    n_neg_train = len(y_train) - n_pos_train
    scale_pos_weight = n_neg_train / max(n_pos_train, 1)
    print(f"  Train: {len(y_train)} ({n_pos_train} pos) | Val: {len(y_val)} ({int(y_val.sum())} pos) | scale_pos_weight={scale_pos_weight:.1f}")

    prefix = f"{model_key}_{resolution}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_summary = {}

    for name in selected_models:
        if name == "xgb":
            clf, y_score, fit_time = train_xgb(X_train, y_train, X_val, y_val, device)
        elif name == "svm":
            (clf, scaler), y_score, fit_time = train_svm(
                X_train, y_train, X_val, y_val, scale_pos_weight
            )
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
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "tpr_at_fpr": metrics.get("tpr_at_fpr", {}),
            "fit_time_s": fit_time,
        }

        # Save model and scaler (for SVM)
        out_path = RESULTS_DIR / f"{prefix}_{name}.json"
        save_obj = {"metrics": results_summary[name], "model_key": model_key, "resolution": resolution}
        if name == "svm":
            import joblib
            joblib.dump({"model": clf, "scaler": scaler}, RESULTS_DIR / f"{prefix}_svm.joblib")
        elif name == "xgb":
            clf.save_model(str(RESULTS_DIR / f"{prefix}_xgb.json"))
        elif name == "catboost":
            clf.save_model(str(RESULTS_DIR / f"{prefix}_catboost.cbm"))
        elif name == "lgbm":
            clf.booster_.save_model(str(RESULTS_DIR / f"{prefix}_lgbm.txt"))

        with open(out_path, "w") as f:
            json.dump(save_obj, f, indent=2)

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Train ML classifiers on cached DINOv3 features")
    parser.add_argument("--model", type=str, default="vits16",
                        help="Model variant (must have cached features)")
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--sweep", action="store_true",
                        help="Run over all cached variant x resolution combos")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit sweep to first N cached variants (for testing)")
    parser.add_argument("--models", nargs="+", default=AVAILABLE_MODELS,
                        choices=AVAILABLE_MODELS,
                        help="Which classifiers to train (default: all)")
    args = parser.parse_args()

    device = _get_device()
    print(f"Device: {device}")

    np.random.seed(SEED)

    if args.sweep:
        cached_files = sorted(CACHE_DIR.glob("*_features.pt"))
        if not cached_files:
            print("No cached features found. Run cache_features.py first.")
            sys.exit(1)
        if getattr(args, "limit", None):
            cached_files = cached_files[: args.limit]

        all_results = {}
        for feat_file in cached_files:
            name = feat_file.stem.replace("_features", "")
            parts = name.rsplit("_", 1)
            mk, res = parts[0], int(parts[1])
            print(f"\n{'='*60}")
            print(f"[{mk} @ {res} | {', '.join(args.models)}]")
            print(f"{'='*60}")
            all_results[name] = run_one(mk, res, args.models, device)

        print(f"\n{'='*60}")
        print("SWEEP SUMMARY (AUC-ROC)")
        print(f"{'='*60}")
        for variant in sorted(all_results.keys()):
            row = all_results[variant]
            line = f"  {variant:<28}"
            for m in args.models:
                line += f"  {m}={row.get(m, {}).get('auc_roc', 0):.4f}"
            print(line)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = RESULTS_DIR / "sweep_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary written to {summary_path}")
        return

    print(f"\n{'='*60}")
    print(f"[{args.model} @ {args.resolution} | {', '.join(args.models)}]")
    print(f"{'='*60}")
    run_one(args.model, args.resolution, args.models, device)
    print(f"\nResults and checkpoints under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
