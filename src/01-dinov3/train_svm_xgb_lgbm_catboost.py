#!/root/.venv/bin/python
"""
Train ML classifiers (SVM, XGBoost, CatBoost, LightGBM) on cached DINOv3
embeddings for any image region.

Features are scaled for SVM only; tree models use raw features.

Usage:
  python train_svm_xgb_lgbm_catboost.py --region care_label --sweep
  python train_svm_xgb_lgbm_catboost.py --region front --model vitb16 --resolution 518
  python train_svm_xgb_lgbm_catboost.py --region care_label --sweep --classifiers xgb svm
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (
    MODEL_VARIANTS,
    REGIONS,
    RESOLUTIONS,
    SEED,
    apply_threshold_auth_positive,
    cache_dir,
    compute_metrics_auth_positive,
    get_or_create_val_split,
    load_cached_features,
    load_metadata,
    results_dir,
)

AVAILABLE_CLASSIFIERS = ["xgb", "svm", "catboost", "lgbm"]


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Classifier training
# ---------------------------------------------------------------------------

def train_xgb(X_train, y_train, X_val, y_val, device, seed=SEED):
    import xgboost as xgb
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
    if device == "cuda":
        params["device"] = "cuda"
    clf = xgb.XGBClassifier(**params)
    t0 = time.time()
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    fit_time = time.time() - t0
    y_score = clf.predict_proba(X_val)[:, 1]
    return clf, y_score, fit_time


def train_svm(X_train, y_train, X_val, y_val, scale_pos_weight, seed=SEED):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    clf = SVC(
        kernel="rbf", C=1.0, gamma="scale", probability=True,
        class_weight={0: 1.0, 1: scale_pos_weight}, random_state=seed,
    )
    t0 = time.time()
    clf.fit(X_train_s, y_train)
    fit_time = time.time() - t0
    y_score = clf.predict_proba(X_val_s)[:, 1]
    return (clf, scaler), y_score, fit_time


def train_catboost(X_train, y_train, X_val, y_val, device, seed=SEED):
    import catboost as cb
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    pool_train = cb.Pool(X_train, label=y_train)
    pool_val = cb.Pool(X_val, label=y_val)
    clf = cb.CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        task_type="GPU" if device == "cuda" else "CPU",
        random_seed=seed, scale_pos_weight=scale_pos_weight, verbose=0,
    )
    t0 = time.time()
    clf.fit(pool_train, eval_set=pool_val)
    fit_time = time.time() - t0
    y_score = clf.predict_proba(X_val)[:, 1]
    return clf, y_score, fit_time


def train_lgbm(X_train, y_train, X_val, y_val, device, seed=SEED):
    import lightgbm as lgb
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    clf = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, random_state=seed, verbosity=-1,
        device="gpu" if device == "cuda" else "cpu",
    )
    t0 = time.time()
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc")
    fit_time = time.time() - t0
    y_score = clf.predict_proba(X_val)[:, 1]
    return clf, y_score, fit_time


# ---------------------------------------------------------------------------
# Run one embedding variant
# ---------------------------------------------------------------------------

def _score_clf(clf, X, name):
    if name == "svm":
        model, scaler = clf
        return model.predict_proba(scaler.transform(X))[:, 1]
    elif name == "lgbm":
        return clf.predict_proba(X)[:, 1]
    else:
        return clf.predict_proba(X)[:, 1]


def run_one(region, model_key, resolution, selected_classifiers, device):
    features, labels, uuids = load_cached_features(region, model_key, resolution)

    records = load_metadata(region, split="train")
    train_idx, val_idx = get_or_create_val_split(region, records)

    uuid_to_idx = {u: i for i, u in enumerate(uuids)}
    feat_train_idx = [uuid_to_idx[records[i]["session_uuid"]] for i in train_idx if records[i]["session_uuid"] in uuid_to_idx]
    feat_val_idx = [uuid_to_idx[records[i]["session_uuid"]] for i in val_idx if records[i]["session_uuid"] in uuid_to_idx]

    X_train = features[feat_train_idx].numpy()
    y_train = labels[feat_train_idx].numpy()
    X_val = features[feat_val_idx].numpy()
    y_val = labels[feat_val_idx].numpy()

    cdir = cache_dir(region)
    test_prefix = f"test_{model_key}_{resolution}"
    test_feat_path = cdir / f"{test_prefix}_features.pt"
    X_test, y_test = None, None
    if test_feat_path.exists():
        test_feats = torch.load(test_feat_path, weights_only=True)
        test_labels_t = torch.load(cdir / f"{test_prefix}_labels.pt", weights_only=True)
        X_test = test_feats.numpy()
        y_test = test_labels_t.numpy()

    n_pos_train = int(y_train.sum())
    n_neg_train = len(y_train) - n_pos_train
    scale_pos_weight = n_neg_train / max(n_pos_train, 1)
    print(f"  Train: {len(y_train)} ({n_pos_train} pos) | Val: {len(y_val)} ({int(y_val.sum())} pos)"
          + (f" | Test: {len(y_test)} ({int(y_test.sum())} pos)" if y_test is not None else ""))

    prefix = f"{model_key}_{resolution}"
    rdir = results_dir(region)
    rdir.mkdir(parents=True, exist_ok=True)
    results_summary = {}

    for name in selected_classifiers:
        if name == "xgb":
            clf, y_score, fit_time = train_xgb(X_train, y_train, X_val, y_val, device)
        elif name == "svm":
            (clf, scaler), y_score, fit_time = train_svm(X_train, y_train, X_val, y_val, scale_pos_weight)
            clf_obj = (clf, scaler)
        elif name == "catboost":
            clf, y_score, fit_time = train_catboost(X_train, y_train, X_val, y_val, device)
        elif name == "lgbm":
            clf, y_score, fit_time = train_lgbm(X_train, y_train, X_val, y_val, device)
        else:
            continue

        clf_for_score = clf_obj if name == "svm" else clf
        train_score = _score_clf(clf_for_score, X_train, name)
        train_m = compute_metrics_auth_positive(y_train, train_score)
        val_m = compute_metrics_auth_positive(y_val, y_score)

        train_auc = train_m.get("auc_roc", 0)
        train_tpr2 = train_m.get("tpr_at_fpr", {}).get("2%", {}).get("tpr", 0)
        val_auc = val_m.get("auc_roc", 0)
        val_tpr2 = val_m.get("tpr_at_fpr", {}).get("2%", {}).get("tpr", 0)

        line = (f"  [{name.upper():8s}] {fit_time:.1f}s | "
                f"train AUC={train_auc:.4f} TPR@2%={train_tpr2:.4f} | "
                f"val AUC={val_auc:.4f} TPR@2%={val_tpr2:.4f}")

        test_info = {}
        if X_test is not None:
            test_score = _score_clf(clf_for_score, X_test, name)
            test_m = compute_metrics_auth_positive(y_test, test_score)
            thresh_orig = val_m.get("tpr_at_fpr", {}).get("2%", {}).get("threshold_orig", 0.5)
            fixed = apply_threshold_auth_positive(y_test, test_score, thresh_orig)
            line += (f" | test AUC={test_m.get('auc_roc', 0):.4f} "
                     f"@2%: TPR={fixed['tpr']:.4f} FPR={fixed['fpr']:.4f}")
            test_info = {"test_auc": test_m.get("auc_roc", 0), "test_fixed_2pct": fixed}

        print(line)

        results_summary[name] = {
            "auc_roc": val_auc, "auc_pr": val_m.get("auc_pr", 0),
            "tpr_at_fpr": val_m.get("tpr_at_fpr", {}),
            "train_auc": train_auc, "train_tpr_at_fpr": train_m.get("tpr_at_fpr", {}),
            "fit_time_s": fit_time,
            **test_info,
        }

        out_path = rdir / f"{prefix}_{name}.json"
        save_obj = {"metrics": results_summary[name], "model_key": model_key,
                     "resolution": resolution, "region": region}

        if name == "svm":
            import joblib
            joblib.dump({"model": clf, "scaler": scaler}, rdir / f"{prefix}_svm.joblib")
        elif name == "xgb":
            clf.save_model(str(rdir / f"{prefix}_xgb.json"))
        elif name == "catboost":
            clf.save_model(str(rdir / f"{prefix}_catboost.cbm"))
        elif name == "lgbm":
            clf.booster_.save_model(str(rdir / f"{prefix}_lgbm.txt"))

        with open(out_path, "w") as f:
            json.dump(save_obj, f, indent=2)

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Train ML classifiers on cached DINOv3 features")
    parser.add_argument("--region", type=str, default="care_label", choices=REGIONS)
    parser.add_argument("--model", type=str, default="vits16", choices=list(MODEL_VARIANTS.keys()))
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--classifiers", nargs="+", default=AVAILABLE_CLASSIFIERS,
                        choices=AVAILABLE_CLASSIFIERS)
    args = parser.parse_args()

    device = _get_device()
    print(f"Device: {device}")
    np.random.seed(SEED)

    region = args.region

    if args.sweep:
        cdir = cache_dir(region)
        cached_files = sorted(cdir.glob("*_features.pt"))
        if not cached_files:
            print(f"No cached features for region '{region}'. Run precompute_embeddings.py first.")
            sys.exit(1)

        all_results = {}
        for feat_file in cached_files:
            name = feat_file.stem.replace("_features", "")
            parts = name.rsplit("_", 1)
            mk, res = parts[0], int(parts[1])
            print(f"\n{'=' * 60}")
            print(f"[{region} | {mk} @ {res} | {', '.join(args.classifiers)}]")
            print(f"{'=' * 60}")
            all_results[name] = run_one(region, mk, res, args.classifiers, device)

        print(f"\n{'=' * 60}")
        print(f"SWEEP SUMMARY (AUC-ROC) — {region}")
        print(f"{'=' * 60}")
        for variant in sorted(all_results.keys()):
            row = all_results[variant]
            line = f"  {variant:<28}"
            for m in args.classifiers:
                line += f"  {m}={row.get(m, {}).get('auc_roc', 0):.4f}"
            print(line)

        rdir = results_dir(region)
        rdir.mkdir(parents=True, exist_ok=True)
        with open(rdir / "sweep_summary.json", "w") as f:
            json.dump(all_results, f, indent=2)
        return

    print(f"\n{'=' * 60}")
    print(f"[{region} | {args.model} @ {args.resolution} | {', '.join(args.classifiers)}]")
    print(f"{'=' * 60}")
    run_one(region, args.model, args.resolution, args.classifiers, device)


if __name__ == "__main__":
    main()
