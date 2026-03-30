#!/root/.venv/bin/python
"""
Concat embeddings from all 4 regions into a single feature vector,
train XGB/SVM/linear classifiers, evaluate on test with auth-positive convention.

Backbone: vitl16_518 (1024d per region). Missing regions zero-filled + 4 mask bits = 4100d.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_metadata, SEED

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cached_features"
REGIONS = ["care_label", "front", "front_exterior_logo", "brand_tag"]
EMBED_KEY = "vitl16_518"
EMBED_DIM = 1024

TARGET_FPRS = [0.005, 0.01, 0.02, 0.05, 0.10]
FPR_NAMES = ["0.5%", "1%", "2%", "5%", "10%"]


def load_features_by_uuid(region, split, feat_key):
    cache = CACHE_DIR / region
    prefix = f"test_{feat_key}" if split == "test" else f"{feat_key}"
    feat_path = cache / f"{prefix}_features.pt"
    label_path = cache / f"{prefix}_labels.pt"
    uuid_path = cache / f"{prefix}_uuids.pt"
    if not feat_path.exists():
        return {}
    features = torch.load(feat_path, weights_only=True)
    labels = torch.load(label_path, weights_only=True)
    uuids = torch.load(uuid_path, weights_only=False) if uuid_path.exists() else []
    return {u: (features[i].numpy(), int(labels[i].item())) for i, u in enumerate(uuids)}


def build_concat_dataset(split, feat_key):
    per_region = {}
    for region in REGIONS:
        per_region[region] = load_features_by_uuid(region, split, feat_key)

    all_uuids = set()
    for region_data in per_region.values():
        all_uuids.update(region_data.keys())
    all_uuids = sorted(all_uuids)

    # Build labels from any available region
    uuid_labels = {}
    for uuid in all_uuids:
        for region in REGIONS:
            if uuid in per_region[region]:
                uuid_labels[uuid] = per_region[region][uuid][1]
                break

    X = np.zeros((len(all_uuids), EMBED_DIM * len(REGIONS) + len(REGIONS)), dtype=np.float32)
    y = np.zeros(len(all_uuids), dtype=np.int64)

    for i, uuid in enumerate(all_uuids):
        y[i] = uuid_labels[uuid]
        for j, region in enumerate(REGIONS):
            if uuid in per_region[region]:
                feat, _ = per_region[region][uuid]
                X[i, j * EMBED_DIM:(j + 1) * EMBED_DIM] = feat
                X[i, EMBED_DIM * len(REGIONS) + j] = 1.0  # mask bit

    print(f"  {split}: {len(all_uuids)} sessions, {int(y.sum())} fake, {len(y) - int(y.sum())} auth")
    return X, y, all_uuids


def auth_positive_metrics(y_true, y_score):
    yt = 1 - np.asarray(y_true)
    ys = 1.0 - np.asarray(y_score)
    if len(np.unique(yt)) < 2:
        return {"auc_roc": 0.0, "tpr_at_fpr": {}}
    auc = float(roc_auc_score(yt, ys))
    fpr_arr, tpr_arr, _ = roc_curve(yt, ys)
    res = {}
    for t, n in zip(TARGET_FPRS, FPR_NAMES):
        d = np.abs(fpr_arr - t)
        tied = np.where(d == d.min())[0]
        idx = tied[np.argmax(tpr_arr[tied])]
        res[n] = {"tpr": float(tpr_arr[idx]), "fpr": float(fpr_arr[idx])}
    return {"auc_roc": auc, "tpr_at_fpr": res}


def main():
    np.random.seed(SEED)

    print("Building train dataset...")
    X_all, y_all, _ = build_concat_dataset("train", EMBED_KEY)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(splitter.split(X_all, y_all))
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    print(f"  Train: {len(X_train)} ({y_train.sum()} fake) | Val: {len(X_val)} ({y_val.sum()} fake)")

    print("\nBuilding test dataset...")
    X_test, y_test, _ = build_concat_dataset("test", EMBED_KEY)

    results = {}

    # XGBoost
    print("\n--- XGBoost ---")
    import xgboost as xgb
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    clf = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        seed=SEED, max_depth=6, learning_rate=0.05, n_estimators=300,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    test_scores = clf.predict_proba(X_test)[:, 1]
    m = auth_positive_metrics(y_test, test_scores)
    results["concat_xgb"] = m
    print(f"  AUC={m['auc_roc']:.4f}  @2%={m['tpr_at_fpr']['2%']['tpr']:.2%}  @5%={m['tpr_at_fpr']['5%']['tpr']:.2%}  @10%={m['tpr_at_fpr']['10%']['tpr']:.2%}")

    # SVM
    print("\n--- SVM ---")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
              class_weight={0: 1.0, 1: spw}, random_state=SEED)
    svm.fit(X_train_s, y_train)
    test_scores = svm.predict_proba(X_test_s)[:, 1]
    m = auth_positive_metrics(y_test, test_scores)
    results["concat_svm"] = m
    print(f"  AUC={m['auc_roc']:.4f}  @2%={m['tpr_at_fpr']['2%']['tpr']:.2%}  @5%={m['tpr_at_fpr']['5%']['tpr']:.2%}  @10%={m['tpr_at_fpr']['10%']['tpr']:.2%}")

    # Linear head (logistic regression via torch)
    print("\n--- Linear Head ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    head = nn.Linear(X_train.shape[1], 1).to(device)
    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    pw = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    head.train()
    for epoch in range(200):
        opt.zero_grad()
        loss = crit(head(X_tr_t).squeeze(-1), y_tr_t)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        test_scores = torch.sigmoid(head(X_te_t).squeeze(-1)).cpu().numpy()
    m = auth_positive_metrics(y_test, test_scores)
    results["concat_linear"] = m
    print(f"  AUC={m['auc_roc']:.4f}  @2%={m['tpr_at_fpr']['2%']['tpr']:.2%}  @5%={m['tpr_at_fpr']['5%']['tpr']:.2%}  @10%={m['tpr_at_fpr']['10%']['tpr']:.2%}")

    # Save
    out = SCRIPT_DIR / "ml_results" / "test_concat_auth_positive.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

    # Summary
    print(f"\n{'Method':<20s} {'AUC':>6} {'@0.5%':>7} {'@1%':>7} {'@2%':>7} {'@5%':>7} {'@10%':>7}")
    for tag in sorted(results, key=lambda t: results[t]["tpr_at_fpr"]["2%"]["tpr"], reverse=True):
        m = results[tag]; tf = m["tpr_at_fpr"]
        def g(n): return tf.get(n, {}).get("tpr", 0)
        print(f"  {tag:<18s} {m['auc_roc']:>6.4f} {g('0.5%'):>7.2%} {g('1%'):>7.2%} {g('2%'):>7.2%} {g('5%'):>7.2%} {g('10%'):>7.2%}")


if __name__ == "__main__":
    main()
