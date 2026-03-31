#!/root/.venv/bin/python
"""
Multi-region fusion: train classifiers on concatenated per-region embeddings.

Models:
  - Attention block (cross-region self-attention pooler)
  - Simple MLP
  - XGBoost
  - SVM

Uses the global session-level val split for train/val.  Test evaluation
applies val-set thresholds and reports auth-positive metrics.

Usage:
  python train_region_fusion.py
  python train_region_fusion.py --embed-key vitl16_714
  python train_region_fusion.py --classifiers attention mlp xgb svm
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from config import (
    REGIONS,
    SEED,
    TARGET_FPRS,
    TARGET_FPR_NAMES,
    _global_val_split_path,
    cache_dir,
    compute_all_metrics,
    compute_metrics_auth_positive,
    create_global_val_split,
)

SCRIPT_DIR = Path(__file__).resolve().parent
EMBED_DIM = 1024
AVAILABLE_CLASSIFIERS = ["attention", "mlp", "xgb", "svm", "lgbm", "catboost"]


# ---------------------------------------------------------------------------
# Data building
# ---------------------------------------------------------------------------

def load_features_by_uuid(region, split, feat_key):
    cdir = cache_dir(region)
    prefix = f"test_{feat_key}" if split == "test" else feat_key
    feat_path = cdir / f"{prefix}_features.pt"
    label_path = cdir / f"{prefix}_labels.pt"
    uuid_path = cdir / f"{prefix}_uuids.pt"
    if not feat_path.exists():
        return {}
    features = torch.load(feat_path, weights_only=True)
    labels = torch.load(label_path, weights_only=True)
    uuids = torch.load(uuid_path, weights_only=False) if uuid_path.exists() else []
    return {u: (features[i].numpy(), int(labels[i].item())) for i, u in enumerate(uuids)}


def build_concat_dataset(split, feat_key):
    per_region = {r: load_features_by_uuid(r, split, feat_key) for r in REGIONS}
    all_uuids = sorted(set().union(*(d.keys() for d in per_region.values())))

    uuid_labels = {}
    for uuid in all_uuids:
        for r in REGIONS:
            if uuid in per_region[r]:
                uuid_labels[uuid] = per_region[r][uuid][1]
                break

    n_regions = len(REGIONS)
    X = np.zeros((len(all_uuids), EMBED_DIM * n_regions + n_regions), dtype=np.float32)
    y = np.zeros(len(all_uuids), dtype=np.int64)

    for i, uuid in enumerate(all_uuids):
        y[i] = uuid_labels[uuid]
        for j, region in enumerate(REGIONS):
            if uuid in per_region[region]:
                feat, _ = per_region[region][uuid]
                X[i, j * EMBED_DIM:(j + 1) * EMBED_DIM] = feat
                X[i, EMBED_DIM * n_regions + j] = 1.0

    print(f"  {split}: {len(all_uuids)} sessions, {int(y.sum())} fake, {len(y) - int(y.sum())} auth")
    return X, y, all_uuids


def split_by_global_val(X, y, uuids):
    """Split using the global session-level val split."""
    split_path = _global_val_split_path()
    if not split_path.exists():
        create_global_val_split()
    with open(split_path) as f:
        data = json.load(f)
    val_set = set(data["val_uuids"])

    train_mask = np.array([u not in val_set for u in uuids])
    val_mask = ~train_mask
    return X[train_mask], y[train_mask], X[val_mask], y[val_mask]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RegionAttentionFusion(nn.Module):
    """Cross-region self-attention followed by pooling and classification."""

    def __init__(self, embed_dim=EMBED_DIM, n_regions=4, n_heads=4,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_regions = n_regions
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, region_embeds, mask_bits):
        """
        region_embeds: (B, n_regions, embed_dim)
        mask_bits:     (B, n_regions)  1=available, 0=missing
        """
        B = region_embeds.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, region_embeds], dim=1)

        cls_mask = torch.zeros(B, 1, device=mask_bits.device, dtype=mask_bits.dtype)
        key_pad_mask = torch.cat([cls_mask, 1 - mask_bits], dim=1).bool()

        out = self.encoder(tokens, src_key_padding_mask=key_pad_mask)
        cls_out = out[:, 0]
        return self.head(cls_out).squeeze(-1)


class MLPFusion(nn.Module):
    def __init__(self, input_dim, hidden1=512, hidden2=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Torch training loop (shared by attention & MLP)
# ---------------------------------------------------------------------------

def _prepare_tensors(X, y, device):
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device))


def train_torch_model(model, X_train, y_train, X_val, y_val, device,
                      epochs=200, lr=1e-3, patience=15, batch_size=256):
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tr, y_tr = _prepare_tensors(X_train, y_train, device)
    X_vl, y_vl = _prepare_tensors(X_val, y_val, device)

    ds = TensorDataset(X_tr, y_tr)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_auc = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in dl:
            optimizer.zero_grad()
            loss = criterion(model.forward_flat(xb) if hasattr(model, "forward_flat") else model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model.forward_flat(X_vl) if hasattr(model, "forward_flat") else model(X_vl)
            val_scores = torch.sigmoid(val_logits).cpu().numpy()
        metrics = compute_all_metrics(y_val, val_scores)
        auc = metrics.get("auc_roc", 0)

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return best_auc


class AttentionWrapper(RegionAttentionFusion):
    """Wraps RegionAttentionFusion to accept flat concat input."""

    def forward_flat(self, x):
        n_r = self.n_regions
        embeds = x[:, :EMBED_DIM * n_r].view(-1, n_r, EMBED_DIM)
        masks = x[:, EMBED_DIM * n_r:]
        return self.forward(embeds, masks)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def train_attention(X_train, y_train, X_val, y_val, device):
    model = AttentionWrapper(EMBED_DIM, len(REGIONS), n_heads=4, n_layers=2).to(device)
    t0 = time.time()
    auc = train_torch_model(model, X_train, y_train, X_val, y_val, device, epochs=200, patience=15)
    return model, auc, time.time() - t0


def train_mlp(X_train, y_train, X_val, y_val, device):
    input_dim = X_train.shape[1]
    model = MLPFusion(input_dim).to(device)
    t0 = time.time()
    auc = train_torch_model(model, X_train, y_train, X_val, y_val, device, epochs=200, patience=15)
    return model, auc, time.time() - t0


def train_xgb(X_train, y_train, X_val, y_val, device):
    import xgboost as xgb
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    clf = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        seed=SEED, max_depth=6, learning_rate=0.05, n_estimators=300,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        device="cuda" if device == "cuda" else "cpu",
    )
    t0 = time.time()
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return clf, float(clf.score(X_val, y_val)), time.time() - t0


def train_svm(X_train, y_train, X_val, y_val):
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_vl_s = scaler.transform(X_val)
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
              class_weight={0: 1.0, 1: spw}, random_state=SEED)
    t0 = time.time()
    clf.fit(X_tr_s, y_train)
    return (clf, scaler), 0.0, time.time() - t0


def train_lgbm(X_train, y_train, X_val, y_val, device):
    import lightgbm as lgb
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    clf = lgb.LGBMClassifier(
        objective="binary", metric="auc", random_state=SEED,
        max_depth=6, learning_rate=0.05, n_estimators=300,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        device="gpu", verbose=-1,
    )
    t0 = time.time()
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return clf, 0.0, time.time() - t0


def train_catboost(X_train, y_train, X_val, y_val, device):
    from catboost import CatBoostClassifier
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    clf = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        scale_pos_weight=spw, random_seed=SEED,
        task_type="GPU", verbose=0,
    )
    t0 = time.time()
    clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    return clf, 0.0, time.time() - t0


def score_model(model_or_obj, X, device, name):
    """Return prediction scores for input X."""
    if name == "attention":
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.sigmoid(model_or_obj.forward_flat(X_t)).cpu().numpy()
    elif name == "mlp":
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.sigmoid(model_or_obj(X_t)).cpu().numpy()
    elif name in ("xgb", "lgbm", "catboost"):
        return model_or_obj.predict_proba(X)[:, 1]
    elif name == "svm":
        clf, scaler = model_or_obj
        return clf.predict_proba(scaler.transform(X))[:, 1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-region fusion classifiers")
    parser.add_argument("--embed-key", type=str, default="vitl16_714")
    parser.add_argument("--classifiers", nargs="+", default=AVAILABLE_CLASSIFIERS,
                        choices=AVAILABLE_CLASSIFIERS)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = args.device

    print(f"{'=' * 70}")
    print(f"REGION FUSION  |  embeddings: {args.embed_key}  |  device: {device}")
    print(f"{'=' * 70}")

    print("\nBuilding train dataset...")
    X_all, y_all, uuids_all = build_concat_dataset("train", args.embed_key)

    print("Splitting with global val split...")
    X_train, y_train, X_val, y_val = split_by_global_val(X_all, y_all, uuids_all)
    print(f"  Train: {len(X_train)} ({int(y_train.sum())} fake) | Val: {len(X_val)} ({int(y_val.sum())} fake)")

    print("\nBuilding test dataset...")
    X_test, y_test, uuids_test = build_concat_dataset("test", args.embed_key)

    results = {}
    out_dir = SCRIPT_DIR / "ml_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.classifiers:
        print(f"\n{'=' * 50}")
        print(f"  Training: {name.upper()}")
        print(f"{'=' * 50}")

        if name == "attention":
            model, val_auc, fit_time = train_attention(X_train, y_train, X_val, y_val, device)
        elif name == "mlp":
            model, val_auc, fit_time = train_mlp(X_train, y_train, X_val, y_val, device)
        elif name == "xgb":
            model, val_auc, fit_time = train_xgb(X_train, y_train, X_val, y_val, device)
        elif name == "svm":
            model, val_auc, fit_time = train_svm(X_train, y_train, X_val, y_val)
        elif name == "lgbm":
            model, val_auc, fit_time = train_lgbm(X_train, y_train, X_val, y_val, device)
        elif name == "catboost":
            model, val_auc, fit_time = train_catboost(X_train, y_train, X_val, y_val, device)
        else:
            continue

        val_scores = score_model(model, X_val, device, name)
        val_metrics = compute_all_metrics(y_val, val_scores)
        val_auth = compute_metrics_auth_positive(y_val, val_scores)

        test_scores = score_model(model, X_test, device, name)
        test_metrics = compute_all_metrics(y_test, test_scores)
        test_auth = compute_metrics_auth_positive(y_test, test_scores)

        tpr2_val = val_metrics.get("tpr_at_fpr", {}).get("2%", {}).get("tpr", 0)
        tpr2_test = test_auth.get("tpr_at_fpr", {}).get("2%", {}).get("tpr", 0)

        print(f"  fit={fit_time:.1f}s | Val AUC={val_metrics.get('auc_roc', 0):.4f} TPR@2%={tpr2_val:.4f}")
        print(f"  Test AUC(auth+)={test_auth.get('auc_roc', 0):.4f}")

        print(f"\n  Auth-positive test metrics:")
        print(f"  {'FPR target':<12} {'TPR':<10} {'Actual FPR':<12}")
        print(f"  {'-' * 34}")
        for fpr_name in TARGET_FPR_NAMES:
            d = test_auth.get("tpr_at_fpr", {}).get(fpr_name, {})
            print(f"  {fpr_name:<12} {d.get('tpr', 0):<10.4f} {d.get('actual_fpr', 0):<12.4f}")

        # Apply val thresholds to test (fixed-threshold evaluation)
        fixed_thresh = {}
        for fpr_name in TARGET_FPR_NAMES:
            vt = val_metrics.get("tpr_at_fpr", {}).get(fpr_name, {})
            if "threshold" not in vt:
                continue
            thresh = vt["threshold"]
            preds = (test_scores >= thresh).astype(int)
            n_pos = int(y_test.sum())
            n_neg = len(y_test) - n_pos
            tp = int(((preds == 1) & (y_test == 1)).sum())
            fp = int(((preds == 1) & (y_test == 0)).sum())
            fixed_thresh[fpr_name] = {
                "val_threshold": float(thresh),
                "test_tpr": float(tp / max(n_pos, 1)),
                "test_fpr": float(fp / max(n_neg, 1)),
                "auth_pass_rate": float(1 - fp / max(n_neg, 1)),
                "fake_miss_rate": float(1 - tp / max(n_pos, 1)),
            }

        results[f"fusion_{name}"] = {
            "val_metrics": val_metrics,
            "val_auth_positive": val_auth,
            "test_metrics": test_metrics,
            "test_auth_positive": test_auth,
            "fixed_threshold": fixed_thresh,
            "fit_time_s": fit_time,
            "embed_key": args.embed_key,
        }

    # Save
    out_path = out_dir / "region_fusion_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary table
    print(f"\n{'=' * 80}")
    print("REGION FUSION SUMMARY (auth-positive test metrics)")
    print(f"{'=' * 80}")
    fmt = "  {:<20s} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}"
    print(fmt.format("Method", "AUC", "@0.5%", "@1%", "@2%", "@5%", "@10%"))
    print(f"  {'-' * 74}")
    for tag in sorted(results, key=lambda t: results[t]["test_auth_positive"].get("tpr_at_fpr", {}).get("2%", {}).get("tpr", 0), reverse=True):
        m = results[tag]["test_auth_positive"]
        tf = m.get("tpr_at_fpr", {})
        def g(n):
            return tf.get(n, {}).get("tpr", 0)
        print(fmt.format(
            tag, f"{m.get('auc_roc', 0):.4f}",
            f"{g('0.5%'):.2%}", f"{g('1%'):.2%}", f"{g('2%'):.2%}",
            f"{g('5%'):.2%}", f"{g('10%'):.2%}"))


if __name__ == "__main__":
    main()
