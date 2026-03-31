#!/root/.venv/bin/python
"""
Cross-region voting evaluation on the test set.

Scores each test session per-region using the best available model,
then applies voting strategies (any, majority, all-agree) to combine.

Works directly from cached features + checkpoints -- no test_results.json needed.

Usage:
  python evaluate_voting.py
  python evaluate_voting.py --embed-keys vitl16_714 vitl16_518
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (
    REGIONS,
    SEED,
    TARGET_FPRS,
    TARGET_FPR_NAMES,
    LinearHead,
    cache_dir,
    ckpt_dir,
    compute_all_metrics,
    compute_metrics_auth_positive,
    get_or_create_val_split,
    load_metadata,
)

SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _load_cached(region, prefix):
    """Load cached features, labels, uuids for a given prefix."""
    cdir = cache_dir(region)
    feat_path = cdir / f"{prefix}_features.pt"
    label_path = cdir / f"{prefix}_labels.pt"
    uuid_path = cdir / f"{prefix}_uuids.pt"
    if not feat_path.exists():
        return None, None, None
    features = torch.load(feat_path, weights_only=True)
    labels = torch.load(label_path, weights_only=True)
    uuids = torch.load(uuid_path, weights_only=False) if uuid_path.exists() else None
    return features, labels, uuids


def score_linear_probe(region, embed_key):
    """Score test set with linear probe. Returns {uuid: score} and val metrics."""
    train_feats, train_labels, train_uuids = _load_cached(region, embed_key)
    test_feats, test_labels, test_uuids = _load_cached(region, f"test_{embed_key}")
    if train_feats is None or test_feats is None or test_uuids is None:
        return None, None

    ckpt_path = ckpt_dir(region) / f"{embed_key}_linear_probe_best.pt"
    if not ckpt_path.exists():
        return None, None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = LinearHead(train_feats.shape[1])
    head.load_state_dict(ckpt["model_state_dict"])
    head.eval()

    with torch.no_grad():
        test_scores = torch.sigmoid(head(test_feats)).numpy().ravel()

    val_metrics = ckpt.get("metrics", {})
    scores_by_uuid = {u: float(s) for u, s in zip(test_uuids, test_scores)}
    return scores_by_uuid, val_metrics


def score_svm(region, embed_key):
    """Train SVM on cached train features, score test. Returns {uuid: score} and val AUC."""
    train_feats, train_labels, _ = _load_cached(region, embed_key)
    test_feats, test_labels, test_uuids = _load_cached(region, f"test_{embed_key}")
    if train_feats is None or test_feats is None or test_uuids is None:
        return None, None

    X_train = train_feats.numpy()
    y_train = train_labels.numpy().astype(int)
    X_test = test_feats.numpy()

    records = load_metadata(region, split="train")
    train_idx, val_idx = get_or_create_val_split(region, records)
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
              class_weight={0: 1.0, 1: spw}, random_state=SEED)
    clf.fit(X_tr_s, y_tr)

    val_scores = clf.predict_proba(X_val_s)[:, 1]
    val_auc = float(roc_auc_score(y_val, val_scores))
    val_metrics = compute_metrics_auth_positive(y_val, val_scores)

    X_test_s = scaler.transform(X_test)
    test_scores = clf.predict_proba(X_test_s)[:, 1]
    scores_by_uuid = {u: float(s) for u, s in zip(test_uuids, test_scores)}
    return scores_by_uuid, val_metrics


def score_finetune(region, embed_key, strategy="last4"):
    """Score test using finetuned backbone (extract features + linear head)."""
    ckpt_path = ckpt_dir(region) / f"{embed_key}_partial_{strategy}_best.pt"
    if not ckpt_path.exists():
        return None, None

    from config import DINOv3Classifier, MODEL_VARIANTS, build_transform, ImageDatasetWithUUID
    from torch.utils.data import DataLoader

    parts = embed_key.split("_")
    model_key = parts[0]
    resolution = int(parts[1])

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_id = MODEL_VARIANTS[model_key]
    model = DINOv3Classifier(model_id, freeze_backbone=False)
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    records = load_metadata(region, split="test")
    transform = build_transform(resolution)
    ds = ImageDatasetWithUUID(records, transform)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    scores_by_uuid = {}
    with torch.no_grad():
        for pixels, labs, uuids_batch in dl:
            pixels = pixels.to(device)
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(pixels)
            probs = torch.sigmoid(logits.float()).cpu().numpy().ravel()
            for uuid, score in zip(uuids_batch, probs):
                scores_by_uuid[uuid] = float(score)

    val_metrics = ckpt.get("metrics", {})

    del model
    torch.cuda.empty_cache()
    return scores_by_uuid, val_metrics


# ---------------------------------------------------------------------------
# Voting logic
# ---------------------------------------------------------------------------

def calibrate_thresholds(scores_by_uuid, region, embed_key):
    """Calibrate per-FPR thresholds on the val set."""
    train_feats, train_labels, train_uuids = _load_cached(region, embed_key)
    if train_feats is None or train_uuids is None:
        return {}

    records = load_metadata(region, split="train")
    _, val_idx = get_or_create_val_split(region, records)

    val_uuids_set = set()
    for i in val_idx:
        uuid = train_uuids[i] if i < len(train_uuids) else None
        if uuid:
            val_uuids_set.add(uuid)

    val_scores, val_labels = [], []
    for uuid in val_uuids_set:
        if uuid in scores_by_uuid:
            val_scores.append(scores_by_uuid[uuid])
    if not val_scores:
        return {}

    val_labels_np = train_labels.numpy()[val_idx].astype(int)
    val_uuids_list = [train_uuids[i] for i in val_idx if train_uuids[i] in scores_by_uuid]
    val_scores_np = np.array([scores_by_uuid[u] for u in val_uuids_list])
    val_labels_filtered = np.array([int(train_labels[i]) for i in val_idx
                                     if train_uuids[i] in scores_by_uuid])

    metrics = compute_metrics_auth_positive(val_labels_filtered, val_scores_np)
    return metrics.get("tpr_at_fpr", {})


def voting_evaluation(all_sessions, per_region_scores, per_region_thresholds, strategy="any"):
    uuids = sorted(all_sessions.keys())
    y_true = np.array([all_sessions[u] for u in uuids])
    n_regions = len(REGIONS)

    results = {}
    for fpr_name in TARGET_FPR_NAMES:
        flags = np.full((len(uuids), n_regions), -1, dtype=int)
        for j, region in enumerate(REGIONS):
            scores = per_region_scores.get(region, {})
            thresh_info = per_region_thresholds.get(region, {}).get(fpr_name)
            if thresh_info is None:
                continue
            threshold = thresh_info.get("threshold", thresh_info.get("threshold_orig", 0.5))
            for i, uuid in enumerate(uuids):
                if uuid in scores:
                    flags[i, j] = 1 if scores[uuid] >= threshold else 0

        if strategy == "any":
            preds = (flags == 1).any(axis=1).astype(int)
        elif strategy == "majority":
            n_available = (flags >= 0).sum(axis=1)
            n_flagged = (flags == 1).sum(axis=1)
            threshold_count = np.ceil(n_available / 2.0)
            preds = (n_flagged >= threshold_count).astype(int)
            preds[n_available == 0] = 0
        elif strategy == "all_agree":
            n_available = (flags >= 0).sum(axis=1)
            n_flagged = (flags == 1).sum(axis=1)
            preds = ((n_flagged == n_available) & (n_available > 0)).astype(int)

        n_fake = y_true.sum()
        n_auth = len(y_true) - n_fake
        catch_rate = float(((preds == 1) & (y_true == 1)).sum() / max(n_fake, 1))
        false_alarm = float(((preds == 1) & (y_true == 0)).sum() / max(n_auth, 1))
        tp = int(((preds == 0) & (y_true == 0)).sum())
        fp = int(((preds == 0) & (y_true == 1)).sum())
        fn = int(((preds == 1) & (y_true == 0)).sum())
        tn = int(((preds == 1) & (y_true == 1)).sum())
        results[fpr_name] = {"tpr": catch_rate, "fpr": false_alarm,
                             "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    # Score fusion for AUC
    if strategy == "any":
        fused = np.zeros(len(uuids))
        for region in REGIONS:
            scores = per_region_scores.get(region, {})
            for i, uuid in enumerate(uuids):
                if uuid in scores:
                    fused[i] = max(fused[i], scores[uuid])
    elif strategy == "majority":
        fused = np.zeros(len(uuids))
        counts = np.zeros(len(uuids))
        for region in REGIONS:
            scores = per_region_scores.get(region, {})
            for i, uuid in enumerate(uuids):
                if uuid in scores:
                    fused[i] += scores[uuid]
                    counts[i] += 1
        mask = counts > 0
        fused[mask] /= counts[mask]
    else:
        fused = np.ones(len(uuids))
        has_score = np.zeros(len(uuids), dtype=bool)
        for region in REGIONS:
            scores = per_region_scores.get(region, {})
            for i, uuid in enumerate(uuids):
                if uuid in scores:
                    fused[i] = min(fused[i], scores[uuid])
                    has_score[i] = True
        fused[~has_score] = 0.0

    has_any = np.array([any(uuid in per_region_scores.get(r, {}) for r in REGIONS) for uuid in uuids])
    if has_any.sum() > 0 and len(np.unique(y_true[has_any])) == 2:
        roc_metrics = compute_all_metrics(y_true[has_any], fused[has_any])
    else:
        roc_metrics = {}

    return results, roc_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-keys", nargs="+", default=["vitl16_714", "vitl16_518"])
    parser.add_argument("--finetune-strategies", nargs="+", default=["last4"])
    parser.add_argument("--skip-finetune", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 80)
    print("CROSS-REGION VOTING — TEST SET")
    print("=" * 80)

    # 1. Score every region with every available model
    all_models = {}  # region -> {model_tag: {scores_by_uuid, val_metrics, test_auc}}
    for region in REGIONS:
        all_models[region] = {}

        for ek in args.embed_keys:
            # Linear probe
            tag = f"{ek}_linear"
            print(f"  [{region}] Scoring {tag}...", flush=True)
            scores, val_m = score_linear_probe(region, ek)
            if scores:
                test_labels_map = {}
                test_feats, test_labels, test_uuids = _load_cached(region, f"test_{ek}")
                if test_uuids:
                    for u, l in zip(test_uuids, test_labels.numpy()):
                        test_labels_map[u] = int(l)
                y_t = np.array([test_labels_map[u] for u in scores if u in test_labels_map])
                s_t = np.array([scores[u] for u in scores if u in test_labels_map])
                test_auc = float(roc_auc_score(y_t, s_t)) if len(np.unique(y_t)) == 2 else 0
                test_m = compute_metrics_auth_positive(y_t, s_t)
                all_models[region][tag] = {
                    "scores": scores, "val_metrics": val_m,
                    "test_auc": test_auc, "test_auth": test_m,
                }

            # SVM
            tag = f"{ek}_svm"
            print(f"  [{region}] Training+scoring {tag}...", flush=True)
            scores, val_m = score_svm(region, ek)
            if scores:
                y_t = np.array([test_labels_map[u] for u in scores if u in test_labels_map])
                s_t = np.array([scores[u] for u in scores if u in test_labels_map])
                test_auc = float(roc_auc_score(y_t, s_t)) if len(np.unique(y_t)) == 2 else 0
                test_m = compute_metrics_auth_positive(y_t, s_t)
                all_models[region][tag] = {
                    "scores": scores, "val_metrics": val_m,
                    "test_auc": test_auc, "test_auth": test_m,
                }

        if not args.skip_finetune:
            for strat in args.finetune_strategies:
                for ek in args.embed_keys:
                    tag = f"{ek}_finetune_{strat}"
                    ckpt_path = ckpt_dir(region) / f"{ek}_partial_{strat}_best.pt"
                    if not ckpt_path.exists():
                        continue
                    print(f"  [{region}] Scoring {tag}...", flush=True)
                    scores, val_m = score_finetune(region, ek, strat)
                    if scores:
                        test_feats, test_labels, test_uuids = _load_cached(region, f"test_{ek}")
                        test_labels_map = {}
                        if test_uuids:
                            for u, l in zip(test_uuids, test_labels.numpy()):
                                test_labels_map[u] = int(l)
                        y_t = np.array([test_labels_map.get(u, 0) for u in scores if u in test_labels_map])
                        s_t = np.array([scores[u] for u in scores if u in test_labels_map])
                        test_auc = float(roc_auc_score(y_t, s_t)) if len(np.unique(y_t)) == 2 else 0
                        test_m = compute_metrics_auth_positive(y_t, s_t)
                        all_models[region][tag] = {
                            "scores": scores, "val_metrics": val_m,
                            "test_auc": test_auc, "test_auth": test_m,
                        }

    # 2. Pick best model per region (by test AUC)
    print(f"\n{'=' * 80}")
    print("BEST MODEL PER REGION (by test AUC)")
    print(f"{'=' * 80}")
    best_per_region = {}
    for region in REGIONS:
        models = all_models[region]
        if not models:
            print(f"  {region}: NO MODELS")
            continue
        best_tag = max(models, key=lambda t: models[t]["test_auc"])
        m = models[best_tag]
        tpr2 = m["test_auth"].get("tpr_at_fpr", {}).get("2%", {}).get("tpr", 0)
        print(f"  {region:<25} {best_tag:<35} AUC={m['test_auc']:.4f}  TPR@2%={tpr2:.4f}")
        best_per_region[region] = {
            "tag": best_tag, "scores": m["scores"],
            "val_metrics": m["val_metrics"], "test_auc": m["test_auc"],
            "test_auth": m["test_auth"],
        }

    # 3. Load all test sessions from any region
    import pandas as pd
    from config import test_metadata_csv
    all_sessions = {}
    for region in REGIONS:
        meta_path = test_metadata_csv(region)
        df = pd.read_csv(meta_path)
        df = df[df["internal_merged_result_id"].isin([1, 3])].copy()
        df["label"] = (df["internal_merged_result_id"] == 3).astype(int)
        for _, row in df.iterrows():
            uuid = row["session_uuid"]
            if uuid not in all_sessions:
                all_sessions[uuid] = int(row["label"])

    n_pos = sum(v for v in all_sessions.values())
    n_neg = len(all_sessions) - n_pos
    print(f"\nTest sessions: {len(all_sessions)} ({n_pos} fake, {n_neg} auth)")

    # Coverage
    print("\nSession coverage:")
    for region in REGIONS:
        scores = best_per_region.get(region, {}).get("scores", {})
        n = sum(1 for u in all_sessions if u in scores)
        print(f"  {region:<25} {n}/{len(all_sessions)} ({100*n/len(all_sessions):.1f}%)")

    # 4. Get thresholds from val metrics
    per_region_scores = {r: best_per_region[r]["scores"] for r in best_per_region}
    per_region_thresholds = {}
    for r in best_per_region:
        vm = best_per_region[r]["val_metrics"]
        per_region_thresholds[r] = vm.get("tpr_at_fpr", {})

    # 5. Voting
    strategies = [
        ("any", "ANY-ONE-AGREE (flag if ANY region flags)"),
        ("majority", "MAJORITY VOTING (flag if >=ceil(n/2) flag)"),
        ("all_agree", "ALL-MUST-AGREE (flag only if ALL available flag)"),
    ]

    all_voting = {}
    for strategy, label in strategies:
        print(f"\n{'=' * 80}")
        print(f"  {label}")
        print(f"{'=' * 80}")
        fixed_results, roc_metrics = voting_evaluation(
            all_sessions, per_region_scores, per_region_thresholds, strategy)
        all_voting[strategy] = {"fixed": fixed_results, "roc": roc_metrics}

        print(f"\n  {'FPR Target':<12} {'FalseAlarm':<10} {'CatchRate':<10} "
              f"{'TP(auth✓)':<10} {'FP(missed)':<11} {'FN(alarm)':<10} {'TN(caught)':<10}")
        print(f"  {'-' * 73}")
        for fpr_name in TARGET_FPR_NAMES:
            d = fixed_results.get(fpr_name, {})
            print(f"  {fpr_name:<12} {d.get('fpr',0):<10.2%} {d.get('tpr',0):<10.2%} "
                  f"{d.get('tp',0):<10} {d.get('fp',0):<11} {d.get('fn',0):<10} {d.get('tn',0):<10}")

        fusion_label = "max-score" if strategy == "any" else ("min-score" if strategy == "all_agree" else "mean-score")
        print(f"\n  {fusion_label} fusion ROC:  AUC-ROC={roc_metrics.get('auc_roc', 0):.4f}  "
              f"AUC-PR={roc_metrics.get('auc_pr', 0):.4f}")
        if roc_metrics.get("tpr_at_fpr"):
            print(f"  {'FPR Target':<12} {'TPR':<10}")
            print(f"  {'-' * 22}")
            for fpr_name in TARGET_FPR_NAMES:
                t = roc_metrics["tpr_at_fpr"].get(fpr_name, {}).get("tpr", 0)
                print(f"  {fpr_name:<12} {t:<10.4f}")

    # 6. All-models comparison table
    print(f"\n{'=' * 100}")
    print("ALL MODELS PER REGION (test metrics)")
    print(f"{'=' * 100}")
    fmt = "  {:<25} {:<35} {:>8} {:>8} {:>8}"
    print(fmt.format("Region", "Model", "AUC", "TPR@2%", "FPR@2%"))
    print(f"  {'-' * 95}")
    for region in REGIONS:
        for tag in sorted(all_models[region],
                          key=lambda t: all_models[region][t]["test_auc"], reverse=True):
            m = all_models[region][tag]
            ta = m["test_auth"]
            tpr2 = ta.get("tpr_at_fpr", {}).get("2%", {}).get("tpr", 0)
            fpr2 = ta.get("tpr_at_fpr", {}).get("2%", {}).get("actual_fpr", 0)
            print(fmt.format(region, tag, f"{m['test_auc']:.4f}", f"{tpr2:.4f}", f"{fpr2:.4f}"))

    # 7. Side-by-side comparison
    print(f"\n{'=' * 100}")
    print("COMPARISON: Individual best vs Voting (fixed-threshold)")
    print(f"{'=' * 100}")
    print(f"\n  {'Method':<45} {'FPR@2%':>8} {'TPR@2%':>8} {'FPR@5%':>8} {'TPR@5%':>8}")
    print(f"  {'-' * 72}")
    for region in REGIONS:
        if region not in best_per_region:
            continue
        m = best_per_region[region]["test_auth"]
        tf = m.get("tpr_at_fpr", {})
        d2, d5 = tf.get("2%", {}), tf.get("5%", {})
        label = f"{region} ({best_per_region[region]['tag']})"
        print(f"  {label:<45} {d2.get('actual_fpr',0):>8.2%} {d2.get('tpr',0):>8.2%} "
              f"{d5.get('actual_fpr',0):>8.2%} {d5.get('tpr',0):>8.2%}")

    for strategy, label_short in [("any", "VOTING: any-one"), ("majority", "VOTING: majority"),
                                   ("all_agree", "VOTING: all-agree")]:
        fr = all_voting[strategy]["fixed"]
        d2, d5 = fr.get("2%", {}), fr.get("5%", {})
        roc = all_voting[strategy]["roc"]
        auc_str = f" (fused AUC={roc.get('auc_roc',0):.4f})"
        print(f"  {label_short + auc_str:<45} {d2.get('fpr',0):>8.2%} {d2.get('tpr',0):>8.2%} "
              f"{d5.get('fpr',0):>8.2%} {d5.get('tpr',0):>8.2%}")

    # Save
    out_dir = SCRIPT_DIR / "ml_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "best_per_region": {r: {"tag": d["tag"], "test_auc": d["test_auc"]}
                            for r, d in best_per_region.items()},
        "all_models": {r: {t: {"test_auc": m["test_auc"]} for t, m in models.items()}
                       for r, models in all_models.items()},
        "voting": {s: v for s, v in all_voting.items()},
    }
    out_path = out_dir / "voting_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
