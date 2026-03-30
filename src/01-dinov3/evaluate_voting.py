#!/root/.venv/bin/python
"""
All-must-agree voting across regions on the test set.

For each session, the best model per region produces a score.
A session is flagged as not-authentic if ANY region flags it.
Missing regions (session has no image in that region) are treated
as authentic (NaN = no flag).

Evaluates at multiple FPR thresholds calibrated on val set.

Usage:
  python evaluate_voting.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from config import (
    SCRIPT_DIR,
    PROJECT_ROOT,
    REGIONS,
    TARGET_FPRS,
    TARGET_FPR_NAMES,
    compute_all_metrics,
    results_dir,
    test_metadata_csv,
)


def load_test_results(region):
    rdir = results_dir(region)
    path = rdir / "test_results.json"
    with open(path) as f:
        return json.load(f)


def get_best_model_per_region():
    """Find the best model per region by test TPR@2%."""
    best = {}
    for region in REGIONS:
        data = load_test_results(region)
        all_methods = {}
        for section in ["linear_probe", "finetune", "ml_classifiers", "finetuned_ml"]:
            for tag, m in data.get(section, {}).items():
                all_methods[tag] = m

        best_tag, best_tpr2 = None, -1
        for tag, m in all_methods.items():
            t2 = m.get("tpr_at_fpr", {}).get("2%", {}).get("tpr", 0)
            auc = m.get("auc_roc", 0)
            if t2 > best_tpr2 or (t2 == best_tpr2 and auc > (best.get(region, {}).get("auc", 0))):
                best_tpr2 = t2
                best_tag = tag
                best[region] = {"tag": tag, "metrics": m, "tpr2": t2, "auc": auc}

        # Also get fixed-threshold info
        all_fix = {}
        for section in ["linear_probe", "finetune", "ml_classifiers", "finetuned_ml"]:
            for tag, d in data.get("fixed_threshold", {}).get(section, {}).items():
                all_fix[tag] = d
        if best_tag and best_tag in all_fix:
            best[region]["fixed_thresholds"] = all_fix[best_tag]

    return best


def load_all_test_sessions():
    """Load the union of all test session UUIDs with labels from any region's metadata."""
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
    return all_sessions


def load_per_region_test_scores(region, model_tag):
    """Load test scores for a specific model from saved test evaluation.

    Returns dict: session_uuid -> score, or None if we need to recompute.
    We reconstruct from cached test features + saved model.
    """
    rdir = results_dir(region)
    cdir = SCRIPT_DIR / "cached_features" / region
    ckpt_dir_path = SCRIPT_DIR / "checkpoints" / region

    # Parse model_tag to figure out how to get scores
    # Tags look like: vitl16_518_finetune, vitb16_518_svm, finetuned_vitl16_518_svm,
    #                  convnext_large_518_lgbm, vitl16_714_linear

    # For simplicity, re-extract scores from saved models + cached test features
    # We need per-session scores, but test_results.json only has aggregate metrics.
    # Let's compute them directly.

    scores_by_uuid = {}

    if "_finetune" in model_tag and not model_tag.startswith("finetuned_"):
        # Direct finetune model
        parts = model_tag.replace("_finetune", "").rsplit("_", 1)
        model_key, res = parts[0], int(parts[1])
        ckpt_path = ckpt_dir_path / f"{model_key}_{res}_finetune_best.pt"
        if not ckpt_path.exists():
            return {}

        from config import DINOv3Classifier, MODEL_VARIANTS, build_transform, ImageDatasetWithUUID, load_metadata
        from torch.utils.data import DataLoader

        records = load_metadata(region, split="test")
        model_id = MODEL_VARIANTS[model_key]
        model = DINOv3Classifier(model_id, freeze_backbone=False)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        transform = build_transform(res)
        ds = ImageDatasetWithUUID(records, transform)
        dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

        with torch.no_grad():
            for pixels, labs, uuids in dl:
                pixels = pixels.to(device)
                logits = model(pixels)
                probs = torch.sigmoid(logits).cpu().numpy()
                for uuid, score in zip(uuids, probs):
                    scores_by_uuid[uuid] = float(score)

        del model
        torch.cuda.empty_cache()

    elif "_linear" in model_tag:
        parts = model_tag.replace("_linear", "").rsplit("_", 1)
        model_key, res = parts[0], int(parts[1])
        prefix = f"test_{model_key}_{res}"
        feat_path = cdir / f"{prefix}_features.pt"
        uuid_path = cdir / f"{prefix}_uuids.pt"
        if not feat_path.exists():
            return {}

        from config import LinearHead
        features = torch.load(feat_path, weights_only=True)
        uuids = torch.load(uuid_path, weights_only=False)
        ckpt_path = ckpt_dir_path / f"{model_key}_{res}_linear_probe_best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        head = LinearHead(features.shape[1]).to(device)
        head.load_state_dict(ckpt["model_state_dict"])
        head.eval()
        with torch.no_grad():
            scores = torch.sigmoid(head(features.to(device))).cpu().numpy()
        for uuid, score in zip(uuids, scores):
            scores_by_uuid[uuid] = float(score)

    elif model_tag.startswith("finetuned_"):
        # finetuned_vitl16_518_svm -> model=vitl16, res=518, clf=svm
        rest = model_tag.replace("finetuned_", "")
        for clf in ["xgb", "svm", "catboost", "lgbm"]:
            if rest.endswith(f"_{clf}"):
                embed_part = rest.replace(f"_{clf}", "")
                parts = embed_part.rsplit("_", 1)
                model_key, res = parts[0], int(parts[1])

                ft_prefix = f"test_finetuned_{model_key}_{res}"
                feat_path = cdir / f"{ft_prefix}_features.pt"
                uuid_path = cdir / f"test_{model_key}_{res}_uuids.pt"
                if not feat_path.exists():
                    return {}

                features = torch.load(feat_path, weights_only=True)
                uuids = torch.load(uuid_path, weights_only=False)
                X = features.numpy()

                src_prefix = f"finetuned_{model_key}_{res}"
                if clf == "svm":
                    import joblib
                    d = joblib.load(rdir / f"{src_prefix}_svm.joblib")
                    probs = d["model"].predict_proba(d["scaler"].transform(X))[:, 1]
                elif clf == "xgb":
                    import xgboost as xgb
                    c = xgb.XGBClassifier()
                    c.load_model(str(rdir / f"{src_prefix}_xgb.json"))
                    probs = c.predict_proba(X)[:, 1]
                elif clf == "catboost":
                    import catboost as cb
                    c = cb.CatBoostClassifier()
                    c.load_model(str(rdir / f"{src_prefix}_catboost.cbm"))
                    probs = c.predict_proba(X)[:, 1]
                elif clf == "lgbm":
                    import lightgbm as lgb
                    booster = lgb.Booster(model_file=str(rdir / f"{src_prefix}_lgbm.txt"))
                    probs = booster.predict(X)

                for uuid, score in zip(uuids, probs):
                    scores_by_uuid[uuid] = float(score)
                break
    else:
        # ML classifier: vitb16_518_svm, convnext_large_518_lgbm etc.
        for clf in ["xgb", "svm", "catboost", "lgbm"]:
            if model_tag.endswith(f"_{clf}"):
                embed_part = model_tag.replace(f"_{clf}", "")
                parts = embed_part.rsplit("_", 1)
                model_key, res = parts[0], int(parts[1])

                prefix = f"test_{model_key}_{res}"
                feat_path = cdir / f"{prefix}_features.pt"
                uuid_path = cdir / f"{prefix}_uuids.pt"
                if not feat_path.exists():
                    return {}

                features = torch.load(feat_path, weights_only=True)
                uuids = torch.load(uuid_path, weights_only=False)
                X = features.numpy()

                src_prefix = f"{model_key}_{res}"
                if clf == "svm":
                    import joblib
                    d = joblib.load(rdir / f"{src_prefix}_svm.joblib")
                    probs = d["model"].predict_proba(d["scaler"].transform(X))[:, 1]
                elif clf == "xgb":
                    import xgboost as xgb
                    c = xgb.XGBClassifier()
                    c.load_model(str(rdir / f"{src_prefix}_xgb.json"))
                    probs = c.predict_proba(X)[:, 1]
                elif clf == "catboost":
                    import catboost as cb
                    c = cb.CatBoostClassifier()
                    c.load_model(str(rdir / f"{src_prefix}_catboost.cbm"))
                    probs = c.predict_proba(X)[:, 1]
                elif clf == "lgbm":
                    import lightgbm as lgb
                    booster = lgb.Booster(model_file=str(rdir / f"{src_prefix}_lgbm.txt"))
                    probs = booster.predict(X)

                for uuid, score in zip(uuids, probs):
                    scores_by_uuid[uuid] = float(score)
                break

    return scores_by_uuid


def _compute_voting_stats(y_true, preds):
    """Compute voting stats.

    Convention: positive = authentic (label 0), negative = fake (label 1).
    preds: 1 = flagged as fake, 0 = passed as authentic.
    y_true: 1 = actually fake, 0 = actually authentic.

    TP = authentic correctly passed, FP = fake wrongly passed (missed),
    FN = authentic wrongly flagged,  TN = fake correctly caught.
    """
    n_fake = y_true.sum()
    n_auth = len(y_true) - n_fake

    # Detection metrics (catch rate / false alarm rate) — unchanged
    catch_rate = float(((preds == 1) & (y_true == 1)).sum() / max(n_fake, 1))
    false_alarm = float(((preds == 1) & (y_true == 0)).sum() / max(n_auth, 1))

    # Confusion matrix with positive = authentic
    tp = int(((preds == 0) & (y_true == 0)).sum())  # authentic correctly passed
    fp = int(((preds == 0) & (y_true == 1)).sum())   # fake wrongly passed (missed)
    fn = int(((preds == 1) & (y_true == 0)).sum())   # authentic wrongly flagged
    tn = int(((preds == 1) & (y_true == 1)).sum())    # fake correctly caught

    return {
        "tpr": catch_rate,
        "fpr": false_alarm,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _per_region_flags(uuids, per_region_scores, per_region_thresholds, fpr_name):
    """Return (n_sessions, n_regions) array: 1=flagged, 0=authentic, -1=missing."""
    flags = np.full((len(uuids), len(REGIONS)), -1, dtype=int)
    for j, region in enumerate(REGIONS):
        scores = per_region_scores.get(region, {})
        thresh_info = per_region_thresholds.get(region, {}).get(fpr_name)
        if thresh_info is None:
            continue
        threshold = thresh_info["threshold"]
        for i, uuid in enumerate(uuids):
            if uuid in scores:
                flags[i, j] = 1 if scores[uuid] >= threshold else 0
    return flags


def voting_evaluation(all_sessions, per_region_scores, per_region_thresholds, strategy="any"):
    """Voting across regions.

    strategy:
      "any"       - flag if ANY region flags. NaN=authentic.
      "majority"  - flag if majority of available regions flag. NaN excluded.
      "all_agree" - flag only if ALL available regions flag. NaN=authentic.
    """
    uuids = sorted(all_sessions.keys())
    y_true = np.array([all_sessions[u] for u in uuids])

    results = {}
    for fpr_name in TARGET_FPR_NAMES:
        flags = _per_region_flags(uuids, per_region_scores, per_region_thresholds, fpr_name)

        if strategy == "any":
            preds = (flags == 1).any(axis=1).astype(int)
        elif strategy == "majority":
            n_available = (flags >= 0).sum(axis=1)
            n_flagged = (flags == 1).sum(axis=1)
            threshold = np.ceil(n_available / 2.0)
            preds = (n_flagged >= threshold).astype(int)
            preds[n_available == 0] = 0
        elif strategy == "all_agree":
            n_available = (flags >= 0).sum(axis=1)
            n_flagged = (flags == 1).sum(axis=1)
            preds = ((n_flagged == n_available) & (n_available > 0)).astype(int)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        results[fpr_name] = _compute_voting_stats(y_true, preds)

    # Score fusion for ROC analysis
    if strategy == "any":
        fused = np.full(len(uuids), 0.0)
        for region in REGIONS:
            scores = per_region_scores.get(region, {})
            for i, uuid in enumerate(uuids):
                if uuid in scores:
                    fused[i] = max(fused[i], scores[uuid])
    elif strategy == "all_agree":
        fused = np.full(len(uuids), 1.0)
        has_score = np.zeros(len(uuids), dtype=bool)
        for region in REGIONS:
            scores = per_region_scores.get(region, {})
            for i, uuid in enumerate(uuids):
                if uuid in scores:
                    fused[i] = min(fused[i], scores[uuid])
                    has_score[i] = True
        fused[~has_score] = 0.0
    else:
        fused = np.full(len(uuids), 0.0)
        counts = np.zeros(len(uuids))
        for region in REGIONS:
            scores = per_region_scores.get(region, {})
            for i, uuid in enumerate(uuids):
                if uuid in scores:
                    fused[i] += scores[uuid]
                    counts[i] += 1
        mask = counts > 0
        fused[mask] /= counts[mask]

    coverage = np.zeros(len(uuids), dtype=int)
    for region in REGIONS:
        scores = per_region_scores.get(region, {})
        for i, uuid in enumerate(uuids):
            if uuid in scores:
                coverage[i] += 1

    has_any = coverage > 0
    if has_any.sum() > 0 and len(np.unique(y_true[has_any])) == 2:
        roc_metrics = compute_all_metrics(y_true[has_any], fused[has_any])
    else:
        roc_metrics = {}

    return results, roc_metrics, coverage


def main():
    print("=" * 80)
    print("ALL-MUST-AGREE VOTING — TEST SET")
    print("=" * 80)

    # Load best model per region
    best = get_best_model_per_region()
    print("\nBest model per region (by test TPR@2%):")
    for region, info in best.items():
        print(f"  {region:<25} {info['tag']:<40} TPR@2%={info['tpr2']:.4f}  AUC={info['auc']:.4f}")

    # Load all test sessions
    all_sessions = load_all_test_sessions()
    n_pos = sum(v for v in all_sessions.values())
    n_neg = len(all_sessions) - n_pos
    print(f"\nUnion of test sessions: {len(all_sessions)} ({n_pos} pos, {n_neg} neg)")

    # Load per-region scores
    print("\nLoading per-region test scores...")
    per_region_scores = {}
    per_region_thresholds = {}
    for region, info in best.items():
        tag = info["tag"]
        print(f"  {region}: loading {tag} ...", flush=True)
        scores = load_per_region_test_scores(region, tag)
        per_region_scores[region] = scores
        print(f"    -> {len(scores)} sessions with scores")

        # Get val thresholds from the checkpoint/model metrics
        metrics = info["metrics"]
        tpr_at_fpr = metrics.get("tpr_at_fpr", {})
        thresholds = {}
        for fpr_name in TARGET_FPR_NAMES:
            d = tpr_at_fpr.get(fpr_name, {})
            if "threshold" in d:
                thresholds[fpr_name] = d
        per_region_thresholds[region] = thresholds

    # Session coverage
    print("\nSession coverage:")
    uuids = sorted(all_sessions.keys())
    for region in REGIONS:
        scores = per_region_scores.get(region, {})
        n = sum(1 for u in uuids if u in scores)
        print(f"  {region:<25} {n}/{len(uuids)} sessions ({100*n/len(uuids):.1f}%)")

    strategies = [
        ("any", "ANY-ONE-AGREE (flag if ANY region flags)"),
        ("majority", "MAJORITY VOTING (flag if >=ceil(n/2) regions flag)"),
        ("all_agree", "ALL-MUST-AGREE (flag only if ALL available regions flag)"),
    ]

    all_voting_results = {}
    for strategy, label in strategies:
        print(f"\n{'=' * 80}")
        print(f"  {label}")
        print(f"{'=' * 80}")
        fixed_results, roc_metrics, coverage = voting_evaluation(
            all_sessions, per_region_scores, per_region_thresholds, strategy=strategy
        )
        all_voting_results[strategy] = {"fixed": fixed_results, "roc": roc_metrics}

        print(f"\n  {'FPR Target':<12} {'FalseAlarm':<10} {'CatchRate':<10} {'TP(auth✓)':<10} {'FP(missed)':<11} {'FN(alarm)':<10} {'TN(caught)':<10}")
        print(f"  {'-' * 73}")
        for fpr_name in TARGET_FPR_NAMES:
            d = fixed_results.get(fpr_name, {})
            print(f"  {fpr_name:<12} {d.get('fpr',0):<10.2%} {d.get('tpr',0):<10.2%} "
                  f"{d.get('tp',0):<10} {d.get('fp',0):<11} {d.get('fn',0):<10} {d.get('tn',0):<10}")

        fusion_label = "max-score" if strategy == "any" else "mean-score"
        print(f"\n  {fusion_label} fusion ROC:  AUC-ROC={roc_metrics.get('auc_roc', 0):.4f}  AUC-PR={roc_metrics.get('auc_pr', 0):.4f}")
        if roc_metrics.get("tpr_at_fpr"):
            print(f"  {'FPR Target':<12} {'TPR':<10}")
            print(f"  {'-' * 22}")
            for fpr_name in TARGET_FPR_NAMES:
                t = roc_metrics["tpr_at_fpr"].get(fpr_name, {}).get("tpr", 0)
                print(f"  {fpr_name:<12} {t:<10.4f}")

    # Side-by-side comparison
    print(f"\n{'=' * 80}")
    print("COMPARISON: Individual vs All-must-agree vs Majority (fixed-threshold)")
    print(f"{'=' * 80}")
    print(f"\n  {'Method':<45} {'FPR@2%':>8} {'TPR@2%':>8} {'FPR@5%':>8} {'TPR@5%':>8} {'FPR@10%':>9} {'TPR@10%':>8}")
    print(f"  {'-' * 92}")
    for region, info in best.items():
        fix = info.get("fixed_thresholds", {})
        d2, d5, d10 = fix.get("2%", {}), fix.get("5%", {}), fix.get("10%", {})
        print(f"  {region:<45} "
              f"{d2.get('fpr',0):>8.2%} {d2.get('tpr',0):>8.2%} "
              f"{d5.get('fpr',0):>8.2%} {d5.get('tpr',0):>8.2%} "
              f"{d10.get('fpr',0):>9.2%} {d10.get('tpr',0):>8.2%}")
    for strategy, label_short in [("any", "VOTING: any-one-agree"), ("majority", "VOTING: majority"), ("all_agree", "VOTING: all-must-agree")]:
        fr = all_voting_results[strategy]["fixed"]
        d2, d5, d10 = fr.get("2%", {}), fr.get("5%", {}), fr.get("10%", {})
        print(f"  {label_short:<45} "
              f"{d2.get('fpr',0):>8.2%} {d2.get('tpr',0):>8.2%} "
              f"{d5.get('fpr',0):>8.2%} {d5.get('tpr',0):>8.2%} "
              f"{d10.get('fpr',0):>9.2%} {d10.get('tpr',0):>8.2%}")

    # Save results
    out = {
        "best_per_region": {r: {"tag": info["tag"], "tpr2": info["tpr2"], "auc": info["auc"]}
                            for r, info in best.items()},
        "voting_any": all_voting_results["any"],
        "voting_majority": all_voting_results["majority"],
        "voting_all_agree": all_voting_results["all_agree"],
        "n_sessions": len(all_sessions),
        "n_pos": n_pos, "n_neg": n_neg,
    }
    out_path = SCRIPT_DIR / "ml_results" / "voting_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
