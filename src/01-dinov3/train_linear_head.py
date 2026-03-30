#!/root/.venv/bin/python
"""
Train a linear probe (Linear(embed_dim, 1)) on cached DINOv3 features.

Uses pre-computed embeddings from precompute_embeddings.py.
Per-region stratified val split is created once and reused.

Usage:
  python train_linear_head.py --region care_label --sweep
  python train_linear_head.py --region front --model vitb16 --resolution 518
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import (
    MODEL_VARIANTS,
    REGIONS,
    RESOLUTIONS,
    SEED,
    CachedFeatureDataset,
    LinearHead,
    apply_threshold_auth_positive,
    cache_dir,
    ckpt_dir,
    compute_metrics_auth_positive,
    format_tpr_at_fpr_inline,
    get_or_create_val_split,
    load_cached_features,
    load_metadata,
    make_weighted_sampler,
    print_final_metrics,
    runs_dir,
)


def train_linear_probe(region, model_key, resolution, args):
    features, labels, uuids = load_cached_features(region, model_key, resolution)
    labels_np = labels.numpy()

    records = load_metadata(region, split="train")
    train_idx, val_idx = get_or_create_val_split(region, records)

    uuid_to_idx = {u: i for i, u in enumerate(uuids)}
    feat_train_idx = [uuid_to_idx[records[i]["session_uuid"]] for i in train_idx if records[i]["session_uuid"] in uuid_to_idx]
    feat_val_idx = [uuid_to_idx[records[i]["session_uuid"]] for i in val_idx if records[i]["session_uuid"] in uuid_to_idx]

    train_feats, train_labels = features[feat_train_idx], labels[feat_train_idx]
    val_feats, val_labels = features[feat_val_idx], labels[feat_val_idx]

    cdir_path = cache_dir(region)
    test_prefix = f"test_{model_key}_{resolution}"
    test_feat_path = cdir_path / f"{test_prefix}_features.pt"
    if test_feat_path.exists():
        test_feats = torch.load(test_feat_path, weights_only=True)
        test_labels_t = torch.load(cdir_path / f"{test_prefix}_labels.pt", weights_only=True)
        print(f"  Test features loaded: {len(test_labels_t)} ({int(test_labels_t.sum())} pos)")
    else:
        test_feats, test_labels_t = None, None
        print("  Test features not cached — skipping test eval")

    n_pos_train = int(train_labels.sum())
    n_neg_train = len(train_labels) - n_pos_train
    pos_weight = torch.tensor([n_neg_train / max(n_pos_train, 1)])

    print(f"  Train: {len(train_labels)} ({n_pos_train} pos) | Val: {len(val_labels)} ({int(val_labels.sum())} pos)")
    print(f"  pos_weight: {pos_weight.item():.1f}")

    embed_dim = features.shape[1]
    model = LinearHead(embed_dim)
    device = torch.device(args.device)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_sampler = make_weighted_sampler(train_labels.numpy())
    train_ds = CachedFeatureDataset(train_feats, train_labels)
    val_ds = CachedFeatureDataset(val_feats, val_labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 4, shuffle=False)

    run_name = f"{model_key}_{resolution}_linear_probe"
    rdir = runs_dir(region)
    rdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(rdir / run_name))
    writer.add_text("config", json.dumps(vars(args), indent=2, default=str))

    best_tpr_2 = -1.0
    best_auc = 0.0
    patience_counter = 0
    cdir = ckpt_dir(region)
    cdir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = cdir / f"{run_name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for feats_batch, labels_batch in train_loader:
            feats_batch = feats_batch.to(device)
            labels_batch = labels_batch.to(device)
            logits = model(feats_batch)
            loss = criterion(logits, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_train_loss = total_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            train_logits = model(train_feats.to(device)).cpu().numpy()
            train_scores_ep = torch.sigmoid(torch.from_numpy(train_logits)).numpy()
            train_m = compute_metrics_auth_positive(train_labels.numpy(), train_scores_ep)

            val_logits_all, val_labels_all = [], []
            for feats_batch, labels_batch in val_loader:
                feats_batch = feats_batch.to(device)
                labels_batch = labels_batch.to(device)
                val_logits_all.append(model(feats_batch).cpu())
                val_labels_all.append(labels_batch.cpu())

        val_logits_cat = torch.cat(val_logits_all).numpy()
        val_labels_cat = torch.cat(val_labels_all).numpy()
        val_scores = torch.sigmoid(torch.from_numpy(val_logits_cat)).numpy()
        val_m = compute_metrics_auth_positive(val_labels_cat, val_scores)

        train_auc = train_m.get("auc_roc", 0)
        train_tpr2 = train_m.get("tpr_at_fpr", {}).get("2%", {}).get("tpr", 0)
        val_auc = val_m.get("auc_roc", 0)
        val_tpr2 = val_m.get("tpr_at_fpr", {}).get("2%", {}).get("tpr", 0)

        writer.add_scalar("loss/train", avg_train_loss, epoch)
        writer.add_scalar("train/auc_roc", train_auc, epoch)
        writer.add_scalar("train/tpr_at_2pct", train_tpr2, epoch)
        writer.add_scalar("val/auc_roc", val_auc, epoch)
        writer.add_scalar("val/tpr_at_2pct", val_tpr2, epoch)

        improved = val_tpr2 > best_tpr_2 or (
            val_tpr2 == best_tpr_2 and val_auc > best_auc
        )
        if improved:
            best_tpr_2 = val_tpr2
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": val_m,
                "config": vars(args),
                "region": region,
            }, best_ckpt_path)
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"train AUC={train_auc:.4f} TPR@2%={train_tpr2:.4f} | "
            f"val AUC={val_auc:.4f} TPR@2%={val_tpr2:.4f}"
            f"{' *' if improved else ''}",
            flush=True,
        )

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    writer.close()
    print(f"\n  Best val TPR@2%: {best_tpr_2:.4f}  (AUC: {best_auc:.4f}) | Checkpoint: {best_ckpt_path}")

    if best_ckpt_path.exists() and test_feats is not None:
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        best_model = LinearHead(features.shape[1]).to(device)
        best_model.load_state_dict(ckpt["model_state_dict"])
        best_model.eval()
        with torch.no_grad():
            test_logits = best_model(test_feats.to(device)).cpu().numpy()
        test_scores = torch.sigmoid(torch.from_numpy(test_logits)).numpy()
        test_labels_np = test_labels_t.numpy()
        test_m = compute_metrics_auth_positive(test_labels_np, test_scores)

        val_thresh_info = ckpt["metrics"].get("tpr_at_fpr", {}).get("2%", {})
        thresh_orig = val_thresh_info.get("threshold_orig", 0.5)
        fixed = apply_threshold_auth_positive(test_labels_np, test_scores, thresh_orig)

        print(f"  TEST | AUC={test_m.get('auc_roc', 0):.4f} | "
              f"@2% val-thresh: TPR(auth passed)={fixed['tpr']:.4f}  FPR(fakes missed)={fixed['fpr']:.4f}")

    return best_auc


def main():
    parser = argparse.ArgumentParser(description="Train linear probe on cached DINOv3 features")
    parser.add_argument("--region", type=str, default="care_label", choices=REGIONS)
    parser.add_argument("--model", type=str, default="vits16", choices=list(MODEL_VARIANTS.keys()))
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--sweep", action="store_true",
                        help="Run over all cached variant+resolution combos for this region")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    region = args.region

    if args.sweep:
        cdir = cache_dir(region)
        cached_files = sorted(cdir.glob("*_features.pt"))
        if not cached_files:
            print(f"No cached features found for region '{region}'. Run precompute_embeddings.py first.")
            sys.exit(1)

        results = {}
        for feat_file in cached_files:
            name = feat_file.stem.replace("_features", "")
            parts = name.rsplit("_", 1)
            mk, res = parts[0], int(parts[1])
            print(f"\n{'=' * 60}")
            print(f"[{region} | {mk} @ {res} | linear_probe]")
            print(f"{'=' * 60}")
            auc_val = train_linear_probe(region, mk, res, args)
            results[name] = auc_val

        print(f"\n{'=' * 60}")
        print(f"SWEEP SUMMARY — {region}")
        print(f"{'=' * 60}")
        for name, auc_val in sorted(results.items(), key=lambda x: -x[1]):
            print(f"  {name:<30s}  AUC-ROC = {auc_val:.4f}")
        return

    print(f"\n{'=' * 60}")
    print(f"[{region} | {args.model} @ {args.resolution} | linear_probe]")
    print(f"{'=' * 60}")
    train_linear_probe(region, args.model, args.resolution, args)


if __name__ == "__main__":
    main()
