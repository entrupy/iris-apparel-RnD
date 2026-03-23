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
    cache_dir,
    ckpt_dir,
    compute_all_metrics,
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
        val_logits_all, val_labels_all = [], []
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for feats_batch, labels_batch in val_loader:
                feats_batch = feats_batch.to(device)
                labels_batch = labels_batch.to(device)
                logits = model(feats_batch)
                val_loss_sum += criterion(logits, labels_batch).item()
                val_batches += 1
                val_logits_all.append(logits.cpu())
                val_labels_all.append(labels_batch.cpu())

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        val_logits_cat = torch.cat(val_logits_all).numpy()
        val_labels_cat = torch.cat(val_labels_all).numpy()
        val_scores = torch.sigmoid(torch.from_numpy(val_logits_cat)).numpy()

        metrics = compute_all_metrics(val_labels_cat, val_scores)

        writer.add_scalar("loss/train", avg_train_loss, epoch)
        writer.add_scalar("loss/val", avg_val_loss, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("metrics/auc_roc", metrics.get("auc_roc", 0), epoch)
        writer.add_scalar("metrics/auc_pr", metrics.get("auc_pr", 0), epoch)
        for fpr_name, fpr_data in metrics.get("tpr_at_fpr", {}).items():
            writer.add_scalar(f"tpr_at_fpr/{fpr_name}", fpr_data["tpr"], epoch)

        current_auc = metrics.get("auc_roc", 0)
        improved = current_auc > best_auc
        if improved:
            best_auc = current_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
                "config": vars(args),
                "region": region,
            }, best_ckpt_path)
        else:
            patience_counter += 1

        tpr_line = format_tpr_at_fpr_inline(metrics)
        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"loss={avg_train_loss:.4f}/{avg_val_loss:.4f} | "
            f"AUC={current_auc:.4f} PR={metrics.get('auc_pr', 0):.4f} | "
            f"TPR@FPR: {tpr_line}"
            f"{' *' if improved else ''}",
            flush=True,
        )

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    writer.add_hparams(
        {"model": model_key, "resolution": resolution, "region": region,
         "lr": args.lr, "batch_size": args.batch_size, "pos_weight": pos_weight.item()},
        {"hparam/best_auc_roc": best_auc},
    )
    writer.close()

    print(f"  Best AUC-ROC: {best_auc:.4f} | Checkpoint: {best_ckpt_path}")
    print_final_metrics(best_ckpt_path)
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
