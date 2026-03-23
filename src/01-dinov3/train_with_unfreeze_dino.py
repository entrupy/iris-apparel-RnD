#!/root/.venv/bin/python
"""
Fine-tune DINOv3 end-to-end: frozen backbone warmup, then full unfreeze.

Best checkpoint is selected by TPR@2%FPR (not AUC-ROC).
A full TPR/FPR table is printed every epoch.

Usage:
  python train_with_unfreeze_dino.py --region care_label --model vitl16 --resolution 518
  python train_with_unfreeze_dino.py --region front --model vitb16 --resolution 714 --epochs 30
"""

import argparse
import json
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
    TARGET_FPR_NAMES,
    DINOv3Classifier,
    ImageDataset,
    build_train_transform,
    build_transform,
    ckpt_dir,
    compute_all_metrics,
    get_or_create_val_split,
    load_metadata,
    make_weighted_sampler,
    print_final_metrics,
    runs_dir,
)


def _print_tpr_fpr_table(metrics, epoch, total_epochs, frozen_str, epoch_time,
                          avg_train_loss, avg_val_loss, improved):
    auc_roc = metrics.get("auc_roc", 0)
    auc_pr = metrics.get("auc_pr", 0)
    tpr_at_fpr = metrics.get("tpr_at_fpr", {})
    tpr_2 = tpr_at_fpr.get("2%", {}).get("tpr", 0)

    marker = " << BEST" if improved else ""
    print(
        f"\n  Epoch {epoch:3d}/{total_epochs} ({frozen_str}, {epoch_time:.1f}s) | "
        f"loss={avg_train_loss:.4f}/{avg_val_loss:.4f} | "
        f"AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  TPR@2%={tpr_2:.4f}{marker}",
        flush=True,
    )
    print(f"  {'FPR target':<12} {'Actual FPR':<12} {'TPR':<10} {'Threshold':<10}")
    print(f"  {'-' * 44}")
    for name in TARGET_FPR_NAMES:
        d = tpr_at_fpr.get(name, {})
        print(f"  {name:<12} {d.get('actual_fpr', 0):<12.4f} {d.get('tpr', 0):<10.4f} {d.get('threshold', 0):<10.4f}")


def train_finetune(region, model_key, resolution, args):
    records = load_metadata(region, split="train")
    train_idx, val_idx = get_or_create_val_split(region, records)

    train_paths = [records[i]["image_path"] for i in train_idx]
    train_labels = np.array([records[i]["label"] for i in train_idx])
    val_paths = [records[i]["image_path"] for i in val_idx]
    val_labels = np.array([records[i]["label"] for i in val_idx])

    n_pos_train = int(train_labels.sum())
    n_neg_train = len(train_labels) - n_pos_train
    pos_weight = torch.tensor([n_neg_train / max(n_pos_train, 1)])

    print(f"  Train: {len(train_labels)} ({n_pos_train} pos) | Val: {len(val_labels)} ({int(val_labels.sum())} pos)")
    print(f"  pos_weight: {pos_weight.item():.1f}")

    train_transform = build_train_transform(resolution)
    val_transform = build_transform(resolution)

    train_ds = ImageDataset(train_paths, train_labels, train_transform)
    val_ds = ImageDataset(val_paths, val_labels, val_transform)

    train_sampler = make_weighted_sampler(train_labels)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    model_id = MODEL_VARIANTS[model_key]
    model = DINOv3Classifier(model_id, freeze_backbone=True).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_name = f"{model_key}_{resolution}_finetune"
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
    backbone_unfrozen = False

    total_epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    for epoch in range(1, total_epochs + 1):
        if epoch == args.warmup_epochs + 1 and not backbone_unfrozen:
            print(f"\n  >> Unfreezing backbone at epoch {epoch}")
            model.unfreeze_backbone()
            backbone_unfrozen = True
            backbone_params = list(model.backbone.parameters())
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": args.lr_backbone},
                {"params": head_params, "lr": args.lr},
            ], weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - args.warmup_epochs,
            )

        model.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.time()

        for pixel_values, labels_batch in train_loader:
            pixel_values = pixel_values.to(device)
            labels_batch = labels_batch.to(device)

            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(pixel_values)
                loss = criterion(logits, labels_batch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_time = time.time() - t0
        avg_train_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_logits_all, val_labels_all = [], []
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for pixel_values, labels_batch in val_loader:
                pixel_values = pixel_values.to(device)
                labels_batch = labels_batch.to(device)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    logits = model(pixel_values)
                    val_loss_sum += criterion(logits, labels_batch).item()
                val_batches += 1
                val_logits_all.append(logits.float().cpu())
                val_labels_all.append(labels_batch.cpu())

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        val_logits_cat = torch.cat(val_logits_all).numpy()
        val_labels_cat = torch.cat(val_labels_all).numpy()
        val_scores = torch.sigmoid(torch.from_numpy(val_logits_cat)).numpy()

        metrics = compute_all_metrics(val_labels_cat, val_scores)

        writer.add_scalar("loss/train", avg_train_loss, epoch)
        writer.add_scalar("loss/val", avg_val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("epoch_time_s", epoch_time, epoch)
        writer.add_scalar("metrics/auc_roc", metrics.get("auc_roc", 0), epoch)
        writer.add_scalar("metrics/auc_pr", metrics.get("auc_pr", 0), epoch)
        for fpr_name, fpr_data in metrics.get("tpr_at_fpr", {}).items():
            writer.add_scalar(f"tpr_at_fpr/{fpr_name}", fpr_data["tpr"], epoch)

        current_tpr_2 = metrics.get("tpr_at_fpr", {}).get("2%", {}).get("tpr", 0)
        current_auc = metrics.get("auc_roc", 0)
        improved = current_tpr_2 > best_tpr_2 or (
            current_tpr_2 == best_tpr_2 and current_auc > best_auc
        )
        if improved:
            best_tpr_2 = current_tpr_2
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

        frozen_str = "frozen" if not backbone_unfrozen else "unfrozen"
        _print_tpr_fpr_table(metrics, epoch, total_epochs, frozen_str, epoch_time,
                             avg_train_loss, avg_val_loss, improved)

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    writer.add_hparams(
        {"model": model_key, "resolution": resolution, "region": region,
         "lr": args.lr, "lr_backbone": args.lr_backbone,
         "warmup_epochs": args.warmup_epochs, "batch_size": args.batch_size,
         "pos_weight": pos_weight.item()},
        {"hparam/best_tpr_2pct": best_tpr_2, "hparam/best_auc_roc": best_auc},
    )
    writer.close()

    print(f"\n  Best TPR@2%FPR: {best_tpr_2:.4f}  (AUC-ROC: {best_auc:.4f})")
    print(f"  Checkpoint: {best_ckpt_path}")
    print_final_metrics(best_ckpt_path)
    return best_tpr_2


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DINOv3 with backbone unfreeze")
    parser.add_argument("--region", type=str, default="care_label", choices=REGIONS)
    parser.add_argument("--model", type=str, default="vitl16", choices=list(MODEL_VARIANTS.keys()))
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n{'=' * 60}")
    print(f"[{args.region} | {args.model} @ {args.resolution} | finetune]")
    print(f"{'=' * 60}")
    train_finetune(args.region, args.model, args.resolution, args)


if __name__ == "__main__":
    main()
