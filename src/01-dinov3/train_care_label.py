"""
Train binary classifier heads on cached DINOv3 features (linear probe)
or directly on images (fine-tune).

Modes:
  linear_probe  - Trains a single Linear(embed_dim, 1) on cached .pt features.
                  Seconds per epoch; run for many variants quickly.
  finetune      - Loads DINOv3 backbone + head, trains end-to-end on images.
                  Unfreezes backbone after --warmup-epochs of frozen training.

Evaluation: TPR @ FPR = {0.5%, 1%, 2%, 5%, 10%} using closest discrete point.

Usage:
  # Linear probe on all cached variants
  python train_care_label.py --mode linear_probe --sweep

  # Linear probe on one variant
  python train_care_label.py --mode linear_probe --model vits16 --resolution 518

  # Fine-tune
  python train_care_label.py --mode finetune --model vitb16 --resolution 518 --epochs 30
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cached_features"
RUNS_DIR = SCRIPT_DIR / "runs"
CKPT_DIR = SCRIPT_DIR / "checkpoints"

DATA_ROOT = SCRIPT_DIR.parent.parent / "resources" / "apparel_supreme_until_dec_2025_care_label"
IMAGE_DIR = DATA_ROOT / "train" / "camera" / "care_label" / "0"

TARGET_FPRS = [0.005, 0.01, 0.02, 0.05, 0.10]
TARGET_FPR_NAMES = ["0.5%", "1%", "2%", "5%", "10%"]

MODEL_VARIANTS = {
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "convnext_tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "convnext_small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "convnext_base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "convnext_large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEED = 42


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class CachedFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].float()


class CareImageDataset(Dataset):
    """For fine-tune mode -- loads images from disk."""

    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.transform(img)
        return pixel_values, float(self.labels[idx])


def load_cached_features(model_key, resolution):
    prefix = f"{model_key}_{resolution}"
    features = torch.load(CACHE_DIR / f"{prefix}_features.pt", weights_only=True)
    labels = torch.load(CACHE_DIR / f"{prefix}_labels.pt", weights_only=True)
    uuids = torch.load(CACHE_DIR / f"{prefix}_uuids.pt", weights_only=False)

    nan_mask = torch.isnan(features).any(dim=1) | torch.isinf(features).any(dim=1)
    if nan_mask.any():
        n_bad = nan_mask.sum().item()
        print(f"  WARNING: {n_bad}/{len(features)} features contain NaN/Inf -- dropping them")
        keep = ~nan_mask
        features = features[keep]
        labels = labels[keep]
        uuids = [u for u, k in zip(uuids, keep.tolist()) if k]

    return features, labels, uuids


def stratified_split(labels, val_ratio=0.2, seed=SEED):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return train_idx, val_idx


def make_weighted_sampler(labels_array):
    """WeightedRandomSampler that oversamples the minority class."""
    class_counts = np.bincount(labels_array.astype(int))
    weights_per_class = 1.0 / class_counts
    sample_weights = weights_per_class[labels_array.astype(int)]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels_array),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_tpr_at_fprs(y_true, y_score, target_fprs=TARGET_FPRS):
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_score)
    results = {}
    for target_fpr, name in zip(target_fprs, TARGET_FPR_NAMES):
        distances = np.abs(fpr_arr - target_fpr)
        min_dist = distances.min()
        tied_indices = np.where(distances == min_dist)[0]
        idx = tied_indices[np.argmax(tpr_arr[tied_indices])]
        thresh_idx = min(idx, len(thresholds) - 1)
        results[name] = {
            "tpr": float(tpr_arr[idx]),
            "actual_fpr": float(fpr_arr[idx]),
            "threshold": float(thresholds[thresh_idx]),
        }
    return results


def compute_all_metrics(y_true, y_score):
    metrics = {}

    if len(np.unique(y_true)) < 2:
        return {"auc_roc": 0.0, "auc_pr": 0.0, "tpr_at_fpr": {}}

    metrics["auc_roc"] = float(roc_auc_score(y_true, y_score))
    metrics["auc_pr"] = float(average_precision_score(y_true, y_score))
    metrics["tpr_at_fpr"] = compute_tpr_at_fprs(y_true, y_score)

    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr_arr - fpr_arr
    best_idx = np.argmax(j_scores)
    thresh_idx = min(best_idx, len(thresholds) - 1)
    metrics["best_threshold_youden"] = {
        "threshold": float(thresholds[thresh_idx]),
        "tpr": float(tpr_arr[best_idx]),
        "fpr": float(fpr_arr[best_idx]),
    }

    return metrics


def format_tpr_at_fpr_inline(metrics):
    """One-line summary of TPR at each FPR target."""
    tpr_at_fpr = metrics.get("tpr_at_fpr", {})
    parts = []
    for name in TARGET_FPR_NAMES:
        d = tpr_at_fpr.get(name, {})
        parts.append(f"{name}={d.get('tpr', 0):.3f}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.head(x).squeeze(-1)


class DINOv3Classifier(nn.Module):
    """Full model for fine-tune mode."""

    def __init__(self, model_id, freeze_backbone=True):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        )
        embed_dim = self.backbone.config.hidden_size
        self.head = nn.Linear(embed_dim, 1)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.freeze_backbone = True

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.freeze_backbone = False

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_features = outputs.pooler_output
        return self.head(cls_features).squeeze(-1)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_linear_probe(model_key, resolution, args):
    features, labels, uuids = load_cached_features(model_key, resolution)
    labels_np = labels.numpy()

    train_idx, val_idx = stratified_split(labels_np, val_ratio=0.2)
    train_feats, train_labels = features[train_idx], labels[train_idx]
    val_feats, val_labels = features[val_idx], labels[val_idx]

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
    writer = SummaryWriter(log_dir=str(RUNS_DIR / run_name))
    writer.add_text("config", json.dumps(vars(args), indent=2, default=str))

    best_auc = 0.0
    patience_counter = 0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = CKPT_DIR / f"{run_name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
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
        val_loss_sum = 0.0
        val_batches = 0
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
            tag = f"tpr_at_fpr/{fpr_name}"
            writer.add_scalar(tag, fpr_data["tpr"], epoch)
            writer.add_scalar(f"actual_fpr/{fpr_name}", fpr_data["actual_fpr"], epoch)

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
        {
            "model": model_key,
            "resolution": resolution,
            "mode": "linear_probe",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "pos_weight": pos_weight.item(),
        },
        {
            "hparam/best_auc_roc": best_auc,
        },
    )
    writer.close()

    print(f"  Best AUC-ROC: {best_auc:.4f} | Checkpoint: {best_ckpt_path}")
    _print_final_metrics(best_ckpt_path)
    return best_auc


def train_finetune(model_key, resolution, args):
    from torchvision import transforms as T
    from PIL import Image

    feat_path = CACHE_DIR / f"{model_key}_{resolution}_labels.pt"
    labels_all = torch.load(feat_path, weights_only=True).numpy()
    uuids_all = torch.load(CACHE_DIR / f"{model_key}_{resolution}_uuids.pt", weights_only=False)

    image_paths = []
    for uuid in uuids_all:
        matches = sorted(IMAGE_DIR.glob(f"{uuid}.macro.care_label.*.jpg"))
        image_paths.append(str(matches[0]))

    train_idx, val_idx = stratified_split(labels_all, val_ratio=0.2)

    train_paths = [image_paths[i] for i in train_idx]
    train_labels = labels_all[train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = labels_all[val_idx]

    n_pos_train = int(train_labels.sum())
    n_neg_train = len(train_labels) - n_pos_train
    pos_weight = torch.tensor([n_neg_train / max(n_pos_train, 1)])

    print(f"  Train: {len(train_labels)} ({n_pos_train} pos) | Val: {len(val_labels)} ({int(val_labels.sum())} pos)")
    print(f"  pos_weight: {pos_weight.item():.1f}")

    train_transform = T.Compose([
        T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_transform = T.Compose([
        T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ds = CareImageDataset(train_paths, train_labels, train_transform)
    val_ds = CareImageDataset(val_paths, val_labels, val_transform)

    train_sampler = make_weighted_sampler(train_labels)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device)
    model_id = MODEL_VARIANTS[model_key]
    model = DINOv3Classifier(model_id, freeze_backbone=True).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=(args.device == "cuda"))

    run_name = f"{model_key}_{resolution}_finetune"
    writer = SummaryWriter(log_dir=str(RUNS_DIR / run_name))
    writer.add_text("config", json.dumps(vars(args), indent=2, default=str))

    best_auc = 0.0
    patience_counter = 0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = CKPT_DIR / f"{run_name}_best.pt"
    backbone_unfrozen = False

    total_epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    for epoch in range(1, total_epochs + 1):
        if epoch == args.warmup_epochs + 1 and not backbone_unfrozen:
            print(f"  >> Unfreezing backbone at epoch {epoch}")
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
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for pixel_values, labels_batch in train_loader:
            pixel_values = pixel_values.to(device)
            labels_batch = labels_batch.to(device)

            with torch.amp.autocast("cuda", enabled=(args.device == "cuda")):
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
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for pixel_values, labels_batch in val_loader:
                pixel_values = pixel_values.to(device)
                labels_batch = labels_batch.to(device)
                with torch.amp.autocast("cuda", enabled=(args.device == "cuda")):
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
            writer.add_scalar(f"actual_fpr/{fpr_name}", fpr_data["actual_fpr"], epoch)

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
            }, best_ckpt_path)
        else:
            patience_counter += 1

        frozen_str = "frozen" if not backbone_unfrozen else "unfrozen"
        tpr_line = format_tpr_at_fpr_inline(metrics)
        print(
            f"  Epoch {epoch:3d}/{total_epochs} ({frozen_str}, {epoch_time:.1f}s) | "
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
        {
            "model": model_key,
            "resolution": resolution,
            "mode": "finetune",
            "lr": args.lr,
            "lr_backbone": args.lr_backbone,
            "warmup_epochs": args.warmup_epochs,
            "batch_size": args.batch_size,
            "pos_weight": pos_weight.item(),
        },
        {"hparam/best_auc_roc": best_auc},
    )
    writer.close()

    print(f"  Best AUC-ROC: {best_auc:.4f} | Checkpoint: {best_ckpt_path}")
    _print_final_metrics(best_ckpt_path)
    return best_auc


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_final_metrics(ckpt_path):
    if not ckpt_path.exists():
        return
    ckpt = torch.load(ckpt_path, weights_only=False)
    metrics = ckpt.get("metrics", {})
    print("\n  --- Best checkpoint metrics ---")
    print(f"  AUC-ROC: {metrics.get('auc_roc', 'N/A')}")
    print(f"  AUC-PR:  {metrics.get('auc_pr', 'N/A')}")
    tpr_at_fpr = metrics.get("tpr_at_fpr", {})
    if tpr_at_fpr:
        print(f"  {'Target FPR':<12} {'Actual FPR':<12} {'TPR':<10} {'Threshold':<10}")
        print(f"  {'-'*44}")
        for name in TARGET_FPR_NAMES:
            d = tpr_at_fpr.get(name, {})
            print(f"  {name:<12} {d.get('actual_fpr', 0):<12.4f} {d.get('tpr', 0):<10.4f} {d.get('threshold', 0):<10.4f}")
    youden = metrics.get("best_threshold_youden", {})
    if youden:
        print(f"  Youden best: threshold={youden.get('threshold', 0):.4f}  "
              f"TPR={youden.get('tpr', 0):.4f}  FPR={youden.get('fpr', 0):.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train care-label authenticity classifier")
    parser.add_argument("--mode", choices=["linear_probe", "finetune"], default="linear_probe")
    parser.add_argument("--model", type=str, default="vits16", choices=list(MODEL_VARIANTS.keys()))
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--sweep", action="store_true",
                        help="Run linear_probe over all cached variant+resolution combos")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5,
                        help="Backbone LR for finetune mode after warmup")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Epochs with frozen backbone before unfreezing (finetune mode)")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if args.sweep:
        cached_files = sorted(CACHE_DIR.glob("*_features.pt"))
        if not cached_files:
            print("No cached features found. Run cache_features.py first.")
            sys.exit(1)

        results = {}
        for feat_file in cached_files:
            name = feat_file.stem.replace("_features", "")
            parts = name.rsplit("_", 1)
            mk, res = parts[0], int(parts[1])
            print(f"\n{'='*60}")
            print(f"[{mk} @ {res} | linear_probe]")
            print(f"{'='*60}")
            auc_val = train_linear_probe(mk, res, args)
            results[name] = auc_val

        print(f"\n{'='*60}")
        print("SWEEP SUMMARY")
        print(f"{'='*60}")
        for name, auc_val in sorted(results.items(), key=lambda x: -x[1]):
            print(f"  {name:<30s}  AUC-ROC = {auc_val:.4f}")
        return

    print(f"\n{'='*60}")
    print(f"[{args.model} @ {args.resolution} | {args.mode}]")
    print(f"{'='*60}")

    if args.mode == "linear_probe":
        train_linear_probe(args.model, args.resolution, args)
    else:
        train_finetune(args.model, args.resolution, args)


if __name__ == "__main__":
    main()
