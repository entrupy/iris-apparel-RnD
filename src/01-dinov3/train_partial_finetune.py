#!/root/.venv/bin/python
"""
Partial finetuning sweep: unfreeze specific backbone components while keeping
the rest frozen. Explores which parts of DINOv3 ViT-L matter most for
authentication at care_label region.

Strategies control which parameters get unfrozen after warmup:
  - Component-type: norm, layer_scale, q, v, qv, qkv, qkvo, mlp, etc.
  - Depth-limited: last1, last4, last8, last12
  - Combinations: mlp_last4, qv_last8+norm, etc.

Usage:
  python train_partial_finetune.py --strategy norm
  python train_partial_finetune.py --strategy qv_last8+norm --batch-size 16
  python train_partial_finetune.py --strategy full  # baseline (unfreeze all)
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
    SEED,
    TARGET_FPR_NAMES,
    DINOv3Classifier,
    ImageDataset,
    apply_threshold_auth_positive,
    build_train_transform,
    build_transform,
    ckpt_dir,
    compute_metrics_auth_positive,
    get_or_create_val_split,
    load_metadata,
    make_weighted_sampler,
    print_final_metrics,
    runs_dir,
)

STRATEGIES = [
    # Group A: Minimal
    "norm", "layer_scale", "norm+ls",
    # Group B: Attention subsets (all layers)
    "q", "v", "qv", "qkv", "qkvo", "qv+norm",
    # Group C: MLP
    "mlp", "mlp+norm",
    # Group D: Depth-limited (all components)
    "last1", "last4", "last8", "last12",
    # Group E: Depth x Component
    "qv_last4", "mlp_last4", "mlp_last8", "attn_last4", "qv_last8+norm",
    # Baseline
    "full",
]

N_LAYERS = 24


def _parse_strategy(strategy):
    """Parse strategy string into (component_patterns, layer_range, extra_patterns).

    Returns a function that takes a parameter name and returns True if it should be unfrozen.
    """
    def _in_layers(name, layer_range):
        if not name.startswith("layer."):
            return False
        idx = int(name.split(".")[1])
        return idx in layer_range

    all_layers = range(N_LAYERS)

    component_map = {
        "norm": ["norm1.", "norm2."],
        "layer_scale": ["layer_scale1.", "layer_scale2."],
        "ls": ["layer_scale1.", "layer_scale2."],
        "q": ["attention.q_proj."],
        "k": ["attention.k_proj."],
        "v": ["attention.v_proj."],
        "o": ["attention.o_proj."],
        "qv": ["attention.q_proj.", "attention.v_proj."],
        "qkv": ["attention.q_proj.", "attention.k_proj.", "attention.v_proj."],
        "qkvo": ["attention.q_proj.", "attention.k_proj.", "attention.v_proj.", "attention.o_proj."],
        "attn": ["attention."],
        "mlp": ["mlp."],
        "full": [""],  # matches everything
    }

    if strategy == "full":
        return lambda name: name.startswith("layer.") or name.startswith("embeddings.") or name.startswith("norm.")

    parts = strategy.replace("+", ",").split(",")
    rules = []

    for part in parts:
        part = part.strip()
        layer_range = all_layers
        comp_key = part

        if "_last" in part:
            comp_key, depth = part.split("_last")
            n = int(depth)
            layer_range = range(N_LAYERS - n, N_LAYERS)
        elif part.startswith("last"):
            n = int(part[4:])
            layer_range = range(N_LAYERS - n, N_LAYERS)
            comp_key = "full"

        patterns = component_map.get(comp_key, [])
        if not patterns:
            raise ValueError(f"Unknown component in strategy: {comp_key}")

        rules.append((patterns, layer_range))

    def should_unfreeze(name):
        if not name.startswith("layer."):
            return False
        idx = int(name.split(".")[1])
        param_suffix = ".".join(name.split(".")[2:])
        for patterns, layer_range in rules:
            if idx in layer_range:
                for pat in patterns:
                    if pat == "" or pat in param_suffix:
                        return True
        return False

    return should_unfreeze


def _count_unfrozen(model, should_unfreeze_fn):
    total, unfrozen = 0, 0
    for name, param in model.backbone.named_parameters():
        total += param.numel()
        if should_unfreeze_fn(name):
            unfrozen += param.numel()
    return unfrozen, total


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


def train_partial(strategy, args):
    region = args.region
    model_key = args.model
    resolution = args.resolution

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

    test_records = load_metadata(region, split="test")
    test_paths = [r["image_path"] for r in test_records]
    test_labels = np.array([r["label"] for r in test_records])
    print(f"  Test: {len(test_labels)} ({int(test_labels.sum())} pos)")

    train_ds = ImageDataset(train_paths, train_labels, train_transform)
    train_eval_ds = ImageDataset(train_paths, train_labels, val_transform)
    val_ds = ImageDataset(val_paths, val_labels, val_transform)
    test_ds = ImageDataset(test_paths, test_labels, val_transform)

    train_sampler = make_weighted_sampler(train_labels)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    model_id = MODEL_VARIANTS[model_key]
    model = DINOv3Classifier(model_id, freeze_backbone=True).to(device)
    model = torch.compile(model)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    should_unfreeze = _parse_strategy(strategy)
    n_unfrozen, n_total = _count_unfrozen(model, should_unfreeze)
    print(f"  Strategy '{strategy}': will unfreeze {n_unfrozen:,} / {n_total:,} backbone params ({n_unfrozen/1e6:.2f}M / {n_total/1e6:.1f}M)")

    run_name = f"{model_key}_{resolution}_partial_{strategy}"
    rdir = runs_dir(region)
    rdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(rdir / run_name))
    writer.add_text("config", json.dumps({"strategy": strategy, **vars(args)}, indent=2, default=str))

    best_tpr_2 = -1.0
    best_auc = 0.0
    patience_counter = 0
    cdir = ckpt_dir(region)
    cdir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = cdir / f"{run_name}_best.pt"
    unfrozen = False

    total_epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    for epoch in range(1, total_epochs + 1):
        if epoch == args.warmup_epochs + 1 and not unfrozen:
            print(f"\n  >> Unfreezing '{strategy}' components at epoch {epoch}")
            backbone_unfrozen_params = []
            for name, param in model.backbone.named_parameters():
                if should_unfreeze(name):
                    param.requires_grad = True
                    backbone_unfrozen_params.append(param)
            unfrozen = True

            actual_unfrozen = sum(p.numel() for p in backbone_unfrozen_params)
            print(f"  >> Actually unfrozen: {actual_unfrozen:,} params ({actual_unfrozen/1e6:.2f}M)")

            optimizer = torch.optim.AdamW([
                {"params": backbone_unfrozen_params, "lr": args.lr_backbone},
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
        def _score_loader(loader):
            all_logits, all_labels = [], []
            with torch.no_grad():
                for pv, lb in loader:
                    pv, lb = pv.to(device), lb.to(device)
                    with torch.amp.autocast(device.type, enabled=use_amp):
                        logits = model(pv)
                    all_logits.append(logits.float().cpu())
                    all_labels.append(lb.cpu())
            logits_cat = torch.cat(all_logits).numpy()
            labels_cat = torch.cat(all_labels).numpy()
            scores = torch.sigmoid(torch.from_numpy(logits_cat)).numpy()
            return labels_cat, scores

        train_labels_ep, train_scores_ep = _score_loader(train_eval_loader)
        val_labels_ep, val_scores_ep = _score_loader(val_loader)

        train_m = compute_metrics_auth_positive(train_labels_ep, train_scores_ep)
        val_m = compute_metrics_auth_positive(val_labels_ep, val_scores_ep)

        train_auc = train_m.get("auc_roc", 0)
        train_tpr2 = train_m.get("tpr_at_fpr", {}).get("2%", {}).get("tpr", 0)
        val_auc = val_m.get("auc_roc", 0)
        val_tpr2 = val_m.get("tpr_at_fpr", {}).get("2%", {}).get("tpr", 0)

        writer.add_scalar("loss/train", avg_train_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("epoch_time_s", epoch_time, epoch)
        writer.add_scalar("train/auc_roc", train_auc, epoch)
        writer.add_scalar("train/tpr_at_2pct", train_tpr2, epoch)
        writer.add_scalar("val/auc_roc", val_auc, epoch)
        writer.add_scalar("val/tpr_at_2pct", val_tpr2, epoch)

        if args.best_metric == "auc":
            improved = val_auc > best_auc or (
                val_auc == best_auc and val_tpr2 > best_tpr_2
            )
        else:
            improved = val_tpr2 > best_tpr_2 or (
                val_tpr2 == best_tpr_2 and val_auc > best_auc
            )
        if improved:
            best_tpr_2 = val_tpr2
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()},
                "metrics": val_m,
                "config": vars(args),
                "region": region,
                "strategy": strategy,
                "n_unfrozen": n_unfrozen,
            }, best_ckpt_path)
        else:
            patience_counter += 1

        frozen_str = "frozen" if not unfrozen else strategy
        marker = " << BEST" if improved else ""
        print(
            f"\n  Epoch {epoch:3d}/{total_epochs} ({frozen_str}, {epoch_time:.1f}s) | "
            f"train AUC={train_auc:.4f} TPR@2%={train_tpr2:.4f} | "
            f"val AUC={val_auc:.4f} TPR@2%={val_tpr2:.4f}{marker}",
            flush=True,
        )

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    writer.close()
    print(f"\n  Best val TPR@2%: {best_tpr_2:.4f}  (AUC: {best_auc:.4f})")
    print(f"  Checkpoint: {best_ckpt_path}")

    # --- Final test evaluation ---
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        load_target = getattr(model, "_orig_mod", model)
        load_target.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        test_labels_all, test_scores_all = _score_loader(test_loader)
        test_m = compute_metrics_auth_positive(test_labels_all, test_scores_all)

        val_thresh_info = ckpt["metrics"].get("tpr_at_fpr", {}).get("2%", {})
        thresh_orig = val_thresh_info.get("threshold_orig", 0.5)
        fixed = apply_threshold_auth_positive(test_labels_all, test_scores_all, thresh_orig)

        print(f"\n  TEST | AUC={test_m.get('auc_roc', 0):.4f} | "
              f"@2% val-thresh: TPR(auth passed)={fixed['tpr']:.4f}  FPR(fakes missed)={fixed['fpr']:.4f}")

    return best_tpr_2, best_auc, n_unfrozen


def main():
    parser = argparse.ArgumentParser(description="Partial finetuning sweep")
    parser.add_argument("--strategy", type=str, required=True, choices=STRATEGIES)
    parser.add_argument("--region", type=str, default="care_label", choices=REGIONS)
    parser.add_argument("--model", type=str, default="vitl16")
    parser.add_argument("--resolution", type=int, default=714)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--best-metric", type=str, default="tpr", choices=["tpr", "auc"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n{'=' * 70}")
    print(f"PARTIAL FINETUNE: {args.strategy}")
    print(f"Model: {args.model} @ {args.resolution} | Region: {args.region}")
    print(f"{'=' * 70}")
    train_partial(args.strategy, args)


if __name__ == "__main__":
    main()
