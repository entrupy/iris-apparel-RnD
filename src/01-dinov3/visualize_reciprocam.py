#!/root/.venv/bin/python
"""
ViT-ReciproCAM (Byun & Lee, 2023) for DINOv3 ViT-L.

Official-style implementation adapted to Hugging Face DINOv3:
hook the pre-attention LayerNorm (`norm1`) of a target transformer block,
expand the single sample into `1 + num_patches` masked samples, then let the
original model continue forward. Spatial masking uses the paper's default
3x3 Gaussian neighborhood and, by default, preserves only the `CLS` token.

Models:
  1. vitl16_714_finetune        (full fine-tune)
  2. vitl16_714_partial_qv_last4 (partial fine-tune)

Usage:
  python visualize_reciprocam.py
  python visualize_reciprocam.py --target-layer -2
  python visualize_reciprocam.py --cam-chunk-size 64
  python visualize_reciprocam.py --n-near 3 --n-confident 3
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MODEL_VARIANTS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DINOv3Classifier,
    build_transform,
    load_metadata,
)

REGION = "care_label"
MODEL_KEY = "vitl16"
RESOLUTION = 714
MODEL_ID = MODEL_VARIANTS[MODEL_KEY]

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / REGION
OUT_DIR = Path(__file__).resolve().parent / "reciprocam_maps"

MODELS_TO_EVAL = {
    "vitl16_714_finetune": CKPT_DIR / f"{MODEL_KEY}_{RESOLUTION}_finetune_best.pt",
    "vitl16_714_partial_qv_last4": CKPT_DIR / f"{MODEL_KEY}_{RESOLUTION}_partial_qv_last4_best.pt",
}


# ---------------------------------------------------------------------------
# ReciproCAM masking hook
# ---------------------------------------------------------------------------
GAUSSIAN_KERNEL = torch.tensor(
    [
        [1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0],
        [1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0],
        [1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0],
    ],
    dtype=torch.float32,
)


class ReciproCAMHook:
    """Replace a `norm1` output with the original sample plus masked variants."""

    def __init__(self, layer_norm_module, patch_start_idx, preserve_prefix_tokens=1, use_gaussian=True):
        self.patch_start_idx = patch_start_idx
        self.preserve_prefix_tokens = preserve_prefix_tokens
        self.use_gaussian = use_gaussian
        self.feature = None
        self.token_indices = None
        self._hook = layer_norm_module.register_forward_hook(self._fn)

    def _fn(self, module, inputs, output):
        self.feature = output.detach()
        if self.token_indices is None:
            return self.feature
        masked = self._generate_masked_features(self.feature, self.token_indices)
        return torch.cat((self.feature, masked), dim=0)

    def set_token_indices(self, token_indices):
        self.token_indices = [int(idx) for idx in token_indices]

    def _generate_masked_features(self, feature, token_indices):
        _, total_tokens, dim = feature.shape
        num_spatial = total_tokens - self.patch_start_idx
        side = int(num_spatial**0.5)
        if side * side != num_spatial:
            raise ValueError(f"Expected square spatial token grid, got {num_spatial} tokens")

        num_masks = len(token_indices)
        masked = torch.zeros(
            num_masks,
            total_tokens,
            dim,
            device=feature.device,
            dtype=feature.dtype,
        )
        if num_masks == 0:
            return masked
        if self.preserve_prefix_tokens > 0:
            masked[:, :self.preserve_prefix_tokens, :] = feature[0, :self.preserve_prefix_tokens, :]

        if not self.use_gaussian:
            for mask_idx, spatial_idx in enumerate(token_indices):
                token_pos = self.patch_start_idx + spatial_idx
                masked[mask_idx, token_pos, :] = feature[0, token_pos, :]
            return masked

        spatial = feature[0, self.patch_start_idx :, :].reshape(side, side, dim)
        masked_spatial = masked[:, self.patch_start_idx :, :].reshape(num_masks, side, side, dim)
        kernel = GAUSSIAN_KERNEL.to(device=feature.device, dtype=feature.dtype).unsqueeze(-1)

        for mask_idx, spatial_idx in enumerate(token_indices):
            y, x = divmod(spatial_idx, side)
            y0 = max(y - 1, 0)
            y1 = min(y + 1, side - 1)
            x0 = max(x - 1, 0)
            x1 = min(x + 1, side - 1)
            patch = spatial[y0 : y1 + 1, x0 : x1 + 1, :]
            ky0 = 1 - (y - y0)
            kx0 = 1 - (x - x0)
            kernel_patch = kernel[
                ky0 : ky0 + patch.shape[0],
                kx0 : kx0 + patch.shape[1],
                :,
            ]
            masked_spatial[mask_idx, y0 : y1 + 1, x0 : x1 + 1, :] = (
                patch * kernel_patch
            )
        return masked

    def clear(self):
        self.feature = None
        self.token_indices = None

    def remove(self):
        self._hook.remove()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(ckpt_path, device):
    """Load DINOv3Classifier.  SDPA attention is fine -- no gradients or
    attention matrices are needed for ReciproCAM."""
    model = DINOv3Classifier(MODEL_ID, freeze_backbone=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Threshold computation (test-set 2% FPR)
# ---------------------------------------------------------------------------
def compute_threshold_at_2pct_fpr(model, records, transform, device):
    """Run inference on full test set and find the threshold at 2% FPR."""
    all_scores, all_labels = [], []
    for rec in records:
        img = Image.open(rec["image_path"]).convert("RGB")
        pixel_values = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(pixel_values)
            score = torch.sigmoid(logit).item()
        all_scores.append(score)
        all_labels.append(rec["label"])

    y_true = np.array(all_labels)
    y_score = np.array(all_scores)

    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_score)
    target_fpr = 0.02
    distances = np.abs(fpr_arr - target_fpr)
    min_dist = distances.min()
    tied = np.where(distances == min_dist)[0]
    idx = tied[np.argmax(tpr_arr[tied])]
    thresh_idx = min(idx, len(thresholds) - 1)

    thresh = float(thresholds[thresh_idx])
    actual_fpr = float(fpr_arr[idx])
    actual_tpr = float(tpr_arr[idx])
    print(f"  Threshold @ ~2% FPR: {thresh:.6f}  (actual FPR={actual_fpr:.4f}, TPR={actual_tpr:.4f})")
    return thresh, y_score, y_true


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------
def denormalize(tensor):
    """Undo ImageNet normalization for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


def make_heatmap_overlay(img_np, heatmap, alpha=0.5):
    cmap = plt.cm.jet
    heatmap_colored = cmap(heatmap)[:, :, :3]
    overlay = (1 - alpha) * img_np + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)


# ---------------------------------------------------------------------------
# ViT-ReciproCAM (Byun & Lee, 2023)
# ---------------------------------------------------------------------------
def resolve_target_layer_index(n_layers, target_layer_idx):
    """Convert negative indices and avoid the unsupported final block."""
    idx = target_layer_idx if target_layer_idx >= 0 else n_layers + target_layer_idx
    if idx == n_layers - 1:
        idx = n_layers - 2
    if idx < 0 or idx >= n_layers:
        raise ValueError(f"Invalid target layer {target_layer_idx} for {n_layers} layers")
    return idx


def infer_spatial_grid(model, pixel_values, patch_start_idx):
    """Infer the actual spatial token grid from the model, not from image size."""
    with torch.inference_mode():
        inputs = pixel_values.to(model.backbone.embeddings.patch_embeddings.weight.dtype)
        total_tokens = model.backbone.embeddings(inputs).shape[1]
    num_patches = total_tokens - patch_start_idx
    side = int(num_patches**0.5)
    if side * side != num_patches:
        raise ValueError(f"Expected square spatial token grid, got {num_patches} spatial tokens")
    return num_patches, side


def compute_reciprocam_map(model, pixel_values, target_layer_idx,
                           patch_start_idx, preserve_prefix_tokens=1,
                           use_gaussian=True, cam_chunk_size=64):
    """
    Official-style ViT-ReciproCAM.

    The model runs once on the original image. At the target block's `norm1`
    output, we inject masked samples for a chunk of spatial tokens, keep
    CLS/register tokens intact, and let the remainder of the model produce
    `1 + chunk` logits. Repeating this over token chunks avoids materializing
    the full `1 + N` batch at high resolution.

    Returns
    -------
    heatmap : (H_patches, W_patches) ndarray in [0, 1]
    base_score : float -- sigmoid score from the normal CLS-pooled forward pass
    """
    chunk_size = max(1, int(cam_chunk_size))
    while True:
        try:
            return _compute_reciprocam_map_chunked(
                model,
                pixel_values,
                target_layer_idx,
                patch_start_idx,
                preserve_prefix_tokens=preserve_prefix_tokens,
                use_gaussian=use_gaussian,
                cam_chunk_size=chunk_size,
            )
        except torch.OutOfMemoryError:
            if pixel_values.device.type == "cuda":
                torch.cuda.empty_cache()
            if chunk_size == 1:
                raise
            new_chunk_size = max(1, chunk_size // 2)
            print(f" OOM at chunk={chunk_size}; retrying with chunk={new_chunk_size}...", end="", flush=True)
            chunk_size = new_chunk_size


def _compute_reciprocam_map_chunked(model, pixel_values, target_layer_idx,
                                    patch_start_idx, preserve_prefix_tokens=1,
                                    use_gaussian=True,
                                    cam_chunk_size=64):
    backbone = model.backbone
    n_layers = len(backbone.layer)
    target_layer_idx = resolve_target_layer_index(n_layers, target_layer_idx)
    target_norm = backbone.layer[target_layer_idx].norm1
    n_patches, patches_per_side = infer_spatial_grid(model, pixel_values, patch_start_idx)

    hook = ReciproCAMHook(
        target_norm,
        patch_start_idx=patch_start_idx,
        preserve_prefix_tokens=preserve_prefix_tokens,
        use_gaussian=use_gaussian,
    )
    all_scores = []
    base_score = None
    try:
        with torch.inference_mode():
            for start in range(0, n_patches, cam_chunk_size):
                end = min(start + cam_chunk_size, n_patches)
                hook.set_token_indices(range(start, end))
                logits = model(pixel_values)
                probs = torch.sigmoid(logits).flatten()
                if base_score is None:
                    base_score = float(probs[0].item())
                chunk_scores = probs[1:]
                if chunk_scores.numel() != end - start:
                    raise RuntimeError(
                        f"Expected {end - start} ReciproCAM scores, got {chunk_scores.numel()}"
                    )
                all_scores.append(chunk_scores.detach().cpu().float())
                hook.clear()
    finally:
        hook.remove()

    scores = torch.cat(all_scores).numpy()

    # ---- reshape to spatial grid ---------------------------------------------
    if len(scores) > n_patches:
        scores = scores[:n_patches]
    elif len(scores) < n_patches:
        scores = np.pad(scores, (0, n_patches - len(scores)))

    heatmap = scores.reshape(patches_per_side, patches_per_side)
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap, base_score


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def create_visualization(img_path, uuid, label, results, out_path):
    """
    Multi-panel figure:
      col 0: original image
      col 1..N: ReciproCAM heatmap overlays for each model / layer variant
    """
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil.resize((RESOLUTION, RESOLUTION), Image.BICUBIC)) / 255.0

    n_cols = 1 + len(results)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5.5))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(img_np)
    gt_str = "FAKE" if label == 1 else "AUTHENTIC"
    axes[0].set_title(f"Original\nGT: {gt_str}\n{uuid[:12]}…", fontsize=9)
    axes[0].axis("off")

    for i, (model_name, heatmap, model_score, threshold) in enumerate(results, 1):
        heatmap_up = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (RESOLUTION, RESOLUTION), Image.BICUBIC
            )
        ) / 255.0

        overlay = make_heatmap_overlay(img_np, heatmap_up, alpha=0.55)
        axes[i].imshow(overlay)

        if label == 0:
            verdict = "✓ TP (auth✓)" if model_score < threshold else "✗ FN (false alarm)"
        else:
            verdict = "✗ FP (missed)" if model_score < threshold else "✓ TN (caught)"

        color = "green" if "✓" in verdict else "red"
        axes[i].set_title(
            f"{model_name}\nscore={model_score:.4f} (thr={threshold:.4f})\n{verdict}",
            fontsize=8,
            color=color,
        )
        axes[i].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------
def select_near_and_confident_above(indices, scores, threshold, n_near=3, n_confident=3):
    """Threshold-near + very confident examples from predicted-positive indices."""
    if len(indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    idx = np.asarray(indices, dtype=int)
    idx_scores = scores[idx]

    margins = idx_scores - threshold
    near_order = np.argsort(margins)
    near_count = min(n_near, len(idx))
    near = idx[near_order[:near_count]]

    conf_order = np.argsort(-idx_scores)
    confident = []
    for pos in conf_order:
        cand = idx[pos]
        if cand in near:
            continue
        confident.append(cand)
        if len(confident) >= n_confident:
            break

    return near.astype(int), np.array(confident, dtype=int)


def select_near_and_confident_below(indices, scores, threshold, n_near=3, n_confident=3):
    """Threshold-near + very confident examples from predicted-negative indices."""
    if len(indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    idx = np.asarray(indices, dtype=int)
    idx_scores = scores[idx]

    margins = threshold - idx_scores
    near_order = np.argsort(margins)
    near_count = min(n_near, len(idx))
    near = idx[near_order[:near_count]]

    conf_order = np.argsort(idx_scores)
    confident = []
    for pos in conf_order:
        cand = idx[pos]
        if cand in near:
            continue
        confident.append(cand)
        if len(confident) >= n_confident:
            break

    return near.astype(int), np.array(confident, dtype=int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ViT-ReciproCAM visualization")
    parser.add_argument("--n-near", type=int, default=3,
                        help="Threshold-near examples per category (TP / FN)")
    parser.add_argument("--n-confident", type=int, default=3,
                        help="Very-confident examples per category (TP / FN)")
    parser.add_argument("--target-layer", type=int, default=-2,
                        help="Target transformer block (negative = from end; last block is remapped to second-last)")
    parser.add_argument("--single-token", action="store_true",
                        help="Disable the paper's default 3x3 Gaussian neighborhood masking")
    parser.add_argument("--keep-register-tokens", action="store_true",
                        help="Preserve DINOv3 register tokens in masked samples (default: preserve CLS only)")
    parser.add_argument("--cam-chunk-size", type=int, default=64,
                        help="Masked tokens per forward pass; lower uses less VRAM")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    transform = build_transform(RESOLUTION)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Region: {REGION}  |  Method: ViT-ReciproCAM")
    print(f"{'='*60}")
    print(f"Chunk size: {args.cam_chunk_size}")
    records = load_metadata(REGION, split="test")

    # -----------------------------------------------------------------------
    # Step 1: Load models & compute thresholds
    # -----------------------------------------------------------------------
    models_info = {}

    for model_name, ckpt_path in MODELS_TO_EVAL.items():
        print(f"\n--- Loading {model_name} ---")
        if not ckpt_path.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        model = load_model(ckpt_path, device)

        num_register = getattr(model.backbone.config, "num_register_tokens", 0)
        patch_start_idx = 1 + num_register
        n_layers = len(list(model.backbone.layer))
        print(f"  Prefix tokens: {patch_start_idx} (1 CLS + {num_register} register)")
        print(f"  Transformer layers: {n_layers}")

        print("  Computing test set scores for threshold...")
        threshold, scores, labels = compute_threshold_at_2pct_fpr(
            model, records, transform, device)

        models_info[model_name] = {
            "model": model,
            "threshold": threshold,
            "scores": scores,
            "labels": labels,
            "patch_start_idx": patch_start_idx,
            "num_register": num_register,
            "n_layers": n_layers,
        }

    if not models_info:
        print("No models loaded. Exiting.")
        return

    # -----------------------------------------------------------------------
    # Step 2: TP / FN split & pick threshold-near + confident examples
    # -----------------------------------------------------------------------
    first_info = models_info[next(iter(models_info))]
    y_true = first_info["labels"]
    y_score = first_info["scores"]
    threshold = first_info["threshold"]

    fake_indices = np.where(y_true == 1)[0]
    tp_indices = fake_indices[y_score[fake_indices] >= threshold]
    fn_indices = fake_indices[y_score[fake_indices] < threshold]

    print(f"\nFake images: {len(fake_indices)} total")
    print(f"  TP (detected):        {len(tp_indices)}")
    print(f"  FN (missed as auth):  {len(fn_indices)}")

    tp_near, tp_conf = select_near_and_confident_above(
        tp_indices, y_score, threshold,
        n_near=args.n_near, n_confident=args.n_confident,
    )
    fn_near, fn_conf = select_near_and_confident_below(
        fn_indices, y_score, threshold,
        n_near=args.n_near, n_confident=args.n_confident,
    )

    print("\nSelected for visualization:")
    print(f"  TP near-threshold:   {len(tp_near)}  (barely caught)")
    print(f"  TP super-confident:  {len(tp_conf)}  (clearly fake)")
    print(f"  FN near-threshold:   {len(fn_near)}  (almost caught)")
    print(f"  FN super-confident:  {len(fn_conf)}  (model thinks very authentic)")

    # -----------------------------------------------------------------------
    # Step 3: Generate ReciproCAM saliency maps
    # -----------------------------------------------------------------------
    all_selected = []
    all_selected.extend([("TP_NEAR", int(i)) for i in tp_near])
    all_selected.extend([("TP_CONF", int(i)) for i in tp_conf])
    all_selected.extend([("FN_NEAR", int(i)) for i in fn_near])
    all_selected.extend([("FN_CONF", int(i)) for i in fn_conf])

    for idx_i, (category, rec_idx) in enumerate(all_selected):
        rec = records[rec_idx]
        uuid = rec["session_uuid"]
        label = rec["label"]
        img_path = rec["image_path"]

        print(f"\n[{idx_i+1}/{len(all_selected)}] {category} — {uuid[:16]}…")

        img = Image.open(img_path).convert("RGB")
        pixel_values = transform(img).unsqueeze(0).to(device)

        results = []

        for model_name, info in models_info.items():
            model = info["model"]
            patch_start_idx = info["patch_start_idx"]
            num_register = info["num_register"]
            thresh = info["threshold"]
            n_layers = info["n_layers"]
            preserve_prefix_tokens = patch_start_idx if args.keep_register_tokens else 1

            # ── 1) ReciproCAM at the requested target layer ───────────
            tgt_abs = resolve_target_layer_index(n_layers, args.target_layer)
            print(f"    {model_name}: layer {tgt_abs}/{n_layers} …",
                  end="", flush=True)
            heatmap, score = compute_reciprocam_map(
                model,
                pixel_values,
                tgt_abs,
                patch_start_idx,
                preserve_prefix_tokens=preserve_prefix_tokens,
                use_gaussian=not args.single_token,
                cam_chunk_size=args.cam_chunk_size,
            )
            results.append((f"{model_name} L{tgt_abs}", heatmap, score, thresh))
            print(f" score={score:.4f}")

            # ── 2) For partial_qv_last4: also target layer n-4 ────────
            if "qv_last4" in model_name:
                tgt4 = n_layers - 4
                print(f"    {model_name}: layer {tgt4}/{n_layers} …",
                      end="", flush=True)
                heatmap4, score4 = compute_reciprocam_map(
                    model,
                    pixel_values,
                    tgt4,
                    patch_start_idx,
                    preserve_prefix_tokens=preserve_prefix_tokens,
                    use_gaussian=not args.single_token,
                    cam_chunk_size=args.cam_chunk_size,
                )
                results.append(
                    (f"{model_name} L{tgt4}", heatmap4, score4, thresh))
                print(f" score={score4:.4f}")

        out_name = f"{category}_{idx_i:02d}_{uuid[:12]}.png"
        create_visualization(img_path, uuid, label, results, OUT_DIR / out_name)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Done! {len(all_selected)} images visualized with ViT-ReciproCAM.")
    print(f"Output directory: {OUT_DIR}")
    print(f"{'='*60}")

    for model_name, info in models_info.items():
        tp_count = np.sum((info["labels"] == 1) & (info["scores"] >= info["threshold"]))
        fn_count = np.sum((info["labels"] == 1) & (info["scores"] < info["threshold"]))
        print(f"  {model_name}: threshold={info['threshold']:.6f}  TP={tp_count}  FN={fn_count}")


if __name__ == "__main__":
    main()
