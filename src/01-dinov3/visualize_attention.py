#!/root/.venv/bin/python
"""
Gradient-Weighted Attention Rollout (Chefer et al., 2021) for DINOv3 ViT-L.

Generates heatmap overlays showing where each model focuses on care_label
test images, comparing TP (correctly detected fakes) vs FN (missed fakes).

Models:
  1. vitl16_714_finetune        (full fine-tune)
  2. vitl16_714_partial_qv_last4 (partial fine-tune)

Usage:
  python visualize_attention.py
  python visualize_attention.py --n-near 3 --n-confident 3
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_curve
from torchvision import transforms

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
PATCH_SIZE = 14
NUM_PATCHES_PER_SIDE = RESOLUTION // PATCH_SIZE  # 51

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / REGION
OUT_DIR = Path(__file__).resolve().parent / "attention_maps"

MODELS_TO_EVAL = {
    "vitl16_714_finetune": CKPT_DIR / f"{MODEL_KEY}_{RESOLUTION}_finetune_best.pt",
    "vitl16_714_partial_qv_last4": CKPT_DIR / f"{MODEL_KEY}_{RESOLUTION}_partial_qv_last4_best.pt",
}


# ---------------------------------------------------------------------------
# Attention hook
# ---------------------------------------------------------------------------
class AttentionHook:
    """Register forward hooks on all DINOv3ViTAttention modules to capture
    attention weight matrices (with grad retained for backward)."""

    def __init__(self, model):
        self.attentions = []  # list of (n_heads, seq, seq) tensors
        self._hooks = []
        # Navigate: model.backbone.layer[i].attention
        for layer in model.backbone.layer:
            h = layer.attention.register_forward_hook(self._hook_fn)
            self._hooks.append(h)

    def _hook_fn(self, module, input, output):
        # output = (attn_output, attn_weights)
        attn_weights = output[1]  # (batch, heads, seq, seq)
        attn_weights.retain_grad()
        self.attentions.append(attn_weights)

    def clear(self):
        self.attentions = []

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ---------------------------------------------------------------------------
# Gradient-weighted attention rollout (Chefer et al. 2021)
# ---------------------------------------------------------------------------
def gradient_attention_rollout(attentions, gradients, start_layer=0):
    """
    Compute class-discriminative attention rollout.

    Args:
        attentions: list of (1, heads, seq, seq) attention weight tensors
        gradients:  list of (1, heads, seq, seq) gradient tensors
        start_layer: first layer to include in rollout (0 = all layers)

    Returns:
        (seq,) relevance vector for the CLS token
    """
    num_tokens = attentions[0].shape[-1]
    device = attentions[0].device

    result = torch.eye(num_tokens, device=device, dtype=torch.float32)

    for i in range(start_layer, len(attentions)):
        attn = attentions[i].detach().float()   # (1, heads, seq, seq)
        grad = gradients[i].detach().float()     # (1, heads, seq, seq)

        # Element-wise: positive relevance only, averaged over heads
        cam = (grad * attn).clamp(min=0).mean(dim=1).squeeze(0)  # (seq, seq)

        # Add identity (residual connection)
        cam = cam + torch.eye(num_tokens, device=device, dtype=torch.float32)
        # Re-normalize rows
        cam = cam / cam.sum(dim=-1, keepdim=True)

        result = result @ cam

    # CLS token row — relevance of each token to CLS
    cls_relevance = result[0]  # (seq,)
    return cls_relevance


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(ckpt_path, device):
    """Load a DINOv3Classifier with eager attention (needed for attn weights)."""
    model = DINOv3Classifier(MODEL_ID, freeze_backbone=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Switch to eager attention so we can extract attention matrices
    model.backbone.config._attn_implementation = "eager"

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
    # Find threshold closest to 2% FPR
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
    """Overlay a heatmap on an image.

    Args:
        img_np: (H, W, 3) float array in [0, 1]
        heatmap: (H, W) float array in [0, 1]
        alpha: blend factor

    Returns:
        (H, W, 3) float array
    """
    cmap = plt.cm.jet
    heatmap_colored = cmap(heatmap)[:, :, :3]  # (H, W, 3)
    overlay = (1 - alpha) * img_np + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)


# ---------------------------------------------------------------------------
# Main attention map computation for one image
# ---------------------------------------------------------------------------
def compute_attention_map(model, pixel_values, hook, num_prefix_tokens, start_layer=0):
    """
    Compute gradient-weighted attention rollout for a single image.

    Returns:
        heatmap: (H_patches, W_patches) numpy array, normalized [0, 1]
        score: float, sigmoid probability
    """
    hook.clear()
    model.zero_grad()

    pixel_values = pixel_values.requires_grad_(False)
    # Forward pass (NOT inference mode — need gradients through attention)
    logit = model(pixel_values)
    score = torch.sigmoid(logit).item()

    # Backward from the "fake" logit
    logit.backward(retain_graph=False)

    # Collect attention weights and their gradients
    attentions = [a for a in hook.attentions]
    gradients = [a.grad for a in hook.attentions]

    # Gradient-weighted attention rollout
    cls_relevance = gradient_attention_rollout(attentions, gradients, start_layer=start_layer)

    # Extract patch relevances (skip CLS + register tokens)
    patch_relevance = cls_relevance[num_prefix_tokens:]

    # Reshape to 2D grid
    n_patches = NUM_PATCHES_PER_SIDE * NUM_PATCHES_PER_SIDE
    if len(patch_relevance) > n_patches:
        patch_relevance = patch_relevance[:n_patches]
    elif len(patch_relevance) < n_patches:
        # Pad if needed (shouldn't happen)
        patch_relevance = F.pad(patch_relevance, (0, n_patches - len(patch_relevance)))

    heatmap = patch_relevance.reshape(NUM_PATCHES_PER_SIDE, NUM_PATCHES_PER_SIDE)
    heatmap = heatmap.detach().cpu().float().numpy()

    # Normalize to [0, 1]
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap, score


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def create_visualization(img_path, uuid, label, results, out_path):
    """
    Create a multi-panel figure:
      col 0: original image
      col 1..N: heatmap overlays for each model variant

    Args:
        img_path: path to original image
        uuid: session UUID
        label: ground truth (0=authentic, 1=fake)
        results: list of (model_name, heatmap, score, threshold) tuples
        out_path: output PNG path
    """
    # Load original image at display resolution
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil.resize((RESOLUTION, RESOLUTION), Image.BICUBIC)) / 255.0

    n_cols = 1 + len(results)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5.5))
    if n_cols == 1:
        axes = [axes]

    # Original image
    axes[0].imshow(img_np)
    gt_str = "FAKE" if label == 1 else "AUTHENTIC"
    axes[0].set_title(f"Original\nGT: {gt_str}\n{uuid[:12]}…", fontsize=9)
    axes[0].axis("off")

    # Heatmap overlays
    for i, (model_name, heatmap, model_score, threshold) in enumerate(results, 1):
        # Upsample heatmap to image resolution
        heatmap_up = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (RESOLUTION, RESOLUTION), Image.BICUBIC
            )
        ) / 255.0

        overlay = make_heatmap_overlay(img_np, heatmap_up, alpha=0.55)
        axes[i].imshow(overlay)

        pred = "FAKE" if model_score >= threshold else "AUTHENTIC"
        if label == 0:  # actually authentic
            verdict = "✓ TP (auth✓)" if model_score < threshold else "✗ FN (false alarm)"
        else:  # actually fake
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
# Main
# ---------------------------------------------------------------------------
def select_near_and_confident_above(indices, scores, threshold, n_near=3, n_confident=3):
    """Pick threshold-near and very confident examples from predicted-positive indices.

    Near-threshold: smallest margin above threshold (barely caught).
    Confident: highest scores (most confident positives).
    """
    if len(indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    idx = np.asarray(indices, dtype=int)
    idx_scores = scores[idx]

    margins = idx_scores - threshold
    near_order = np.argsort(margins)  # closest above threshold first
    near_count = min(n_near, len(idx))
    near = idx[near_order[:near_count]]

    conf_order = np.argsort(-idx_scores)  # highest score first
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
    """Pick threshold-near and very confident examples from predicted-negative indices.

    Near-threshold: smallest margin below threshold (almost caught).
    Confident: lowest scores (model was most sure they're authentic).
    """
    if len(indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    idx = np.asarray(indices, dtype=int)
    idx_scores = scores[idx]

    margins = threshold - idx_scores  # positive = below threshold
    near_order = np.argsort(margins)  # closest below threshold first
    near_count = min(n_near, len(idx))
    near = idx[near_order[:near_count]]

    conf_order = np.argsort(idx_scores)  # lowest score first (most "authentic"-looking)
    confident = []
    for pos in conf_order:
        cand = idx[pos]
        if cand in near:
            continue
        confident.append(cand)
        if len(confident) >= n_confident:
            break

    return near.astype(int), np.array(confident, dtype=int)


def main():
    parser = argparse.ArgumentParser(description="Gradient-weighted attention rollout visualization")
    parser.add_argument(
        "--n-near",
        type=int,
        default=3,
        help="Number of threshold-near examples to show per category (TP and FN)",
    )
    parser.add_argument(
        "--n-confident",
        type=int,
        default=3,
        help="Number of very confident examples to show per category (TP and FN)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    transform = build_transform(RESOLUTION)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load test metadata
    print(f"\n{'='*60}")
    print(f"Region: {REGION}")
    print(f"{'='*60}")
    records = load_metadata(REGION, split="test")

    # -----------------------------------------------------------------------
    # Step 1: Load models & compute thresholds
    # -----------------------------------------------------------------------
    models_info = {}  # name -> {model, threshold, scores, hook, num_prefix}

    for model_name, ckpt_path in MODELS_TO_EVAL.items():
        print(f"\n--- Loading {model_name} ---")
        if not ckpt_path.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        model = load_model(ckpt_path, device)

        # Determine number of prefix tokens (CLS + register)
        num_register = getattr(model.backbone.config, "num_register_tokens", 0)
        num_prefix = 1 + num_register
        print(f"  Prefix tokens: {num_prefix} (1 CLS + {num_register} register)")

        print("  Computing test set scores for threshold...")
        threshold, scores, labels = compute_threshold_at_2pct_fpr(model, records, transform, device)

        models_info[model_name] = {
            "model": model,
            "threshold": threshold,
            "scores": scores,
            "labels": labels,
            "num_prefix": num_prefix,
        }

    if not models_info:
        print("No models loaded. Exiting.")
        return

    # -----------------------------------------------------------------------
    # Step 2: Identify TP/FP and pick threshold-near + very confident examples
    # -----------------------------------------------------------------------
    # Use the first model to determine TP/FP split for image selection
    first_model_name = list(models_info.keys())[0]
    first_info = models_info[first_model_name]
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
        tp_indices,
        y_score,
        threshold,
        n_near=args.n_near,
        n_confident=args.n_confident,
    )
    fn_near, fn_conf = select_near_and_confident_below(
        fn_indices,
        y_score,
        threshold,
        n_near=args.n_near,
        n_confident=args.n_confident,
    )

    print("\nSelected for visualization:")
    print(f"  TP near-threshold:   {len(tp_near)}  (barely caught)")
    print(f"  TP super-confident:  {len(tp_conf)}  (clearly fake)")
    print(f"  FN near-threshold:   {len(fn_near)}  (almost caught)")
    print(f"  FN super-confident:  {len(fn_conf)}  (model thinks very authentic)")

    # -----------------------------------------------------------------------
    # Step 3: Generate attention maps
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

        # Prepare image tensor
        img = Image.open(img_path).convert("RGB")
        pixel_values = transform(img).unsqueeze(0).to(device)

        results = []

        for model_name, info in models_info.items():
            model = info["model"]
            num_prefix = info["num_prefix"]
            thresh = info["threshold"]

            # ── 1) Full rollout (all layers) ──────────────────────────
            hook = AttentionHook(model)
            heatmap_full, score = compute_attention_map(
                model, pixel_values, hook, num_prefix, start_layer=0
            )
            results.append((model_name, heatmap_full, score, thresh))
            hook.remove()

            # ── 2) Last-4-layer rollout for partial_qv_last4 ──────────
            if "qv_last4" in model_name:
                hook2 = AttentionHook(model)
                n_layers = len(model.backbone.layer)
                heatmap_l4, score_l4 = compute_attention_map(
                    model, pixel_values, hook2, num_prefix,
                    start_layer=n_layers - 4
                )
                results.append((f"{model_name}_last4only", heatmap_l4, score_l4, thresh))
                hook2.remove()

        # Save figure
        out_name = f"{category}_{idx_i:02d}_{uuid[:12]}.png"
        create_visualization(img_path, uuid, label, results, OUT_DIR / out_name)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Done! {len(all_selected)} images visualized.")
    print(f"Output directory: {OUT_DIR}")
    print(f"{'='*60}")

    for model_name, info in models_info.items():
        tp_count = np.sum((info["labels"] == 1) & (info["scores"] >= info["threshold"]))
        fn_count = np.sum((info["labels"] == 1) & (info["scores"] < info["threshold"]))
        print(f"  {model_name}: threshold={info['threshold']:.6f}  TP={tp_count}  FN={fn_count}")


if __name__ == "__main__":
    main()
