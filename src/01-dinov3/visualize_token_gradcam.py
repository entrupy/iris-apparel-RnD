#!/root/.venv/bin/python
"""
Token Grad-CAM for DINOv3 ViT-L with configurable layer submodule targets.

This is a more flexible alternative to attention rollout / ReciproCAM for
probing DINOv3 internals. It lets you compare token-level saliency maps from
specific transformer submodules such as:
  - `norm2`            : semantic token features before the MLP
  - `attention.o_proj` : attention write-back into the token stream
  - `attention.v_proj` : value/content stream (especially useful for qv tuning)

Usage:
  python visualize_token_gradcam.py
  python visualize_token_gradcam.py --targets "-2:norm2,-2:attn.o,-2:attn.v"
  python visualize_token_gradcam.py --targets "-4:norm2,-2:norm2,-4:attn.v"
  python visualize_token_gradcam.py --n-near 1 --n-confident 0
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
    DINOv3Classifier,
    build_transform,
    load_metadata,
)

REGION = "care_label"
MODEL_KEY = "vitl16"
RESOLUTION = 714
MODEL_ID = MODEL_VARIANTS[MODEL_KEY]

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / REGION
OUT_DIR = Path(__file__).resolve().parent / "token_gradcam_maps"

MODELS_TO_EVAL = {
    "vitl16_714_finetune": CKPT_DIR / f"{MODEL_KEY}_{RESOLUTION}_finetune_best.pt",
    "vitl16_714_partial_qv_last4": CKPT_DIR / f"{MODEL_KEY}_{RESOLUTION}_partial_qv_last4_best.pt",
}

MODULE_ALIASES = {
    "norm2": "norm2",
    "attn.o": "attention.o_proj",
    "attn.v": "attention.v_proj",
    "attn.q": "attention.q_proj",
    "attn.k": "attention.k_proj",
    "attention.o_proj": "attention.o_proj",
    "attention.v_proj": "attention.v_proj",
    "attention.q_proj": "attention.q_proj",
    "attention.k_proj": "attention.k_proj",
}

MODULE_SHORT_NAMES = {
    "norm2": "norm2",
    "attention.o_proj": "attn.o",
    "attention.v_proj": "attn.v",
    "attention.q_proj": "attn.q",
    "attention.k_proj": "attn.k",
}

DEFAULT_TARGETS = "-2:norm2,-2:attn.o,-2:attn.v"


# ---------------------------------------------------------------------------
# Activation hook
# ---------------------------------------------------------------------------
class ActivationHook:
    """Capture the output tensor of a target module for Grad-CAM."""

    def __init__(self, module):
        self.activation = None
        self._hook = module.register_forward_hook(self._fn)

    def _fn(self, module, inputs, output):
        self.activation = output[0] if isinstance(output, tuple) else output

    def remove(self):
        self._hook.remove()


# ---------------------------------------------------------------------------
# Target parsing / module lookup
# ---------------------------------------------------------------------------
def parse_target_specs(spec_string):
    """Parse comma-separated `layer_idx:module_name` target specs."""
    targets = []
    for raw_spec in spec_string.split(","):
        spec = raw_spec.strip()
        if not spec:
            continue
        if ":" not in spec:
            raise ValueError(
                f"Invalid target spec `{spec}`. Use `layer_idx:module_name`, "
                "e.g. `-2:norm2` or `-2:attn.o`."
            )
        layer_str, module_key = spec.split(":", 1)
        layer_idx = int(layer_str)
        module_path = MODULE_ALIASES.get(module_key.strip(), module_key.strip())
        if module_path not in MODULE_SHORT_NAMES:
            valid = ", ".join(sorted(MODULE_ALIASES))
            raise ValueError(f"Unsupported module `{module_key}`. Valid aliases: {valid}")
        targets.append({
            "layer_idx": layer_idx,
            "module_path": module_path,
            "short_name": MODULE_SHORT_NAMES[module_path],
        })
    if not targets:
        raise ValueError("No valid Grad-CAM targets parsed from --targets.")
    return targets


def resolve_target_layer_index(n_layers, target_layer_idx):
    """Convert negative indices and keep them in range."""
    idx = target_layer_idx if target_layer_idx >= 0 else n_layers + target_layer_idx
    if idx < 0 or idx >= n_layers:
        raise ValueError(f"Invalid target layer {target_layer_idx} for {n_layers} layers")
    return idx


def get_submodule(root_module, dotted_path):
    """Resolve a dotted attribute path like `attention.o_proj`."""
    module = root_module
    for part in dotted_path.split("."):
        module = getattr(module, part)
    return module


def infer_spatial_grid(num_spatial_tokens):
    side = int(num_spatial_tokens**0.5)
    if side * side != num_spatial_tokens:
        raise ValueError(f"Expected square token grid, got {num_spatial_tokens} spatial tokens")
    return side


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(ckpt_path, device):
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
    print(
        f"  Threshold @ ~2% FPR: {thresh:.6f}  "
        f"(actual FPR={actual_fpr:.4f}, TPR={actual_tpr:.4f})"
    )
    return thresh, y_score, y_true


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def compute_token_gradcam(model, pixel_values, target_layer_idx, module_path, patch_start_idx):
    """
    Compute token Grad-CAM for a target submodule.

    The target module must output a `(batch, tokens, channels)` tensor.
    Prefix tokens (CLS + registers) are excluded from the final heatmap.
    """
    backbone = model.backbone
    n_layers = len(backbone.layer)
    layer_idx = resolve_target_layer_index(n_layers, target_layer_idx)
    target_module = get_submodule(backbone.layer[layer_idx], module_path)

    hook = ActivationHook(target_module)
    try:
        model.zero_grad(set_to_none=True)
        logit = model(pixel_values)
        score = torch.sigmoid(logit).item()

        activation = hook.activation
        if activation is None:
            raise RuntimeError(f"Hook did not capture activation for L{layer_idx}:{module_path}")
        if activation.ndim != 3:
            raise RuntimeError(
                f"Expected 3D token activation from L{layer_idx}:{module_path}, "
                f"got shape {tuple(activation.shape)}"
            )

        grad = torch.autograd.grad(
            logit.sum(),
            activation,
            retain_graph=False,
            create_graph=False,
        )[0]
    finally:
        hook.remove()

    act_spatial = activation[0, patch_start_idx:, :].detach().float()
    grad_spatial = grad[0, patch_start_idx:, :].detach().float()

    num_spatial = act_spatial.shape[0]
    side = infer_spatial_grid(num_spatial)

    # Grad-CAM weights: average gradient over spatial tokens, per channel.
    channel_weights = grad_spatial.mean(dim=0)  # (C,)
    cam = torch.relu((act_spatial * channel_weights.unsqueeze(0)).sum(dim=-1))  # (T,)

    heatmap = cam.reshape(side, side).cpu().numpy()
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap, score, layer_idx


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def make_heatmap_overlay(img_np, heatmap, alpha=0.5):
    cmap = plt.cm.jet
    heatmap_colored = cmap(heatmap)[:, :, :3]
    overlay = (1 - alpha) * img_np + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)


def create_visualization(img_path, uuid, label, results, out_path):
    """
    Multi-panel figure:
      col 0: original image
      col 1..N: Grad-CAM heatmap overlays for each model / target
    """
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil.resize((RESOLUTION, RESOLUTION), Image.BICUBIC)) / 255.0

    n_cols = 1 + len(results)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.6 * n_cols, 5.5))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(img_np)
    gt_str = "FAKE" if label == 1 else "AUTHENTIC"
    axes[0].set_title(f"Original\nGT: {gt_str}\n{uuid[:12]}…", fontsize=9)
    axes[0].axis("off")

    for i, (panel_name, heatmap, model_score, threshold) in enumerate(results, 1):
        heatmap_up = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (RESOLUTION, RESOLUTION), Image.BICUBIC
            )
        ) / 255.0

        overlay = make_heatmap_overlay(img_np, heatmap_up, alpha=0.55)
        axes[i].imshow(overlay)

        pred_is_fake = model_score >= threshold
        pred = "FAKE" if pred_is_fake else "AUTHENTIC"
        correct = (label == 1 and pred_is_fake) or (label == 0 and not pred_is_fake)
        color = "green" if correct else "red"
        status = "correct" if correct else "wrong"

        axes[i].set_title(
            f"{panel_name}\nscore={model_score:.4f} (thr={threshold:.4f})\npred={pred} · {status}",
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
    """Pick threshold-near and very confident examples from predicted-positive indices."""
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
    """Pick threshold-near and very confident examples from predicted-negative indices."""
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
    parser = argparse.ArgumentParser(description="Token Grad-CAM visualization for DINOv3")
    parser.add_argument(
        "--targets",
        type=str,
        default=DEFAULT_TARGETS,
        help=(
            "Comma-separated `layer_idx:module_name` specs. "
            "Example: `-2:norm2,-2:attn.o,-2:attn.v`"
        ),
    )
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
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    targets = parse_target_specs(args.targets)
    device = torch.device(args.device)
    transform = build_transform(RESOLUTION)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Region: {REGION}  |  Method: Token Grad-CAM")
    print(f"{'='*60}")
    print("Targets:")
    for target in targets:
        print(f"  {target['layer_idx']:+d}:{target['short_name']}")

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
        n_layers = len(model.backbone.layer)
        print(f"  Prefix tokens: {patch_start_idx} (1 CLS + {num_register} register)")
        print(f"  Transformer layers: {n_layers}")

        print("  Computing test set scores for threshold...")
        threshold, scores, labels = compute_threshold_at_2pct_fpr(model, records, transform, device)

        models_info[model_name] = {
            "model": model,
            "threshold": threshold,
            "scores": scores,
            "labels": labels,
            "patch_start_idx": patch_start_idx,
            "n_layers": n_layers,
        }

    if not models_info:
        print("No models loaded. Exiting.")
        return

    # -----------------------------------------------------------------------
    # Step 2: Identify TP/FN and pick threshold-near + very confident examples
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
    # Step 3: Generate Grad-CAM maps
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
            thresh = info["threshold"]
            n_layers = info["n_layers"]

            for target in targets:
                target_abs = resolve_target_layer_index(n_layers, target["layer_idx"])
                print(
                    f"    {model_name}: L{target_abs} {target['short_name']} …",
                    end="",
                    flush=True,
                )
                heatmap, score, actual_layer = compute_token_gradcam(
                    model,
                    pixel_values,
                    target["layer_idx"],
                    target["module_path"],
                    patch_start_idx,
                )
                panel_name = f"{model_name} L{actual_layer} {target['short_name']}"
                results.append((panel_name, heatmap, score, thresh))
                print(f" score={score:.4f}")

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
