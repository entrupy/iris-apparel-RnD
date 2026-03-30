#!/root/.venv/bin/python
"""
Token Grad-CAM for DINOv3 ViT-L with configurable layer submodule targets.

Generates heatmap overlays for all 4 auth-positive categories:
  TP = authentic correctly passed
  FP = fake wrongly passed (missed)
  FN = authentic wrongly flagged
  TN = fake correctly caught

Auto-discovers partial finetune checkpoints for the given region.

Usage:
  python visualize_token_gradcam.py
  python visualize_token_gradcam.py --region front
  python visualize_token_gradcam.py --targets "-2:norm2,-2:attn.o,-2:attn.v"
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MODEL_VARIANTS,
    REGIONS,
    DINOv3Classifier,
    build_transform,
    load_metadata,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_KEY = "vitl16"
RESOLUTION = 714
MODEL_ID = MODEL_VARIANTS[MODEL_KEY]

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
    targets = []
    for raw_spec in spec_string.split(","):
        spec = raw_spec.strip()
        if not spec:
            continue
        if ":" not in spec:
            raise ValueError(f"Invalid target spec `{spec}`. Use `layer_idx:module_name`.")
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
    idx = target_layer_idx if target_layer_idx >= 0 else n_layers + target_layer_idx
    if idx < 0 or idx >= n_layers:
        raise ValueError(f"Invalid target layer {target_layer_idx} for {n_layers} layers")
    return idx


def get_submodule(root_module, dotted_path):
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
# Model loading + val threshold
# ---------------------------------------------------------------------------
def load_model_and_threshold(ckpt_path, device):
    model = DINOv3Classifier(MODEL_ID, freeze_backbone=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    metrics = ckpt.get("metrics", {})
    thresh_info = metrics.get("tpr_at_fpr", {}).get("2%", {})
    threshold = thresh_info.get("threshold_orig", 0.5)
    print(f"  Val threshold @2%: {threshold:.6f}")
    return model, threshold


def score_test_set(model, records, transform, device):
    all_scores, all_labels = [], []
    for rec in records:
        img = Image.open(rec["image_path"]).convert("RGB")
        pv = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            score = torch.sigmoid(model(pv)).item()
        all_scores.append(score)
        all_labels.append(rec["label"])
    return np.array(all_scores), np.array(all_labels)


def discover_checkpoints(region):
    ckpt_dir = SCRIPT_DIR / "checkpoints" / region
    found = {}
    for p in sorted(ckpt_dir.glob(f"{MODEL_KEY}_{RESOLUTION}_partial_*_best.pt")):
        name = p.stem.replace("_best", "")
        found[name] = p
    return found


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def compute_token_gradcam(model, pixel_values, target_layer_idx, module_path, patch_start_idx):
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
            raise RuntimeError(f"Expected 3D activation, got shape {tuple(activation.shape)}")

        grad = torch.autograd.grad(logit.sum(), activation,
                                   retain_graph=False, create_graph=False)[0]
    finally:
        hook.remove()

    act_spatial = activation[0, patch_start_idx:, :].detach().float()
    grad_spatial = grad[0, patch_start_idx:, :].detach().float()

    num_spatial = act_spatial.shape[0]
    side = infer_spatial_grid(num_spatial)

    channel_weights = grad_spatial.mean(dim=0)
    cam = torch.relu((act_spatial * channel_weights.unsqueeze(0)).sum(dim=-1))

    heatmap = cam.reshape(side, side).cpu().numpy()
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap, score, layer_idx


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def make_heatmap_overlay(img_np, heatmap, alpha=0.5):
    cmap = plt.cm.jet
    heatmap_colored = cmap(heatmap)[:, :, :3]
    overlay = (1 - alpha) * img_np + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)


def create_visualization(img_path, uuid, label, results, out_path):
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil.resize((RESOLUTION, RESOLUTION), Image.BICUBIC)) / 255.0

    n_cols = 1 + len(results)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.6 * n_cols, 5.5))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(img_np)
    gt_str = "FAKE" if label == 1 else "AUTHENTIC"
    axes[0].set_title(f"Original\nGT: {gt_str}\n{uuid[:12]}...", fontsize=9)
    axes[0].axis("off")

    for i, (panel_name, heatmap, model_score, threshold) in enumerate(results, 1):
        heatmap_up = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (RESOLUTION, RESOLUTION), Image.BICUBIC)
        ) / 255.0
        overlay = make_heatmap_overlay(img_np, heatmap_up, alpha=0.55)
        axes[i].imshow(overlay)

        if label == 0:
            verdict = "TP (auth passed)" if model_score < threshold else "FN (auth flagged)"
        else:
            verdict = "FP (fake missed)" if model_score < threshold else "TN (fake caught)"

        correct = ("TP" in verdict or "TN" in verdict)
        color = "green" if correct else "red"
        axes[i].set_title(
            f"{panel_name}\nscore={model_score:.4f} (thr={threshold:.4f})\n{verdict}",
            fontsize=8, color=color)
        axes[i].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------
def select_near_and_confident(indices, scores, threshold, above, n_near=3, n_confident=3):
    if len(indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    idx = np.asarray(indices, dtype=int)
    s = scores[idx]
    if above:
        margins = s - threshold
        near_order = np.argsort(margins)
        conf_order = np.argsort(-s)
    else:
        margins = threshold - s
        near_order = np.argsort(margins)
        conf_order = np.argsort(s)
    near_count = min(n_near, len(idx))
    near = idx[near_order[:near_count]]
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
    parser.add_argument("--region", type=str, default="care_label", choices=REGIONS)
    parser.add_argument("--targets", type=str, default=DEFAULT_TARGETS)
    parser.add_argument("--n-near", type=int, default=3)
    parser.add_argument("--n-confident", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    targets = parse_target_specs(args.targets)
    device = torch.device(args.device)
    transform = build_transform(RESOLUTION)
    out_dir = SCRIPT_DIR / "token_gradcam_maps" / args.region
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Region: {args.region}  |  Method: Token Grad-CAM")
    print(f"{'='*60}")
    print("Targets:")
    for t in targets:
        print(f"  {t['layer_idx']:+d}:{t['short_name']}")

    records = load_metadata(args.region, split="test")
    ckpts = discover_checkpoints(args.region)
    if not ckpts:
        print("No checkpoints found. Exiting.")
        return
    print(f"Found {len(ckpts)} checkpoints: {', '.join(ckpts.keys())}")

    models_info = {}
    for name, path in ckpts.items():
        print(f"\n--- {name} ---")
        model, threshold = load_model_and_threshold(path, device)
        num_register = getattr(model.backbone.config, "num_register_tokens", 0)
        patch_start_idx = 1 + num_register
        n_layers = len(model.backbone.layer)

        print("  Scoring test set...")
        scores, labels = score_test_set(model, records, transform, device)
        models_info[name] = {
            "model": model, "threshold": threshold,
            "scores": scores, "labels": labels,
            "patch_start_idx": patch_start_idx, "n_layers": n_layers,
        }

    first = models_info[next(iter(models_info))]
    y_true, y_score, threshold = first["labels"], first["scores"], first["threshold"]

    auth_idx = np.where(y_true == 0)[0]
    fake_idx = np.where(y_true == 1)[0]
    tp_idx = auth_idx[y_score[auth_idx] < threshold]
    fn_idx = auth_idx[y_score[auth_idx] >= threshold]
    fp_idx = fake_idx[y_score[fake_idx] < threshold]
    tn_idx = fake_idx[y_score[fake_idx] >= threshold]

    print(f"\nAuth-positive split:")
    print(f"  TP (auth passed):   {len(tp_idx)}")
    print(f"  FN (auth flagged):  {len(fn_idx)}")
    print(f"  FP (fake missed):   {len(fp_idx)}")
    print(f"  TN (fake caught):   {len(tn_idx)}")

    categories = [
        ("TP", tp_idx, False), ("FN", fn_idx, True),
        ("FP", fp_idx, False), ("TN", tn_idx, True),
    ]
    all_selected = []
    for cat_name, cat_idx, above in categories:
        near, conf = select_near_and_confident(
            cat_idx, y_score, threshold, above=above,
            n_near=args.n_near, n_confident=args.n_confident)
        all_selected.extend([(f"{cat_name}_NEAR", int(i)) for i in near])
        all_selected.extend([(f"{cat_name}_CONF", int(i)) for i in conf])

    print(f"\nVisualizing {len(all_selected)} images...")

    for idx_i, (category, rec_idx) in enumerate(all_selected):
        rec = records[rec_idx]
        uuid, label, img_path = rec["session_uuid"], rec["label"], rec["image_path"]
        print(f"\n[{idx_i+1}/{len(all_selected)}] {category} -- {uuid[:16]}...")

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
                print(f"    {model_name}: L{target_abs} {target['short_name']} ...",
                      end="", flush=True)
                heatmap, score, actual_layer = compute_token_gradcam(
                    model, pixel_values, target["layer_idx"],
                    target["module_path"], patch_start_idx)
                panel_name = f"{model_name} L{actual_layer} {target['short_name']}"
                results.append((panel_name, heatmap, score, thresh))
                print(f" score={score:.4f}")

        out_name = f"{category}_{idx_i:02d}_{uuid[:12]}.png"
        create_visualization(img_path, uuid, label, results, out_dir / out_name)

    print(f"\n{'='*60}")
    print(f"Done! {len(all_selected)} images visualized.")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
