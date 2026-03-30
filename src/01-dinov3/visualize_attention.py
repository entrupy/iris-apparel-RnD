#!/root/.venv/bin/python
"""
Gradient-Weighted Attention Rollout (Chefer et al., 2021) for DINOv3 ViT-L.

Generates heatmap overlays for all 4 auth-positive categories:
  TP = authentic correctly passed
  FP = fake wrongly passed (missed)
  FN = authentic wrongly flagged
  TN = fake correctly caught

Auto-discovers partial finetune checkpoints for the given region.

Usage:
  python visualize_attention.py
  python visualize_attention.py --region front --n-near 3 --n-confident 3
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MODEL_VARIANTS,
    REGIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DINOv3Classifier,
    build_transform,
    load_metadata,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_KEY = "vitl16"
RESOLUTION = 714
MODEL_ID = MODEL_VARIANTS[MODEL_KEY]
PATCH_SIZE = 14
NUM_PATCHES_PER_SIDE = RESOLUTION // PATCH_SIZE


# ---------------------------------------------------------------------------
# Attention hook
# ---------------------------------------------------------------------------
class AttentionHook:
    def __init__(self, model):
        self.attentions = []
        self._hooks = []
        for layer in model.backbone.layer:
            h = layer.attention.register_forward_hook(self._hook_fn)
            self._hooks.append(h)

    def _hook_fn(self, module, input, output):
        attn_weights = output[1]
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
    num_tokens = attentions[0].shape[-1]
    device = attentions[0].device
    result = torch.eye(num_tokens, device=device, dtype=torch.float32)
    for i in range(start_layer, len(attentions)):
        attn = attentions[i].detach().float()
        grad = gradients[i].detach().float()
        cam = (grad * attn).clamp(min=0).mean(dim=1).squeeze(0)
        cam = cam + torch.eye(num_tokens, device=device, dtype=torch.float32)
        cam = cam / cam.sum(dim=-1, keepdim=True)
        result = result @ cam
    return result[0]


# ---------------------------------------------------------------------------
# Model loading + threshold from checkpoint
# ---------------------------------------------------------------------------
def load_model_and_threshold(ckpt_path, device):
    model = DINOv3Classifier(MODEL_ID, freeze_backbone=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.backbone.config._attn_implementation = "eager"
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


# ---------------------------------------------------------------------------
# Discover checkpoints
# ---------------------------------------------------------------------------
def discover_checkpoints(region):
    ckpt_dir = SCRIPT_DIR / "checkpoints" / region
    found = {}
    for p in sorted(ckpt_dir.glob(f"{MODEL_KEY}_{RESOLUTION}_partial_*_best.pt")):
        name = p.stem.replace("_best", "")
        found[name] = p
    return found


# ---------------------------------------------------------------------------
# Attention map computation
# ---------------------------------------------------------------------------
def compute_attention_map(model, pixel_values, hook, num_prefix_tokens, start_layer=0):
    hook.clear()
    model.zero_grad()
    pixel_values = pixel_values.requires_grad_(False)
    logit = model(pixel_values)
    score = torch.sigmoid(logit).item()
    logit.backward(retain_graph=False)

    attentions = list(hook.attentions)
    gradients = [a.grad for a in hook.attentions]
    cls_relevance = gradient_attention_rollout(attentions, gradients, start_layer=start_layer)

    patch_relevance = cls_relevance[num_prefix_tokens:]
    n_patches = NUM_PATCHES_PER_SIDE * NUM_PATCHES_PER_SIDE
    if len(patch_relevance) > n_patches:
        patch_relevance = patch_relevance[:n_patches]
    elif len(patch_relevance) < n_patches:
        patch_relevance = F.pad(patch_relevance, (0, n_patches - len(patch_relevance)))

    heatmap = patch_relevance.reshape(NUM_PATCHES_PER_SIDE, NUM_PATCHES_PER_SIDE)
    heatmap = heatmap.detach().cpu().float().numpy()
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap, score


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
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5.5))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(img_np)
    gt_str = "FAKE" if label == 1 else "AUTHENTIC"
    axes[0].set_title(f"Original\nGT: {gt_str}\n{uuid[:12]}...", fontsize=9)
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
            verdict = "TP (auth passed)" if model_score < threshold else "FN (auth flagged)"
        else:
            verdict = "FP (fake missed)" if model_score < threshold else "TN (fake caught)"

        correct = ("TP" in verdict or "TN" in verdict)
        color = "green" if correct else "red"
        axes[i].set_title(
            f"{model_name}\nscore={model_score:.4f} (thr={threshold:.4f})\n{verdict}",
            fontsize=8, color=color,
        )
        axes[i].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------
def select_near_and_confident(indices, scores, threshold, above, n_near=3, n_confident=3):
    """Pick threshold-near and confident examples.

    above=True: items above threshold (score >= threshold).
    above=False: items below threshold (score < threshold).
    """
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
    parser = argparse.ArgumentParser(description="Gradient-weighted attention rollout visualization")
    parser.add_argument("--region", type=str, default="care_label", choices=REGIONS)
    parser.add_argument("--n-near", type=int, default=3)
    parser.add_argument("--n-confident", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    transform = build_transform(RESOLUTION)
    out_dir = SCRIPT_DIR / "attention_maps" / args.region
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Region: {args.region}  |  Method: Attention Rollout")
    print(f"{'='*60}")

    records = load_metadata(args.region, split="test")
    ckpts = discover_checkpoints(args.region)
    if not ckpts:
        print("No checkpoints found. Exiting.")
        return
    print(f"Found {len(ckpts)} checkpoints: {', '.join(ckpts.keys())}")

    # Load models + val thresholds
    models_info = {}
    for name, path in ckpts.items():
        print(f"\n--- {name} ---")
        model, threshold = load_model_and_threshold(path, device)
        num_register = getattr(model.backbone.config, "num_register_tokens", 0)
        num_prefix = 1 + num_register

        print("  Scoring test set...")
        scores, labels = score_test_set(model, records, transform, device)
        models_info[name] = {
            "model": model, "threshold": threshold,
            "scores": scores, "labels": labels,
            "num_prefix": num_prefix,
        }

    # Use first model for image selection
    first = models_info[next(iter(models_info))]
    y_true, y_score, threshold = first["labels"], first["scores"], first["threshold"]

    auth_idx = np.where(y_true == 0)[0]
    fake_idx = np.where(y_true == 1)[0]

    tp_idx = auth_idx[y_score[auth_idx] < threshold]
    fn_idx = auth_idx[y_score[auth_idx] >= threshold]
    fp_idx = fake_idx[y_score[fake_idx] < threshold]
    tn_idx = fake_idx[y_score[fake_idx] >= threshold]

    print(f"\nAuth-positive split (threshold={threshold:.4f}):")
    print(f"  TP (auth passed):   {len(tp_idx)}")
    print(f"  FN (auth flagged):  {len(fn_idx)}")
    print(f"  FP (fake missed):   {len(fp_idx)}")
    print(f"  TN (fake caught):   {len(tn_idx)}")

    categories = [
        ("TP", tp_idx, False),
        ("FN", fn_idx, True),
        ("FP", fp_idx, False),
        ("TN", tn_idx, True),
    ]

    all_selected = []
    for cat_name, cat_idx, above in categories:
        near, conf = select_near_and_confident(
            cat_idx, y_score, threshold, above=above,
            n_near=args.n_near, n_confident=args.n_confident,
        )
        all_selected.extend([(f"{cat_name}_NEAR", int(i)) for i in near])
        all_selected.extend([(f"{cat_name}_CONF", int(i)) for i in conf])

    print(f"\nVisualizing {len(all_selected)} images...")

    for idx_i, (category, rec_idx) in enumerate(all_selected):
        rec = records[rec_idx]
        uuid = rec["session_uuid"]
        label = rec["label"]
        img_path = rec["image_path"]

        print(f"\n[{idx_i+1}/{len(all_selected)}] {category} -- {uuid[:16]}...")
        img = Image.open(img_path).convert("RGB")
        pixel_values = transform(img).unsqueeze(0).to(device)

        results = []
        for model_name, info in models_info.items():
            model = info["model"]
            num_prefix = info["num_prefix"]
            thresh = info["threshold"]

            hook = AttentionHook(model)
            heatmap, score = compute_attention_map(
                model, pixel_values, hook, num_prefix, start_layer=0)
            results.append((model_name, heatmap, score, thresh))
            hook.remove()

        out_name = f"{category}_{idx_i:02d}_{uuid[:12]}.png"
        create_visualization(img_path, uuid, label, results, out_dir / out_name)

    print(f"\n{'='*60}")
    print(f"Done! {len(all_selected)} images visualized.")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
