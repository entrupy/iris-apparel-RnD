#!/root/.venv/bin/python
"""
ViT-ReciproCAM (Byun & Lee, 2023) for DINOv3 ViT-L.

Generates heatmap overlays for all 4 auth-positive categories:
  TP = authentic correctly passed
  FP = fake wrongly passed (missed)
  FN = authentic wrongly flagged
  TN = fake correctly caught

Auto-discovers partial finetune checkpoints for the given region.

Usage:
  python visualize_reciprocam.py
  python visualize_reciprocam.py --region front --target-layer -2
  python visualize_reciprocam.py --cam-chunk-size 64
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


# ---------------------------------------------------------------------------
# ReciproCAM masking hook
# ---------------------------------------------------------------------------
GAUSSIAN_KERNEL = torch.tensor(
    [[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]],
    dtype=torch.float32,
)


class ReciproCAMHook:
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
        masked = torch.zeros(num_masks, total_tokens, dim,
                             device=feature.device, dtype=feature.dtype)
        if num_masks == 0:
            return masked
        if self.preserve_prefix_tokens > 0:
            masked[:, :self.preserve_prefix_tokens, :] = feature[0, :self.preserve_prefix_tokens, :]

        if not self.use_gaussian:
            for mask_idx, spatial_idx in enumerate(token_indices):
                token_pos = self.patch_start_idx + spatial_idx
                masked[mask_idx, token_pos, :] = feature[0, token_pos, :]
            return masked

        spatial = feature[0, self.patch_start_idx:, :].reshape(side, side, dim)
        masked_spatial = masked[:, self.patch_start_idx:, :].reshape(num_masks, side, side, dim)
        kernel = GAUSSIAN_KERNEL.to(device=feature.device, dtype=feature.dtype).unsqueeze(-1)

        for mask_idx, spatial_idx in enumerate(token_indices):
            y, x = divmod(spatial_idx, side)
            y0, y1 = max(y - 1, 0), min(y + 1, side - 1)
            x0, x1 = max(x - 1, 0), min(x + 1, side - 1)
            patch = spatial[y0:y1+1, x0:x1+1, :]
            ky0, kx0 = 1 - (y - y0), 1 - (x - x0)
            kernel_patch = kernel[ky0:ky0+patch.shape[0], kx0:kx0+patch.shape[1], :]
            masked_spatial[mask_idx, y0:y1+1, x0:x1+1, :] = patch * kernel_patch
        return masked

    def clear(self):
        self.feature = None
        self.token_indices = None

    def remove(self):
        self._hook.remove()


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
# ReciproCAM computation
# ---------------------------------------------------------------------------
def resolve_target_layer_index(n_layers, target_layer_idx):
    idx = target_layer_idx if target_layer_idx >= 0 else n_layers + target_layer_idx
    if idx == n_layers - 1:
        idx = n_layers - 2
    if idx < 0 or idx >= n_layers:
        raise ValueError(f"Invalid target layer {target_layer_idx} for {n_layers} layers")
    return idx


def infer_spatial_grid(model, pixel_values, patch_start_idx):
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
    chunk_size = max(1, int(cam_chunk_size))
    while True:
        try:
            return _compute_reciprocam_map_chunked(
                model, pixel_values, target_layer_idx, patch_start_idx,
                preserve_prefix_tokens=preserve_prefix_tokens,
                use_gaussian=use_gaussian, cam_chunk_size=chunk_size)
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
                                    use_gaussian=True, cam_chunk_size=64):
    backbone = model.backbone
    n_layers = len(backbone.layer)
    target_layer_idx = resolve_target_layer_index(n_layers, target_layer_idx)
    target_norm = backbone.layer[target_layer_idx].norm1
    n_patches, patches_per_side = infer_spatial_grid(model, pixel_values, patch_start_idx)

    hook = ReciproCAMHook(target_norm, patch_start_idx=patch_start_idx,
                          preserve_prefix_tokens=preserve_prefix_tokens,
                          use_gaussian=use_gaussian)
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
                    raise RuntimeError(f"Expected {end-start} scores, got {chunk_scores.numel()}")
                all_scores.append(chunk_scores.detach().cpu().float())
                hook.clear()
    finally:
        hook.remove()

    scores = torch.cat(all_scores).numpy()
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
            f"{model_name}\nscore={model_score:.4f} (thr={threshold:.4f})\n{verdict}",
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
    parser = argparse.ArgumentParser(description="ViT-ReciproCAM visualization")
    parser.add_argument("--region", type=str, default="care_label", choices=REGIONS)
    parser.add_argument("--n-near", type=int, default=3)
    parser.add_argument("--n-confident", type=int, default=3)
    parser.add_argument("--target-layer", type=int, default=-2)
    parser.add_argument("--single-token", action="store_true")
    parser.add_argument("--keep-register-tokens", action="store_true")
    parser.add_argument("--cam-chunk-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    transform = build_transform(RESOLUTION)
    out_dir = SCRIPT_DIR / "reciprocam_maps" / args.region
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Region: {args.region}  |  Method: ViT-ReciproCAM")
    print(f"{'='*60}")

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
        n_layers = len(list(model.backbone.layer))

        print("  Scoring test set...")
        scores, labels = score_test_set(model, records, transform, device)
        models_info[name] = {
            "model": model, "threshold": threshold,
            "scores": scores, "labels": labels,
            "patch_start_idx": patch_start_idx,
            "num_register": num_register, "n_layers": n_layers,
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
            preserve = patch_start_idx if args.keep_register_tokens else 1

            tgt_abs = resolve_target_layer_index(n_layers, args.target_layer)
            print(f"    {model_name}: layer {tgt_abs}/{n_layers} ...", end="", flush=True)
            heatmap, score = compute_reciprocam_map(
                model, pixel_values, tgt_abs, patch_start_idx,
                preserve_prefix_tokens=preserve,
                use_gaussian=not args.single_token,
                cam_chunk_size=args.cam_chunk_size)
            results.append((f"{model_name} L{tgt_abs}", heatmap, score, thresh))
            print(f" score={score:.4f}")

        out_name = f"{category}_{idx_i:02d}_{uuid[:12]}.png"
        create_visualization(img_path, uuid, label, results, out_dir / out_name)

    print(f"\n{'='*60}")
    print(f"Done! {len(all_selected)} images visualized.")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
