"""
Extract and cache DINOv3 CLS token features for care-label images.

Iterates over model variants x resolutions, saves .pt files to cached_features/.
Each cache file contains:
  - features: Tensor[N, embed_dim]
  - labels: Tensor[N]  (0 = authentic, 1 = not-authentic)
  - session_uuids: list[str]

Usage:
  python cache_features.py                          # all variants x all resolutions
  python cache_features.py --models vits16 vitb16   # specific models
  python cache_features.py --resolutions 518        # specific resolution
  python cache_features.py --batch-size 8           # reduce for OOM
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent.parent / "resources" / "apparel_supreme_until_dec_2025_care_label"
METADATA_CSV = DATA_ROOT / "train" / "metadata.csv"
IMAGE_DIR = DATA_ROOT / "train" / "camera" / "care_label" / "0"
CACHE_DIR = SCRIPT_DIR / "cached_features"

MODEL_VARIANTS = {
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "convnext_tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "convnext_small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "convnext_base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "convnext_large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
}

RESOLUTIONS = [518, 714]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_metadata():
    df = pd.read_csv(METADATA_CSV)
    df = df[df["internal_merged_result_id"].isin([1, 3])].copy()
    df["label"] = (df["internal_merged_result_id"] == 3).astype(int)

    records = []
    for _, row in df.iterrows():
        uuid = row["session_uuid"]
        pattern = f"{uuid}.macro.care_label.*.jpg"
        matches = sorted(IMAGE_DIR.glob(pattern))
        if matches:
            records.append({
                "session_uuid": uuid,
                "image_path": str(matches[0]),
                "label": row["label"],
            })

    print(f"Metadata: {len(df)} rows in CSV, {len(records)} with images")
    labels = [r["label"] for r in records]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"  Authentic (0): {n_neg}  |  Not-authentic (1): {n_pos}  |  Ratio: {n_neg / max(n_pos, 1):.1f}:1")
    return records


class CareImageDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        pixel_values = self.transform(img)
        return pixel_values, rec["label"], rec["session_uuid"]


def build_transform(resolution):
    return transforms.Compose([
        transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.inference_mode()
def extract_features(model, dataloader, device):
    all_features = []
    all_labels = []
    all_uuids = []

    for batch_idx, (pixel_values, labels, uuids) in enumerate(dataloader):
        pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
        outputs = model(pixel_values=pixel_values)
        cls_features = outputs.pooler_output.float().cpu()
        cls_features = torch.nan_to_num(cls_features, nan=0.0, posinf=0.0, neginf=0.0)

        all_features.append(cls_features)
        all_labels.extend(labels)
        all_uuids.extend(uuids)

        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            print(f"    Batch {batch_idx + 1}/{len(dataloader)}", flush=True)

    features = torch.cat(all_features, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)
    return features, labels, all_uuids


def cache_one(model_key, model_id, resolution, records, device, batch_size, num_workers):
    cache_name = f"{model_key}_{resolution}"
    feat_path = CACHE_DIR / f"{cache_name}_features.pt"
    label_path = CACHE_DIR / f"{cache_name}_labels.pt"
    uuid_path = CACHE_DIR / f"{cache_name}_uuids.pt"

    if feat_path.exists() and label_path.exists() and uuid_path.exists():
        print(f"  [SKIP] {cache_name} already cached")
        return

    print(f"  Loading model {model_id} ...")
    load_kwargs = {"dtype": torch.bfloat16}
    if not model_key.startswith("convnext"):
        load_kwargs["attn_implementation"] = "sdpa"
    model = AutoModel.from_pretrained(model_id, **load_kwargs).to(device).eval()

    if hasattr(model.config, "hidden_size"):
        embed_dim = model.config.hidden_size
    else:
        embed_dim = model.config.hidden_sizes[-1]
    patch_size = getattr(model.config, "patch_size", None)
    print(f"  Embed dim: {embed_dim}, Patch size: {patch_size}")

    transform = build_transform(resolution)
    dataset = CareImageDataset(records, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    print(f"  Extracting features at {resolution}x{resolution} ...")
    t0 = time.time()
    features, labels, uuids = extract_features(model, dataloader, device)
    elapsed = time.time() - t0
    print(f"  Done: {features.shape} in {elapsed:.1f}s")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(features, feat_path)
    torch.save(labels, label_path)
    torch.save(uuids, uuid_path)
    print(f"  Saved to {CACHE_DIR}/{cache_name}_*.pt")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Cache DINOv3 features for care-label images")
    parser.add_argument("--models", nargs="+", default=list(MODEL_VARIANTS.keys()),
                        choices=list(MODEL_VARIANTS.keys()), help="Model variants to cache")
    parser.add_argument("--resolutions", nargs="+", type=int, default=RESOLUTIONS,
                        help="Image resolutions to cache")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    records = load_metadata()

    combos = [(m, r) for m in args.models for r in args.resolutions]
    print(f"\nWill cache {len(combos)} variant(s): {combos}\n")

    for model_key, resolution in combos:
        model_id = MODEL_VARIANTS[model_key]
        print(f"[{model_key} @ {resolution}]")
        cache_one(model_key, model_id, resolution, records, args.device, args.batch_size, args.num_workers)
        print()

    print("All done.")


if __name__ == "__main__":
    main()
