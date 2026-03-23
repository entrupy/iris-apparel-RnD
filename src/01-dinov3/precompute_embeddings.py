#!/root/.venv/bin/python
"""
Extract and cache DINOv3 CLS-token features for any image region.

All inference is done in fp32 to avoid the bfloat16 overflow issue
observed in earlier experiments with vitl16.

Usage:
  python precompute_embeddings.py                                    # all regions, all models
  python precompute_embeddings.py --regions care_label front         # specific regions
  python precompute_embeddings.py --models vits16 vitb16 --resolutions 518
  python precompute_embeddings.py --batch-size 8                     # reduce for OOM
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from config import (
    MODEL_VARIANTS,
    REGIONS,
    RESOLUTIONS,
    ImageDatasetWithUUID,
    build_transform,
    cache_dir,
    load_metadata,
)


@torch.inference_mode()
def extract_features(model, dataloader, device):
    all_features, all_labels, all_uuids = [], [], []

    for batch_idx, (pixel_values, labels, uuids) in enumerate(dataloader):
        pixel_values = pixel_values.to(device, dtype=torch.float32)
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


def cache_one(model_key, model_id, resolution, records, region, device, batch_size, num_workers):
    cdir = cache_dir(region)
    cache_name = f"{model_key}_{resolution}"
    feat_path = cdir / f"{cache_name}_features.pt"
    label_path = cdir / f"{cache_name}_labels.pt"
    uuid_path = cdir / f"{cache_name}_uuids.pt"

    if feat_path.exists() and label_path.exists() and uuid_path.exists():
        print(f"  [SKIP] {cache_name} already cached")
        return

    print(f"  Loading model {model_id} (fp32) ...")
    load_kwargs = {"torch_dtype": torch.float32}
    if not model_key.startswith("convnext"):
        load_kwargs["attn_implementation"] = "sdpa"
    model = AutoModel.from_pretrained(model_id, **load_kwargs).to(device).eval()

    if hasattr(model.config, "hidden_size"):
        embed_dim = model.config.hidden_size
    else:
        embed_dim = model.config.hidden_sizes[-1]
    print(f"  Embed dim: {embed_dim}")

    transform = build_transform(resolution)
    dataset = ImageDatasetWithUUID(records, transform)
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

    cdir.mkdir(parents=True, exist_ok=True)
    torch.save(features, feat_path)
    torch.save(labels, label_path)
    torch.save(uuids, uuid_path)
    print(f"  Saved to {cdir}/{cache_name}_*.pt")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Cache DINOv3 features for image regions")
    parser.add_argument("--regions", nargs="+", default=REGIONS, choices=REGIONS)
    parser.add_argument("--models", nargs="+", default=list(MODEL_VARIANTS.keys()),
                        choices=list(MODEL_VARIANTS.keys()))
    parser.add_argument("--resolutions", nargs="+", type=int, default=RESOLUTIONS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    for region in args.regions:
        print(f"\n{'=' * 60}")
        print(f"REGION: {region}")
        print(f"{'=' * 60}")

        records = load_metadata(region, split="train")
        combos = [(m, r) for m in args.models for r in args.resolutions]
        print(f"  Will cache {len(combos)} variant(s)\n")

        for model_key, resolution in combos:
            model_id = MODEL_VARIANTS[model_key]
            print(f"  [{model_key} @ {resolution}]")
            cache_one(model_key, model_id, resolution, records, region,
                      args.device, args.batch_size, args.num_workers)
            print()

    print("All done.")


if __name__ == "__main__":
    main()
