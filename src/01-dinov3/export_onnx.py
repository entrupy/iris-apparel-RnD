#!/root/.venv/bin/python
"""
Export the full region-fusion pipeline to ONNX:
  1. DINOv3 ViT-L backbone -> onnx_models/dinov3_vitl16_714.onnx
  2. StandardScaler + SVM  -> onnx_models/fusion_svm.onnx

Usage:
  python export_onnx.py
  python export_onnx.py --embed-key vitl16_714 --skip-backbone
  python export_onnx.py --skip-svm
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from config import (
    MODEL_VARIANTS,
    REGIONS,
    SEED,
    _global_val_split_path,
    cache_dir,
    create_global_val_split,
)

SCRIPT_DIR = Path(__file__).resolve().parent
EMBED_DIM = 1024
N_REGIONS = len(REGIONS)
CONCAT_DIM = EMBED_DIM * N_REGIONS + N_REGIONS  # 4100


class BackboneWrapper(nn.Module):
    """Wraps HuggingFace backbone to output only pooler_output tensor."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        return outputs.pooler_output


def load_features_by_uuid(region, split, feat_key):
    cdir = cache_dir(region)
    prefix = f"test_{feat_key}" if split == "test" else feat_key
    feat_path = cdir / f"{prefix}_features.pt"
    label_path = cdir / f"{prefix}_labels.pt"
    uuid_path = cdir / f"{prefix}_uuids.pt"
    if not feat_path.exists():
        return {}
    features = torch.load(feat_path, weights_only=True)
    labels = torch.load(label_path, weights_only=True)
    uuids = torch.load(uuid_path, weights_only=False) if uuid_path.exists() else []
    return {u: (features[i].numpy(), int(labels[i].item())) for i, u in enumerate(uuids)}


def build_concat_dataset(split, feat_key):
    per_region = {r: load_features_by_uuid(r, split, feat_key) for r in REGIONS}
    all_uuids = sorted(set().union(*(d.keys() for d in per_region.values())))

    uuid_labels = {}
    for uuid in all_uuids:
        for r in REGIONS:
            if uuid in per_region[r]:
                uuid_labels[uuid] = per_region[r][uuid][1]
                break

    X = np.zeros((len(all_uuids), CONCAT_DIM), dtype=np.float32)
    y = np.zeros(len(all_uuids), dtype=np.int64)

    for i, uuid in enumerate(all_uuids):
        y[i] = uuid_labels[uuid]
        for j, region in enumerate(REGIONS):
            if uuid in per_region[region]:
                feat, _ = per_region[region][uuid]
                X[i, j * EMBED_DIM:(j + 1) * EMBED_DIM] = feat
                X[i, EMBED_DIM * N_REGIONS + j] = 1.0

    return X, y, all_uuids


def split_by_global_val(X, y, uuids):
    split_path = _global_val_split_path()
    if not split_path.exists():
        create_global_val_split()
    with open(split_path) as f:
        data = json.load(f)
    val_set = set(data["val_uuids"])
    train_mask = np.array([u not in val_set for u in uuids])
    val_mask = ~train_mask
    return X[train_mask], y[train_mask], X[val_mask], y[val_mask]


# ---------------------------------------------------------------------------
# Stage 1: Backbone export
# ---------------------------------------------------------------------------

def export_backbone(model_key, resolution, out_dir, device="cuda"):
    model_id = MODEL_VARIANTS[model_key]
    out_path = out_dir / f"dinov3_{model_key}_{resolution}.onnx"

    for attn_impl in ("sdpa", "eager"):
        print(f"\n  Trying attn_implementation='{attn_impl}'...")
        try:
            from transformers import AutoModel
            backbone = AutoModel.from_pretrained(
                model_id,
                dtype=torch.float32,
                attn_implementation=attn_impl,
            )
            wrapper = BackboneWrapper(backbone).to(device).eval()

            dummy = torch.randn(1, 3, resolution, resolution, device=device)

            print(f"  Exporting to {out_path} ...")
            t0 = time.time()
            torch.onnx.export(
                wrapper,
                (dummy,),
                str(out_path),
                opset_version=17,
                input_names=["pixel_values"],
                output_names=["pooler_output"],
                dynamic_axes={
                    "pixel_values": {0: "batch"},
                    "pooler_output": {0: "batch"},
                },
                dynamo=False,
            )
            elapsed = time.time() - t0
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"  Backbone ONNX exported in {elapsed:.1f}s ({size_mb:.0f} MB)")

            onnx_model = onnx.load(str(out_path))
            onnx.checker.check_model(onnx_model, full_check=True)
            print("  ONNX model passed validation check")

            ref_output = wrapper(dummy).detach().cpu().numpy()
            del wrapper, backbone
            torch.cuda.empty_cache()

            return out_path, ref_output, dummy.cpu().numpy()

        except Exception as e:
            print(f"  Failed with '{attn_impl}': {e}")
            if attn_impl == "eager":
                raise
            print("  Falling back...")
            continue

    raise RuntimeError("Both sdpa and eager export failed")


# ---------------------------------------------------------------------------
# Stage 2: SVM export
# ---------------------------------------------------------------------------

def train_and_export_svm(embed_key, out_dir):
    out_path = out_dir / "fusion_svm.onnx"

    print("\n  Building fusion dataset...")
    X_all, y_all, uuids_all = build_concat_dataset("train", embed_key)
    X_train, y_train, X_val, y_val = split_by_global_val(X_all, y_all, uuids_all)
    print(f"  Train: {len(X_train)} ({int(y_train.sum())} pos) | Val: {len(X_val)} ({int(y_val.sum())} pos)")

    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    print("  Training StandardScaler + SVM pipeline...")
    t0 = time.time()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                     class_weight={0: 1.0, 1: spw}, random_state=SEED)),
    ])
    pipeline.fit(X_train, y_train)
    print(f"  SVM trained in {time.time() - t0:.1f}s")

    ref_scores = pipeline.predict_proba(X_val[:5])

    print(f"  Converting to ONNX...")
    t0 = time.time()
    onnx_model = convert_sklearn(
        pipeline,
        "fusion_svm",
        initial_types=[("X", FloatTensorType([None, CONCAT_DIM]))],
        target_opset=17,
    )
    onnx.save_model(onnx_model, str(out_path))
    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  SVM ONNX exported in {elapsed:.1f}s ({size_mb:.1f} MB)")

    onnx_loaded = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_loaded)
    print("  ONNX model passed validation check")

    return out_path, ref_scores, X_val[:5]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_backbone(onnx_path, ref_output, ref_input):
    print("\n  Validating backbone ONNX with ONNX Runtime...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    active = sess.get_providers()
    print(f"  Active providers: {active}")

    ort_output = sess.run(None, {"pixel_values": ref_input})[0]
    max_diff = np.abs(ref_output - ort_output).max()
    mean_diff = np.abs(ref_output - ort_output).mean()
    print(f"  Max abs diff: {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")

    if max_diff < 1e-3:
        print("  PASS: backbone outputs match within tolerance")
    else:
        print(f"  WARNING: max diff {max_diff:.6f} exceeds 1e-3 (may be acceptable for fp32 ViT)")

    return max_diff


def validate_svm(onnx_path, ref_scores, ref_input):
    print("\n  Validating SVM ONNX with ONNX Runtime...")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    ort_output = sess.run(None, {"X": ref_input.astype(np.float32)})
    ort_probs = ort_output[1]

    if isinstance(ort_probs, list):
        ort_probs_arr = np.array([[d[0], d[1]] for d in ort_probs])
    else:
        ort_probs_arr = ort_probs

    max_diff = np.abs(ref_scores - ort_probs_arr).max()
    mean_diff = np.abs(ref_scores - ort_probs_arr).mean()
    print(f"  Max abs diff: {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")

    if max_diff < 1e-4:
        print("  PASS: SVM outputs match within tolerance")
    else:
        print(f"  WARNING: max diff {max_diff:.6f} exceeds 1e-4")

    return max_diff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export region-fusion pipeline to ONNX")
    parser.add_argument("--model", type=str, default="vitl16")
    parser.add_argument("--resolution", type=int, default=714)
    parser.add_argument("--embed-key", type=str, default=None,
                        help="Feature key for SVM (default: {model}_{resolution})")
    parser.add_argument("--skip-backbone", action="store_true")
    parser.add_argument("--skip-svm", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    embed_key = args.embed_key or f"{args.model}_{args.resolution}"

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = SCRIPT_DIR / "onnx_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"ONNX EXPORT | model={args.model} res={args.resolution} embed={embed_key}")
    print("=" * 70)

    if not args.skip_backbone:
        print("\n--- Stage 1: Backbone ---")
        bb_path, bb_ref, bb_input = export_backbone(
            args.model, args.resolution, out_dir, args.device)
        validate_backbone(bb_path, bb_ref, bb_input)

    if not args.skip_svm:
        print("\n--- Stage 2: Fusion SVM ---")
        svm_path, svm_ref, svm_input = train_and_export_svm(embed_key, out_dir)
        validate_svm(svm_path, svm_ref, svm_input)

    print("\n" + "=" * 70)
    print("ONNX export complete. Files:")
    for f in sorted(out_dir.glob("*.onnx")):
        print(f"  {f}  ({f.stat().st_size / (1024*1024):.1f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
