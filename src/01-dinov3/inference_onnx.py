#!/root/.venv/bin/python
"""
End-to-end ONNX inference: raw images -> fake probability + decision.

Loads the exported DINOv3 backbone + fusion SVM ONNX models and runs
inference on a set of region crops for one session.

Usage:
  # Score a single session (provide one image per region)
  python inference_onnx.py \
      --care-label /path/to/care_label.jpg \
      --front /path/to/front.jpg \
      --front-exterior-logo /path/to/logo.jpg \
      --brand-tag /path/to/brand_tag.jpg

  # Score with missing regions (missing = zero embedding + mask=0)
  python inference_onnx.py \
      --front /path/to/front.jpg \
      --brand-tag /path/to/brand_tag.jpg

  # Force CPU
  python inference_onnx.py --device cpu --front /path/to/front.jpg

  # Batch inference from a CSV
  python inference_onnx.py --csv sessions.csv
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_DIR = SCRIPT_DIR / "onnx_models"
THRESHOLDS_PATH = ONNX_DIR / "thresholds.json"

REGIONS = ["care_label", "front", "front_exterior_logo", "brand_tag"]
EMBED_DIM = 1024
N_REGIONS = len(REGIONS)
CONCAT_DIM = EMBED_DIM * N_REGIONS + N_REGIONS

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_RESOLUTION = 714
DEFAULT_THRESHOLD_FPR = "2%"


def preprocess_image(image_path, resolution=DEFAULT_RESOLUTION):
    """Load, resize, normalize an image to (1, 3, H, W) float32 numpy array."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((resolution, resolution), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr[np.newaxis]  # add batch dim


def load_thresholds():
    """Load val-calibrated thresholds from JSON."""
    if THRESHOLDS_PATH.exists():
        with open(THRESHOLDS_PATH) as f:
            return json.load(f)
    return {}


class ONNXFusionPipeline:
    """End-to-end ONNX inference pipeline for region fusion."""

    def __init__(self, device="auto", resolution=DEFAULT_RESOLUTION):
        self.resolution = resolution
        self.thresholds = load_thresholds()

        if device == "auto":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        bb_path = ONNX_DIR / "dinov3_vitl16_714.onnx"
        svm_path = ONNX_DIR / "fusion_svm.onnx"

        if not bb_path.exists() or not svm_path.exists():
            raise FileNotFoundError(
                f"ONNX models not found in {ONNX_DIR}. Run export_onnx.py first.")

        self.bb_sess = ort.InferenceSession(str(bb_path), providers=providers)
        self.svm_sess = ort.InferenceSession(
            str(svm_path), providers=["CPUExecutionProvider"])

        self.active_providers = self.bb_sess.get_providers()

    def extract_embedding(self, image_path):
        """Extract 1024-d CLS embedding from a single image."""
        pv = preprocess_image(image_path, self.resolution)
        out = self.bb_sess.run(None, {"pixel_values": pv})
        return out[0][0]  # (1024,)

    def extract_embeddings_batch(self, image_paths):
        """Extract embeddings for multiple images in one forward pass."""
        batch = np.concatenate(
            [preprocess_image(p, self.resolution) for p in image_paths], axis=0)
        out = self.bb_sess.run(None, {"pixel_values": batch})
        return out[0]  # (N, 1024)

    def predict(self, region_images):
        """
        Run full pipeline for one session.

        Args:
            region_images: dict mapping region name -> image path (or None).
                           e.g. {"care_label": "img1.jpg", "front": "img2.jpg"}

        Returns:
            dict with score, decision at each FPR threshold, and per-region info.
        """
        t0 = time.time()

        concat = np.zeros(CONCAT_DIM, dtype=np.float32)
        region_info = {}

        available = [(r, region_images[r]) for r in REGIONS
                     if r in region_images and region_images[r] is not None]

        if available:
            paths = [p for _, p in available]
            embeddings = self.extract_embeddings_batch(paths)

            for idx, (region, path) in enumerate(available):
                j = REGIONS.index(region)
                concat[j * EMBED_DIM:(j + 1) * EMBED_DIM] = embeddings[idx]
                concat[EMBED_DIM * N_REGIONS + j] = 1.0
                region_info[region] = {"available": True, "path": str(path)}

        for r in REGIONS:
            if r not in region_info:
                region_info[r] = {"available": False}

        embed_time = time.time() - t0

        t1 = time.time()
        X = concat[np.newaxis].astype(np.float32)
        svm_out = self.svm_sess.run(None, {"X": X})

        if isinstance(svm_out[1], list):
            probs = {int(k): float(v) for k, v in svm_out[1][0].items()}
        elif isinstance(svm_out[1], np.ndarray):
            probs = {0: float(svm_out[1][0][0]), 1: float(svm_out[1][0][1])}
        else:
            probs = {0: 0.0, 1: float(svm_out[1])}

        score = probs.get(1, 0.0)
        svm_time = time.time() - t1

        decisions = {}
        for fpr_name, thresh in self.thresholds.items():
            decisions[fpr_name] = {
                "threshold": thresh,
                "is_fake": score >= thresh,
                "confidence": abs(score - thresh),
            }

        return {
            "score": score,
            "decisions": decisions,
            "regions": region_info,
            "n_regions_available": sum(1 for r in region_info.values() if r["available"]),
            "latency_ms": {
                "backbone": round(embed_time * 1000, 1),
                "svm": round(svm_time * 1000, 1),
                "total": round((embed_time + svm_time) * 1000, 1),
            },
            "providers": self.active_providers,
        }


def main():
    parser = argparse.ArgumentParser(
        description="ONNX inference: region images -> authenticity score")
    parser.add_argument("--care-label", type=str, default=None)
    parser.add_argument("--front", type=str, default=None)
    parser.add_argument("--front-exterior-logo", type=str, default=None)
    parser.add_argument("--brand-tag", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV with columns: session_id, care_label, front, "
                             "front_exterior_logo, brand_tag (image paths)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--threshold-fpr", type=str, default=DEFAULT_THRESHOLD_FPR,
                        help="FPR target for binary decision (default: 2%%)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    pipeline = ONNXFusionPipeline(device=args.device, resolution=args.resolution)
    print(f"Loaded ONNX pipeline (providers: {pipeline.active_providers})")

    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        results = []
        for _, row in df.iterrows():
            region_images = {}
            for r in REGIONS:
                col = r.replace("_", "-") if r.replace("_", "-") in df.columns else r
                if col in df.columns and pd.notna(row.get(col)):
                    region_images[r] = row[col]
            result = pipeline.predict(region_images)
            result["session_id"] = row.get("session_id", row.name)
            results.append(result)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            fpr_key = args.threshold_fpr
            print(f"\n{'Session':<40} {'Score':>8} {'Decision':>10} "
                  f"{'Regions':>8} {'Latency':>10}")
            print("-" * 80)
            for r in results:
                dec = r["decisions"].get(fpr_key, {})
                label = "FAKE" if dec.get("is_fake", False) else "AUTH"
                print(f"{str(r['session_id']):<40} {r['score']:>8.4f} "
                      f"{label:>10} {r['n_regions_available']:>8} "
                      f"{r['latency_ms']['total']:>8.0f}ms")
    else:
        region_images = {}
        if args.care_label:
            region_images["care_label"] = args.care_label
        if args.front:
            region_images["front"] = args.front
        if args.front_exterior_logo:
            region_images["front_exterior_logo"] = args.front_exterior_logo
        if args.brand_tag:
            region_images["brand_tag"] = args.brand_tag

        if not region_images:
            parser.error("Provide at least one region image or --csv")

        result = pipeline.predict(region_images)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\nScore: {result['score']:.6f}")
            print(f"Regions available: {result['n_regions_available']}/4")
            for r, info in result["regions"].items():
                status = "available" if info["available"] else "missing"
                print(f"  {r}: {status}")

            print(f"\nDecisions (score >= threshold = FAKE):")
            for fpr_name, dec in result["decisions"].items():
                label = "FAKE" if dec["is_fake"] else "AUTH"
                print(f"  @{fpr_name} FPR: {label} "
                      f"(thresh={dec['threshold']:.6f}, score={result['score']:.6f})")

            lat = result["latency_ms"]
            print(f"\nLatency: {lat['total']:.0f}ms "
                  f"(backbone={lat['backbone']:.0f}ms, svm={lat['svm']:.0f}ms)")
            print(f"Providers: {result['providers']}")


if __name__ == "__main__":
    main()
