"""
Shared configuration, data loading, metrics, and model definitions for
multi-region DINOv3 authentication experiments.

All scripts import from here to avoid duplication.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"

SEED = 42

REGIONS = ["care_label", "front", "front_exterior_logo", "brand_tag"]

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

TARGET_FPRS = [0.005, 0.01, 0.02, 0.05, 0.10]
TARGET_FPR_NAMES = ["0.5%", "1%", "2%", "5%", "10%"]


# ---------------------------------------------------------------------------
# Region-aware paths
# ---------------------------------------------------------------------------

def _train_data_root(region: str) -> Path:
    return RESOURCES_DIR / f"apparel_supreme_until_dec_2025_{region}"


def _test_data_root(region: str) -> Path:
    return RESOURCES_DIR / f"apparel_supreme_jan_to_feb_2026_{region}"


def train_image_dir(region: str) -> Path:
    return _train_data_root(region) / "train" / "camera" / region / "0"


def test_image_dir(region: str) -> Path:
    return _test_data_root(region) / "test" / "camera" / region / "0"


def train_metadata_csv(region: str) -> Path:
    return _train_data_root(region) / "train" / "metadata.csv"


def test_metadata_csv(region: str) -> Path:
    return _test_data_root(region) / "test" / "metadata.csv"


def cache_dir(region: str) -> Path:
    return SCRIPT_DIR / "cached_features" / region


def ckpt_dir(region: str) -> Path:
    return SCRIPT_DIR / "checkpoints" / region


def results_dir(region: str) -> Path:
    return SCRIPT_DIR / "ml_results" / region


def runs_dir(region: str) -> Path:
    return SCRIPT_DIR / "runs" / region


def val_split_path(region: str) -> Path:
    return SCRIPT_DIR / "val_splits" / f"{region}_split.json"


def _global_val_split_path() -> Path:
    return SCRIPT_DIR / "val_splits" / "global_session_split.json"


# ---------------------------------------------------------------------------
# Metadata & val-split
# ---------------------------------------------------------------------------

def _find_image(img_dir: Path, uuid: str, region: str):
    """Return the first matching image path for a session UUID, or None."""
    pattern = f"{uuid}.macro.{region}.*.jpg"
    matches = sorted(img_dir.glob(pattern))
    return str(matches[0]) if matches else None


def load_metadata(region: str, split: str = "train"):
    """Load labeled records that have matching images.

    Returns list[dict] with keys: session_uuid, image_path, label.
    """
    if split == "train":
        meta_csv = train_metadata_csv(region)
        img_dir = train_image_dir(region)
    else:
        meta_csv = test_metadata_csv(region)
        img_dir = test_image_dir(region)

    df = pd.read_csv(meta_csv)
    df = df[df["internal_merged_result_id"].isin([1, 3])].copy()
    df["label"] = (df["internal_merged_result_id"] == 3).astype(int)

    records = []
    for _, row in df.iterrows():
        path = _find_image(img_dir, row["session_uuid"], region)
        if path:
            records.append({
                "session_uuid": row["session_uuid"],
                "image_path": path,
                "label": int(row["label"]),
            })

    n_pos = sum(r["label"] for r in records)
    n_neg = len(records) - n_pos
    print(f"[{region}/{split}] {len(df)} labeled rows, {len(records)} with images")
    print(f"  Authentic (0): {n_neg}  |  Not-authentic (1): {n_pos}  |  Ratio: {n_neg / max(n_pos, 1):.1f}:1")
    return records


def create_global_val_split(force=False):
    """Build a single session-level train/val split across all regions.

    Stratifies by (label x region-availability-pattern) so rare region
    combinations are represented proportionally in both splits.  Falls back
    to label-only stratification if any compound stratum has < 2 samples.

    The split is cached to disk; pass force=True to regenerate.
    """
    from collections import Counter

    split_path = _global_val_split_path()
    if split_path.exists() and not force:
        with open(split_path) as f:
            return json.load(f)

    all_sessions = {}
    region_records = {}

    for region in REGIONS:
        records = load_metadata(region, split="train")
        region_records[region] = records
        for r in records:
            uuid = r["session_uuid"]
            if uuid not in all_sessions:
                all_sessions[uuid] = {"label": r["label"], "regions": []}
            if region not in all_sessions[uuid]["regions"]:
                all_sessions[uuid]["regions"].append(region)

    uuids = list(all_sessions.keys())
    labels = np.array([all_sessions[u]["label"] for u in uuids])

    strat_keys = []
    for u in uuids:
        pattern = "+".join(sorted(all_sessions[u]["regions"]))
        strat_keys.append(f"{all_sessions[u]['label']}_{pattern}")

    pattern_counts = Counter(strat_keys)
    use_compound = min(pattern_counts.values()) >= 2
    strat_target = np.array(strat_keys) if use_compound else labels
    strat_desc = ("label x region-availability" if use_compound
                  else "label-only (some compound strata too small)")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(splitter.split(np.zeros(len(uuids)), strat_target))

    train_uuids = [uuids[i] for i in train_idx]
    val_uuids = [uuids[i] for i in val_idx]
    val_set = set(val_uuids)

    per_region = {}
    for region in REGIONS:
        t_pos = t_neg = v_pos = v_neg = 0
        for r in region_records[region]:
            is_val = r["session_uuid"] in val_set
            is_pos = r["label"] == 1
            if is_val:
                if is_pos:
                    v_pos += 1
                else:
                    v_neg += 1
            else:
                if is_pos:
                    t_pos += 1
                else:
                    t_neg += 1
        per_region[region] = {
            "total": t_pos + t_neg + v_pos + v_neg,
            "train": t_pos + t_neg, "train_pos": t_pos, "train_neg": t_neg,
            "val": v_pos + v_neg, "val_pos": v_pos, "val_neg": v_neg,
        }

    data = {
        "seed": SEED,
        "stratification": strat_desc,
        "n_total_sessions": len(uuids),
        "n_train_sessions": len(train_uuids),
        "n_val_sessions": len(val_uuids),
        "per_region": per_region,
        "train_uuids": train_uuids,
        "val_uuids": val_uuids,
    }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n  Global val split created ({strat_desc})")
    print(f"  Sessions: {len(uuids)} total | {len(train_uuids)} train | {len(val_uuids)} val")
    print(f"  Saved to {split_path}")
    return data


def get_or_create_val_split(region: str, records=None):
    """Return (train_idx, val_idx) arrays using the global session-level split.

    The global split assigns each session UUID to train or val consistently
    across all regions, stratified by label and region-availability pattern.
    """
    split_path = _global_val_split_path()

    if split_path.exists():
        with open(split_path) as f:
            data = json.load(f)
    else:
        data = create_global_val_split()

    if records is None:
        raise ValueError("records are required to compute per-region indices")

    val_set = set(data["val_uuids"])
    train_idx, val_idx = [], []
    for i, r in enumerate(records):
        if r["session_uuid"] in val_set:
            val_idx.append(i)
        else:
            train_idx.append(i)

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    print(f"  Val split [{region}]: train={len(train_idx)}, val={len(val_idx)}")
    return train_idx, val_idx


def print_val_split_distributions():
    """Print per-region train/val distributions from the global split."""
    split_path = _global_val_split_path()
    if not split_path.exists():
        print("  No global split found. Run create_global_val_split() first.")
        return

    with open(split_path) as f:
        data = json.load(f)

    strat = data.get("stratification", "unknown")
    n_total = data["n_total_sessions"]
    n_train = data["n_train_sessions"]
    n_val = data["n_val_sessions"]
    per_region = data.get("per_region", {})

    print(f"\n{'=' * 78}")
    print(f"  GLOBAL VAL SPLIT  |  stratified by: {strat}  |  seed: {data['seed']}")
    print(f"  Unique sessions: {n_total} total  |  "
          f"{n_train} train ({100*n_train/n_total:.0f}%)  |  "
          f"{n_val} val ({100*n_val/n_total:.0f}%)")
    print(f"{'=' * 78}")

    fmt = "  {:<25} {:>6} {:>6} {:>6} {:>7}  | {:>6} {:>6} {:>6} {:>7}"
    print(fmt.format(
        "Region", "Train", "Pos", "Neg", "Pos%", "Val", "Pos", "Neg", "Pos%"))
    print(f"  {'-' * 75}")

    for region in REGIONS:
        s = per_region.get(region)
        if not s:
            continue
        t_pct = f"{100.0 * s['train_pos'] / max(s['train'], 1):.1f}%"
        v_pct = f"{100.0 * s['val_pos'] / max(s['val'], 1):.1f}%"
        print(fmt.format(
            region, s["train"], s["train_pos"], s["train_neg"], t_pct,
            s["val"], s["val_pos"], s["val_neg"], v_pct))

    print(f"  {'-' * 75}\n")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class CachedFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].float()


class ImageDataset(Dataset):
    """Loads images from disk given (path, label) pairs."""

    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.transform(img)
        return pixel_values, float(self.labels[idx])


class ImageDatasetWithUUID(Dataset):
    """Like ImageDataset but also returns the session UUID (for caching)."""

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


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_train_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_tpr_at_fprs(y_true, y_score):
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_score)
    results = {}
    for target_fpr, name in zip(TARGET_FPRS, TARGET_FPR_NAMES):
        distances = np.abs(fpr_arr - target_fpr)
        min_dist = distances.min()
        tied_indices = np.where(distances == min_dist)[0]
        idx = tied_indices[np.argmax(tpr_arr[tied_indices])]
        thresh_idx = min(idx, len(thresholds) - 1)
        results[name] = {
            "tpr": float(tpr_arr[idx]),
            "actual_fpr": float(fpr_arr[idx]),
            "threshold": float(thresholds[thresh_idx]),
        }
    return results


def compute_all_metrics(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return {"auc_roc": 0.0, "auc_pr": 0.0, "tpr_at_fpr": {}}

    metrics = {
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "auc_pr": float(average_precision_score(y_true, y_score)),
        "tpr_at_fpr": compute_tpr_at_fprs(y_true, y_score),
    }

    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr_arr - fpr_arr
    best_idx = np.argmax(j_scores)
    thresh_idx = min(best_idx, len(thresholds) - 1)
    metrics["best_threshold_youden"] = {
        "threshold": float(thresholds[thresh_idx]),
        "tpr": float(tpr_arr[best_idx]),
        "fpr": float(fpr_arr[best_idx]),
    }
    return metrics


def format_tpr_at_fpr_inline(metrics):
    tpr_at_fpr = metrics.get("tpr_at_fpr", {})
    parts = []
    for name in TARGET_FPR_NAMES:
        d = tpr_at_fpr.get(name, {})
        parts.append(f"{name}={d.get('tpr', 0):.3f}")
    return " | ".join(parts)


def compute_metrics_auth_positive(y_true_orig, y_score_orig):
    """Compute metrics with positive=authentic convention.

    Input uses original labels (0=authentic, 1=fake) and original scores
    (higher = more likely fake).  Internally flips both so that
    TPR = authentic pass rate and FPR = fake miss rate.
    """
    y_true = 1 - np.asarray(y_true_orig)
    y_score = 1.0 - np.asarray(y_score_orig)

    if len(np.unique(y_true)) < 2:
        return {"auc_roc": 0.0, "auc_pr": 0.0, "tpr_at_fpr": {}}

    auc_roc = float(roc_auc_score(y_true, y_score))
    auc_pr = float(average_precision_score(y_true, y_score))

    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_score)
    tpr_at_fpr = {}
    for target_fpr, name in zip(TARGET_FPRS, TARGET_FPR_NAMES):
        distances = np.abs(fpr_arr - target_fpr)
        min_dist = distances.min()
        tied = np.where(distances == min_dist)[0]
        idx = tied[np.argmax(tpr_arr[tied])]
        thresh_idx = min(idx, len(thresholds) - 1)
        auth_thresh = float(thresholds[thresh_idx])
        tpr_at_fpr[name] = {
            "tpr": float(tpr_arr[idx]),
            "actual_fpr": float(fpr_arr[idx]),
            "threshold": auth_thresh,
            "threshold_orig": float(1.0 - auth_thresh),
        }

    return {"auc_roc": auc_roc, "auc_pr": auc_pr, "tpr_at_fpr": tpr_at_fpr}


def apply_threshold_auth_positive(y_true, y_score, threshold_orig):
    """Apply original-space threshold and return auth-positive TPR/FPR.

    threshold_orig is in the original score space (higher = more likely fake).
    Items with score >= threshold_orig are flagged as fake.
    """
    preds = (np.asarray(y_score) >= threshold_orig).astype(int)
    y_true = np.asarray(y_true)
    n_auth = int((y_true == 0).sum())
    n_fake = int((y_true == 1).sum())
    auth_passed = int(((preds == 0) & (y_true == 0)).sum())
    fakes_missed = int(((preds == 0) & (y_true == 1)).sum())
    return {
        "tpr": float(auth_passed / max(n_auth, 1)),
        "fpr": float(fakes_missed / max(n_fake, 1)),
    }


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def make_weighted_sampler(labels_array):
    """WeightedRandomSampler that oversamples the minority class."""
    class_counts = np.bincount(labels_array.astype(int))
    weights_per_class = 1.0 / class_counts
    sample_weights = weights_per_class[labels_array.astype(int)]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels_array),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Feature loading (cached .pt files)
# ---------------------------------------------------------------------------

def load_cached_features(region: str, model_key: str, resolution: int):
    cdir = cache_dir(region)
    prefix = f"{model_key}_{resolution}"
    features = torch.load(cdir / f"{prefix}_features.pt", weights_only=True)
    labels = torch.load(cdir / f"{prefix}_labels.pt", weights_only=True)
    uuids = torch.load(cdir / f"{prefix}_uuids.pt", weights_only=False)

    nan_mask = torch.isnan(features).any(dim=1) | torch.isinf(features).any(dim=1)
    if nan_mask.any():
        n_bad = nan_mask.sum().item()
        print(f"  WARNING: {n_bad}/{len(features)} features contain NaN/Inf -- dropping them")
        keep = ~nan_mask
        features = features[keep]
        labels = labels[keep]
        uuids = [u for u, k in zip(uuids, keep.tolist()) if k]

    return features, labels, uuids


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.head(x).squeeze(-1)


class DINOv3Classifier(nn.Module):
    """Full backbone + head for fine-tuning."""

    def __init__(self, model_id, freeze_backbone=True):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        )
        embed_dim = self.backbone.config.hidden_size
        self.head = nn.Linear(embed_dim, 1)
        self._frozen = freeze_backbone
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._frozen = True

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._frozen = False

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_features = outputs.pooler_output
        return self.head(cls_features).squeeze(-1)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_final_metrics(ckpt_path):
    if not Path(ckpt_path).exists():
        return
    ckpt = torch.load(ckpt_path, weights_only=False)
    metrics = ckpt.get("metrics", {})
    print("\n  --- Best checkpoint metrics ---")
    print(f"  AUC-ROC: {metrics.get('auc_roc', 'N/A')}")
    print(f"  AUC-PR:  {metrics.get('auc_pr', 'N/A')}")
    tpr_at_fpr = metrics.get("tpr_at_fpr", {})
    if tpr_at_fpr:
        print(f"  {'Target FPR':<12} {'Actual FPR':<12} {'TPR':<10} {'Threshold':<10}")
        print(f"  {'-' * 44}")
        for name in TARGET_FPR_NAMES:
            d = tpr_at_fpr.get(name, {})
            print(f"  {name:<12} {d.get('actual_fpr', 0):<12.4f} {d.get('tpr', 0):<10.4f} {d.get('threshold', 0):<10.4f}")
    youden = metrics.get("best_threshold_youden", {})
    if youden:
        print(f"  Youden best: threshold={youden.get('threshold', 0):.4f}  "
              f"TPR={youden.get('tpr', 0):.4f}  FPR={youden.get('fpr', 0):.4f}")
    print()
