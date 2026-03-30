# DINOv3 Authentication Experiments

Binary classification: **authentic** (label 0) vs **not-authentic** (label 1) across 4 garment regions.

## Regions

| Region | Description |
|---|---|
| `care_label` | Care/wash label |
| `front` | Front of garment |
| `front_exterior_logo` | Front logo area |
| `brand_tag` | Brand tag |

## Data

- **Train:** `resources/apparel_supreme_until_dec_2025_{region}/` (~7k-10k images per region)
- **Test:** `resources/apparel_supreme_jan_to_feb_2026_{region}/` (~1k-1.4k images per region)
- **Val split:** 80/20 stratified from train, global across all regions (same session always in same split). Stored in `val_splits/global_session_split.json`.

## Metric Conventions

All scripts use **auth-positive** convention (positive = authentic):

- **TPR** = authentic pass rate (fraction of authentic items correctly passed)
- **FPR** = fake miss rate (fraction of fakes that slipped through)
- **TPR@2%** = auth pass rate when only 2% of fakes are missed

Training labels unchanged (0=auth, 1=fake). Scores flipped only for metric computation via `compute_metrics_auth_positive()` in config. Each `tpr_at_fpr` entry stores both `threshold` (auth-positive space) and `threshold_orig` (original score space, for downstream `score >= threshold` application).

---

## Scripts

### Core Configuration

**`config.py`** — Shared across all scripts. Contains:
- Paths: `train_image_dir()`, `test_image_dir()`, `cache_dir()`, `ckpt_dir()`, `results_dir()`
- Constants: `REGIONS`, `MODEL_VARIANTS`, `RESOLUTIONS`, `SEED`, `TARGET_FPRS`
- Data: `load_metadata()`, `get_or_create_val_split()`, `create_global_val_split()`
- Datasets: `ImageDataset`, `CachedFeatureDataset`, `ImageDatasetWithUUID`
- Models: `DINOv3Classifier` (backbone + head), `LinearHead`
- Metrics: `compute_all_metrics()`, `compute_metrics_auth_positive()`, `compute_tpr_at_fprs()`
- Utilities: `make_weighted_sampler()`, `load_cached_features()`, `print_final_metrics()`

### Phase 1 — Frozen Backbone

**`precompute_embeddings.py`** — Extract and cache DINOv3 CLS-token features (fp32) per region.
```bash
python precompute_embeddings.py --models vitl16 --resolutions 518 714 --regions care_label front front_exterior_logo brand_tag
```
Outputs: `cached_features/{region}/{model}_{resolution}_{features,labels,uuids}.pt`

**`train_linear_head.py`** — Train a BCE linear probe on cached embeddings with early stopping on val AUC.
```bash
python train_linear_head.py --region care_label --model vitl16 --resolution 714
python train_linear_head.py --region care_label --sweep  # all cached variants
```
Outputs: `checkpoints/{region}/{model}_{res}_linear_probe_best.pt`

**`train_svm_xgb_lgbm_catboost.py`** — Train SVM, XGBoost, CatBoost, LightGBM on cached embeddings.
```bash
python train_svm_xgb_lgbm_catboost.py --region care_label --model vitl16 --resolution 714
python train_svm_xgb_lgbm_catboost.py --region care_label --sweep
```
Outputs: `ml_results/{region}/{model}_{res}_{clf}.json` + model artifacts (`.joblib`, `.json`, `.cbm`, `.txt`)

### Phase 2 — Partial Finetune

**`train_partial_finetune.py`** — Warmup frozen backbone, then unfreeze selected ViT components per strategy.
```bash
python train_partial_finetune.py --strategy q --region care_label     # Q projections, all layers
python train_partial_finetune.py --strategy qv --region care_label    # Q+V projections
python train_partial_finetune.py --strategy last4 --region care_label # all components, last 4 layers
python train_partial_finetune.py --strategy norm --region care_label  # layer norms only
```
Strategies: `norm`, `layer_scale`, `q`, `v`, `qv`, `qkv`, `qkvo`, `mlp`, `last1`, `last4`, `last8`, `last12`, `qv_last4`, `full`, and more. See `STRATEGIES` in the script.

Outputs: `checkpoints/{region}/vitl16_714_partial_{strategy}_best.pt`

**`train_with_svm_xgb_lgbm_catboost_with_unfrozen_trained_dino.py`** — Extract embeddings from a finetuned checkpoint, then train ML classifiers on those features.
```bash
python train_with_svm_xgb_lgbm_catboost_with_unfrozen_trained_dino.py \
    --region care_label --model vitl16 --resolution 714 --ckpt-tag partial_q
```
`--ckpt-tag` selects the checkpoint: looks for `checkpoints/{region}/vitl16_714_{ckpt_tag}_best.pt`.

Outputs: `ml_results/{region}/{ckpt_tag}_{model}_{res}_{clf}.*`

### Phase 3 — Multi-Region Fusion

**`train_region_fusion.py`** — Concatenate per-region embeddings (4 × 1024d + 4 mask bits = 4100d), train fusion classifiers.
```bash
python train_region_fusion.py --embed-key vitl16_714
```
Models: `attention` (cross-region self-attention), `mlp` (3-layer), `xgb`, `svm`.
Uses global val split. Reports auth-positive test metrics.

Outputs: `ml_results/region_fusion_results.json`

### Phase 4 — Test Evaluation

**`evaluate_test.py`** — Full test pipeline for one region: extract test features, score all models, apply val thresholds.
```bash
python evaluate_test.py --region care_label --models vitl16 --resolutions 518 714
```
Outputs: `ml_results/{region}/test_results.json`

**`evaluate_voting.py`** — Multi-region voting using best model per region. Three strategies:
- **any-one-agree (OR):** flag if ANY region flags — aggressive, high catch rate
- **majority:** flag if >= ceil(n/2) regions flag — balanced
- **all-must-agree (AND):** flag only if ALL available regions flag — conservative, low false alarm
```bash
python evaluate_voting.py
```
Outputs: `ml_results/voting_results.json`

### Visualization

**`visualize_attention.py`** — Gradient-weighted attention rollout maps (Chefer et al.) for finetune vs partial finetune checkpoints.
```bash
python visualize_attention.py --n-confident 5 --n-near 5
```

**`visualize_reciprocam.py`** — ViT-ReciproCAM saliency maps via norm1 token masking.
```bash
python visualize_reciprocam.py --target-layer 23 --n-confident 5
```

**`visualize_token_gradcam.py`** — Token-level Grad-CAM with configurable layer/module targets.
```bash
python visualize_token_gradcam.py --targets "layer.23.mlp" --n-confident 5
```

### Orchestration

**`run_pipeline.sh`** — Master script running all phases sequentially.
```bash
bash run_pipeline.sh              # full pipeline
bash run_pipeline.sh --phase 2    # start from phase 2
```

**`run_all.sh`** — Multi-GPU parallel orchestration (Branch A: frozen pipeline, Branch B: finetune pipeline). Requires 4 GPUs.
```bash
bash run_all.sh
bash run_all.sh care_label front  # specific regions only
```

---

## Directory Structure

```
src/01-dinov3/
├── config.py                          # shared config, models, metrics
├── precompute_embeddings.py           # phase 1: cache frozen features
├── train_linear_head.py               # phase 1: linear probe
├── train_svm_xgb_lgbm_catboost.py    # phase 1: ML classifiers
├── train_partial_finetune.py          # phase 2: partial finetune
├── train_with_svm_...py              # phase 2: ML on finetuned features
├── train_region_fusion.py             # phase 3: multi-region fusion
├── evaluate_test.py                   # phase 4: per-region test eval
├── evaluate_voting.py                 # phase 4: multi-region voting
├── visualize_attention.py             # viz: attention rollout
├── visualize_reciprocam.py            # viz: reciprocam maps
├── visualize_token_gradcam.py         # viz: token grad-cam
├── run_pipeline.sh                    # orchestration (single GPU)
├── run_all.sh                         # orchestration (multi GPU)
├── experiment_results.md              # latest results summary
│
├── val_splits/
│   └── global_session_split.json      # 80/20 stratified, session-level
│
├── cached_features/{region}/          # .pt files: features, labels, uuids
├── checkpoints/{region}/              # .pt model checkpoints
├── ml_results/{region}/               # .json metrics + model artifacts
├── ml_results/
│   ├── region_fusion_results.json
│   └── voting_results.json
│
├── attention_maps/                    # visualization outputs
├── reciprocam_maps/                   # visualization outputs
└── logs/pipeline/                     # training logs
```

## Results

See `experiment_results.md` for the latest test TPR/FPR tables across all experiments.
