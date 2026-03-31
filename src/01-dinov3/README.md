# DINOv3 Authentication Experiments

Binary classification: **authentic** (label 0) vs **not-authentic** (label 1) across four garment regions.

## Regions


| Region                | Description      |
| --------------------- | ---------------- |
| `care_label`          | Care/wash label  |
| `front`               | Front of garment |
| `front_exterior_logo` | Front logo area  |
| `brand_tag`           | Brand tag        |


## Data

- **Train:** `resources/apparel_supreme_until_dec_2025_{region}/` (~7k–10k labeled rows per region, depending on region)
- **Test:** `resources/apparel_supreme_jan_to_feb_2026_{region}/` (~1k–1.4k labeled rows)
- **Val split:** 80/20 stratified from train, global across all regions (same session always in the same split). Stored in `val_splits/global_session_split.json`.

## Metric conventions

All scripts use the **auth-positive** convention (positive = authentic):

- **TPR** = authentic pass rate (fraction of authentic items correctly passed)
- **FPR** = fake miss rate (fraction of fakes that slip through)
- **TPR@2%** = authentic pass rate when the operating point targets **2%** fake miss rate (see `TARGET_FPRS` in `config.py`)

Training labels stay 0 = auth, 1 = fake. Scores are flipped only inside metric helpers via `compute_metrics_auth_positive()` in `config.py`. Each `tpr_at_fpr` entry stores both `threshold` (auth-positive space) and `threshold_orig` (original score space).

---

## Scripts

### Core configuration

`**config.py`** — Shared across all scripts. Contains:

- Paths: `train_image_dir()`, `test_image_dir()`, `cache_dir()`, `ckpt_dir()`, `results_dir()`
- Constants: `REGIONS`, `MODEL_VARIANTS`, `RESOLUTIONS`, `SEED`, `TARGET_FPRS`
- Data: `load_metadata()`, `get_or_create_val_split()`, `create_global_val_split()`
- Datasets: `ImageDataset`, `CachedFeatureDataset`, `ImageDatasetWithUUID`
- Models: `DINOv3Classifier` (backbone + head), `LinearHead`
- Metrics: `compute_all_metrics()`, `compute_metrics_auth_positive()`, `compute_tpr_at_fprs()`
- Utilities: `make_weighted_sampler()`, `load_cached_features()`, `print_final_metrics()`

### Phase 1 — Frozen backbone

`**precompute_embeddings.py**` — Extract and cache DINOv3 CLS-token features (fp32) per region.

```bash
python precompute_embeddings.py --models vitl16 --resolutions 518 714 --regions care_label front front_exterior_logo brand_tag
```

Outputs: `cached_features/{region}/{model}_{resolution}_{features,labels,uuids}.pt`

`**train_linear_head.py**` — Train a BCE linear probe on cached embeddings with early stopping on val AUC.

```bash
python train_linear_head.py --region care_label --model vitl16 --resolution 714
python train_linear_head.py --region care_label --sweep  # all cached variants
```

Outputs: `checkpoints/{region}/{model}_{res}_linear_probe_best.pt`

`**train_svm_xgb_lgbm_catboost.py**` — Train SVM, XGBoost, CatBoost, LightGBM on cached embeddings.

```bash
python train_svm_xgb_lgbm_catboost.py --region care_label --model vitl16 --resolution 714
python train_svm_xgb_lgbm_catboost.py --region care_label --sweep
```

Outputs: `ml_results/{region}/{model}_{res}_{clf}.json` plus model artifacts (`.joblib`, `.json`, `.cbm`, `.txt`)

### Phase 2 — Partial finetune

`**train_partial_finetune.py**` — Warm up a frozen backbone, then unfreeze selected ViT blocks per strategy.

```bash
python train_partial_finetune.py --strategy q --region care_label       # Q projections, all layers
python train_partial_finetune.py --strategy qv --region care_label     # Q+V projections
python train_partial_finetune.py --strategy last4 --region care_label  # all components, last 4 layers
python train_partial_finetune.py --strategy norm --region care_label   # layer norms only
```

Strategies include `norm`, `layer_scale`, `q`, `v`, `qv`, `qkv`, `qkvo`, `mlp`, `last1`, `last4`, `last8`, `last12`, `qv_last4`, `full`, and more. See `STRATEGIES` in the script.

Outputs: `checkpoints/{region}/vitl16_714_partial_{strategy}_best.pt`

`**train_with_svm_xgb_lgbm_catboost_with_unfrozen_trained_dino.py**` — Extract embeddings from a finetuned checkpoint, then train ML classifiers on those features.

```bash
python train_with_svm_xgb_lgbm_catboost_with_unfrozen_trained_dino.py \
    --region care_label --model vitl16 --resolution 714 --ckpt-tag partial_q
```

`--ckpt-tag` selects `checkpoints/{region}/vitl16_714_{ckpt_tag}_best.pt`.

Outputs: `ml_results/{region}/{ckpt_tag}_{model}_{res}_{clf}.*`

### Phase 3 — Multi-region fusion

`**train_region_fusion.py**` — Concatenate per-region embeddings (4 × 1024-d + 4 mask bits = 4100-d) and train fusion classifiers. Uses the global val split and reports auth-positive test metrics.

```bash
python train_region_fusion.py --embed-key vitl16_714 --classifiers mlp svm xgb lgbm catboost
```

Available `--classifiers`: `attention`, `mlp`, `xgb`, `svm`, `lgbm`, `catboost`.

Outputs: `ml_results/region_fusion_results.json`

### Phase 4 — Test evaluation

`**evaluate_test.py**` — Full test pipeline for one region: extract test features, score models, apply val thresholds.

```bash
python evaluate_test.py --region care_label --models vitl16 --resolutions 518 714
```

Outputs: `ml_results/{region}/test_results.json`

`**evaluate_voting.py**` — Multi-region voting using the best model per region. Strategies: any-one-agree (OR), majority, all-must-agree (AND).

```bash
python evaluate_voting.py
```

Outputs: `ml_results/voting_results.json`

### Visualization

`**visualize_attention.py**` — Gradient-weighted attention rollout (Chefer et al.).

```bash
python visualize_attention.py --n-confident 5 --n-near 5
```

`**visualize_reciprocam.py**` — ViT-ReciproCAM saliency via norm1 token masking.

```bash
python visualize_reciprocam.py --target-layer 23 --n-confident 5
```

`**visualize_token_gradcam.py**` — Token-level Grad-CAM with configurable layer/module targets.

```bash
python visualize_token_gradcam.py --targets "layer.23.mlp" --n-confident 5
```

### Orchestration

`**run_pipeline.sh**` — Sequential pipeline.

```bash
bash run_pipeline.sh              # full pipeline
bash run_pipeline.sh --phase 2    # start from phase 2
```

`**run_all.sh**` — Multi-GPU parallel runs (frozen vs finetune branches). Expects four GPUs.

```bash
bash run_all.sh
bash run_all.sh care_label front  # limit regions
```

---

## Directory structure

```
src/01-dinov3/
├── config.py
├── precompute_embeddings.py
├── train_linear_head.py
├── train_svm_xgb_lgbm_catboost.py
├── train_partial_finetune.py
├── train_with_svm_xgb_lgbm_catboost_with_unfrozen_trained_dino.py
├── train_region_fusion.py
├── evaluate_test.py
├── evaluate_voting.py
├── visualize_attention.py
├── visualize_reciprocam.py
├── visualize_token_gradcam.py
├── run_pipeline.sh
├── run_all.sh
├── experiment_results.md          # optional older / alternate summary
├── val_splits/global_session_split.json
├── cached_features/{region}/
├── checkpoints/{region}/
├── ml_results/{region}/
├── ml_results/region_fusion_results.json
├── ml_results/voting_results.json
├── attention_maps/
├── reciprocam_maps/
└── logs/pipeline/
```

---

## Results (from `nohup/` logs)

All metrics use the **auth-positive** convention above. **Test TPR@2%** and **FPR@2%** use the **val-calibrated threshold** applied to the test set (not test-optimal). Train TPR@2% is often ~100% (memorization) and is omitted.

### Linear probe on frozen `vitl16` embeddings


| Region              | Res | Test AUC   | Test TPR@2% | Test FPR@2% | Best val TPR@2% |
| ------------------- | --- | ---------- | ----------- | ----------- | --------------- |
| care_label          | 518 | 0.7969     | 21.1%       | 0.0%        | 21.3%           |
| care_label          | 714 | 0.8016     | 19.8%       | 0.0%        | 21.8%           |
| front               | 518 | 0.6849     | 15.0%       | 2.4%        | 13.2%           |
| front               | 714 | **0.8474** | 16.4%       | 2.4%        | 17.3%           |
| front_exterior_logo | 518 | **0.8325** | 30.1%       | 5.7%        | 34.8%           |
| front_exterior_logo | 714 | 0.8220     | 30.4%       | 5.7%        | 33.3%           |
| brand_tag           | 518 | **0.8113** | 24.9%       | 0.0%        | 24.8%           |
| brand_tag           | 714 | 0.7962     | 20.6%       | 0.0%        | 21.1%           |


### Frozen embeddings + classifiers (`train_svm_xgb_lgbm_catboost.py`)

**care_label**


| Clf      | 518 AUC    | 518 TPR@2% | 518 FPR | 714 AUC | 714 TPR@2% | 714 FPR |
| -------- | ---------- | ---------- | ------- | ------- | ---------- | ------- |
| XGB      | 0.8346     | 13.9%      | 3.0%    | 0.8160  | 15.5%      | 3.0%    |
| **SVM**  | **0.8911** | **28.5%**  | 0.0%    | 0.8712  | 24.4%      | 0.0%    |
| CatBoost | 0.7989     | 13.2%      | 0.0%    | 0.7477  | 0.9%       | 0.0%    |
| LGBM     | 0.8128     | 14.2%      | 3.0%    | 0.7643  | 15.6%      | 3.0%    |


**front**


| Clf      | 518 AUC    | 518 TPR@2% | 518 FPR | 714 AUC | 714 TPR@2% | 714 FPR |
| -------- | ---------- | ---------- | ------- | ------- | ---------- | ------- |
| XGB      | 0.9126     | 11.1%      | 0.0%    | 0.8755  | 18.9%      | 4.9%    |
| **SVM**  | **0.9246** | 9.7%       | 0.0%    | 0.8995  | 7.7%       | 0.0%    |
| CatBoost | 0.8357     | 17.3%      | 2.4%    | 0.8261  | 18.8%      | 2.4%    |
| LGBM     | 0.9000     | 2.7%       | 0.0%    | 0.8699  | 19.1%      | 0.0%    |


**front_exterior_logo**


| Clf      | 518 AUC    | 518 TPR@2% | 518 FPR | 714 AUC | 714 TPR@2% | 714 FPR |
| -------- | ---------- | ---------- | ------- | ------- | ---------- | ------- |
| XGB      | 0.8248     | 38.2%      | 2.9%    | 0.8172  | 26.1%      | 5.7%    |
| **SVM**  | **0.8635** | 16.7%      | 2.9%    | 0.8368  | 26.2%      | 2.9%    |
| CatBoost | 0.7308     | 36.4%      | 14.3%   | 0.8056  | 20.1%      | 8.6%    |
| LGBM     | 0.8172     | 24.5%      | 2.9%    | 0.8020  | 6.8%       | 0.0%    |


**brand_tag**


| Clf      | 518 AUC | 518 TPR@2% | 518 FPR | 714 AUC    | 714 TPR@2% | 714 FPR |
| -------- | ------- | ---------- | ------- | ---------- | ---------- | ------- |
| XGB      | 0.8226  | 18.5%      | 3.4%    | **0.8282** | 14.5%      | 0.0%    |
| SVM      | 0.7951  | 10.6%      | 0.0%    | 0.7977     | 10.2%      | 0.0%    |
| CatBoost | 0.7868  | 9.9%       | 0.0%    | 0.7726     | 20.2%      | 0.0%    |
| LGBM     | 0.7874  | 17.8%      | 0.0%    | 0.8196     | 16.7%      | 0.0%    |


### Partial finetune `last4` + linear head (`vitl16` @ 714)

Checkpoint chosen by best **val TPR@2%**; test uses that threshold.


| Region              | Test AUC   | Test TPR@2% | Test FPR@2% | Best val TPR@2% |
| ------------------- | ---------- | ----------- | ----------- | --------------- |
| care_label          | 0.8178     | 25.5%       | 0.0%        | 27.1%           |
| front               | **0.8886** | 14.5%       | 0.0%        | 13.9%           |
| front_exterior_logo | 0.8155     | 36.6%       | 5.7%        | 40.1%           |
| brand_tag           | **0.8352** | 25.0%       | 0.0%        | 24.3%           |


A separate AUC-selected run for `front_exterior_logo` gave test AUC **0.8230**, TPR@2% **35.7%**, FPR **5.7%**.

### Finetuned `last4` embeddings + ML classifiers (`vitl16` @ 714)

Features extracted from the `last4` finetuned backbone, then SVM/XGB/CatBoost/LGBM trained on top.

**care_label**


| Clf      | Test AUC   | TPR@2%    | FPR@2% |
| -------- | ---------- | --------- | ------ |
| XGB      | 0.7947     | 19.7%     | 3.0%   |
| **SVM**  | **0.8712** | **24.4%** | 0.0%   |
| CatBoost | 0.7300     | 14.4%     | 0.0%   |
| LGBM     | 0.7504     | 9.7%      | 6.1%   |


**front**


| Clf      | Test AUC   | TPR@2% | FPR@2% |
| -------- | ---------- | ------ | ------ |
| XGB      | 0.8745     | 16.9%  | 0.0%   |
| **SVM**  | **0.8994** | 7.7%   | 0.0%   |
| CatBoost | 0.8333     | 21.2%  | 2.4%   |
| LGBM     | 0.8617     | 7.4%   | 0.0%   |


**front_exterior_logo**


| Clf      | Test AUC   | TPR@2%    | FPR@2% |
| -------- | ---------- | --------- | ------ |
| XGB      | 0.8387     | 24.1%     | 2.9%   |
| **SVM**  | **0.8368** | **26.1%** | 2.9%   |
| CatBoost | 0.8274     | 21.9%     | 8.6%   |
| LGBM     | 0.8070     | 11.6%     | 0.0%   |


**brand_tag**


| Clf      | Test AUC   | TPR@2%    | FPR@2% |
| -------- | ---------- | --------- | ------ |
| **XGB**  | **0.8235** | **16.9%** | 0.0%   |
| SVM      | 0.7978     | 10.2%     | 0.0%   |
| CatBoost | 0.7708     | 12.8%     | 3.4%   |
| LGBM     | 0.8155     | 14.6%     | 0.0%   |


### Region fusion (`train_region_fusion.py`, `vitl16_714` frozen embeddings)

Concatenated 4-region embeddings (4 × 1024-d + 4 mask bits). **Auth-positive:** each cell is **TPR** (authentics passed) and **actual test FPR** in parentheses — fake miss rate on test at the val threshold chosen for that FPR target (discrete fake count often prevents hitting the target exactly).


| Method   | Test AUC   | @0.5% target      | @1% target        | @2% target        | @5% target        | @10% target       |
| -------- | ---------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| **SVM**  | **0.8890** | 48.7% (0.0%)      | 48.7% (0.0%)      | **58.1% (2.4%)**  | 63.6% (4.9%)      | 64.6% (9.8%)      |
| XGB      | 0.8489     | 32.7% (0.0%)      | 32.7% (0.0%)      | 51.6% (2.4%)      | 52.4% (7.3%)      | 52.9% (9.8%)      |
| MLP      | 0.8787     | 13.8% (0.0%)      | 13.8% (0.0%)      | 37.6% (2.4%)      | 52.4% (4.9%)      | **68.4% (9.8%)**  |
| LGBM     | 0.8245     | 34.6% (0.0%)      | 34.6% (0.0%)      | 36.9% (2.4%)      | 42.2% (4.9%)      | 57.5% (9.8%)      |
| CatBoost | 0.8181     | 11.5% (0.0%)      | 11.5% (0.0%)      | 12.7% (2.4%)      | 27.9% (4.9%)      | 52.4% (9.8%)      |


**Fusion SVM at the 2% target** gives the highest single TPR (**58.1%** authentics passed). The `region_fusion_last4_714` run had zero test sessions (finetuned test features not cached); treat as incomplete.

### Best model per region (by test TPR@2%)

| Region | Best Model | Test AUC | Test TPR@2% | FPR@2% |
|--------|------------|----------|-------------|--------|
| front_exterior_logo | Frozen XGB 518 | 0.8248 | **38.2%** | 2.9% |
| care_label | Frozen SVM 518 | 0.8911 | **28.5%** | 0.0% |
| brand_tag | Finetune last4 | 0.8352 | **25.0%** | 0.0% |
| front | Unfrozen last4 CatBoost | 0.8333 | **21.2%** | 2.4% |


### Notes

- Multi-region **voting** (`evaluate_voting.py`) is implemented with any/majority/all-agree strategies but threshold calibration across regions needs refinement. Score-level fusion (e.g. fusion SVM) currently outperforms fixed-threshold voting.
- Region coverage varies: `front` covers 98.2% of test sessions, `front_exterior_logo` only 61.4%. Missing regions get zero embeddings + mask bit = 0 in fusion.

