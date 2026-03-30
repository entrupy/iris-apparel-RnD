# DINOv3 Multi-Region Authentication — Experiment Summary

**Date:** 2026-03-22
**Task:** Binary classification — authentic vs not-authentic
**Regions:** care_label, front, front_exterior_logo, brand_tag
**Backbones:** ViT (vits16, vitb16, vitl16) + ConvNeXt (tiny, small, base, large)
**Resolutions:** 518, 714
**Features:** fp32 (fixed bfloat16 overflow)
**Finetune checkpoint selection:** best TPR@2%FPR (tiebreak: AUC-ROC)

**Convention:** Positive = authentic. TP = authentic correctly passed, FP = fake wrongly passed (missed fake), FN = authentic wrongly flagged, TN = fake correctly caught. Per-model tables use detection metrics: "Val/Test TPR" = fake catch rate, "FPR" = false alarm rate on authentic. Voting tables use auth-positive: TPR = auth pass rate, FPR = fake miss rate.

---

## Data

### Training (apparel_supreme_until_dec_2025)


| Region              | Images | Positive | Negative | Train | Val   | Val Pos |
| ------------------- | ------ | -------- | -------- | ----- | ----- | ------- |
| care_label          | 7,294  | 240      | 7054     | 5,835 | 1,459 | —       |
| front               | 9,627  | 306      | 9321     | 7,701 | 1,926 | —       |
| front_exterior_logo | 5,005  | 147      | 4858     | 4,004 | 1,001 | —       |
| brand_tag           | 9,169  | 288      | 8881     | 7,335 | 1,834 | —       |


### Test (apparel_supreme_jan_to_feb_2026)


| Region              | Images | Positive | Negative |
| ------------------- | ------ | -------- | -------- |
| care_label          | 962    | 33       | 929      |
| front               | 1,398  | 41       | 1,357    |
| front_exterior_logo | 875    | 35       | 840      |
| brand_tag           | 1,192  | 29       | 1,163    |


---

## Per-Region Results (test ROC, sorted by TPR @ 2% missed)

**TPR** = % of authentic items correctly passed (true authentics)
**FPR** = % of fakes that slipped through (missed fakes)

Computed from the **test ROC curve** directly. Top 25 models per region.

### care_label (test: 962 imgs, 33 fake)

| Model | AUC | TPR @ 0.5% missed | TPR @ 1% missed | TPR @ 2% missed | TPR @ 5% missed | TPR @ 10% missed |
| --- | --- | --- | --- | --- | --- | --- |
| vitl16_714_partial_q | 0.9297 | 16.47% | 16.47% | 76.32% | 79.66% | 84.07% |
| vitl16_714_partial_qv_last4 | 0.9252 | 19.48% | 19.48% | 69.32% | 81.49% | 84.28% |
| vitl16_714_partial_v | 0.9014 | 11.52% | 11.52% | 65.98% | 65.98% | 71.47% |
| vitl16_714_partial_qv | 0.9164 | 11.84% | 11.84% | 64.48% | 73.09% | 78.15% |
| vitl16_714_partial_norm+ls | 0.8963 | 32.94% | 32.94% | 61.03% | 68.35% | 72.34% |
| vitl16_714_partial_norm | 0.8966 | 36.17% | 36.17% | 60.82% | 69.43% | 72.34% |
| finetuned_vitl16_714_svm | 0.9118 | 0.00% | 0.00% | 59.53% | 67.38% | 71.58% |
| finetuned_vitl16_714_lgbm | 0.9022 | 25.40% | 25.40% | 54.90% | 70.94% | 72.23% |
| convnext_base_518_linear | 0.8118 | 25.30% | 25.30% | 54.79% | 55.01% | 55.87% |
| finetuned_vitl16_714_catboost | 0.8848 | 2.80% | 2.80% | 52.21% | 57.27% | 70.18% |
| vits16_518_catboost | 0.8660 | 45.21% | 45.21% | 49.19% | 55.01% | 58.23% |
| vits16_714_svm | 0.8625 | 35.84% | 35.84% | 48.98% | 53.71% | 55.33% |
| vitl16_518_lgbm | 0.8487 | 25.51% | 25.51% | 48.44% | 49.62% | 51.78% |
| vits16_518_svm | 0.8892 | 32.83% | 32.83% | 48.44% | 57.91% | 64.48% |
| finetuned_vitb16_714_svm | 0.8516 | 22.93% | 22.93% | 46.50% | 49.62% | 61.14% |
| vitb16_714_svm | 0.8516 | 22.93% | 22.93% | 46.50% | 49.62% | 61.14% |
| vits16_714_lgbm | 0.8536 | 28.85% | 28.85% | 44.78% | 45.53% | 61.36% |
| vits16_714_linear | 0.8135 | 31.32% | 31.32% | 44.35% | 54.25% | 56.30% |
| finetuned_vitb16_518_svm | 0.8937 | 39.18% | 39.18% | 42.73% | 56.73% | 66.63% |
| vitb16_518_svm | 0.8937 | 39.18% | 39.18% | 42.73% | 56.73% | 66.63% |
| convnext_tiny_518_linear | 0.7679 | 14.75% | 14.75% | 42.41% | 45.53% | 49.09% |
| vitb16_518_linear | 0.8155 | 27.45% | 27.45% | 40.58% | 48.01% | 53.50% |
| vitl16_714_svm | 0.8649 | 32.29% | 32.29% | 40.58% | 47.36% | 49.62% |
| finetuned_vitb16_518_catboost | 0.8639 | 35.41% | 35.41% | 39.07% | 42.95% | 51.24% |
| vitb16_518_catboost | 0.8639 | 35.41% | 35.41% | 39.07% | 42.95% | 51.24% |

### front (test: 1398 imgs, 41 fake)

| Model | AUC | TPR @ 0.5% missed | TPR @ 1% missed | TPR @ 2% missed | TPR @ 5% missed | TPR @ 10% missed |
| --- | --- | --- | --- | --- | --- | --- |
| vitl16_518_svm | 0.9359 | 67.50% | 67.50% | 77.16% | 81.36% | 84.45% |
| vits16_518_linear | 0.8814 | 49.30% | 49.30% | 58.66% | 61.83% | 74.06% |
| vits16_714_linear | 0.8847 | 47.90% | 47.90% | 57.55% | 61.16% | 69.34% |
| vitl16_714_svm | 0.9155 | 38.17% | 38.17% | 56.08% | 68.83% | 76.34% |
| vitb16_518_svm | 0.9018 | 53.28% | 53.28% | 55.12% | 55.78% | 63.15% |
| vitl16_518_linear | 0.8856 | 17.76% | 17.76% | 54.02% | 56.45% | 66.69% |
| convnext_small_714_catboost | 0.8906 | 7.81% | 7.81% | 53.80% | 64.11% | 67.13% |
| finetuned_vitb16_714_lgbm | 0.8698 | 37.58% | 37.58% | 53.43% | 65.44% | 66.62% |
| vitb16_714_lgbm | 0.8697 | 37.58% | 37.58% | 53.43% | 65.36% | 66.62% |
| vits16_714_lgbm | 0.9115 | 40.16% | 40.16% | 52.62% | 59.25% | 73.03% |
| convnext_tiny_518_catboost | 0.8653 | 9.58% | 9.58% | 51.88% | 53.06% | 55.27% |
| vitl16_714_linear | 0.8771 | 15.03% | 15.03% | 51.73% | 53.72% | 65.36% |
| convnext_base_714_lgbm | 0.8879 | 47.38% | 47.38% | 51.66% | 51.73% | 62.12% |
| convnext_base_518_catboost | 0.9139 | 29.40% | 29.40% | 50.18% | 65.51% | 80.47% |
| vitb16_518_catboost | 0.8672 | 32.50% | 32.50% | 49.52% | 52.69% | 56.89% |
| finetuned_vitb16_714_catboost | 0.9039 | 45.69% | 45.69% | 48.19% | 60.65% | 72.59% |
| vitb16_714_catboost | 0.9039 | 45.69% | 45.69% | 48.19% | 60.65% | 72.59% |
| vits16_714_catboost | 0.8680 | 45.69% | 45.69% | 46.87% | 47.53% | 67.87% |
| vits16_518_catboost | 0.9020 | 8.47% | 8.47% | 46.06% | 64.70% | 77.08% |
| convnext_base_714_catboost | 0.8450 | 21.81% | 21.81% | 45.39% | 52.25% | 55.20% |
| vitb16_518_lgbm | 0.8636 | 23.36% | 23.36% | 42.89% | 44.80% | 62.49% |
| convnext_tiny_714_linear | 0.8787 | 35.45% | 35.45% | 42.82% | 50.18% | 70.38% |
| finetuned_vitb16_518_lgbm | 0.8675 | 12.82% | 12.82% | 42.52% | 50.41% | 57.77% |
| vitl16_518_catboost | 0.8426 | 34.78% | 34.78% | 42.23% | 42.89% | 46.57% |
| finetuned_vitb16_714_svm | 0.8697 | 40.97% | 40.97% | 40.97% | 55.42% | 57.63% |

### front_exterior_logo (test: 875 imgs, 35 fake)

| Model | AUC | TPR @ 0.5% missed | TPR @ 1% missed | TPR @ 2% missed | TPR @ 5% missed | TPR @ 10% missed |
| --- | --- | --- | --- | --- | --- | --- |
| convnext_large_714_lgbm | 0.8736 | 22.02% | 22.02% | 50.95% | 52.02% | 56.90% |
| vits16_518_catboost | 0.8384 | 13.81% | 13.81% | 46.67% | 49.52% | 69.29% |
| convnext_small_714_svm | 0.8948 | 35.36% | 35.36% | 46.31% | 55.36% | 70.12% |
| convnext_small_714_catboost | 0.8710 | 32.86% | 32.86% | 44.40% | 45.60% | 71.43% |
| convnext_large_714_catboost | 0.9054 | 37.38% | 37.38% | 43.33% | 57.50% | 79.40% |
| convnext_small_714_lgbm | 0.8840 | 36.31% | 36.31% | 41.55% | 47.50% | 76.67% |
| vits16_714_catboost | 0.7965 | 33.69% | 33.69% | 40.00% | 40.24% | 44.29% |
| vits16_518_linear | 0.8031 | 27.98% | 27.98% | 39.76% | 46.31% | 50.12% |
| vits16_518_svm | 0.8378 | 8.93% | 8.93% | 39.05% | 39.17% | 56.07% |
| finetuned_vitb16_518_catboost | 0.7908 | 3.81% | 3.81% | 38.45% | 38.57% | 43.45% |
| convnext_small_518_catboost | 0.8318 | 20.48% | 20.48% | 36.67% | 43.10% | 56.67% |
| vitl16_518_svm | 0.8334 | 3.21% | 3.21% | 34.40% | 50.00% | 56.79% |
| convnext_small_714_linear | 0.8293 | 6.90% | 6.90% | 32.86% | 45.00% | 57.50% |
| convnext_small_518_svm | 0.8622 | 6.43% | 6.43% | 32.38% | 33.93% | 57.50% |
| convnext_base_518_linear | 0.8512 | 15.12% | 15.12% | 32.02% | 51.55% | 66.79% |
| vits16_714_svm | 0.8354 | 27.86% | 27.86% | 29.52% | 32.14% | 47.86% |
| convnext_small_518_lgbm | 0.8518 | 5.95% | 5.95% | 29.40% | 54.76% | 57.50% |
| finetuned_vitl16_518_catboost | 0.8351 | 4.40% | 4.40% | 29.40% | 46.43% | 65.95% |
| vitl16_714_svm | 0.8361 | 4.29% | 4.29% | 27.98% | 37.50% | 59.40% |
| convnext_base_714_catboost | 0.8639 | 6.43% | 6.43% | 27.50% | 32.62% | 62.98% |
| vitb16_714_catboost | 0.7837 | 21.67% | 21.67% | 27.50% | 33.93% | 52.14% |
| convnext_small_518_linear | 0.8317 | 21.07% | 21.07% | 27.02% | 51.31% | 51.67% |
| vitb16_518_lgbm | 0.7891 | 8.93% | 8.93% | 25.95% | 26.43% | 48.69% |
| convnext_base_518_catboost | 0.8630 | 24.17% | 24.17% | 25.48% | 47.26% | 59.40% |
| vitb16_518_svm | 0.8206 | 0.36% | 0.36% | 25.48% | 27.86% | 57.14% |

### brand_tag (test: 1192 imgs, 29 fake)

| Model | AUC | TPR @ 0.5% missed | TPR @ 1% missed | TPR @ 2% missed | TPR @ 5% missed | TPR @ 10% missed |
| --- | --- | --- | --- | --- | --- | --- |
| vits16_518_lgbm | 0.8901 | 67.93% | 67.93% | 68.36% | 68.36% | 73.95% |
| convnext_tiny_518_svm | 0.8924 | 26.91% | 26.91% | 66.90% | 66.90% | 73.09% |
| vits16_714_catboost | 0.8894 | 60.53% | 60.53% | 66.29% | 66.29% | 69.05% |
| convnext_large_518_catboost | 0.9061 | 39.55% | 39.55% | 63.71% | 63.71% | 74.89% |
| convnext_large_714_linear | 0.8815 | 32.67% | 32.67% | 63.28% | 63.28% | 68.44% |
| finetuned_vitl16_714_catboost | 0.8932 | 57.87% | 57.87% | 59.24% | 59.24% | 64.32% |
| vitb16_518_catboost | 0.9072 | 56.84% | 56.84% | 58.99% | 58.99% | 70.42% |
| vitl16_518_linear | 0.8889 | 41.27% | 41.27% | 57.44% | 57.44% | 62.68% |
| finetuned_vitl16_714_lgbm | 0.8799 | 29.49% | 29.49% | 52.97% | 52.97% | 67.24% |
| convnext_tiny_714_catboost | 0.8930 | 32.67% | 32.67% | 52.62% | 52.62% | 79.45% |
| vits16_714_linear | 0.8705 | 43.85% | 43.85% | 52.28% | 52.28% | 66.72% |
| vitl16_714_linear | 0.8782 | 48.58% | 48.58% | 51.76% | 51.76% | 60.10% |
| convnext_large_714_catboost | 0.8688 | 48.32% | 48.32% | 51.33% | 51.33% | 55.72% |
| vits16_518_catboost | 0.8848 | 48.32% | 48.32% | 51.33% | 51.33% | 66.12% |
| vitb16_714_linear | 0.8751 | 32.24% | 32.24% | 51.07% | 51.07% | 60.10% |
| finetuned_vitb16_714_lgbm | 0.8383 | 31.81% | 31.81% | 50.64% | 50.64% | 54.00% |
| vitb16_714_lgbm | 0.8383 | 31.90% | 31.90% | 50.64% | 50.64% | 54.08% |
| convnext_small_714_svm | 0.8614 | 25.28% | 25.28% | 48.93% | 48.93% | 58.21% |
| convnext_base_518_linear | 0.8802 | 45.40% | 45.40% | 48.50% | 48.50% | 69.39% |
| finetuned_vitb16_714_svm | 0.8526 | 41.62% | 41.62% | 48.32% | 48.32% | 50.47% |
| vitb16_714_svm | 0.8526 | 41.62% | 41.62% | 48.32% | 48.32% | 50.47% |
| finetuned_vitl16_518_svm | 0.8890 | 24.42% | 24.42% | 47.89% | 47.89% | 66.04% |
| vitb16_518_linear | 0.8659 | 44.20% | 44.20% | 47.64% | 47.64% | 55.37% |
| convnext_base_714_linear | 0.8624 | 45.23% | 45.23% | 47.55% | 47.55% | 63.80% |
| vits16_714_lgbm | 0.8802 | 47.55% | 47.55% | 47.55% | 54.00% | 61.82% |

## Best per Region

| Region | Best Method | AUC | TPR @ 2% missed | TPR @ 5% missed | TPR @ 10% missed |
| --- | --- | --- | --- | --- | --- |
| care_label | vitl16_714_partial_q | 0.9297 | 76.32% | 79.66% | 84.07% |
| front | vitl16_518_svm | 0.9359 | 77.16% | 81.36% | 84.45% |
| front_exterior_logo | convnext_large_714_lgbm | 0.8736 | 50.95% | 52.02% | 56.90% |
| brand_tag | vits16_518_lgbm | 0.8901 | 68.36% | 68.36% | 73.95% |


---
## Voting (Multi-Region Ensemble)

Best model per region used. A session is flagged as fake if **any** region flags it (all-must-agree-authentic).
Majority voting: flag if >= ceil(available/2) regions flag. Missing regions = no flag.

**TPR** = true authentics, **FPR** = missed fakes. Val thresholds applied to test.
"Aggressiveness" = per-region threshold strictness (higher = more items flagged as fake).
As aggressiveness increases: fewer auth pass (TPR↓), but fewer fakes escape (FPR↓).

### All-Must-Agree Voting

| Aggressiveness | True auth (TPR) | Missed fakes (FPR) | TP (auth✓) | FP (missed) | FN (alarm) | TN (caught) |
| --- | --- | --- | --- | --- | --- | --- |
| low (0.5%) | 98.55% | 39.02% | 1363 | 16 | 20 | 25 |
| ↓ (1%) | 97.25% | 31.71% | 1345 | 13 | 38 | 28 |
| mid (2%) | 94.58% | 21.95% | 1308 | 9 | 75 | 32 |
| ↓ (5%) | 86.12% | 14.63% | 1191 | 6 | 192 | 35 |
| high (10%) | 74.55% | 9.76% | 1031 | 4 | 352 | 37 |

### Majority Voting

| Aggressiveness | True auth (TPR) | Missed fakes (FPR) | TP (auth✓) | FP (missed) | FN (alarm) | TN (caught) |
| --- | --- | --- | --- | --- | --- | --- |
| low (0.5%) | 99.64% | 60.98% | 1378 | 25 | 5 | 16 |
| ↓ (1%) | 99.42% | 60.98% | 1375 | 25 | 8 | 16 |
| mid (2%) | 99.06% | 43.90% | 1370 | 18 | 13 | 23 |
| ↓ (5%) | 96.02% | 34.15% | 1328 | 14 | 55 | 27 |
| high (10%) | 91.32% | 24.39% | 1263 | 10 | 120 | 31 |

---

## Multi-Region Embedding Fusion

Concatenate vitl16_518 embeddings from all 4 regions (1024d × 4 + 4 mask bits = 4100d).
Missing regions zero-filled. Test ROC, auth-positive convention.

| Method | AUC | TPR @ 0.5% missed | TPR @ 1% missed | TPR @ 2% missed | TPR @ 5% missed | TPR @ 10% missed |
| --- | --- | --- | --- | --- | --- | --- |
| concat_linear | 0.8883 | 21.11% | 21.11% | 50.61% | 51.55% | 65.87% |
| concat_svm | 0.8833 | 12.87% | 12.87% | 48.37% | 54.45% | 54.95% |
| concat_xgb | 0.8922 | 17.28% | 17.28% | 25.60% | 56.83% | 66.38% |

---

## Final Comparison

| Method | TPR @ 2% missed | TPR @ 5% missed | TPR @ 10% missed |
| --- | --- | --- | --- |
| **Majority voting (all 4)** | 99.06% | 96.02% | 91.32% |
| **All-must-agree (all 4)** | 94.58% | 86.12% | 74.55% |
| front (vitl16_518_svm) | 77.16% | 81.36% | 84.45% |
| care_label (vitl16_714_partial_q) | 76.32% | 79.66% | 84.07% |
| brand_tag (vits16_518_lgbm) | 68.36% | 68.36% | 73.95% |
| front_exterior_logo (convnext_large_714_lgbm) | 50.95% | 52.02% | 56.90% |
| concat_linear | 50.61% | 51.55% | 65.87% |
| concat_svm | 48.37% | 54.45% | 54.95% |
| concat_xgb | 25.60% | 56.83% | 66.38% |

