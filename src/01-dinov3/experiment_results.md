# Experiment Results — DINOv3 Authentication (Test Set)

> **Convention:** Val thresholds applied to test. TPR = fake catch rate, FPR = false alarm on authentics.

---

## Per-Region — Frozen DINOv3 (vitl16)

### care_label

| Model | Test AUC | Fixed FPR@2% | Fixed TPR@2% | Fixed FPR@5% | Fixed TPR@5% | Fixed FPR@10% | Fixed TPR@10% |
|---|---|---|---|---|---|---|---|
| vitl16_518_linear | 0.8111 | 0.54% | 6.06% | 2.80% | 18.18% | 8.60% | 42.42% |
| vitl16_714_linear | 0.8212 | 1.40% | 9.09% | 3.55% | 21.21% | 7.20% | 39.39% |
| vitl16_518_svm | 0.8911 | 2.15% | 45.45% | 4.73% | 54.55% | 10.32% | 69.70% |
| vitl16_714_svm | 0.8712 | 2.05% | 33.33% | 3.98% | 48.48% | 10.32% | 69.70% |
| vitl16_518_xgb | — | — | — | — | — | — | — |
| vitl16_714_xgb | — | — | — | — | — | — | — |
| vitl16_518_catboost | 0.7989 | 2.15% | 18.18% | 4.41% | 33.33% | 8.71% | 48.48% |
| vitl16_714_catboost | 0.7477 | 1.40% | 15.15% | 4.09% | 33.33% | 8.17% | 42.42% |
| vitl16_518_lgbm | 0.8130 | 1.61% | 15.15% | 4.52% | 24.24% | 8.60% | 42.42% |
| vitl16_714_lgbm | 0.7635 | 2.48% | 21.21% | 5.59% | 33.33% | 8.60% | 42.42% |
| vitl16_518_finetune | 0.9031 | 2.69% | 57.58% | 5.38% | 63.64% | 11.61% | 78.79% |
| vitl16_714_finetune | 0.9012 | 1.94% | 54.55% | 5.48% | 63.64% | 11.40% | 75.76% |

### front

| Model | Test AUC | Fixed FPR@2% | Fixed TPR@2% | Fixed FPR@5% | Fixed TPR@5% | Fixed FPR@10% | Fixed TPR@10% |
|---|---|---|---|---|---|---|---|
| vitl16_518_linear | 0.8859 | 2.87% | 41.46% | 5.31% | 56.10% | 9.58% | 63.41% |
| vitl16_714_linear | 0.8713 | 2.87% | 46.34% | 5.16% | 53.66% | 9.65% | 60.98% |
| vitl16_518_svm | 0.9246 | 0.96% | 41.46% | 3.91% | 60.98% | 9.51% | 75.61% |
| vitl16_714_svm | 0.9155 | 1.55% | 46.34% | 4.57% | 60.98% | 9.51% | 73.17% |
| vitl16_518_catboost | 0.8357 | 1.25% | 21.95% | 4.27% | 53.66% | 9.28% | 65.85% |
| vitl16_714_catboost | 0.8639 | 1.55% | 34.15% | 4.72% | 46.34% | 9.36% | 65.85% |
| vitl16_518_lgbm | 0.9000 | 1.40% | 46.34% | 3.98% | 56.10% | 8.99% | 68.29% |
| vitl16_714_lgbm | 0.8713 | 1.77% | 43.90% | 4.20% | 58.54% | 8.55% | 68.29% |
| vitl16_518_finetune | 0.8747 | 1.40% | 31.71% | 5.01% | 53.66% | 9.95% | 68.29% |
| vitl16_714_finetune | 0.8771 | 1.84% | 41.46% | 4.72% | 60.98% | 9.51% | 73.17% |

### front_exterior_logo

| Model | Test AUC | Fixed FPR@2% | Fixed TPR@2% | Fixed FPR@5% | Fixed TPR@5% | Fixed FPR@10% | Fixed TPR@10% |
|---|---|---|---|---|---|---|---|
| vitl16_518_linear | 0.8155 | 1.43% | 25.71% | 4.52% | 34.29% | 8.57% | 51.43% |
| vitl16_714_linear | 0.8030 | 2.26% | 31.43% | 4.76% | 34.29% | 9.52% | 42.86% |
| vitl16_518_svm | 0.8334 | 1.07% | 28.57% | 4.17% | 48.57% | 9.29% | 51.43% |
| vitl16_714_svm | 0.8361 | 0.71% | 25.71% | 4.17% | 42.86% | 9.29% | 54.29% |
| vitl16_518_catboost | 0.8171 | 1.07% | 31.43% | 4.29% | 42.86% | 8.81% | 62.86% |
| vitl16_714_catboost | 0.8277 | 1.19% | 28.57% | 4.52% | 51.43% | 8.81% | 60.00% |
| vitl16_518_lgbm | 0.8081 | 2.74% | 37.14% | 5.48% | 51.43% | 9.52% | 57.14% |
| vitl16_714_lgbm | 0.8145 | 0.71% | 20.00% | 4.52% | 51.43% | 8.57% | 57.14% |
| vitl16_518_finetune | 0.8148 | 2.26% | 31.43% | 4.76% | 45.71% | 9.76% | 60.00% |
| vitl16_714_finetune | 0.7845 | 1.67% | 25.71% | 4.52% | 45.71% | 9.29% | 57.14% |

### brand_tag

| Model | Test AUC | Fixed FPR@2% | Fixed TPR@2% | Fixed FPR@5% | Fixed TPR@5% | Fixed FPR@10% | Fixed TPR@10% |
|---|---|---|---|---|---|---|---|
| vitl16_518_linear | 0.8889 | 2.06% | 44.83% | 5.35% | 58.62% | 9.98% | 68.97% |
| vitl16_714_linear | 0.8782 | 2.24% | 51.72% | 4.61% | 55.17% | 10.23% | 72.41% |
| vitl16_518_svm | 0.7905 | 2.32% | 34.48% | 5.52% | 44.83% | 10.32% | 58.62% |
| vitl16_714_svm | 0.8334 | 2.84% | 41.38% | 5.52% | 41.38% | 10.57% | 55.17% |
| vitl16_518_catboost | 0.7486 | 2.75% | 31.03% | 5.35% | 37.93% | 10.49% | 44.83% |
| vitl16_714_catboost | 0.7981 | 2.49% | 20.69% | 5.18% | 34.48% | 9.72% | 37.93% |
| vitl16_518_lgbm | 0.8235 | 2.67% | 41.38% | 5.10% | 58.62% | 9.89% | 62.07% |
| vitl16_714_lgbm | 0.8636 | 1.29% | 48.28% | 4.70% | 55.17% | 10.15% | 58.62% |
| vitl16_518_finetune | 0.8978 | 2.49% | 55.17% | 6.10% | 58.62% | 11.78% | 65.52% |
| vitl16_714_finetune | 0.8789 | 2.92% | 44.83% | 5.35% | 58.62% | 10.49% | 68.97% |

---

## Region Fusion (concatenated vitl16_714 embeddings, auth-positive test)

> **Auth-positive:** TPR = authentic pass rate, FPR = fake miss rate

| Method | Test AUC | TPR@0.5% | TPR@1% | TPR@2% | TPR@5% | TPR@10% |
|---|---|---|---|---|---|---|
| fusion_svm | 0.8890 | 48.73% | 48.73% | 58.13% | 63.56% | 64.64% |
| fusion_xgb | 0.8489 | 32.68% | 32.68% | 51.63% | 52.35% | 52.86% |
| fusion_mlp | 0.8795 | 21.62% | 21.62% | 41.72% | 57.19% | 65.58% |
| fusion_attention | 0.7829 | 25.02% | 25.02% | 34.13% | 35.00% | 41.43% |

### Reading the table above

FPR targets are in auth-positive convention: **% of fakes missed**.
At 2% fakes missed, fusion\_svm lets 58.13% of authentics through.
At 10% fakes missed, fusion\_mlp lets 65.58% of authentics through.

---

## Voting (per-region best models, val thresholds on test)

### Best model per region

| Region | Best Model | Test TPR@2% | Test AUC |
|---|---|---|---|
| care_label | finetuned_vitl16_714_svm | 60.61% | 0.9118 |
| front | vitl16_518_svm | 51.22% | 0.9246 |
| front_exterior_logo | finetuned_vitl16_518_svm | 45.71% | 0.8068 |
| brand_tag | vitl16_518_finetune | 55.17% | 0.8978 |

### Voting strategies (fixed-threshold)

| Strategy | FPR@2% | TPR@2% | FPR@5% | TPR@5% | FPR@10% | TPR@10% |
|---|---|---|---|---|---|---|
| Any-one-agree (OR) | 5.35% | 82.93% | 13.30% | 82.93% | 23.79% | 87.80% |
| Majority vote | 1.08% | 56.10% | 4.34% | 65.85% | 9.26% | 80.49% |
| All-must-agree (AND) | 0.07% | 24.39% | 0.43% | 39.02% | 1.01% | 43.90% |

### Score fusion ROC (continuous)

| Strategy | Fusion | AUC | TPR@2% | TPR@5% | TPR@10% |
|---|---|---|---|---|---|
| Any-one-agree | max-score | 0.9442 | 63.41% | 78.05% | 85.37% |
| Majority | mean-score | 0.9487 | 68.29% | 80.49% | 82.93% |
| All-must-agree | min-score | 0.8787 | 41.46% | 43.90% | 56.10% |
