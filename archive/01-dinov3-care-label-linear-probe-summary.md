# DINOv3 Care Label Authentication — Experiment Summary

**Date:** 2026-03-17
**Task:** Binary classification — authentic vs not-authentic care labels
**Dataset:** `apparel_supreme_until_dec_2025_front` (front camera images)

---

## Data


| Stat            | Value                |
| --------------- | -------------------- |
| Total images    | 7,294                |
| Train           | 5,835 (192 positive) |
| Val             | 1,459 (48 positive)  |
| Split           | 80/20 stratified     |
| Class imbalance | 29.4:1 (neg:pos)     |
| `pos_weight`    | 29.4                 |


---

## 1. Linear Probe (Frozen Backbone → Trained Head)

50 epochs, early stopping (patience 7), best checkpoint by AUC-ROC.

### AUC-ROC & AUC-PR (sorted by AUC-ROC)


| Model              | AUC-ROC    | AUC-PR |
| ------------------ | ---------- | ------ |
| convnext_small_518 | **0.7586** | 0.1473 |
| vitb16_518         | 0.7558     | 0.1224 |
| vitb16_714         | 0.7555     | 0.1229 |
| vits16_714         | 0.7514     | 0.0909 |
| vitl16_714         | 0.7482     | 0.1026 |
| convnext_small_714 | 0.7470     | 0.1039 |
| convnext_base_518  | 0.7434     | 0.0937 |
| convnext_large_518 | 0.7409     | 0.0988 |
| vits16_518         | 0.7391     | 0.1705 |
| vitl16_518         | 0.7320     | 0.1166 |
| convnext_tiny_518  | 0.7292     | 0.0915 |
| convnext_large_714 | 0.7271     | 0.1217 |
| convnext_base_714  | 0.7247     | 0.0770 |
| convnext_tiny_714  | 0.7218     | 0.1076 |


### TPR at Target FPR (best checkpoint)


| Model              | AUC-ROC | [TPR@0.5](mailto:TPR@0.5)% | TPR@1%     | TPR@2%     | TPR@5%     | TPR@10%    |
| ------------------ | ------- | -------------------------- | ---------- | ---------- | ---------- | ---------- |
| convnext_small_518 | 0.7586  | 0.0833                     | 0.0833     | 0.1042     | 0.2500     | **0.4583** |
| vitb16_518         | 0.7558  | 0.0833                     | 0.1042     | 0.1250     | 0.1875     | 0.2708     |
| vitb16_714         | 0.7555  | **0.1250**                 | **0.1458** | 0.1667     | 0.2083     | 0.2708     |
| vits16_714         | 0.7514  | 0.0208                     | 0.0833     | 0.0833     | 0.1875     | 0.3333     |
| vitl16_714         | 0.7482  | 0.0417                     | 0.0625     | 0.1458     | 0.2292     | 0.3542     |
| convnext_small_714 | 0.7470  | 0.0417                     | 0.0625     | 0.1250     | 0.2708     | 0.3542     |
| convnext_base_518  | 0.7434  | 0.0625                     | 0.0833     | 0.1042     | 0.1458     | 0.3125     |
| convnext_large_518 | 0.7409  | 0.0208                     | 0.0625     | 0.1250     | 0.2708     | 0.3542     |
| vits16_518         | 0.7391  | 0.1042                     | 0.1667     | **0.2083** | **0.3333** | 0.3542     |
| vitl16_518         | 0.7320  | 0.0833                     | 0.1250     | 0.1667     | 0.2500     | 0.3125     |
| convnext_tiny_518  | 0.7292  | 0.0417                     | 0.0625     | 0.0833     | 0.2292     | 0.2917     |
| convnext_large_714 | 0.7271  | 0.0833                     | 0.1250     | 0.1667     | 0.2917     | 0.3750     |
| convnext_base_714  | 0.7247  | 0.0417                     | 0.0625     | 0.0833     | 0.1458     | 0.2708     |
| convnext_tiny_714  | 0.7218  | 0.0417                     | 0.0625     | 0.1042     | 0.1667     | 0.3125     |


---

## 2. ML Classifiers on Same Embeddings (2026-03-17)

Trained XGBoost (GPU), SVM (RBF), CatBoost, and LightGBM on the same cached DINOv3/ConvNeXt embeddings. Same 80/20 stratified split.

### Best ML Classifier per Embedding (sorted by best AUC-ROC)


| Embedding          | Best ML  | AUC-ROC    | AUC-PR | TPR@2% | TPR@5% | TPR@10% |
| ------------------ | -------- | ---------- | ------ | ------ | ------ | ------- |
| vitb16_518         | SVM      | **0.8493** | 0.3162 | 0.2708 | 0.5208 | 0.6042  |
| vits16_518         | SVM      | 0.8488     | 0.3126 | 0.2500 | 0.5000 | 0.6458  |
| vitl16_714         | SVM      | 0.8351     | 0.2908 | 0.2708 | 0.3750 | 0.5208  |
| vitb16_518         | XGB      | 0.8153     | 0.2598 | 0.2708 | 0.3542 | 0.5625  |
| vits16_714         | SVM      | 0.8064     | 0.2689 | 0.2292 | 0.3542 | 0.5208  |
| vitb16_714         | SVM      | 0.8063     | 0.2847 | 0.2500 | 0.2917 | 0.5625  |
| vitb16_714         | LGBM     | 0.8048     | 0.2175 | 0.1875 | 0.3333 | 0.4583  |
| vitl16_714         | LGBM     | 0.7981     | 0.2297 | 0.2292 | 0.3750 | 0.5000  |
| vitb16_714         | XGB      | 0.7985     | 0.2445 | 0.2708 | 0.3542 | 0.5208  |
| vitb16_518         | LGBM     | 0.7936     | 0.2117 | 0.1667 | 0.2500 | 0.5417  |
| convnext_large_518 | LGBM     | 0.7747     | 0.1552 | 0.1875 | 0.2500 | 0.3958  |
| vitl16_714         | CatBoost | 0.7754     | 0.1593 | 0.1875 | 0.3125 | 0.4375  |
| vits16_518         | XGB      | 0.7732     | 0.1999 | 0.1667 | 0.3125 | 0.4583  |
| vitl16_714         | XGB      | 0.7700     | 0.2380 | 0.2292 | 0.3750 | 0.4375  |


### Full ML Results Table (all 14 embeddings × 4 classifiers, best AUC-ROC per embedding)


| Embedding          | XGB        | SVM        | CatBoost   | LGBM       | Linear Probe |
| ------------------ | ---------- | ---------- | ---------- | ---------- | ------------ |
| vitb16_518         | 0.8153     | **0.8493** | 0.7549     | 0.7936     | 0.7558       |
| vitb16_714         | 0.7985     | **0.8063** | 0.7729     | 0.8048     | 0.7555       |
| vits16_518         | 0.7732     | **0.8488** | 0.6778     | 0.7467     | 0.7391       |
| vits16_714         | 0.7726     | **0.8064** | 0.7530     | 0.7637     | 0.7514       |
| vitl16_518         | 0.7565     | 0.7518     | **0.7628** | 0.7175     | 0.7320       |
| vitl16_714         | 0.7700     | **0.8351** | 0.7754     | 0.7981     | 0.7482       |
| convnext_base_518  | 0.6981     | 0.7026     | **0.7174** | 0.7068     | 0.7434       |
| convnext_base_714  | 0.7145     | **0.7255** | 0.6975     | 0.6968     | 0.7247       |
| convnext_large_518 | 0.7504     | 0.7367     | 0.7256     | **0.7747** | 0.7409       |
| convnext_large_714 | 0.7355     | **0.7388** | 0.7284     | 0.7315     | 0.7271       |
| convnext_small_518 | **0.7082** | 0.7002     | 0.7137     | 0.6909     | 0.7586       |
| convnext_small_714 | 0.7113     | 0.6954     | **0.7243** | 0.6877     | 0.7470       |
| convnext_tiny_518  | 0.7354     | 0.7029     | 0.7123     | **0.7567** | 0.7292       |
| convnext_tiny_714  | **0.7377** | 0.7112     | 0.7143     | 0.6845     | 0.7218       |


---

## 3. Fine-tuning (Backbone Unfrozen)

30 epochs, 5-epoch warmup (head only), then full backbone unfreeze. `lr=1e-3` (head), `lr_backbone=1e-5`, batch size 4, mixed precision.

### Results (best checkpoint)


| Model      | AUC-ROC    | AUC-PR     | [TPR@0.5](mailto:TPR@0.5)% | TPR@1%     | TPR@2%     | TPR@5%     | TPR@10%    |
| ---------- | ---------- | ---------- | -------------------------- | ---------- | ---------- | ---------- | ---------- |
| vitl16_518 | **0.8928** | 0.3478     | 0.2292                     | 0.3125     | 0.3333     | 0.5000     | 0.6458     |
| vitl16_714 | 0.8806     | **0.3719** | 0.2292                     | 0.2917     | **0.3958** | **0.5625** | **0.6250** |
| vitb16_518 | 0.8130     | 0.2655     | **0.2292**                 | **0.2708** | 0.3125     | 0.3333     | 0.4792     |
| vitb16_714 | 0.8015     | 0.1532     | 0.0833                     | 0.1042     | 0.1667     | 0.2708     | 0.4375     |


**Notes:**

- vitb16_518 finetune collapsed at epoch 8 after unfreeze (AUC dropped to 0.46) but recovered partially; best checkpoint was epoch 7 (AUC 0.8130).
- vitb16_714 finetune never recovered after unfreeze collapse; best checkpoint was epoch 5 frozen-only (AUC 0.8015).
- vitl16 models benefited most from fine-tuning — vitl16_518 reached 0.8928 AUC-ROC (vs 0.7320 linear probe).

---

## Key Takeaways

1. **Linear probe AUC-ROC range is narrow** (0.7218–0.7586) across all 14 backbone+resolution combos, suggesting the pre-trained features carry similar discriminative signal for this task.
2. **ML classifiers significantly outperform linear probes on ViT embeddings.** SVM on vitb16_518 embeddings achieves 0.8493 AUC-ROC (vs 0.7558 linear probe) — a +9.4pp jump. ConvNeXt embeddings see smaller or no gains from ML classifiers.
3. **Fine-tuning gives the best absolute results.** vitl16_518 finetune hits 0.8928 AUC-ROC — the best in the entire experiment. However, training is unstable (backbone unfreeze can cause collapse) and slow.
4. **Best overall at each operating point:**


| FPR budget | Best method                                                     | TPR   |
| ---------- | --------------------------------------------------------------- | ----- |
| 0.5%       | vitb16_518 finetune / vitl16_518 finetune / vitl16_714 finetune | 22.9% |
| 1%         | vitl16_518 finetune                                             | 31.3% |
| 2%         | vitl16_714 finetune                                             | 39.6% |
| 5%         | vitl16_714 finetune                                             | 56.3% |
| 10%        | vits16_518 SVM                                                  | 64.6% |


1. **Caveat:** Validation set has only 48 positives — each positive is worth ~2.1% TPR. All TPR figures have high granularity noise.
2. **fp16 overflow bug:** Early vitl16 cached features had Inf values from fp16 overflow. Fixed by re-extracting in fp32.


---

## 4. Test Set Evaluation — Jan-Feb 2026

**TPR** = % of authentic items correctly passed (true authentics)
**FPR** = % of fakes that slipped through (missed fakes)

Read as: "TPR @ X% FPR" = at a threshold where X% of fakes are missed, what % of authentic items pass correctly.
Computed from the **test ROC curve** directly.

**Dataset:** `apparel_supreme_jan_to_feb_2026_care_label/test/` — 962 images (929 authentic, 33 fake)

### All Models (sorted by AUC-ROC)

| Model | AUC | TPR @ 0.5% missed | TPR @ 1% missed | TPR @ 2% missed | TPR @ 5% missed | TPR @ 10% missed |
|-------|-----|-------------------|----------------|----------------|----------------|-----------------|
| finetuned_vitl16_714_svm | 0.9118 | 0.00% | 0.00% | 59.53% | 67.38% | 71.58% |
| finetuned_vitl16_518_svm | 0.9060 | 0.00% | 0.00% | 37.67% | 67.28% | 72.66% |
| vitl16_518_finetune | 0.9033 | 27.23% | 27.23% | 48.98% | 57.80% | 68.35% |
| finetuned_vitl16_714_lgbm | 0.9022 | 25.40% | 25.40% | 54.90% | 70.94% | 72.23% |
| vitl16_714_finetune | 0.8983 | 0.00% | 0.00% | 47.58% | 58.67% | 69.64% |
| finetuned_vitb16_518_svm | 0.8937 | 39.18% | 39.18% | 42.73% | 56.73% | 66.63% |
| vitb16_518_svm | 0.8937 | 39.18% | 39.18% | 42.73% | 56.73% | 66.63% |
| vits16_518_svm | 0.8892 | 32.83% | 32.83% | 48.44% | 57.91% | 64.48% |
| vitl16_518_svm | 0.8856 | 21.31% | 21.31% | 34.02% | 59.85% | 60.28% |
| finetuned_vitl16_714_catboost | 0.8848 | 2.80% | 2.80% | 52.21% | 57.27% | 70.18% |
| finetuned_vitl16_518_lgbm | 0.8800 | 3.98% | 3.98% | 27.45% | 30.46% | 58.56% |
| vits16_518_catboost | 0.8660 | 45.21% | 45.21% | 49.19% | 55.01% | 58.23% |
| finetuned_vitl16_518_catboost | 0.8653 | 4.95% | 4.95% | 7.53% | 38.75% | 58.88% |
| vitl16_714_svm | 0.8649 | 32.29% | 32.29% | 40.58% | 47.36% | 49.62% |
| finetuned_vitb16_518_catboost | 0.8639 | 35.41% | 35.41% | 39.07% | 42.95% | 51.24% |
| vitb16_518_catboost | 0.8639 | 35.41% | 35.41% | 39.07% | 42.95% | 51.24% |
| vits16_714_svm | 0.8625 | 35.84% | 35.84% | 48.98% | 53.71% | 55.33% |
| vits16_518_lgbm | 0.8544 | 28.85% | 28.85% | 36.92% | 56.62% | 58.67% |
| finetuned_vitb16_518_lgbm | 0.8537 | 30.68% | 30.68% | 38.64% | 52.21% | 52.64% |
| vitb16_518_lgbm | 0.8537 | 30.68% | 30.68% | 38.64% | 52.10% | 52.64% |
| vits16_714_lgbm | 0.8536 | 28.85% | 28.85% | 44.78% | 45.53% | 61.36% |
| finetuned_vitb16_714_svm | 0.8516 | 22.93% | 22.93% | 46.50% | 49.62% | 61.14% |
| vitb16_714_svm | 0.8516 | 22.93% | 22.93% | 46.50% | 49.62% | 61.14% |
| vitl16_518_lgbm | 0.8487 | 25.51% | 25.51% | 48.44% | 49.62% | 51.78% |
| vitb16_518_finetune | 0.8365 | 47.36% | 47.36% | 48.44% | 49.73% | 53.39% |
| vitl16_714_lgbm | 0.8334 | 8.93% | 8.93% | 31.75% | 32.72% | 47.36% |
| vitb16_714_lgbm | 0.8292 | 11.30% | 11.30% | 16.04% | 36.81% | 41.98% |
| vitb16_714_finetune | 0.8259 | 24.54% | 24.54% | 29.92% | 33.26% | 34.45% |
| finetuned_vitb16_714_lgbm | 0.8249 | 7.00% | 7.00% | 17.76% | 50.05% | 52.53% |
| vitl16_714_linear | 0.8249 | 31.54% | 31.54% | 32.29% | 35.41% | 47.79% |
| vitl16_518_linear | 0.8194 | 25.62% | 25.62% | 34.34% | 36.06% | 39.72% |
| convnext_base_714_lgbm | 0.8165 | 24.97% | 24.97% | 31.65% | 32.62% | 36.49% |
| vitb16_518_linear | 0.8155 | 27.45% | 27.45% | 40.58% | 48.01% | 53.50% |
| convnext_large_714_lgbm | 0.8144 | 10.98% | 10.98% | 15.29% | 26.05% | 42.73% |
| vits16_714_linear | 0.8135 | 31.32% | 31.32% | 44.35% | 54.25% | 56.30% |
| convnext_base_518_linear | 0.8118 | 25.30% | 25.30% | 54.79% | 55.01% | 55.87% |
| vits16_518_linear | 0.8101 | 22.07% | 22.07% | 22.71% | 41.01% | 55.97% |
| convnext_base_518_lgbm | 0.8094 | 17.55% | 17.55% | 34.88% | 36.06% | 44.46% |
| vitb16_714_linear | 0.8084 | 20.45% | 20.45% | 36.06% | 37.24% | 48.12% |
| convnext_base_518_svm | 0.8034 | 24.11% | 24.11% | 31.32% | 35.52% | 42.52% |
| convnext_base_714_svm | 0.7971 | 12.92% | 12.92% | 17.87% | 38.64% | 44.35% |
| convnext_tiny_518_svm | 0.7948 | 23.25% | 23.25% | 26.16% | 32.83% | 33.91% |
| convnext_tiny_714_svm | 0.7934 | 7.21% | 7.21% | 20.67% | 25.62% | 35.95% |
| convnext_base_714_catboost | 0.7920 | 34.12% | 34.12% | 36.38% | 37.24% | 39.94% |
| convnext_large_714_svm | 0.7886 | 18.08% | 18.08% | 27.66% | 28.53% | 34.34% |
| vitl16_714_catboost | 0.7859 | 7.86% | 7.86% | 11.52% | 14.96% | 40.90% |
| convnext_large_518_svm | 0.7847 | 9.90% | 9.90% | 11.52% | 20.34% | 46.50% |
| convnext_small_518_svm | 0.7820 | 5.38% | 5.38% | 10.55% | 21.42% | 31.11% |
| convnext_small_518_catboost | 0.7813 | 8.40% | 8.40% | 30.57% | 37.57% | 43.70% |
| convnext_small_714_lgbm | 0.7794 | 9.58% | 9.58% | 23.90% | 25.30% | 37.89% |
| convnext_large_518_lgbm | 0.7786 | 12.49% | 12.49% | 29.28% | 29.49% | 33.91% |
| convnext_large_714_catboost | 0.7737 | 18.73% | 18.73% | 30.36% | 34.77% | 34.77% |
| convnext_small_714_svm | 0.7709 | 1.61% | 1.61% | 5.60% | 14.42% | 32.40% |
| convnext_base_518_catboost | 0.7702 | 7.32% | 7.32% | 23.47% | 38.11% | 38.64% |
| convnext_tiny_518_linear | 0.7679 | 14.75% | 14.75% | 42.41% | 45.53% | 49.09% |
| vitl16_518_catboost | 0.7659 | 3.23% | 3.23% | 8.18% | 15.18% | 17.65% |
| vits16_714_catboost | 0.7647 | 10.55% | 10.55% | 13.99% | 21.10% | 22.93% |
| convnext_tiny_518_lgbm | 0.7625 | 9.69% | 9.69% | 10.55% | 11.30% | 30.14% |
| convnext_base_714_linear | 0.7582 | 33.58% | 33.58% | 38.75% | 39.29% | 44.67% |
| finetuned_vitb16_714_catboost | 0.7525 | 8.83% | 8.83% | 36.17% | 36.49% | 36.81% |
| vitb16_714_catboost | 0.7525 | 8.83% | 8.83% | 36.17% | 36.49% | 36.81% |
| convnext_tiny_714_lgbm | 0.7507 | 14.53% | 14.53% | 20.02% | 27.66% | 36.17% |
| convnext_tiny_714_catboost | 0.7504 | 5.49% | 5.49% | 12.06% | 32.19% | 35.52% |
| convnext_small_518_linear | 0.7434 | 22.17% | 22.17% | 23.79% | 39.61% | 41.66% |
| convnext_large_518_linear | 0.7417 | 12.49% | 12.49% | 22.50% | 37.14% | 40.47% |
| convnext_small_714_catboost | 0.7330 | 14.85% | 14.85% | 16.15% | 29.82% | 29.92% |
| convnext_large_714_linear | 0.7320 | 21.10% | 21.10% | 26.26% | 34.23% | 39.07% |
| convnext_small_518_lgbm | 0.7308 | 9.90% | 9.90% | 13.99% | 18.08% | 29.06% |
| convnext_tiny_714_linear | 0.7234 | 30.14% | 30.14% | 34.12% | 37.24% | 43.81% |
| convnext_small_714_linear | 0.7067 | 20.99% | 20.99% | 24.54% | 34.55% | 44.03% |
| convnext_large_518_catboost | 0.7007 | 7.10% | 7.10% | 18.41% | 29.60% | 30.46% |
| convnext_tiny_518_catboost | 0.6879 | 2.37% | 2.37% | 6.03% | 15.72% | 16.36% |

### Best at Each Operating Point (test ROC)

| Missed fakes | Best Method | True authentics |
|-------------|------------|----------------|
| 0.5% | vitb16_518_finetune | 47.36% |
| 1% | vitb16_518_finetune | 47.36% |
| 2% | finetuned_vitl16_714_svm | 59.53% |
| 5% | finetuned_vitl16_714_lgbm | 70.94% |
| 10% | finetuned_vitl16_518_svm | 72.66% |

