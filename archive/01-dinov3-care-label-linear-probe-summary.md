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

**Dataset:** `apparel_supreme_jan_to_feb_2026_care_label/test/`


| Stat                     | Value  |
| ------------------------ | ------ |
| Images with labels       | 962    |
| Not-authentic (positive) | 33     |
| Authentic (negative)     | 929    |
| Imbalance ratio          | 28.2:1 |


### Linear Probes (sorted by AUC-ROC)


| Model              | AUC-ROC | AUC-PR | [TPR@0.5](mailto:TPR@0.5)% | TPR@1% | TPR@2% | TPR@5% | TPR@10% |
| ------------------ | ------- | ------ | -------------------------- | ------ | ------ | ------ | ------- |
| vitl16_714         | 0.8232  | 0.2773 | 0.1515                     | 0.2121 | 0.3333 | 0.3939 | 0.5758  |
| vitl16_518         | 0.8205  | 0.3137 | 0.2424                     | 0.2727 | 0.3030 | 0.4242 | 0.6667  |
| vits16_714         | 0.8162  | 0.1442 | 0.0606                     | 0.0606 | 0.1212 | 0.3939 | 0.4545  |
| vitb16_518         | 0.8161  | 0.1693 | 0.0909                     | 0.1212 | 0.1212 | 0.2727 | 0.4848  |
| convnext_base_518  | 0.8126  | 0.1482 | 0.0303                     | 0.1212 | 0.1515 | 0.3030 | 0.4242  |
| vits16_518         | 0.8117  | 0.1637 | 0.0606                     | 0.0909 | 0.2121 | 0.3939 | 0.4848  |
| vitb16_714         | 0.8093  | 0.1533 | 0.0303                     | 0.0606 | 0.1818 | 0.3333 | 0.4848  |
| convnext_base_714  | 0.7572  | 0.1280 | 0.0909                     | 0.1212 | 0.1818 | 0.2727 | 0.3030  |
| convnext_small_518 | 0.7427  | 0.1317 | 0.0909                     | 0.1515 | 0.1818 | 0.2121 | 0.3030  |
| convnext_large_518 | 0.7421  | 0.1495 | 0.0909                     | 0.1515 | 0.2121 | 0.3030 | 0.4242  |
| convnext_large_714 | 0.7326  | 0.1116 | 0.0303                     | 0.0303 | 0.1212 | 0.3030 | 0.4545  |
| convnext_tiny_518  | 0.7242  | 0.0806 | 0.0606                     | 0.0606 | 0.0909 | 0.0909 | 0.2727  |
| convnext_tiny_714  | 0.7211  | 0.0661 | 0.0000                     | 0.0303 | 0.0303 | 0.1212 | 0.2121  |
| convnext_small_714 | 0.6926  | 0.0613 | 0.0000                     | 0.0303 | 0.0303 | 0.0909 | 0.1818  |


### Fine-tuned Models


| Model               | AUC-ROC | AUC-PR | [TPR@0.5](mailto:TPR@0.5)% | TPR@1% | TPR@2% | TPR@5% | TPR@10% |
| ------------------- | ------- | ------ | -------------------------- | ------ | ------ | ------ | ------- |
| vitl16_714_finetune | 0.8874  | 0.3705 | 0.3333                     | 0.3636 | 0.3939 | 0.5758 | 0.7273  |
| vitb16_518_finetune | 0.8609  | 0.2002 | 0.0303                     | 0.1818 | 0.2727 | 0.4848 | 0.6364  |
| vitl16_518_finetune | 0.8563  | 0.4298 | 0.2727                     | 0.3333 | 0.4242 | 0.5758 | 0.6364  |
| vitb16_714_finetune | 0.8264  | 0.2074 | 0.0606                     | 0.0909 | 0.2424 | 0.4242 | 0.6364  |


### ML Classifiers — Best per Embedding (sorted by AUC-ROC)


| Embedding          | Best ML  | AUC-ROC | AUC-PR | TPR@2% | TPR@5% | TPR@10% |
| ------------------ | -------- | ------- | ------ | ------ | ------ | ------- |
| vitb16_518         | SVM      | 0.8935  | 0.5037 | 0.4545 | 0.5758 | 0.6667  |
| vits16_518         | SVM      | 0.8883  | 0.4579 | 0.4545 | 0.5758 | 0.6667  |
| vitl16_518         | SVM      | 0.8864  | 0.4993 | 0.4848 | 0.5758 | 0.6970  |
| vits16_714         | LGBM     | 0.8685  | 0.4309 | 0.4545 | 0.6061 | 0.7273  |
| vitl16_714         | SVM      | 0.8653  | 0.4692 | 0.4848 | 0.6061 | 0.6667  |
| vitb16_714         | SVM      | 0.8510  | 0.4814 | 0.4545 | 0.5152 | 0.5758  |
| convnext_base_714  | CATBOOST | 0.8171  | 0.3488 | 0.3030 | 0.3939 | 0.4545  |
| convnext_large_714 | LGBM     | 0.8113  | 0.4545 | 0.4545 | 0.5758 | 0.6061  |
| convnext_base_518  | SVM      | 0.8054  | 0.2234 | 0.2424 | 0.3636 | 0.4848  |
| convnext_tiny_518  | SVM      | 0.7926  | 0.3658 | 0.3636 | 0.4848 | 0.5758  |
| convnext_large_518 | LGBM     | 0.7915  | 0.3969 | 0.4242 | 0.5152 | 0.5758  |
| convnext_tiny_714  | SVM      | 0.7915  | 0.3453 | 0.3636 | 0.4545 | 0.5152  |
| convnext_small_518 | SVM      | 0.7819  | 0.3053 | 0.3030 | 0.4242 | 0.5758  |
| convnext_small_714 | CATBOOST | 0.7775  | 0.2835 | 0.3333 | 0.4242 | 0.5152  |


### Fixed-Threshold Evaluation (val-set thresholds on test data)

Thresholds calibrated on the validation set are applied unchanged to test scores.
This shows realistic deployment performance including probability shift.
Grouped by FPR target, sorted by test TPR descending.

| Val FPR | Model | Val TPR | Test FPR | Test TPR |
|---------|-------|---------|----------|----------|
| 0.5% | vitl16_714_svm | 20.83% | 1.08% | 39.39% |
| 0.5% | vitb16_518_svm | 18.75% | 0.54% | 36.36% |
| 0.5% | vitl16_518_svm | 16.67% | 0.65% | 36.36% |
| 0.5% | vits16_518_lgbm | 12.50% | 1.29% | 36.36% |
| 0.5% | vits16_714_svm | 16.67% | 0.32% | 36.36% |
| 0.5% | vitb16_714_catboost | 14.58% | 0.22% | 33.33% |
| 0.5% | vitb16_714_lgbm | 14.58% | 0.43% | 33.33% |
| 0.5% | vitb16_714_svm | 20.83% | 0.32% | 33.33% |
| 0.5% | vits16_714_catboost | 8.33% | 0.54% | 33.33% |
| 0.5% | vitl16_714_finetune | 22.92% | 0.43% | 30.30% |
| 0.5% | vits16_518_catboost | 14.58% | 1.08% | 30.30% |
| 0.5% | vits16_518_svm | 18.75% | 1.08% | 30.30% |
| 0.5% | convnext_large_714_svm | 12.50% | 0.11% | 27.27% |
| 0.5% | convnext_tiny_518_lgbm | 10.42% | 0.75% | 27.27% |
| 0.5% | vitl16_518_finetune | 22.92% | 0.54% | 27.27% |
| 0.5% | vitl16_518_lgbm | 8.33% | 0.11% | 27.27% |
| 0.5% | vitl16_714_catboost | 8.33% | 0.32% | 27.27% |
| 0.5% | vitl16_714_lgbm | 10.42% | 0.43% | 27.27% |
| 0.5% | vits16_714_lgbm | 16.67% | 0.43% | 27.27% |
| 0.5% | convnext_tiny_518_svm | 14.58% | 0.54% | 24.24% |
| 0.5% | vitb16_518_lgbm | 12.50% | 0.65% | 24.24% |
| 0.5% | convnext_large_518_lgbm | 10.42% | 0.11% | 21.21% |
| 0.5% | convnext_large_714_catboost | 12.50% | 0.22% | 21.21% |
| 0.5% | convnext_large_714_lgbm | 6.25% | 0.00% | 21.21% |
| 0.5% | vitb16_518_catboost | 12.50% | 0.22% | 21.21% |
| 0.5% | convnext_base_714_catboost | 10.42% | 0.00% | 18.18% |
| 0.5% | convnext_base_714_lgbm | 8.33% | 0.00% | 18.18% |
| 0.5% | convnext_large_518_svm | 6.25% | 0.11% | 18.18% |
| 0.5% | convnext_small_518_lgbm | 10.42% | 0.11% | 18.18% |
| 0.5% | convnext_small_714_svm | 12.50% | 0.11% | 18.18% |
| 0.5% | convnext_tiny_714_svm | 12.50% | 0.32% | 18.18% |
| 0.5% | vitb16_518_finetune | 22.92% | 1.18% | 18.18% |
| 0.5% | convnext_base_518_catboost | 10.42% | 0.65% | 15.15% |
| 0.5% | convnext_base_518_lgbm | 10.42% | 0.43% | 15.15% |
| 0.5% | convnext_small_518_svm | 10.42% | 0.22% | 15.15% |
| 0.5% | convnext_small_714_lgbm | 8.33% | 0.22% | 15.15% |
| 0.5% | convnext_base_518_svm | 10.42% | 0.54% | 12.12% |
| 0.5% | convnext_small_518_catboost | 12.50% | 0.00% | 12.12% |
| 0.5% | convnext_tiny_714_lgbm | 12.50% | 0.43% | 12.12% |
| 0.5% | vitl16_714 (linear) | 4.17% | 0.32% | 12.12% |
| 0.5% | vitb16_518 (linear) | 8.33% | 0.65% | 9.09% |
| 0.5% | vitl16_518 (linear) | 8.33% | 0.00% | 9.09% |
| 0.5% | convnext_base_714_svm | 10.42% | 0.00% | 6.06% |
| 0.5% | convnext_large_518_catboost | 8.33% | 0.00% | 6.06% |
| 0.5% | convnext_small_714_catboost | 8.33% | 0.11% | 6.06% |
| 0.5% | convnext_tiny_518 (linear) | 4.17% | 0.75% | 6.06% |
| 0.5% | vitl16_518_catboost | 4.17% | 0.75% | 6.06% |
| 0.5% | vits16_518 (linear) | 10.42% | 0.97% | 6.06% |
| 0.5% | convnext_large_518 (linear) | 2.08% | 0.32% | 3.03% |
| 0.5% | convnext_small_518 (linear) | 8.33% | 0.11% | 3.03% |
| 0.5% | convnext_tiny_518_catboost | 6.25% | 0.22% | 3.03% |
| 0.5% | vitb16_714 (linear) | 12.50% | 0.75% | 3.03% |
| 0.5% | vitb16_714_finetune | 8.33% | 0.32% | 3.03% |
| 0.5% | vits16_714 (linear) | 2.08% | 0.43% | 3.03% |
| 0.5% | convnext_base_518 (linear) | 6.25% | 0.22% | 0.00% |
| 0.5% | convnext_base_714 (linear) | 4.17% | 0.11% | 0.00% |
| 0.5% | convnext_large_714 (linear) | 8.33% | 0.22% | 0.00% |
| 0.5% | convnext_small_714 (linear) | 4.17% | 0.65% | 0.00% |
| 0.5% | convnext_tiny_714 (linear) | 4.17% | 0.00% | 0.00% |
| 0.5% | convnext_tiny_714_catboost | 4.17% | 0.00% | 0.00% |
| 1% | vits16_714_svm | 18.75% | 1.18% | 45.45% |
| 1% | vitb16_518_svm | 20.83% | 0.86% | 42.42% |
| 1% | vitl16_518_svm | 18.75% | 1.29% | 42.42% |
| 1% | vitl16_714_svm | 25.00% | 1.29% | 42.42% |
| 1% | vits16_518_svm | 22.92% | 1.51% | 42.42% |
| 1% | vits16_714_lgbm | 18.75% | 1.40% | 42.42% |
| 1% | vitb16_714_svm | 22.92% | 0.54% | 39.39% |
| 1% | vitb16_714_catboost | 25.00% | 0.86% | 36.36% |
| 1% | vits16_518_lgbm | 16.67% | 1.61% | 36.36% |
| 1% | vitb16_714_lgbm | 14.58% | 0.43% | 33.33% |
| 1% | vitl16_518_finetune | 31.25% | 1.29% | 33.33% |
| 1% | vitl16_518_lgbm | 12.50% | 0.43% | 33.33% |
| 1% | vitl16_714_catboost | 12.50% | 0.65% | 33.33% |
| 1% | vitl16_714_finetune | 29.17% | 0.86% | 33.33% |
| 1% | vits16_714_catboost | 12.50% | 0.65% | 33.33% |
| 1% | convnext_large_714_catboost | 14.58% | 0.22% | 30.30% |
| 1% | convnext_large_714_svm | 14.58% | 0.32% | 30.30% |
| 1% | vitb16_518_catboost | 16.67% | 1.72% | 30.30% |
| 1% | vits16_518_catboost | 16.67% | 1.18% | 30.30% |
| 1% | convnext_base_714_lgbm | 10.42% | 0.65% | 27.27% |
| 1% | convnext_tiny_518_lgbm | 12.50% | 0.86% | 27.27% |
| 1% | convnext_tiny_714_svm | 14.58% | 0.54% | 27.27% |
| 1% | vitl16_714_lgbm | 16.67% | 0.54% | 27.27% |
| 1% | convnext_base_714_catboost | 12.50% | 0.43% | 24.24% |
| 1% | convnext_large_518_catboost | 12.50% | 0.54% | 24.24% |
| 1% | convnext_large_518_svm | 10.42% | 0.43% | 24.24% |
| 1% | convnext_tiny_518_svm | 16.67% | 0.65% | 24.24% |
| 1% | vitb16_518_lgbm | 12.50% | 0.65% | 24.24% |
| 1% | vitl16_518 (linear) | 12.50% | 0.43% | 24.24% |
| 1% | convnext_large_518_lgbm | 14.58% | 0.32% | 21.21% |
| 1% | convnext_large_714_lgbm | 8.33% | 0.00% | 21.21% |
| 1% | convnext_small_518_svm | 14.58% | 0.43% | 21.21% |
| 1% | convnext_small_714_catboost | 10.42% | 0.65% | 21.21% |
| 1% | convnext_base_518_catboost | 12.50% | 1.18% | 18.18% |
| 1% | convnext_base_518_lgbm | 18.75% | 0.86% | 18.18% |
| 1% | convnext_base_714_svm | 12.50% | 0.54% | 18.18% |
| 1% | convnext_small_518_lgbm | 12.50% | 0.43% | 18.18% |
| 1% | convnext_small_714_lgbm | 10.42% | 0.32% | 18.18% |
| 1% | convnext_small_714_svm | 14.58% | 0.43% | 18.18% |
| 1% | vitb16_518_finetune | 27.08% | 1.18% | 18.18% |
| 1% | convnext_small_518_catboost | 14.58% | 0.54% | 15.15% |
| 1% | convnext_base_518_svm | 10.42% | 0.54% | 12.12% |
| 1% | convnext_tiny_714_lgbm | 12.50% | 0.43% | 12.12% |
| 1% | vitl16_714 (linear) | 6.25% | 0.43% | 12.12% |
| 1% | convnext_tiny_714_catboost | 8.33% | 0.97% | 9.09% |
| 1% | vitb16_518 (linear) | 10.42% | 0.75% | 9.09% |
| 1% | vitb16_714_finetune | 10.42% | 1.18% | 9.09% |
| 1% | vits16_518 (linear) | 16.67% | 1.08% | 9.09% |
| 1% | convnext_large_518 (linear) | 6.25% | 0.54% | 6.06% |
| 1% | convnext_tiny_518 (linear) | 6.25% | 1.40% | 6.06% |
| 1% | convnext_tiny_518_catboost | 16.67% | 0.54% | 6.06% |
| 1% | vitl16_518_catboost | 8.33% | 0.97% | 6.06% |
| 1% | vits16_714 (linear) | 8.33% | 0.86% | 6.06% |
| 1% | convnext_base_518 (linear) | 8.33% | 0.65% | 3.03% |
| 1% | convnext_large_714 (linear) | 12.50% | 0.22% | 3.03% |
| 1% | convnext_small_518 (linear) | 8.33% | 0.11% | 3.03% |
| 1% | vitb16_714 (linear) | 14.58% | 0.86% | 3.03% |
| 1% | convnext_base_714 (linear) | 6.25% | 0.22% | 0.00% |
| 1% | convnext_small_714 (linear) | 6.25% | 0.75% | 0.00% |
| 1% | convnext_tiny_714 (linear) | 6.25% | 0.11% | 0.00% |
| 2% | vitl16_714_svm | 27.08% | 2.91% | 48.48% |
| 2% | vits16_714_svm | 22.92% | 1.29% | 48.48% |
| 2% | vitb16_714_lgbm | 18.75% | 1.40% | 45.45% |
| 2% | vitl16_518_svm | 25.00% | 1.72% | 45.45% |
| 2% | vitb16_518_svm | 27.08% | 1.29% | 42.42% |
| 2% | vitb16_714_svm | 25.00% | 1.94% | 42.42% |
| 2% | vitl16_518_finetune | 33.33% | 2.05% | 42.42% |
| 2% | vitl16_714_lgbm | 22.92% | 2.05% | 42.42% |
| 2% | vits16_518_svm | 25.00% | 1.94% | 42.42% |
| 2% | vits16_714_lgbm | 20.83% | 2.26% | 42.42% |
| 2% | vitb16_714_catboost | 29.17% | 2.05% | 39.39% |
| 2% | vitl16_518_lgbm | 20.83% | 1.29% | 39.39% |
| 2% | vitl16_714_finetune | 39.58% | 2.15% | 39.39% |
| 2% | vits16_518_catboost | 18.75% | 1.61% | 39.39% |
| 2% | vits16_518_lgbm | 20.83% | 2.58% | 36.36% |
| 2% | vits16_714_catboost | 20.83% | 1.18% | 36.36% |
| 2% | convnext_large_714_lgbm | 12.50% | 0.75% | 33.33% |
| 2% | convnext_tiny_518_svm | 18.75% | 2.05% | 33.33% |
| 2% | convnext_tiny_714_svm | 16.67% | 1.18% | 33.33% |
| 2% | vitl16_714_catboost | 18.75% | 1.51% | 33.33% |
| 2% | convnext_large_714_catboost | 18.75% | 0.22% | 30.30% |
| 2% | convnext_large_714_svm | 16.67% | 0.65% | 30.30% |
| 2% | convnext_tiny_518_lgbm | 18.75% | 2.26% | 30.30% |
| 2% | vitb16_518_catboost | 25.00% | 2.58% | 30.30% |
| 2% | vitb16_518_lgbm | 16.67% | 0.97% | 30.30% |
| 2% | vitl16_714 (linear) | 14.58% | 1.29% | 30.30% |
| 2% | convnext_base_714_catboost | 14.58% | 0.97% | 27.27% |
| 2% | convnext_base_714_lgbm | 12.50% | 0.86% | 27.27% |
| 2% | convnext_base_714_svm | 18.75% | 1.72% | 27.27% |
| 2% | convnext_large_518_lgbm | 18.75% | 0.86% | 27.27% |
| 2% | convnext_large_518_svm | 16.67% | 1.08% | 27.27% |
| 2% | convnext_small_518_lgbm | 14.58% | 0.97% | 27.27% |
| 2% | vitl16_518 (linear) | 16.67% | 0.75% | 27.27% |
| 2% | convnext_large_518_catboost | 16.67% | 1.18% | 24.24% |
| 2% | convnext_tiny_714_lgbm | 14.58% | 1.18% | 24.24% |
| 2% | vitb16_518_finetune | 31.25% | 2.05% | 24.24% |
| 2% | convnext_small_518_svm | 18.75% | 0.86% | 21.21% |
| 2% | convnext_small_714_catboost | 14.58% | 1.18% | 21.21% |
| 2% | convnext_small_714_lgbm | 12.50% | 0.86% | 21.21% |
| 2% | convnext_small_714_svm | 16.67% | 1.08% | 21.21% |
| 2% | convnext_base_518_catboost | 18.75% | 2.05% | 18.18% |
| 2% | convnext_base_518_lgbm | 18.75% | 0.86% | 18.18% |
| 2% | convnext_base_518_svm | 14.58% | 1.18% | 18.18% |
| 2% | convnext_large_518 (linear) | 12.50% | 1.18% | 18.18% |
| 2% | vitb16_714_finetune | 16.67% | 1.83% | 18.18% |
| 2% | convnext_small_518 (linear) | 10.42% | 1.29% | 15.15% |
| 2% | convnext_small_518_catboost | 14.58% | 0.54% | 15.15% |
| 2% | vitb16_714 (linear) | 16.67% | 2.15% | 15.15% |
| 2% | convnext_base_518 (linear) | 10.42% | 1.29% | 12.12% |
| 2% | convnext_tiny_714_catboost | 12.50% | 1.83% | 12.12% |
| 2% | vitb16_518 (linear) | 12.50% | 1.94% | 12.12% |
| 2% | vitl16_518_catboost | 16.67% | 1.51% | 12.12% |
| 2% | vits16_518 (linear) | 20.83% | 1.18% | 12.12% |
| 2% | convnext_tiny_518 (linear) | 8.33% | 1.94% | 6.06% |
| 2% | convnext_tiny_518_catboost | 16.67% | 0.54% | 6.06% |
| 2% | vits16_714 (linear) | 8.33% | 0.86% | 6.06% |
| 2% | convnext_base_714 (linear) | 8.33% | 0.32% | 3.03% |
| 2% | convnext_large_714 (linear) | 16.67% | 1.29% | 3.03% |
| 2% | convnext_small_714 (linear) | 12.50% | 1.61% | 0.00% |
| 2% | convnext_tiny_714 (linear) | 10.42% | 1.51% | 0.00% |
| 5% | vitl16_714_svm | 37.50% | 5.27% | 60.61% |
| 5% | vitl16_518_finetune | 50.00% | 4.84% | 57.58% |
| 5% | vitl16_714_lgbm | 37.50% | 4.95% | 57.58% |
| 5% | vits16_518_svm | 50.00% | 4.74% | 57.58% |
| 5% | vits16_714_lgbm | 35.42% | 4.41% | 57.58% |
| 5% | vitl16_518_lgbm | 29.17% | 4.41% | 54.55% |
| 5% | vitl16_518_svm | 35.42% | 4.20% | 54.55% |
| 5% | convnext_large_714_lgbm | 25.00% | 2.80% | 51.52% |
| 5% | vitb16_518_lgbm | 25.00% | 3.98% | 51.52% |
| 5% | vitb16_518_svm | 52.08% | 4.31% | 51.52% |
| 5% | vitb16_714_catboost | 35.42% | 4.95% | 51.52% |
| 5% | vitb16_714_svm | 29.17% | 4.84% | 51.52% |
| 5% | vitl16_714_catboost | 31.25% | 4.63% | 51.52% |
| 5% | vitl16_714_finetune | 56.25% | 4.84% | 51.52% |
| 5% | vits16_518_catboost | 25.00% | 4.84% | 51.52% |
| 5% | vits16_714_svm | 35.42% | 4.84% | 51.52% |
| 5% | vitb16_714_lgbm | 33.33% | 4.09% | 48.48% |
| 5% | convnext_base_714_lgbm | 20.83% | 4.52% | 45.45% |
| 5% | convnext_large_518_lgbm | 25.00% | 2.80% | 45.45% |
| 5% | convnext_tiny_518_svm | 25.00% | 4.20% | 45.45% |
| 5% | vits16_518_lgbm | 35.42% | 4.95% | 45.45% |
| 5% | vits16_714_catboost | 31.25% | 4.95% | 45.45% |
| 5% | convnext_large_518_catboost | 27.08% | 3.55% | 42.42% |
| 5% | convnext_small_518_lgbm | 20.83% | 3.23% | 42.42% |
| 5% | convnext_base_518_lgbm | 27.08% | 4.52% | 39.39% |
| 5% | convnext_base_714_svm | 25.00% | 3.55% | 39.39% |
| 5% | convnext_small_714_catboost | 25.00% | 3.88% | 39.39% |
| 5% | convnext_small_714_svm | 29.17% | 4.31% | 39.39% |
| 5% | convnext_tiny_518_catboost | 20.83% | 4.41% | 39.39% |
| 5% | convnext_tiny_518_lgbm | 25.00% | 4.41% | 39.39% |
| 5% | vitb16_714_finetune | 27.08% | 4.41% | 39.39% |
| 5% | convnext_large_714_catboost | 27.08% | 1.94% | 36.36% |
| 5% | convnext_large_714_svm | 25.00% | 2.91% | 36.36% |
| 5% | convnext_tiny_714_svm | 22.92% | 2.37% | 36.36% |
| 5% | convnext_small_518_svm | 27.08% | 4.63% | 33.33% |
| 5% | vitb16_518_finetune | 33.33% | 3.01% | 33.33% |
| 5% | vitl16_714 (linear) | 22.92% | 3.01% | 33.33% |
| 5% | vits16_518 (linear) | 33.33% | 3.12% | 33.33% |
| 5% | convnext_base_518_svm | 22.92% | 3.44% | 30.30% |
| 5% | convnext_base_714_catboost | 20.83% | 3.55% | 30.30% |
| 5% | convnext_large_518 (linear) | 27.08% | 3.34% | 30.30% |
| 5% | convnext_tiny_714_lgbm | 27.08% | 4.84% | 30.30% |
| 5% | vitb16_518_catboost | 33.33% | 5.17% | 30.30% |
| 5% | convnext_base_518 (linear) | 14.58% | 2.80% | 27.27% |
| 5% | convnext_base_518_catboost | 25.00% | 5.17% | 27.27% |
| 5% | convnext_large_518_svm | 22.92% | 2.05% | 27.27% |
| 5% | vitl16_518 (linear) | 25.00% | 2.26% | 27.27% |
| 5% | convnext_small_518_catboost | 18.75% | 3.44% | 24.24% |
| 5% | convnext_small_714_lgbm | 14.58% | 1.72% | 24.24% |
| 5% | vitb16_714 (linear) | 20.83% | 3.55% | 24.24% |
| 5% | vitl16_518_catboost | 29.17% | 2.58% | 24.24% |
| 5% | convnext_small_518 (linear) | 25.00% | 3.01% | 18.18% |
| 5% | vitb16_518 (linear) | 18.75% | 3.55% | 18.18% |
| 5% | vits16_714 (linear) | 18.75% | 2.37% | 18.18% |
| 5% | convnext_base_714 (linear) | 14.58% | 1.40% | 15.15% |
| 5% | convnext_tiny_714_catboost | 25.00% | 4.09% | 15.15% |
| 5% | convnext_large_714 (linear) | 29.17% | 3.77% | 12.12% |
| 5% | convnext_small_714 (linear) | 27.08% | 3.55% | 9.09% |
| 5% | convnext_tiny_518 (linear) | 22.92% | 6.14% | 9.09% |
| 5% | convnext_tiny_714 (linear) | 16.67% | 3.55% | 3.03% |
| 10% | vitl16_518_svm | 43.75% | 9.90% | 69.70% |
| 10% | vitl16_714_finetune | 62.50% | 8.83% | 69.70% |
| 10% | vitl16_714_svm | 52.08% | 10.98% | 66.67% |
| 10% | vits16_714_lgbm | 43.75% | 9.04% | 66.67% |
| 10% | vitb16_518_svm | 60.42% | 9.04% | 63.64% |
| 10% | vitl16_714_catboost | 43.75% | 9.47% | 63.64% |
| 10% | vits16_518_svm | 64.58% | 10.23% | 63.64% |
| 10% | vitb16_518_lgbm | 54.17% | 8.29% | 60.61% |
| 10% | vitl16_518_finetune | 64.58% | 8.93% | 60.61% |
| 10% | vitl16_518_lgbm | 35.42% | 7.64% | 60.61% |
| 10% | vits16_714_svm | 52.08% | 11.84% | 60.61% |
| 10% | vitb16_518_catboost | 43.75% | 8.50% | 57.58% |
| 10% | vitb16_714_catboost | 47.92% | 11.63% | 57.58% |
| 10% | vitb16_714_svm | 56.25% | 9.58% | 57.58% |
| 10% | vitl16_714_lgbm | 50.00% | 9.36% | 57.58% |
| 10% | vits16_518_catboost | 31.25% | 6.46% | 57.58% |
| 10% | convnext_large_714_lgbm | 35.42% | 6.03% | 54.55% |
| 10% | convnext_small_714_svm | 37.50% | 7.00% | 54.55% |
| 10% | vitb16_714_finetune | 43.75% | 8.50% | 54.55% |
| 10% | vitb16_714_lgbm | 45.83% | 9.26% | 54.55% |
| 10% | vits16_518_lgbm | 41.67% | 10.98% | 54.55% |
| 10% | convnext_base_518_lgbm | 33.33% | 9.80% | 51.52% |
| 10% | convnext_large_518_lgbm | 39.58% | 8.40% | 51.52% |
| 10% | convnext_tiny_518_svm | 33.33% | 8.40% | 51.52% |
| 10% | vitb16_518_finetune | 47.92% | 6.57% | 51.52% |
| 10% | vits16_714_catboost | 41.67% | 9.26% | 51.52% |
| 10% | convnext_large_518_catboost | 39.58% | 7.10% | 48.48% |
| 10% | convnext_large_714_catboost | 39.58% | 6.67% | 48.48% |
| 10% | convnext_small_518_lgbm | 33.33% | 7.00% | 48.48% |
| 10% | convnext_small_518_svm | 33.33% | 7.43% | 48.48% |
| 10% | convnext_tiny_518_catboost | 27.08% | 9.69% | 48.48% |
| 10% | convnext_base_518_svm | 37.50% | 9.04% | 45.45% |
| 10% | convnext_base_714_lgbm | 27.08% | 7.21% | 45.45% |
| 10% | convnext_base_714_svm | 37.50% | 6.03% | 45.45% |
| 10% | convnext_small_714_catboost | 33.33% | 7.97% | 45.45% |
| 10% | convnext_tiny_518_lgbm | 37.50% | 10.66% | 45.45% |
| 10% | convnext_tiny_714_svm | 31.25% | 7.64% | 45.45% |
| 10% | vitb16_714 (linear) | 27.08% | 9.04% | 45.45% |
| 10% | convnext_large_714 (linear) | 37.50% | 8.07% | 42.42% |
| 10% | vitb16_518 (linear) | 27.08% | 8.50% | 42.42% |
| 10% | vitl16_518 (linear) | 31.25% | 5.60% | 42.42% |
| 10% | vitl16_714 (linear) | 35.42% | 6.35% | 42.42% |
| 10% | convnext_base_714_catboost | 33.33% | 7.86% | 39.39% |
| 10% | convnext_large_518_svm | 33.33% | 7.97% | 39.39% |
| 10% | convnext_large_714_svm | 33.33% | 7.21% | 39.39% |
| 10% | convnext_small_714_lgbm | 31.25% | 5.60% | 39.39% |
| 10% | convnext_tiny_714_lgbm | 35.42% | 8.07% | 39.39% |
| 10% | vits16_518 (linear) | 35.42% | 5.17% | 39.39% |
| 10% | convnext_base_518 (linear) | 31.25% | 7.21% | 36.36% |
| 10% | convnext_base_518_catboost | 41.67% | 10.23% | 36.36% |
| 10% | convnext_large_518 (linear) | 35.42% | 7.75% | 33.33% |
| 10% | convnext_small_518_catboost | 31.25% | 7.97% | 33.33% |
| 10% | vitl16_518_catboost | 45.83% | 5.06% | 30.30% |
| 10% | vits16_714 (linear) | 33.33% | 4.63% | 30.30% |
| 10% | convnext_base_714 (linear) | 27.08% | 4.63% | 27.27% |
| 10% | convnext_small_518 (linear) | 45.83% | 7.21% | 24.24% |
| 10% | convnext_tiny_518 (linear) | 29.17% | 8.72% | 21.21% |
| 10% | convnext_tiny_714_catboost | 31.25% | 7.97% | 21.21% |
| 10% | convnext_tiny_714 (linear) | 31.25% | 6.89% | 18.18% |
| 10% | convnext_small_714 (linear) | 35.42% | 7.53% | 9.09% |


### Best Fixed-Threshold TPR at Each FPR Target


| Val FPR Target | Best Method         | Val TPR | Test FPR | Test TPR |
| -------------- | ------------------- | ------- | -------- | -------- |
| 0.5%           | vitl16_714_svm      | 20.83%  | 1.08%    | 39.39%   |
| 1%             | vits16_714_svm      | 18.75%  | 1.18%    | 45.45%   |
| 2%             | vits16_714_svm      | 22.92%  | 1.29%    | 48.48%   |
| 5%             | vitl16_714_svm      | 37.50%  | 5.27%    | 60.61%   |
| 10%            | vitl16_714_finetune | 62.50%  | 8.83%    | 69.70%   |


### Best Overall at Each FPR (test ROC curve, for reference)


| FPR Budget | Best Method         | Val TPR | Test TPR |
| ---------- | ------------------- | ------- | -------- |
| 0.5%       | vits16_714_svm      | 16.67%  | 39.4%    |
| 1%         | vits16_714_svm      | 18.75%  | 45.5%    |
| 2%         | vits16_714_svm      | 22.92%  | 51.5%    |
| 5%         | vits16_714_lgbm     | 35.42%  | 60.6%    |
| 10%        | vitl16_714_finetune | 62.50%  | 72.7%    |


