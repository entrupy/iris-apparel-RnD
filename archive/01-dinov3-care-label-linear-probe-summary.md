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

