#!/bin/bash
# Fully parallel execution respecting GPU memory:
#
# Per region/GPU:
#   - Job 1 (GPU, serial): xgb sweep → catboost sweep → lgbm sweep → missing finetunes → finetuned-ML
#   - Job 2 (CPU, parallel): svm sweep (runs alongside Job 1 since SVM is CPU-only)
#
# 4 regions x 2 jobs = 8 total processes, 4 GPUs fully utilized.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="/root/.venv/bin/python -u"
LOG_DIR="./logs/parallel"
mkdir -p "$LOG_DIR"

REGIONS=("care_label" "front" "front_exterior_logo" "brand_tag")
GPUS=(0 1 2 3)
FINETUNE_MODELS=("vitb16" "vitl16")
FINETUNE_RESOLUTIONS=(518 714)

PIDS=()
LABELS=()

launch() {
    PIDS+=($!)
    LABELS+=("$1")
    echo "  [$!] $1"
}

echo "=== Launching parallel jobs ==="
echo ""

for i in "${!REGIONS[@]}"; do
    region="${REGIONS[$i]}"
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"

    echo "--- $region (GPU $gpu) ---"

    # Job 1: GPU work serialized (xgb → catboost → lgbm → finetune → finetuned-ML)
    (
        export CUDA_VISIBLE_DEVICES=$gpu

        echo ">>> GPU classifiers: XGBoost sweep"
        $PYTHON train_svm_xgb_lgbm_catboost.py --region "$region" --sweep --classifiers xgb

        echo ">>> GPU classifiers: CatBoost sweep"
        $PYTHON train_svm_xgb_lgbm_catboost.py --region "$region" --sweep --classifiers catboost

        echo ">>> GPU classifiers: LightGBM sweep"
        $PYTHON train_svm_xgb_lgbm_catboost.py --region "$region" --sweep --classifiers lgbm

        # Missing finetunes
        for model in "${FINETUNE_MODELS[@]}"; do
            for res in "${FINETUNE_RESOLUTIONS[@]}"; do
                ckpt="checkpoints/${region}/${model}_${res}_finetune_best.pt"
                if [ ! -f "$ckpt" ]; then
                    echo ">>> Finetune: $model @ $res"
                    $PYTHON train_with_unfreeze_dino.py \
                        --region "$region" --model "$model" --resolution "$res" \
                        --epochs 30 --batch-size 4 --patience 7
                else
                    echo ">>> Finetune: $model @ $res -- already done"
                fi
            done
        done

        # Finetuned-ML for all completed finetunes
        for model in "${FINETUNE_MODELS[@]}"; do
            for res in "${FINETUNE_RESOLUTIONS[@]}"; do
                ckpt="checkpoints/${region}/${model}_${res}_finetune_best.pt"
                if [ -f "$ckpt" ]; then
                    echo ">>> Finetuned ML: $model @ $res"
                    $PYTHON train_with_svm_xgb_lgbm_catboost_with_unfrozen_trained_dino.py \
                        --region "$region" --model "$model" --resolution "$res" \
                        --batch-size 16
                fi
            done
        done

        echo ">>> GPU JOBS DONE: $region  $(date)"
    ) > "$LOG_DIR/${region}_gpu.log" 2>&1 &
    launch "${region}/gpu"

    # Job 2: SVM sweep (CPU-only, runs in parallel with GPU jobs)
    (
        $PYTHON train_svm_xgb_lgbm_catboost.py --region "$region" --sweep --classifiers svm
        echo ">>> SVM DONE: $region  $(date)"
    ) > "$LOG_DIR/${region}_svm.log" 2>&1 &
    launch "${region}/svm"

    echo ""
done

echo ""
echo "Total jobs: ${#PIDS[@]} (4 GPU + 4 SVM)"
echo "Waiting for all..."
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}" 2>/dev/null; then
        echo "[OK]     ${LABELS[$i]}"
    else
        echo "[FAILED] ${LABELS[$i]}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "ALL DONE. $(date)"
else
    echo "$FAILED job(s) failed. Check $LOG_DIR/. $(date)"
fi
