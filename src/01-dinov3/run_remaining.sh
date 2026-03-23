#!/bin/bash
# Resume: run only the remaining work across 4 GPUs.
# Skips already-completed embeddings, linear probes, and finetune checkpoints.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="/root/.venv/bin/python -u"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

REGIONS=("care_label" "front" "front_exterior_logo" "brand_tag")
GPUS=(0 1 2 3)
FINETUNE_MODELS=("vitb16" "vitl16")
FINETUNE_RESOLUTIONS=(518 714)

run_remaining() {
    local region=$1
    local gpu=$2
    local log="$LOG_DIR/${region}_resume.log"

    (
        export CUDA_VISIBLE_DEVICES=$gpu
        echo "============================================================"
        echo "RESUME: $region  |  GPU $gpu  |  $(date)"
        echo "============================================================"

        # ML sweep (will overwrite existing but that's fine -- fast for tree models)
        echo ""
        echo ">>> ML classifier sweep"
        $PYTHON train_svm_xgb_lgbm_catboost.py --region "$region" --sweep

        # Missing finetunes
        for model in "${FINETUNE_MODELS[@]}"; do
            for res in "${FINETUNE_RESOLUTIONS[@]}"; do
                ckpt="checkpoints/${region}/${model}_${res}_finetune_best.pt"
                if [ ! -f "$ckpt" ]; then
                    echo ""
                    echo ">>> Finetune: $model @ $res (missing)"
                    $PYTHON train_with_unfreeze_dino.py \
                        --region "$region" --model "$model" --resolution "$res" \
                        --epochs 30 --batch-size 4 --patience 7
                else
                    echo ">>> Finetune: $model @ $res -- already done, skipping"
                fi
            done
        done

        # Finetuned ML for all completed finetunes
        for model in "${FINETUNE_MODELS[@]}"; do
            for res in "${FINETUNE_RESOLUTIONS[@]}"; do
                ckpt="checkpoints/${region}/${model}_${res}_finetune_best.pt"
                if [ -f "$ckpt" ]; then
                    echo ""
                    echo ">>> Finetuned ML: $model @ $res"
                    $PYTHON train_with_svm_xgb_lgbm_catboost_with_unfrozen_trained_dino.py \
                        --region "$region" --model "$model" --resolution "$res" \
                        --batch-size 16
                fi
            done
        done

        echo ""
        echo "============================================================"
        echo "RESUME DONE: $region  |  $(date)"
        echo "============================================================"
    ) > "$log" 2>&1
}

echo "Resuming remaining work for ${#REGIONS[@]} regions"
echo ""

PIDS=()
for i in "${!REGIONS[@]}"; do
    gpu_idx=$((i % ${#GPUS[@]}))
    run_remaining "${REGIONS[$i]}" "${GPUS[$gpu_idx]}" &
    PIDS+=($!)
    echo "[GPU ${GPUS[$gpu_idx]}] ${REGIONS[$i]} -> PID $!"
done

echo ""
echo "Waiting for all to finish..."

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[OK] ${REGIONS[$i]}"
    else
        echo "[FAILED] ${REGIONS[$i]}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All done. $(date)"
else
    echo "$FAILED region(s) failed. Check $LOG_DIR/*_resume.log. $(date)"
fi
