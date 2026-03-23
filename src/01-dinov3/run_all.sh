#!/bin/bash
# Run the full pipeline for all 4 regions in parallel, one region per GPU.
# Within each region, two branches run in parallel on the same GPU:
#
#   Branch A: precompute → linear_head → ml_classifiers
#   Branch B: finetune   → finetune_ml
#
# Dependency graph:
#   precompute ──→ linear_head ──→ ml_classifiers
#   finetune   ──→ finetune_ml       (no dependency on precompute)
#
# GPU assignment:
#   GPU 0: care_label
#   GPU 1: front
#   GPU 2: front_exterior_logo
#   GPU 3: brand_tag

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/root/.venv/bin/python -u"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

REGIONS=("care_label" "front" "front_exterior_logo" "brand_tag")
GPUS=(0 1 2 3)

FINETUNE_MODELS=("vitb16" "vitl16")
FINETUNE_RESOLUTIONS=(518 714)

if [ $# -gt 0 ]; then
    REGIONS=("$@")
fi

# Step 0: pre-create val splits for all regions (avoids race conditions)
echo ">>> Step 0: Creating val splits for all regions..."
for region in "${REGIONS[@]}"; do
    $PYTHON -c "
from config import load_metadata, get_or_create_val_split
records = load_metadata('$region', split='train')
get_or_create_val_split('$region', records)
"
done
echo ""

# Branch A: precompute → linear_head → ml_classifiers
run_branch_a() {
    local region=$1
    local gpu=$2
    local log="$LOG_DIR/${region}_branch_a.log"

    (
        export CUDA_VISIBLE_DEVICES=$gpu

        echo "============================================================"
        echo "BRANCH A: $region  |  GPU $gpu  |  $(date)"
        echo "============================================================"

        echo ""
        echo ">>> A1: Precompute embeddings (fp32)"
        echo "============================================================"
        $PYTHON precompute_embeddings.py --regions "$region" --batch-size 16

        echo ""
        echo ">>> A2: Train linear probes (sweep)"
        echo "============================================================"
        $PYTHON train_linear_head.py --region "$region" --sweep

        echo ""
        echo ">>> A3: Train ML classifiers (sweep)"
        echo "============================================================"
        $PYTHON train_svm_xgb_lgbm_catboost.py --region "$region" --sweep

        echo ""
        echo "============================================================"
        echo "BRANCH A DONE: $region  |  $(date)"
        echo "============================================================"
    ) > "$log" 2>&1
}

# Branch B: finetune → finetune_ml
run_branch_b() {
    local region=$1
    local gpu=$2
    local log="$LOG_DIR/${region}_branch_b.log"

    (
        export CUDA_VISIBLE_DEVICES=$gpu

        echo "============================================================"
        echo "BRANCH B: $region  |  GPU $gpu  |  $(date)"
        echo "============================================================"

        echo ""
        echo ">>> B1: Fine-tune DINOv3 (vitb16 + vitl16 x 518 + 714)"
        echo "============================================================"
        for model in "${FINETUNE_MODELS[@]}"; do
            for res in "${FINETUNE_RESOLUTIONS[@]}"; do
                echo ""
                echo "--- Finetune: $model @ $res ---"
                $PYTHON train_with_unfreeze_dino.py \
                    --region "$region" --model "$model" --resolution "$res" \
                    --epochs 30 --batch-size 4 --patience 7
            done
        done

        echo ""
        echo ">>> B2: ML on finetuned backbone features"
        echo "============================================================"
        for model in "${FINETUNE_MODELS[@]}"; do
            for res in "${FINETUNE_RESOLUTIONS[@]}"; do
                echo ""
                echo "--- Finetuned ML: $model @ $res ---"
                $PYTHON train_with_svm_xgb_lgbm_catboost_with_unfrozen_trained_dino.py \
                    --region "$region" --model "$model" --resolution "$res" \
                    --batch-size 16
            done
        done

        echo ""
        echo "============================================================"
        echo "BRANCH B DONE: $region  |  $(date)"
        echo "============================================================"
    ) > "$log" 2>&1
}

run_region() {
    local region=$1
    local gpu=$2

    echo "[GPU $gpu] Starting $region (branch A + B in parallel)"

    run_branch_a "$region" "$gpu" &
    local pid_a=$!

    run_branch_b "$region" "$gpu" &
    local pid_b=$!

    local failed=0
    if wait $pid_a; then
        echo "[GPU $gpu] $region branch A (precompute→linear→ML) DONE"
    else
        echo "[GPU $gpu] $region branch A FAILED"
        failed=1
    fi

    if wait $pid_b; then
        echo "[GPU $gpu] $region branch B (finetune→finetune_ML) DONE"
    else
        echo "[GPU $gpu] $region branch B FAILED"
        failed=1
    fi

    return $failed
}

echo "Starting pipeline: ${#REGIONS[@]} regions x 2 branches = $((${#REGIONS[@]} * 2)) parallel jobs"
echo "Logs in $LOG_DIR/"
echo ""

PIDS=()
for i in "${!REGIONS[@]}"; do
    gpu_idx=$((i % ${#GPUS[@]}))
    run_region "${REGIONS[$i]}" "${GPUS[$gpu_idx]}" &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} region jobs (each with 2 branches): PIDs ${PIDS[*]}"
echo "Waiting for all to finish..."
echo ""

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
    echo "All regions completed successfully. $(date)"
else
    echo "$FAILED region(s) failed. Check logs in $LOG_DIR/. $(date)"
fi
