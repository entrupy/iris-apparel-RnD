#!/bin/bash
# Full experiment pipeline:
#   Phase 1: Frozen DINOv3 -> linear head + ML classifiers (vitl16 x 518,714 x 4 regions)
#   Phase 2: Partial finetune last4 -> ML on finetuned features (all 4 regions)
#   Phase 3: Region fusion (attention, MLP, XGB, SVM) + voting
#   Phase 4: Test evaluation with auth-positive TPR/FPR
#
# Usage:
#   bash run_pipeline.sh              # run everything
#   bash run_pipeline.sh --phase 2    # run from phase 2 onwards
#   bash run_pipeline.sh --phase 3    # run from phase 3 onwards

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/root/.venv/bin/python -u"
LOG_DIR="$SCRIPT_DIR/logs/pipeline"
mkdir -p "$LOG_DIR"

REGIONS=("care_label" "front" "front_exterior_logo" "brand_tag")

START_PHASE=1
if [[ "${1:-}" == "--phase" ]]; then
    START_PHASE=${2:-1}
fi

# ---------------------------------------------------------------------------
# Phase 0: Ensure global val split exists
# ---------------------------------------------------------------------------
echo ">>> Phase 0: Ensuring global val split..."
$PYTHON -c "
from config import create_global_val_split, print_val_split_distributions
create_global_val_split()
print_val_split_distributions()
"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: Frozen backbone — precompute + linear head + ML classifiers
# ---------------------------------------------------------------------------
if [ "$START_PHASE" -le 1 ]; then
    echo ">>> Phase 1: Frozen DINOv3 classifiers"
    echo "============================================================"

    echo "  1a. Precompute embeddings (vitl16 x 518,714)..."
    $PYTHON precompute_embeddings.py \
        --models vitl16 --resolutions 518 714 \
        --regions "${REGIONS[@]}" --batch-size 16 \
        2>&1 | tee "$LOG_DIR/phase1_precompute.log"

    for region in "${REGIONS[@]}"; do
        for res in 518 714; do
            echo ""
            echo "  1b. Linear head: $region / vitl16@$res"
            $PYTHON train_linear_head.py --region "$region" --model vitl16 --resolution "$res" \
                2>&1 | tee -a "$LOG_DIR/phase1_linear_${region}.log"

            echo ""
            echo "  1c. ML classifiers: $region / vitl16@$res"
            $PYTHON train_svm_xgb_lgbm_catboost.py --region "$region" --model vitl16 --resolution "$res" \
                2>&1 | tee -a "$LOG_DIR/phase1_ml_${region}.log"
        done
    done

    echo ""
    echo ">>> Phase 1 DONE"
    echo ""
fi

# ---------------------------------------------------------------------------
# Phase 2: Partial finetune (last4) + ML on finetuned features
# ---------------------------------------------------------------------------
if [ "$START_PHASE" -le 2 ]; then
    echo ">>> Phase 2: Partial finetune (last4) + ML on finetuned features"
    echo "============================================================"

    for region in "${REGIONS[@]}"; do
        echo ""
        echo "  2a. Finetune last4: $region"
        $PYTHON train_partial_finetune.py \
            --strategy last4 --region "$region" \
            --batch-size 16 --epochs 30 --patience 7 \
            2>&1 | tee "$LOG_DIR/phase2_finetune_${region}.log"

        echo ""
        echo "  2b. ML on finetuned features: $region"
        $PYTHON train_with_svm_xgb_lgbm_catboost_with_unfrozen_trained_dino.py \
            --region "$region" --model vitl16 --resolution 714 \
            --ckpt-tag partial_last4 --batch-size 16 \
            2>&1 | tee "$LOG_DIR/phase2_ml_${region}.log"
    done

    echo ""
    echo ">>> Phase 2 DONE"
    echo ""
fi

# ---------------------------------------------------------------------------
# Phase 3: Region fusion
# ---------------------------------------------------------------------------
if [ "$START_PHASE" -le 3 ]; then
    echo ">>> Phase 3: Region fusion (attention, MLP, XGB, SVM)"
    echo "============================================================"

    $PYTHON train_region_fusion.py --embed-key vitl16_714 \
        2>&1 | tee "$LOG_DIR/phase3_fusion.log"

    echo ""
    echo ">>> Phase 3 DONE"
    echo ""
fi

# ---------------------------------------------------------------------------
# Phase 4: Test evaluation + voting
# ---------------------------------------------------------------------------
if [ "$START_PHASE" -le 4 ]; then
    echo ">>> Phase 4: Test evaluation"
    echo "============================================================"

    for region in "${REGIONS[@]}"; do
        echo ""
        echo "  4a. Test eval: $region"
        $PYTHON evaluate_test.py --region "$region" \
            --models vitl16 --resolutions 518 714 \
            2>&1 | tee "$LOG_DIR/phase4_test_${region}.log"
    done

    echo ""
    echo "  4b. Voting evaluation"
    $PYTHON evaluate_voting.py \
        2>&1 | tee "$LOG_DIR/phase4_voting.log"

    echo ""
    echo ">>> Phase 4 DONE"
    echo ""
fi

echo "============================================================"
echo "PIPELINE COMPLETE  $(date)"
echo "Logs: $LOG_DIR/"
echo "============================================================"
