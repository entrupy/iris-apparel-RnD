#!/bin/bash
# Run all 20 partial finetuning strategies serially on a single GPU.
# Uses nohup for persistence. Large batch size to maximize VRAM utilization.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="/root/.venv/bin/python -u"
LOG_DIR="./logs/partial"
mkdir -p "$LOG_DIR"

BATCH_SIZE=32

STRATEGIES=(
    norm layer_scale "norm+ls"
    q v qv qkv qkvo "qv+norm"
    mlp "mlp+norm"
    last1 last4 last8 last12
    mlp_last4 mlp_last8 attn_last4 "qv_last8+norm"
    full
)

echo "=========================================="
echo "PARTIAL FINETUNE SWEEP"
echo "Model: vitl16 @ 714 | Region: care_label"
echo "Batch size: $BATCH_SIZE | Strategies: ${#STRATEGIES[@]}"
echo "=========================================="
echo ""

RESULTS_FILE="$LOG_DIR/sweep_results.txt"
echo "Strategy,Unfrozen_Params,Val_TPR2,Val_AUC,Status" > "$RESULTS_FILE"

for i in "${!STRATEGIES[@]}"; do
    strat="${STRATEGIES[$i]}"
    n=$((i + 1))
    log="$LOG_DIR/${strat}.log"

    echo "[$n/${#STRATEGIES[@]}] Strategy: $strat -> $log"

    if $PYTHON train_partial_finetune.py \
        --strategy "$strat" \
        --batch-size "$BATCH_SIZE" \
        --epochs 30 \
        --patience 7 \
        > "$log" 2>&1; then
        echo "  [OK] $strat"
        # Extract best TPR from log
        best_line=$(grep "Best TPR@2%FPR" "$log" | tail -1)
        echo "$strat,$best_line,OK" >> "$RESULTS_FILE"
    else
        echo "  [FAILED] $strat"
        echo "$strat,,,FAILED" >> "$RESULTS_FILE"
    fi
    echo ""
done

echo "=========================================="
echo "SWEEP COMPLETE. $(date)"
echo "=========================================="

# Print summary
echo ""
echo "RESULTS SUMMARY:"
echo ""
printf "%-20s %12s %10s %10s\n" "Strategy" "Unfrozen" "Val TPR@2%" "Val AUC"
echo "------------------------------------------------------------"
for strat in "${STRATEGIES[@]}"; do
    log="$LOG_DIR/${strat}.log"
    if [ -f "$log" ]; then
        unfrozen=$(grep "will unfreeze" "$log" | grep -oP '\d+\.\d+M' | head -1)
        tpr2=$(grep "Best TPR@2%FPR" "$log" | grep -oP '\d+\.\d+' | head -1)
        auc=$(grep "Best TPR@2%FPR" "$log" | grep -oP 'AUC-ROC: \K\d+\.\d+' | head -1)
        printf "%-20s %12s %10s %10s\n" "$strat" "${unfrozen:-?}" "${tpr2:-?}" "${auc:-?}"
    fi
done
