#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/root/.venv/bin/python"
TRAIN="$SCRIPT_DIR/train_partial_finetune.py"

STRATEGY="last4"
REGIONS=("front" "front_exterior_logo" "brand_tag")

for region in "${REGIONS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Starting: $STRATEGY on $region"
    echo "  $(date)"
    echo "============================================================"
    $PYTHON "$TRAIN" --strategy "$STRATEGY" --region "$region" 2>&1 | tee "$SCRIPT_DIR/logs/partial/${STRATEGY}_${region}.log"
    echo ""
    echo "  Finished $region at $(date)"
done

echo ""
echo "All done."
