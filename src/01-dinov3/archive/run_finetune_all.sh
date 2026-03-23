#!/bin/bash
set -e
cd /root/iris
source /root/.venv/bin/activate

COMMON="--mode finetune --epochs 30 --batch-size 4 --lr 1e-3 --lr-backbone 1e-5 --warmup-epochs 5"
SCRIPT="src/01-dinov3/train_care_label.py"
NOHUP_DIR="src/01-dinov3/nohup"

mkdir -p "$NOHUP_DIR"

echo "=== Starting 4 sequential finetune runs ==="

for MODEL in vitb16 vitl16; do
  for RES in 518 714; do
    OUT="$NOHUP_DIR/finetune_${MODEL}_${RES}.out"
    echo ""
    echo ">>> Starting $MODEL @ $RES -> $OUT"
    python -u "$SCRIPT" $COMMON --model "$MODEL" --resolution "$RES" > "$OUT" 2>&1
    echo ">>> Finished $MODEL @ $RES (exit=$?)"
  done
done

echo ""
echo "=== All 4 finetune runs complete ==="
