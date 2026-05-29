#!/bin/bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
MODEL_PATH="${MODEL_PATH:-/path/to/MiniCPM-o-4_5}"
GPU_IDS=(0 1)
NUM_WORKERS=${#GPU_IDS[@]}
OUTPUT_DIR="offline_math_eval/results"
MANIFEST="offline_math_eval/eval_manifest.json"

# -- Step 1: extract video clips --
if [ ! -f "$MANIFEST" ]; then
    echo "[Step 1] extracting video clips ..."
    $PYTHON offline_math_eval/prepare_clips.py
else
    echo "[Step 1] eval_manifest.json already exists; skipping extraction"
fi

# -- Step 2: multi-GPU inference --
echo ""
echo "[Step 2] starting inference (GPUs: ${GPU_IDS[*]})"
mkdir -p "$OUTPUT_DIR"

rm -f "$OUTPUT_DIR"/results_worker*.jsonl

pids=()
for ((i=0; i<NUM_WORKERS; i++)); do
    gpu=${GPU_IDS[$i]}
    echo "  worker $i -> GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON offline_math_eval/run_offline_inference.py \
        --manifest "$MANIFEST" \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --worker_id "$i" \
        --num_workers "$NUM_WORKERS" \
        > "$OUTPUT_DIR/worker_${i}.log" 2>&1 &
    pids+=($!)
done

echo "  PIDs: ${pids[*]}"
echo "  Logs: tail -f $OUTPUT_DIR/worker_*.log"
echo ""

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "  worker PID $pid failed"
        ((failed++))
    fi
done

if [ "$failed" -gt 0 ]; then
    echo "Warning: $failed worker(s) failed"
    for ((i=0; i<NUM_WORKERS; i++)); do
        echo "--- worker_${i}.log (last 20 lines) ---"
        tail -20 "$OUTPUT_DIR/worker_${i}.log"
    done
    exit 1
fi

echo "[Step 2] inference done!"

# -- Step 3: merge results --
echo ""
echo "[Step 3] merging results ..."
$PYTHON offline_math_eval/merge_results.py

echo ""
echo "=== All done ==="
echo "Result: offline_math_eval/all_results.json"
