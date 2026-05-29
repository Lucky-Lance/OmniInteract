#!/bin/bash
# Multi-GPU launcher for offline math-eval inference.
#
# Usage:
#   # Step 1: extract video clips
#   python offline_math_eval/prepare_clips.py
#
#   # Step 2: run inference
#   bash offline_math_eval/launch.sh /path/to/MiniCPM-o-4_5
#
#   # Step 3: merge results
#   python offline_math_eval/merge_results.py

set -euo pipefail

PYTHON="${PYTHON:-python3}"

MODEL_PATH="${1:-/path/to/MiniCPM-o-4_5}"
OUTPUT_DIR="offline_math_eval/results"
MANIFEST="offline_math_eval/eval_manifest.json"

GPU_IDS=(0 1)
NUM_WORKERS=${#GPU_IDS[@]}

mkdir -p "$OUTPUT_DIR"

echo "=== Offline math-eval inference ==="
echo "Model:    $MODEL_PATH"
echo "GPUs:     ${GPU_IDS[*]}"
echo "Manifest: $MANIFEST"
echo ""

pids=()
for ((i=0; i<NUM_WORKERS; i++)); do
    gpu=${GPU_IDS[$i]}
    echo "Starting worker $i (GPU $gpu) ..."
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON offline_math_eval/run_offline_inference.py \
        --manifest "$MANIFEST" \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --worker_id "$i" \
        --num_workers "$NUM_WORKERS" \
        > "$OUTPUT_DIR/worker_${i}.log" 2>&1 &
    pids+=($!)
done

echo ""
echo "All workers started, PIDs: ${pids[*]}"
echo "Logs: tail -f $OUTPUT_DIR/worker_*.log"
echo ""

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "Worker PID $pid failed"
        ((failed++))
    fi
done

if [ "$failed" -gt 0 ]; then
    echo "Warning: $failed worker(s) failed; check the logs"
else
    echo "All workers done!"
fi

echo ""
echo "Next: python offline_math_eval/merge_results.py"
