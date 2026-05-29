#!/usr/bin/env bash
#
# Multi-GPU launcher for MiniCPM-o 4.5 full-duplex batch inference.
#
# Usage:
#   bash launch_batch_minicpmo.sh [--model_path /path/to/model] [--output_root outputs/minicpmo]
#
# Defaults:
#   model_path:  /path/to/MiniCPM-o-4_5
#   output_root: outputs/minicpmo
#   num_gpus:    10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# -- Parse args --
MODEL_PATH="${MODEL_PATH:-/path/to/MiniCPM-o-4_5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/minicpmo}"
NUM_GPUS="${NUM_GPUS:-10}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-Streaming Omni Conversation.}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --system_prompt) SYSTEM_PROMPT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  MiniCPM-o 4.5 Batch Inference Launcher"
echo "============================================"
echo "  model_path:  $MODEL_PATH"
echo "  output_root: $OUTPUT_ROOT"
echo "  num_gpus:    $NUM_GPUS"
echo ""

# -- Step 1: build the video list --
WORK_DIR="$OUTPUT_ROOT/.work"
mkdir -p "$WORK_DIR"

python3 -c "
import json, sys
sys.path.insert(0, '.')
from batch_inference_minicpmo import collect_all_videos
from pathlib import Path

videos = collect_all_videos(Path('data'))
print(f'Found {len(videos)} videos', file=sys.stderr)

all_list = Path('$WORK_DIR/all_videos.jsonl')
with open(all_list, 'w') as f:
    for v in videos:
        f.write(json.dumps(v, ensure_ascii=False) + '\n')

num_gpus = int('$NUM_GPUS')
splits = [[] for _ in range(num_gpus)]
for i, v in enumerate(videos):
    splits[i % num_gpus].append(v)

for gpu_idx, split in enumerate(splits):
    fpath = Path('$WORK_DIR') / f'videos_gpu{gpu_idx}.jsonl'
    with open(fpath, 'w') as f:
        for v in split:
            f.write(json.dumps(v, ensure_ascii=False) + '\n')
    print(f'  GPU {gpu_idx}: {len(split)} videos -> {fpath}', file=sys.stderr)
"

echo ""
echo "Video list sharded across $NUM_GPUS GPU(s)"
echo ""

# -- Step 2: launch workers --
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

declare -A PID_MAP  # PID -> GPU_IDX
PIDS=()
for GPU_IDX in $(seq 0 $((NUM_GPUS - 1))); do
    VIDEO_LIST="$WORK_DIR/videos_gpu${GPU_IDX}.jsonl"
    LOG_FILE="$LOG_DIR/gpu${GPU_IDX}.log"

    if [ ! -s "$VIDEO_LIST" ]; then
        echo "[GPU $GPU_IDX] no tasks; skipping"
        continue
    fi

    echo "[GPU $GPU_IDX] launching worker -> $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$GPU_IDX python3 batch_inference_minicpmo.py \
        --video_list "$VIDEO_LIST" \
        --model_path "$MODEL_PATH" \
        --output_root "$OUTPUT_ROOT" \
        --system_prompt "$SYSTEM_PROMPT" \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    PIDS+=($PID)
    PID_MAP[$PID]=$GPU_IDX
done

echo ""
echo "All workers launched (${#PIDS[@]} processes)"
for PID in "${PIDS[@]}"; do
    echo "  GPU ${PID_MAP[$PID]} -> PID $PID"
done
echo ""
echo "Tail logs:    tail -f $LOG_DIR/gpu*.log"
echo "Progress:     grep -c 'done\\|failed' $LOG_DIR/gpu*.log"
echo ""

# -- Step 3: wait for workers --
FAILED_GPUS=()
for PID in "${PIDS[@]}"; do
    GPU_IDX=${PID_MAP[$PID]}
    if wait "$PID"; then
        echo "[GPU $GPU_IDX] worker done (PID $PID)"
    else
        echo "[GPU $GPU_IDX] worker failed (PID $PID, exit=$?)"
        FAILED_GPUS+=("$GPU_IDX")
    fi
done

echo ""
echo "============================================"
echo "  All workers finished"
echo "============================================"

if [ ${#FAILED_GPUS[@]} -gt 0 ]; then
    echo "  Failed GPUs: ${FAILED_GPUS[*]}"
    for FGPU in "${FAILED_GPUS[@]}"; do
        echo "  Log: cat $LOG_DIR/gpu${FGPU}.log"
    done
else
    echo "  All ok!"
fi

# -- Step 4: merge summaries --
echo ""
echo "Merging result summaries..."
python3 -c "
import json
from pathlib import Path

output_root = Path('$OUTPUT_ROOT')

# summaries are append-only; per video, keep the latest record
all_records = []
for sf in sorted(output_root.glob('summary_gpu*.jsonl')):
    with open(sf) as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))

# dedupe by video, keeping the last record (most recent run)
by_video = {}
for r in all_records:
    by_video[r.get('video', r.get('output_dir', ''))] = r
merged = list(by_video.values())

ok = [r for r in merged if r.get('status') == 'ok']
err = [r for r in merged if r.get('status') == 'error']

summary = {
    'total': len(merged),
    'success': len(ok),
    'failed': len(err),
    'results': merged,
}

out = output_root / 'batch_summary.json'
with open(out, 'w') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f'  total: {len(merged)}, ok: {len(ok)}, failed: {len(err)}')
print(f'  summary written: {out}')
if err:
    print(f'  failures:')
    for e in err:
        print(f'    - {e[\"video\"]}: {e.get(\"error\", \"unknown\")}')
"

echo ""
echo "Done! Output root: $OUTPUT_ROOT"
