#!/bin/bash
# ---------------------------------------------------------------------------
# Qwen-Omni Realtime batch inference launcher.
#
# Usage:
#   export DASHSCOPE_API_KEY=sk-xxx
#   bash launch_batch_qwen.sh [--workers 10] [--model qwen3.5-omni-flash-realtime] ...
#
# Resume-safe: any video with a .done marker is skipped automatically.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -- Defaults --
WORKERS=10
MODEL="qwen3.5-omni-flash-realtime"
OUTPUT_ROOT="outputs/qwen"
VOICE="Tina"
TURN_MODE="server_vad"
PROBE_EVERY_SEC="1.0"
TURN_SILENCE_MS="800"
VIDEO_FPS="2.0"
MAX_FRAME_EDGE="640"
JPEG_QUALITY="80"
NO_REALTIME=""

# -- Parse CLI args --
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)       WORKERS="$2"; shift 2;;
        --model)         MODEL="$2"; shift 2;;
        --output_root)   OUTPUT_ROOT="$2"; shift 2;;
        --voice)         VOICE="$2"; shift 2;;
        --turn_mode)     TURN_MODE="$2"; shift 2;;
        --probe_every_sec)  PROBE_EVERY_SEC="$2"; shift 2;;
        --turn_silence_ms)  TURN_SILENCE_MS="$2"; shift 2;;
        --video_fps)     VIDEO_FPS="$2"; shift 2;;
        --max_frame_edge)   MAX_FRAME_EDGE="$2"; shift 2;;
        --jpeg_quality)     JPEG_QUALITY="$2"; shift 2;;
        --no_realtime)   NO_REALTIME="--no_realtime"; shift;;
        *)               echo "unknown arg: $1"; exit 1;;
    esac
done

# -- Check API key --
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
    echo "error: please set DASHSCOPE_API_KEY first"
    echo "  export DASHSCOPE_API_KEY=sk-xxx"
    exit 1
fi

echo "========================================"
echo "Qwen-Omni Realtime batch inference"
echo "========================================"
echo "  workers:    $WORKERS"
echo "  model:      $MODEL"
echo "  output:     $OUTPUT_ROOT"
echo "  turn_mode:  $TURN_MODE"
echo "  voice:      $VOICE"
echo "========================================"

mkdir -p "$OUTPUT_ROOT/logs"

LOG_FILE="$OUTPUT_ROOT/logs/batch_$(date +%Y%m%d_%H%M%S).log"

python3 batch_inference_qwen.py \
    --workers "$WORKERS" \
    --output_root "$OUTPUT_ROOT" \
    --model "$MODEL" \
    --voice "$VOICE" \
    --turn_mode "$TURN_MODE" \
    --probe_every_sec "$PROBE_EVERY_SEC" \
    --turn_silence_ms "$TURN_SILENCE_MS" \
    --video_fps "$VIDEO_FPS" \
    --max_frame_edge "$MAX_FRAME_EDGE" \
    --jpeg_quality "$JPEG_QUALITY" \
    $NO_REALTIME \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Batch inference done!"
else
    echo "Batch inference failed (exit code: $EXIT_CODE)"
fi
echo "log:     $LOG_FILE"
echo "summary: $OUTPUT_ROOT/batch_summary.json"
echo "========================================"

exit $EXIT_CODE
