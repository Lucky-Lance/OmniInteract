#!/bin/bash
# ---------------------------------------------------------------------------
# Gemini Live API batch inference launcher.
#
# Usage (API key):
#   export GEMINI_API_KEY=xxx
#   bash launch_batch_gemini.sh [--workers 10] ...
#
# Usage (Vertex AI):
#   gcloud auth application-default login
#   bash launch_batch_gemini.sh --use_vertex --project_id YOUR_PROJECT ...
#
# Resume-safe: any video with a .done marker is skipped automatically.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -- Defaults --
WORKERS=10
MODEL="gemini-live-2.5-flash-native-audio"
OUTPUT_ROOT="outputs/gemini"
VOICE="Puck"
VIDEO_FPS="2.0"
MAX_FRAME_EDGE="640"
JPEG_QUALITY="80"
NO_REALTIME=""
USE_VERTEX=""
PROJECT_ID=""
LOCATION="us-central1"
SYSTEM_PROMPT="You are a helpful and friendly AI assistant. RESPOND IN THE SAME LANGUAGE AS THE USER. YOU MUST RESPOND UNMISTAKABLY IN THE USER'S LANGUAGE."

# -- Parse CLI args --
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)         WORKERS="$2"; shift 2;;
        --model)           MODEL="$2"; shift 2;;
        --output_root)     OUTPUT_ROOT="$2"; shift 2;;
        --voice)           VOICE="$2"; shift 2;;
        --video_fps)       VIDEO_FPS="$2"; shift 2;;
        --max_frame_edge)  MAX_FRAME_EDGE="$2"; shift 2;;
        --jpeg_quality)    JPEG_QUALITY="$2"; shift 2;;
        --no_realtime)     NO_REALTIME="--no_realtime"; shift;;
        --use_vertex)      USE_VERTEX="--use_vertex"; shift;;
        --project_id)      PROJECT_ID="$2"; shift 2;;
        --location)        LOCATION="$2"; shift 2;;
        --system_prompt)   SYSTEM_PROMPT="$2"; shift 2;;
        *)                 echo "unknown arg: $1"; exit 1;;
    esac
done

# -- Check auth --
if [[ -z "$USE_VERTEX" ]]; then
    if [[ -z "${GEMINI_API_KEY:-}" ]] && [[ -z "${GOOGLE_API_KEY:-}" ]]; then
        echo "error: please set GEMINI_API_KEY or GOOGLE_API_KEY first"
        echo "  export GEMINI_API_KEY=xxx"
        echo "or use --use_vertex mode"
        exit 1
    fi
fi

echo "========================================"
echo "Gemini Live API batch inference"
echo "========================================"
echo "  workers:    $WORKERS"
echo "  model:      $MODEL"
echo "  output:     $OUTPUT_ROOT"
echo "  voice:      $VOICE"
if [[ -n "$USE_VERTEX" ]]; then
    echo "  auth:       Vertex AI (project=$PROJECT_ID, location=$LOCATION)"
else
    echo "  auth:       API Key"
fi
echo "========================================"

mkdir -p "$OUTPUT_ROOT/logs"

LOG_FILE="$OUTPUT_ROOT/logs/batch_$(date +%Y%m%d_%H%M%S).log"

EXTRA_ARGS=""
if [[ -n "$USE_VERTEX" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --use_vertex --location $LOCATION"
    if [[ -n "$PROJECT_ID" ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --project_id $PROJECT_ID"
    fi
fi

python3 batch_inference_gemini.py \
    --workers "$WORKERS" \
    --output_root "$OUTPUT_ROOT" \
    --model "$MODEL" \
    --voice "$VOICE" \
    --video_fps "$VIDEO_FPS" \
    --max_frame_edge "$MAX_FRAME_EDGE" \
    --jpeg_quality "$JPEG_QUALITY" \
    --system_prompt "$SYSTEM_PROMPT" \
    $NO_REALTIME \
    $EXTRA_ARGS \
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
