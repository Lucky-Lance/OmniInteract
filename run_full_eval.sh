#!/usr/bin/env bash
# One-shot OmniInteract scoring pipeline.
#
# Usage:
#   bash run_full_eval.sh <model>            # e.g. gemini / minicpmo / qwen
#   bash run_full_eval.sh <model> "0,1,2,3"  # restrict GPUs for Step 1
#
# Runs, for a single model under outputs/<model>/:
#   Step 1  ASR + forced alignment       -> output.json / precise_truncation.json
#   Step 2  LLM judge + slot scoring      -> outputs/<model>/unified_eval/
#   Step 3  prints paper-table metrics (IA-QTF1 / IDS / NCCS)
#
# Configure everything via environment variables (no secrets are baked in):
#   ASR_MODEL        path to Qwen3-ASR-1.7B
#   ALIGN_MODEL      path to Qwen3-ForcedAligner-0.6B
#   JUDGE_API_URL    OpenAI-compatible chat-completions endpoint
#   JUDGE_API_MODEL  judge model name
#   JUDGE_API_KEY    judge API key (required)
#   EVAL_WORKERS     concurrent samples in Step 2 (default 4)
set -euo pipefail

MODEL="${1:?usage: bash run_full_eval.sh <model> [gpu_ids]}"
GPU_IDS="${2:-0,1,2,3,4,5,6,7,8,9}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# --- model paths (Qwen3 ASR + forced aligner) -------------------------------
ASR_MODEL="${ASR_MODEL:-/path/to/Qwen3-ASR-1.7B}"
ALIGN_MODEL="${ALIGN_MODEL:-/path/to/Qwen3-ForcedAligner-0.6B}"

# --- LLM judge (OpenAI-compatible) ------------------------------------------
JUDGE_API_URL="${JUDGE_API_URL:-https://api.openai.com/v1/chat/completions}"
JUDGE_API_MODEL="${JUDGE_API_MODEL:-gpt-4o-2024-08-06}"
JUDGE_API_KEY="${JUDGE_API_KEY:?set JUDGE_API_KEY in your environment}"

# --- eval parallelism for Step 2 --------------------------------------------
EVAL_WORKERS="${EVAL_WORKERS:-4}"

OUT_ROOT="outputs/${MODEL}"
BATCH_SUMMARY="${OUT_ROOT}/batch_summary.json"
EVAL_DIR="${OUT_ROOT}/unified_eval"

if [[ ! -f "${BATCH_SUMMARY}" ]]; then
  echo "[ERROR] ${BATCH_SUMMARY} not found. Did inference for '${MODEL}' finish?" >&2
  exit 1
fi

echo "=============================================================="
echo " OmniInteract scoring | model=${MODEL}"
echo " out_root=${OUT_ROOT}"
echo " gpu_ids=${GPU_IDS}"
echo " judge=${JUDGE_API_MODEL} @ ${JUDGE_API_URL}"
echo "=============================================================="

# --- Step 1: data preparation (ASR + forced alignment) ----------------------
echo ""
echo ">>> [Step 1/3] Data preparation (ASR + forced alignment)"
python eval/data_prep/data_prep_batch.py \
  --batch_summary_json "${BATCH_SUMMARY}" \
  --output_root "${OUT_ROOT}" \
  --asr_model "${ASR_MODEL}" \
  --align_model "${ALIGN_MODEL}" \
  --gpu_ids "${GPU_IDS}"

# --- Step 2: unified evaluation (judge + scoring + aggregation) -------------
echo ""
echo ">>> [Step 2/3] Unified evaluation (LLM judge + slot scoring)"
python eval/run_eval.py \
  --batch_summary_json "${BATCH_SUMMARY}" \
  --output_root "${OUT_ROOT}" \
  --model_json_name precise_truncation.json \
  --out_dir "${EVAL_DIR}" \
  --num_workers "${EVAL_WORKERS}" \
  --skip_existing \
  --judge_api_url "${JUDGE_API_URL}" \
  --judge_api_model "${JUDGE_API_MODEL}" \
  --judge_api_key "${JUDGE_API_KEY}"

# --- Step 3: print paper-table metrics --------------------------------------
echo ""
echo ">>> [Step 3/3] Paper-table metrics"
SUMMARY_JSON="${EVAL_DIR}/unified_eval_summary.json"
if [[ -f "${SUMMARY_JSON}" ]]; then
  python - "${SUMMARY_JSON}" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
pm = data.get("summary", {}).get("paper_metrics", {})
print(f"\nSummary: {sys.argv[1]}\n")
print(json.dumps(pm, indent=2, ensure_ascii=False))
PY
else
  echo "[WARN] ${SUMMARY_JSON} not found." >&2
fi

echo ""
echo "Done. Per-model results: ${EVAL_DIR}/"
echo "To build a multi-model markdown report once all models are scored:"
echo "  python eval/evaluation/summarize_unified_eval.py \\"
echo "    --model_eval_dir gemini=outputs/gemini/unified_eval \\"
echo "    --model_eval_dir minicpmo=outputs/minicpmo/unified_eval \\"
echo "    --model_eval_dir qwen=outputs/qwen/unified_eval \\"
echo "    --report_md outputs/unified_eval_report.md"
