# OmniInteract Evaluation

`eval/` provides the reproducible evaluation pipeline for OmniInteract. Starting from a directory of model outputs, it performs data preparation, LLM judging, slot-level scoring, and global aggregation, yielding the exact metrics reported in the paper tables.

## Pipeline overview

```text
Model output
  ├── output.wav
  └── wav_transcript.json
        │
        ▼
Data preparation
  ├── output.json                 # ASR result of output.wav
  └── precise_truncation.json     # Truncated text that actually played + word-level timestamps
        │
        ▼
Unified evaluation
  ├── *.unified_match.json        # Chunk-to-slot matching
  ├── *.unified_eval.json         # Per-slot detailed scoring
  └── unified_eval_summary.json   # Global metrics + paper-table numbers
```

We recommend using `precise_truncation.json` as the evaluation input. If you can verify that the text in `wav_transcript.json` already matches the audio that actually played in `output.wav`, you may instead generate and use `wav_transcript_aligned.json`.

## Input format

Each sample's model-output directory should contain:

```text
outputs/<model>/<subset>/<video_name>/        # e.g. outputs/qwen/1q1a/videos__0001/
├── output.wav              # 24 kHz, mono, 16-bit PCM; same duration as the input video
├── wav_transcript.json     # Model-native text, compatible with the output.json schema
├── model_output.jsonl      # Optional: per-second aggregated text/audio records
├── model_output.txt        # Optional: text preview
├── responses.jsonl         # Optional: response/turn-level records
├── events.jsonl            # Optional: raw event log
└── audio_per_second/       # Optional: per-second PCM files
```

The transcript JSON schema consumed by the evaluator:

```json
{
  "text": "full model response",
  "chunks": [
    {
      "text": "one model utterance",
      "timestamp": [3.21, 5.43],
      "aligned_words": [
        {"text": "one", "start": 3.21, "end": 3.50}
      ]
    }
  ]
}
```

`aligned_words` is optional on raw `wav_transcript.json` but is critical for precise scoring: without word-level timestamps, chunks that straddle the `t_a` boundary cannot be split cleanly into early / core stages.

## Dependencies

Basic evaluation only needs:

```bash
pip install requests
```

To re-run ASR and forced alignment on `output.wav` you also need:

```bash
pip install numpy soundfile torch tqdm
```

The data-preparation scripts also depend on a local `qwen_asr` package and require an ASR + forced-aligner model on disk, e.g.:

- `/path/to/Qwen3-ASR-1.7B`
- `/path/to/Qwen3-ForcedAligner-0.6B`

## Step 1: prepare evaluation transcripts

The data-preparation stage reads:

```text
output.wav + wav_transcript.json
```

and produces:

```text
output.json                # ASR of output.wav, with word-level timestamps
precise_truncation.json    # Recommended transcript for evaluation, with aligned_words
```

Recommended batch entry point:

```bash
python eval/data_prep/data_prep_batch.py \
  --batch_summary_json outputs/qwen/batch_summary.json \
  --output_root outputs/qwen \
  --asr_model /path/to/Qwen3-ASR-1.7B \
  --align_model /path/to/Qwen3-ForcedAligner-0.6B \
  --gpu_ids 0 \
  --num_workers 1 \
  --summary_json outputs/qwen/data_prep_summary.json
```

For each sample this script:

1. Runs ASR on `output.wav` to produce `output.json`.
2. Aligns `wav_transcript.json` against `output.json` and truncates text that never actually played.
3. Runs forced alignment on the truncated model-native text to produce `precise_truncation.json`.

For single-sample debugging, you can run the two stages separately:

```bash
python eval/data_prep/ASR.py \
  --wav_dir outputs/qwen/<video_name> \
  --asr_model /path/to/Qwen3-ASR-1.7B \
  --align_model /path/to/Qwen3-ForcedAligner-0.6B \
  --device cuda:0

python eval/data_prep/get_precise_truncation.py \
  --wav_transcript outputs/qwen/<video_name>/wav_transcript.json \
  --asr_output outputs/qwen/<video_name>/output.json \
  --audio outputs/qwen/<video_name>/output.wav \
  --out outputs/qwen/<video_name>/precise_truncation.json \
  --align_model /path/to/Qwen3-ForcedAligner-0.6B \
  --device cuda:0
```

If you don't need ASR-based truncation and only want word-level timestamps on `wav_transcript.json`:

```bash
python eval/data_prep/align_wav_transcript.py \
  --batch_summary_json outputs/qwen/batch_summary.json \
  --output_root outputs/qwen \
  --out_name wav_transcript_aligned.json \
  --align_model /path/to/Qwen3-ForcedAligner-0.6B \
  --device cuda:0 \
  --summary_json outputs/qwen/wav_transcript_align_summary.json
```

## Step 2: prepare a manifest

A `manifest` is the per-sample evaluation list, recommended as JSONL (one sample per line):

```jsonl
{"sample_id":"video_0001","gt_json":"data/1q1a/annotations/0001.json","model_json":"outputs/qwen/1q1a/videos__0001/precise_truncation.json","scene_type":"multi_turn"}
{"sample_id":"video_0001_math","gt_json":"data/1q1a_math/annotations/0001.json","model_json":"outputs/qwen/1q1a_math/videos__0001/precise_truncation.json","scene_type":"multi_turn"}
{"sample_id":"video_0001_1qna","gt_json":"data/1qna/annotations/captaincook4d/Breakfast_Burritos-7_135.json","model_json":"outputs/qwen/1qna/videos_bench__captaincook4d__Breakfast_Burritos-7_135/precise_truncation.json","scene_type":"1QnA"}
```

| Field | Description |
|---|---|
| `sample_id` | Unique sample id, used to name output files. Must not contain `/` or `\`. |
| `gt_json` | Path to the ground-truth annotation file. |
| `model_json` | Model transcript produced by data preparation; `precise_truncation.json` is recommended. |
| `scene_type` | Task type. One of `multi_turn`, `nested`, `1QnA`. |

Rules for `scene_type`:

- `multi_turn`: standard 1Q1A real-time / proactive / interruption samples. Covers all of `data/1q1a_math/` and the 90 non-nested videos in `data/1q1a/`.
- `nested`: nested-interaction samples. Covers the 60 nested videos in `data/1q1a/` (identified by `scene_type == "nested"` in `data/1q1a/video_json_map.json`).
- `1QnA`: long-task continuous-monitoring samples. Covers all of `data/1qna/`.

If your inference already produced a `batch_summary.json` whose paths resolve into this repository's `data/`, you can skip the manifest and let the runner infer everything (it reads `scene_type` directly from each subset's `video_json_map.json`):

```bash
python eval/run_eval.py \
  --batch_summary_json outputs/qwen/batch_summary.json \
  --output_root outputs/qwen \
  --model_json_name precise_truncation.json \
  --out_dir outputs/qwen/unified_eval
```

If automatic inference cannot find the GT, switch to a manifest with explicit `gt_json` per sample.

## Step 3: run unified evaluation

Set the LLM-judge API key:

```bash
export JUDGE_API_KEY=sk-...
```

Run with a manifest:

```bash
python eval/run_eval.py \
  --manifest manifests/qwen_eval.jsonl \
  --out_dir outputs/qwen/unified_eval \
  --num_workers 4 \
  --skip_existing
```

The judge defaults to an OpenAI-compatible Chat Completions endpoint:

```bash
--judge_api_url https://api.openai.com/v1/chat/completions
--judge_api_model gpt-4o-2024-08-06
```

For other compatible providers, override these two flags.

## Output files

When evaluation finishes, `out_dir` contains:

```text
outputs/qwen/unified_eval/
├── <sample_id>.unified_match.json
├── <sample_id>.unified_eval.json
├── unified_eval_progress.json
└── unified_eval_summary.json
```

`*.unified_match.json` records slot construction, chunk matching, and the early / core split.

`*.unified_eval.json` records the per-slot details:

- Slot times: `start`, `t_a`, `end`
- Stage text: `stage_early.actual_text`, `stage_core.actual_text`
- Stage anchors: `stage_early.answer_start`, `stage_core.answer_start`
- Judge output: `quality_score`, `timeliness_score`, `judge_rationale`
- Scoring: `TP_ack`, `TP_core`, `TP_n`, `FP_delta`, `FN_delta`
- Interrupt diagnostics: `interruption_diagnostic`
- Nested info: `nested_group_id`, `nested_role`

`unified_eval_summary.json` records global metrics. The paper-table values are at:

```text
summary.paper_metrics.exp_f1
summary.paper_metrics.exp_interruption
summary.paper_metrics.exp_nested
```

## Metric definitions

All questions are converted into a unified interaction slot:

```text
slot = [start, t_a, end)
```

Output before `t_a` is the early stage (mostly judging whether the model holds the floor appropriately); output after `t_a` is the core stage (judging core-answer quality and timeliness).

Per-slot soft TP:

```text
TP_n = clamp(Score_ack + Score_core, 0, 1)
```

Global aggregates:

```text
Global_TP = sum(TP_n)
Global_FP = sum(FP_delta) + unmatched_chunks
Global_FN = sum(FN_delta)
IA_QTF1   = 2 * Global_TP / (2 * Global_TP + Global_FP + Global_FN)
```

Interrupted slots are not required to complete the original answer, so `FN_delta = 0`. Continued output after the interrupt is still counted as FP via the spill mechanism and reported under interruption diagnostics.

Nested questions report the slot-level IA-QTF1 plus `NCCS` (Nested Chain Completion Score), which measures whether the model answers the inner Q2 first and then resumes the outer Q1.

## Script reference

| Script | Purpose |
|---|---|
| `eval/run_eval.py` | Recommended entry point: matching, judging, scoring, and summarization. |
| `eval/data_prep/data_prep_batch.py` | Batch-generate `precise_truncation.json`. |
| `eval/data_prep/ASR.py` | Run ASR on `output.wav` to produce `output.json`. |
| `eval/data_prep/get_precise_truncation.py` | Combine `wav_transcript.json`, ASR, and audio into the precise transcript. |
| `eval/data_prep/align_wav_transcript.py` | Add word-level timestamps to `wav_transcript.json` without ASR truncation. |
| `eval/evaluation/build_slots.py` | Convert GT annotations into unified slots. |
| `eval/evaluation/match_slots.py` | Match model chunks to slots and split early / core. |
| `eval/evaluation/compute_unified_eval.py` | Call the LLM judge and score a single sample. |
| `eval/evaluation/summarize_unified_eval.py` | Rebuild the global summary (and the multi-model report) from existing `*.unified_eval.json`. |
| `eval/evaluation/analyze_interrupt_gpt4o_eval2.py` | Diagnostic analysis for interrupted slots. |
| `eval/evaluation/compute_1q1a_exclusive_f1.py` | Compute 1Q1A mutually exclusive F1 metrics from `unified_eval_summary.json`. |

## FAQ

- If data preparation cannot find a sample directory, double-check `output_dir` in `batch_summary.json`, or pass `--output_root outputs/<model>` explicitly.
- If the evaluator cannot find a GT, switch to a manifest and fill `gt_json` by hand.
- If `precise_truncation.json` is missing, the evaluator falls back to `wav_transcript_aligned.json`, `wav_transcript.json`, and `output.json` in that order, but timeliness scoring becomes less stable.
- To rebuild the summary without re-judging:

```bash
python eval/evaluation/summarize_unified_eval.py \
  --eval_dir outputs/qwen/unified_eval \
  --out_json outputs/qwen/unified_eval/unified_eval_summary.json
```

To regenerate the multi-model report (markdown):

```bash
python eval/evaluation/summarize_unified_eval.py \
  --model_eval_dir aura=outputs/aura/unified_eval \
  --model_eval_dir gemini=outputs/gemini/unified_eval \
  --model_eval_dir minicpmo=outputs/minicpmo/unified_eval \
  --model_eval_dir qwen=outputs/qwen/unified_eval \
  --report_md outputs/unified_eval_report.md
```
