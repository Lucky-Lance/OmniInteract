# OmniInteract

**Benchmarking Real-World Streaming Interaction for Real-Time Omnimodal Assistants**

Xudong Lu\*†, Xueying Li\*, Annan Wang\*, Yang Bo, Jinpeng Chen, Zengliang Li, Nianzu Yang, Rui Liu, Xue Yang, Jingwen Hou✉, Hongsheng Li

<sub>\*Equal contribution · †Project lead · ✉Corresponding author</sub>
<sub>CUHK MMLab · SJTU · NTU · McMaster · CityUHK · JUFE</sub>

[![arXiv](https://img.shields.io/badge/arXiv-2605.26485-b31b1b.svg)](https://arxiv.org/abs/2605.26485)
[![Project](https://img.shields.io/badge/Project-OmniInteract-blue.svg)](https://github.com/Lucky-Lance/OmniInteract)

---

## TL;DR

OmniInteract is a streaming benchmark for **real-time omnimodal LLMs**, evaluated through their **native online inference** over continuous audio-visual streams. User queries and ambient sounds live in the audio track, visual events live in the video, and models must decide **whether**, **when**, and **what** to respond — all without lookahead to future content.

This repository contains the **inference adapters** that drive the four evaluated models through their native streaming SDKs against the OmniInteract dataset, along with batch orchestrators and the offline math-reasoning evaluation pipeline.

## Key Features

- **Native streaming evaluation.** Spoken queries, visual events, and background sounds remain in the original audio-visual stream — no text prompts, no pre-segmented clips, no offline lookahead.
- **Two complementary splits.** **1Q1A** for localized single-response interaction (real-time, proactive, nested), and **1QnA** for long-horizon continuous task monitoring.
- **Interaction slot formulation.** Every response opportunity is a triple `(trigger, response window, target answer)`, making continuous streams measurable while preserving their temporal structure.
- **Interaction-aware metrics.** **IA-QTF1** (Interaction-Aware Quality-Timeliness F1), **IDS** (Interruption Diagnostic Suite: NOR / PAQ / CSM-SR / CSM-AS), and **NCCS** (Nested Chain Completion Score) jointly evaluate content, timing, and conversational continuity.
- **Full-duplex stress.** 192 interrupted slots and 240 nested slots (120 pairs) probe stopping behavior and context switch / resumption.

## Dataset at a Glance

- **250 videos**, **1,430 temporally grounded response slots**
- **1Q1A** (1,062 slots / 210 videos): 638 real-time, 184 proactive, 240 nested
- **1QnA** (368 slots / 40 videos): continuous task monitoring with one instruction → many time-grounded answers
- **Interruptions**: 147 in 1Q1A + 45 in 1QnA
- **Domains**: Chinese daily-life interaction (home, gym, museum, shopping, …) and English mathematical reasoning

## Evaluated Models

We evaluate four representative real-time omnimodal models, all available as inference adapters in this repository. The AURA adapter lives on the [`aura`](https://github.com/Lucky-Lance/OmniInteract/tree/aura) branch (it ships its own ASR + TTS + vLLM service stack); the other three are on `main`.

| Model                            | Adapter                                                                                                                                                              | Backend            |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| **AURA**                         | [`launch_batch_aura.sh`](https://github.com/Lucky-Lance/OmniInteract/blob/aura/launch_batch_aura.sh) on the [`aura`](https://github.com/Lucky-Lance/OmniInteract/tree/aura) branch                  | local GPU          |
| **Gemini 2.5 Flash Live**        | [`generative-ai/.../run_with_video_file.py`](generative-ai/gemini/multimodal-live-api/native-audio-websocket-demo-apps/plain-js-python-sdk-demo-app/run_with_video_file.py) | Google / Vertex AI |
| **MiniCPM-o 4.5**                | [`MiniCPM-o-Demo/run_with_video_file.py`](MiniCPM-o-Demo/run_with_video_file.py)                                                                                     | local GPU          |
| **Qwen3.5-Omni Flash Realtime**  | [`alibabacloud-bailian-speech-demo/.../run_with_video_file.py`](alibabacloud-bailian-speech-demo/samples/conversation/omni/python/run_with_video_file.py)            | DashScope API      |

### Pseudo-online inference on top of official SDKs

Every adapter implements pseudo-online inference using the model's **official SDK and reference online demo**: video and audio are fed into the streaming API at real-time cadence, while the model side's full-duplex logic (VAD, interruption handling, multi-turn state) is identical to a live deployment. The only difference from the official online demos is the I/O adaptation layer: input comes from a video file instead of camera + microphone, and output is written to WAV + JSONL instead of being played back live.

- **Qwen3.5-Omni Flash Realtime**: same `OmniRealtimeConversation` SDK and `OmniRealtimeCallback` event loop as `run_with_camera.py`. Identical session config (`server_vad`, `PCM_16000HZ_MONO_16BIT` in / `PCM_24000HZ_MONO_16BIT` out, `gummy-realtime-v1` transcription). Interruption semantics match: on `speech_started` we drop the pending buffer and go silent.
- **Gemini 2.5 Flash Live**: same `genai.Client` + `aio.live.connect` setup as the official `gemini_live.py`. Identical `LiveConnectConfig` (`response_modalities=[AUDIO]`, input/output transcription, `proactivity(proactive_audio=True)`). The four async tasks (`send_audio`, `send_video`, `send_text`, `receive_loop`) mirror the official demo line for line. Audio is sent in ~85 ms chunks to emulate the browser AudioWorklet.
- **MiniCPM-o 4.5**: same `UnifiedProcessor` → `set_duplex_mode()` flow as `worker.py`. The one-step-per-second loop of `prefill(audio, frames)` → `generate()` → `finalize()` mirrors the WebSocket protocol used by the online demo. Sliding window enabled (`basic` mode, high = 7000 / low = 5000 tokens) so 5-minute+ videos do not blow up the KV cache.

## Main Findings

| Setting        | Metric      | Best Model                  | Score     |
| -------------- | ----------- | --------------------------- | --------- |
| 1Q1A real-time | IA-QTF1     | Gemini 2.5 Flash Live       | 0.553     |
| 1Q1A proactive | IA-QTF1     | MiniCPM-o 4.5               | 0.607     |
| 1Q1A nested    | IA-QTF1     | MiniCPM-o 4.5               | 0.599     |
| 1Q1A global    | IA-QTF1     | AURA                        | 0.467     |
| Nested chain   | NCCS        | MiniCPM-o 4.5               | 0.284     |
| 1QnA           | IA-QTF1     | AURA                        | 0.052     |
| **All Global** | **IA-QTF1** | **MiniCPM-o 4.5**           | **0.368** |

- Long-horizon **1QnA monitoring** is the hardest setting — all four models score below 0.06.
- Many models treat nested queries as **permanent context switches**: Gemini misses the outer query in 119/120 pairs and Qwen3.5-Omni in 116/120, while MiniCPM-o (55/120) and AURA (54/120) resume far more reliably.
- Interruption handling exposes a clear trade-off: Gemini stays silent (NOR 85.94%) but spills little (CSM 40.74% / 0.312 s), while MiniCPM-o answers more (PAQ 0.571) yet over-generates by **10.067 s** on average after `t_end`.
- A focused full-duplex study on MiniCPM-o 4.5 mathematical reasoning shows pure quality drops from **0.6833 offline → 0.3475 online** (Δ −0.3358), confirming that strong offline reasoning does *not* transfer to native streaming interaction.

## Quick Start

### AURA

The AURA inference adapter lives on the [`aura`](https://github.com/Lucky-Lance/OmniInteract/tree/aura) branch (it bundles its own ASR + TTS + vLLM streaming service stack, with installation and weight details in that branch's `README.md`). It is evaluated on the same `data/` layout described below:

```bash
git checkout aura
MODEL_PATH=/path/to/AURA-8b \
ASR_MODEL=/path/to/Qwen3-ASR-1.7B \
TTS_MODEL=/path/to/Qwen3-TTS-12Hz-1.7B-Base \
bash launch_batch_aura.sh
```

### Gemini 2.5 Flash Live

```bash
export GEMINI_API_KEY=xxx
cd generative-ai/gemini/multimodal-live-api/native-audio-websocket-demo-apps/plain-js-python-sdk-demo-app
python run_with_video_file.py --video /path/to/video.mp4
```

### MiniCPM-o 4.5

```bash
cd MiniCPM-o-Demo
CUDA_VISIBLE_DEVICES=0 python run_with_video_file.py \
    --video /path/to/video.mp4 \
    --model_path /path/to/MiniCPM-o-4_5
```

### Qwen3.5-Omni Flash Realtime

```bash
export DASHSCOPE_API_KEY=sk-xxx
cd alibabacloud-bailian-speech-demo/samples/conversation/omni/python
python run_with_video_file.py --video /path/to/video.mp4
```

## Dataset Layout

The benchmark dataset is hosted on Hugging Face: [**lucky-lance/OmniInteract**](https://huggingface.co/datasets/lucky-lance/OmniInteract). Download and extract it into `data/` at the repository root:

```bash
# Option 1: huggingface-cli
huggingface-cli download lucky-lance/OmniInteract data.tar.gz --repo-type dataset --local-dir .
tar -xf data.tar.gz   # creates ./data/

# Option 2: git + git-lfs
git lfs install
git clone https://huggingface.co/datasets/lucky-lance/OmniInteract
tar -xf OmniInteract/data.tar.gz -C .   # creates ./data/
```

The dataset under `data/` is anonymized and ships in three flat subsets:

```
data/
├── 1q1a/                    # Chinese daily-life QA, 150 videos
│   ├── videos/0001.mp4 ... 0150.mp4
│   ├── annotations/0001.json ... 0150.json
│   └── video_json_map.json
├── 1q1a_math/               # English math reasoning, 60 videos (covered by both online batch inference and the offline math comparison)
│   ├── videos/0001.mp4 ... 0060.mp4
│   ├── annotations/0001.json ... 0060.json
│   └── video_json_map.json
└── 1qna/                    # Long-horizon cooking task monitoring, 40 videos
    ├── videos_bench/
    │   ├── captaincook4d/*.mp4
    │   └── egoper/*.mp4
    └── annotations/
        ├── captaincook4d/*.json
        └── egoper/*.json
```

Each `video_json_map.json` has the shape:

```json
{
  "total": 150,
  "entries": [
    {"video": "videos/0001.mp4", "annotation": "annotations/0001.json", "scene_type": "nested"},
    {"video": "videos/0091.mp4", "annotation": "annotations/0091.json", "scene_type": "multi_turn"},
    ...
  ]
}
```

The `scene_type` field is `"nested"` for the 60 videos that probe context switch / resumption (paper "Nested" category) and `"multi_turn"` otherwise. It is consumed by `eval/evaluation/run_batch_unified_eval.py` to dispatch the right slot builder per video, so that aggregated counts reproduce Tab. data_statistics in the paper.

Each `data/1q1a/annotations/*.json` is a list of QA entries with the schema:

```json
[
  {
    "question_time": "00:01",
    "question_text": "...",
    "answer_time": "00:36",
    "answer_text": "...",
    "question_type": "realtime",
    "is_interrupted": false
  }
]
```

`data/1q1a_math/annotations/*.json` additionally carries an attribution `"source"` URL per entry pointing to the original math-problem page.

Together, `1q1a` (150) + `1q1a_math` (60) = **210 videos / 1,062 slots** make up the 1Q1A split; `1qna` (40) = **40 videos / 368 slots** make up the 1QnA split. The 1Q1A split decomposes as 60 nested videos (240 slots = 120 nested pairs, each pair = 1 outer + 1 inner) and 150 non-nested videos: 90 general 1q1a videos contribute 518 real-time + 184 proactive slots, and 60 1q1a_math videos contribute 120 real-time slots. Aggregating: **638 real-time + 184 proactive + 240 nested = 1,062**, exactly matching paper Tab. data_statistics.

## Output Format

Each adapter emits the same files under `outputs/<model>/<subset>/<video_name>/`. The subset comes from `data/` (`1q1a`, `1q1a_math`, or `1qna`); `<video_name>` is the original video path with `/` replaced by `__` (e.g. `data/1q1a/videos/0001.mp4` → `outputs/qwen/1q1a/videos__0001/`).

```
outputs/<model>/<subset>/<video_name>/
├── output.wav              # 24 kHz mono 16-bit PCM, duration == input duration
├── wav_transcript.json     # WAV-aligned native model text (compatible with output.json)
├── model_output.jsonl      # per-second text / audio records
├── model_output.txt        # human-readable text preview
├── responses.jsonl         # one record per response/turn (delta-level timestamps)
├── events.jsonl            # raw event log (Qwen / Gemini)
└── audio_per_second/       # per-second PCM files
```

- **`output.wav`**: 24 kHz, mono, 16-bit PCM. Duration matches the input video (silence-padded to align). On interruption, the pending buffer is dropped and audio goes silent; playback resumes when the next response arrives.
- **`wav_transcript.json`**: format compatible with the `output.json` schema produced by [`eval/data_prep/ASR.py`](eval/data_prep/ASR.py). Text source is the model's native output (100% accurate, not ASR). Time source is frame-level WAV alignment (30 ms precision).

## Batch Inference

The three `main`-branch orchestrators (Qwen / Gemini / MiniCPM-o) auto-discover videos under `data/1q1a/`, `data/1q1a_math/`, and `data/1qna/videos_bench/`. They support resume-from-crash via `.done` marker files in each output directory. AURA has its own orchestrator on the `aura` branch (shown last below).

```bash
# Qwen3.5-Omni Flash Realtime
export DASHSCOPE_API_KEY=sk-xxx
bash launch_batch_qwen.sh --workers 10

# Gemini 2.5 Flash Live
export GEMINI_API_KEY=xxx
bash launch_batch_gemini.sh --workers 10

# MiniCPM-o 4.5 (multi-GPU)
bash launch_batch_minicpmo.sh --model_path /path/to/MiniCPM-o-4_5 --num_gpus 8

# AURA (on the `aura` branch; spins up its own ASR + TTS + vLLM stack)
git checkout aura
MODEL_PATH=/path/to/AURA-8b bash launch_batch_aura.sh
```

## Evaluation

`eval/` is the reproducible scoring pipeline. Given a directory of model outputs (see "Output Format" above), it runs data preparation, LLM judging, slot-level scoring, and global aggregation, and produces the exact paper-table numbers (IA-QTF1, IDS, NCCS).

### One-shot scoring (recommended)

`run_full_eval.sh` runs the whole per-model pipeline (data prep → judging → metrics). Pass a single model name; it auto-discovers `outputs/<model>/batch_summary.json` and prints the paper-table metrics at the end:

```bash
bash run_full_eval.sh gemini            # use all 10 GPUs for Step 1
bash run_full_eval.sh qwen "0,1,2,3"    # optional 2nd arg restricts GPUs
```

The ASR/aligner paths, GPU list, and LLM-judge config are baked in as defaults and can be overridden via environment variables:

```bash
ASR_MODEL=/path/to/Qwen3-ASR-1.7B \
ALIGN_MODEL=/path/to/Qwen3-ForcedAligner-0.6B \
JUDGE_API_URL=https://api.openai.com/v1/chat/completions \
JUDGE_API_MODEL=gpt-4o-2024-08-06 \
JUDGE_API_KEY=sk-... \
EVAL_WORKERS=4 \
bash run_full_eval.sh minicpmo
```

### Manual stages

The full workflow has three stages — full details in [`eval/README.md`](eval/README.md):

```bash
# Step 1: precise transcripts (ASR truncation + forced alignment)
python eval/data_prep/data_prep_batch.py \
  --batch_summary_json outputs/qwen/batch_summary.json \
  --output_root outputs/qwen \
  --asr_model /path/to/Qwen3-ASR-1.7B \
  --align_model /path/to/Qwen3-ForcedAligner-0.6B \
  --gpu_ids 0 --num_workers 1

# Step 2: unified evaluation (LLM judge + slot scoring + aggregation)
export JUDGE_API_KEY=sk-...
python eval/run_eval.py \
  --batch_summary_json outputs/qwen/batch_summary.json \
  --output_root outputs/qwen \
  --model_json_name precise_truncation.json \
  --out_dir outputs/qwen/unified_eval \
  --num_workers 4 --skip_existing

# Step 3 (optional): multi-model markdown report
python eval/evaluation/summarize_unified_eval.py \
  --model_eval_dir aura=outputs/aura/unified_eval \
  --model_eval_dir gemini=outputs/gemini/unified_eval \
  --model_eval_dir minicpmo=outputs/minicpmo/unified_eval \
  --model_eval_dir qwen=outputs/qwen/unified_eval \
  --report_md outputs/unified_eval_report.md
```

The three judge prompts (early stage, core stage, interrupted partial-answer quality) in `eval/evaluation/llm_judge.py` are the verbatim English versions of the Chinese prompts used in the paper experiments, matching Listings 1–3 in OmniInteract Appendix §B (`Detailed Scoring Definitions` / `LLM Judge Evaluation Protocol`).

The paper-table metrics live at:

```text
unified_eval_summary.json
└── summary.paper_metrics
    ├── exp_f1            # realtime / proactive / nested + one_q1a_global / one_qna / all_global IA-QTF1
    ├── exp_interruption  # NOR, PAQ, CSM-SR, CSM-AS over interrupted slots
    └── exp_nested        # NCCS + inner / outer IA-QTF1
```

## Offline Math Evaluation

`offline_math_eval/` runs the 60-video `data/1q1a_math/` subset through MiniCPM-o 4.5 in half-duplex chat-with-video mode for offline scoring, supporting the offline vs. online comparison reported in the paper.

```bash
# Step 1: extract evaluation clips (first and last minute of every video)
python offline_math_eval/prepare_clips.py

# Step 2: multi-GPU inference
bash offline_math_eval/launch.sh /path/to/MiniCPM-o-4_5

# Step 3: merge results
python offline_math_eval/merge_results.py
# Output: offline_math_eval/all_results.json
```

## Repository Layout

```
OmniInteract/
├── alibabacloud-bailian-speech-demo/   # Qwen3.5-Omni adapter  (upstream: aliyun/alibabacloud-bailian-speech-demo)
├── generative-ai/                      # Gemini 2.5 Live adapter (upstream: GoogleCloudPlatform/generative-ai)
├── MiniCPM-o-Demo/                     # MiniCPM-o 4.5 adapter (full-duplex demo)
├── eval/                               # Unified evaluation pipeline (IA-QTF1, IDS, NCCS)
├── offline_math_eval/                  # Offline math-reasoning evaluation pipeline
├── data/                               # Anonymized benchmark dataset (see "Dataset Layout")
├── batch_inference_*.py                # Batch orchestrators (one per model)
├── launch_batch_*.sh                   # Shell launchers for the orchestrators
├── run_full_eval.sh                    # One-shot per-model scoring (data prep -> judge -> metrics)
├── merge_video_response.py             # Mux model audio back into the source video
└── README.md
```

## Dependencies

```bash
# Common
pip install opencv-python

# Qwen3.5-Omni Flash Realtime
pip install dashscope

# Gemini 2.5 Flash Live
pip install google-genai python-dotenv

# MiniCPM-o 4.5
pip install "transformers==4.51.0" accelerate "torch>=2.3.0,<=2.8.0" "torchaudio<=2.8.0" "minicpmo-utils[all]>=1.0.5" librosa "setuptools<81"
# Note 1: `minicpmo-utils` is the PyPI distribution name; the importable Python
#         module is `minicpmo` (not `minicpmo_utils`).
# Note 2: `setuptools<81` is required because setuptools >= 81 removes
#         `pkg_resources`, which `librosa` (and therefore `minicpmo`) still
#         imports at module load time.

# Evaluation (unified pipeline)
# IMPORTANT: install this in a SEPARATE environment from MiniCPM-o 4.5.
# `qwen-asr` (the ASR + forced-aligner backend) HARD-PINS transformers==4.57.6,
# which is incompatible with the `transformers==4.51.0` pin required by
# MiniCPM-o inference. The two cannot coexist in one env. (Inference and
# evaluation are independent stages, so separate envs are fine.)
pip install requests                                   # judge API + slot scoring
pip install numpy soundfile tqdm "torch>=2.3.0"        # data preparation (ASR + forced alignment)
pip install qwen-asr soxr                              # ASR + forced aligner backend
# Notes:
# - `qwen-asr` (validated: 0.0.6, which pins transformers==4.57.6) also pulls in
#   librosa / accelerate / soundfile. The importable module is `qwen_asr`.
# - `soxr` must be installed explicitly: transformers 4.57.x imports it in
#   `transformers.audio_utils`, but does not always pull it in as a hard dep.
# - `pip install "setuptools<81"` silences the `pkg_resources` deprecation
#   warning from librosa (optional; warning only, does not affect results).
# Plus the model weights for Qwen3-ASR-1.7B and Qwen3-ForcedAligner-0.6B on disk.
```

System dependencies: `ffmpeg` and `ffprobe` must be on `PATH`.

## Upstream Sources

- [AURA (`aura` branch)](https://github.com/Lucky-Lance/OmniInteract/tree/aura) — AURA streaming inference adapter (ASR + TTS + vLLM); model weights at [aurateam/AURA](https://huggingface.co/aurateam/AURA)
- [aliyun/alibabacloud-bailian-speech-demo](https://github.com/aliyun/alibabacloud-bailian-speech-demo) — DashScope Qwen-Omni SDK
- [GoogleCloudPlatform/generative-ai](https://github.com/GoogleCloudPlatform/generative-ai) — Google Gemini Live demos
- [openbmb/MiniCPM-o-4_5](https://huggingface.co/openbmb/MiniCPM-o-4_5) — MiniCPM-o 4.5 model weights

## Citation

```bibtex
@article{lu2026omniinteract,
  title   = {OmniInteract: Benchmarking Real-World Streaming Interaction for Real-Time Omnimodal Assistants},
  author  = {Lu, Xudong and Li, Xueying and Wang, Annan and Bo, Yang and Chen, Jinpeng and Li, Zengliang and Yang, Nianzu and Liu, Rui and Yang, Xue and Hou, Jingwen and Li, Hongsheng},
  journal = {arXiv preprint arXiv:2605.26485},
  year    = {2026}
}
```
