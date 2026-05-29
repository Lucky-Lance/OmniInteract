# AURA 全双工离线评测驱动

本目录给 AURA（`Qwen3_VL_online_streaming_v2_ContextManaged.py` + ASR + TTS）
增加了一套**本地视频文件驱动器**，用来在不依赖浏览器前端的情况下，
把 `<person>/<id>/<id>.mp4` 当作用户的音视频输入，按 2 fps 流式打到
AURA 主服务上，并把回流的文字与 TTS 音频按
[Full-Duplex-Bench-Omni](../Full-Duplex-Bench-Omni/README.md) 规定的格式落盘，
便于直接跑现成的 FDB-Omni 评测脚本。

## 依赖

- 复用仓库根目录的 [`requirements.txt`](../requirements.txt)，新增
  `silero-vad`、`scipy`。
- 需要在 PATH 中能找到 `ffmpeg` / `ffprobe`（用来抽音轨）。
- 需要一个可连通的 AURA 服务栈：ASR @ `:8001`、TTS @ `:8002`、vLLM 主服务
  @ `:12346`，用 `bash start_all.sh` 启动即可（或 `launch_batch_aura.sh` 启动双实例）。

## 模块构成

| 文件 | 作用 |
| --- | --- |
| `silero_vad_helper.py` | 对 `silero-vad` 的 `VADIterator` 做流式封装，产出 `(start, end, pcm)` 语音片段 |
| `video_file_driver.py` | 单个视频的驱动器：cv2 抽帧 → mp4 chunk → Type 1；ffmpeg 抽音轨 → VAD → WAV → Type 2；独立 RX 线程处理 Type 8/9/10；`FDBAggregator` 写 `output.wav` / `wav_transcript.json` / `responses.jsonl` / `events.jsonl` / `model_output.*` / `audio_per_second/` |
| `run_dataset.py` | 批跑器：递归发现 `<root>` 下所有 `.mp4`（跟随符号链接），对每个视频建立新 TCP 连接，先发 `Type 6` + `Type 4` 重置会话，顺序调用驱动器，拷贝 GT JSON 为 `gt.json`，并记录 `run_summary.jsonl`。支持 `--num_shards` / `--shard_index` 分片并行和 `--gt_root` 指定独立标注目录 |

## 数据流

```
mp4 ──┬─ cv2.VideoCapture ─ 2 fps frames ─ cv2.VideoWriter(mp4) ─ Type 1 ┐
      │                                                                 │
      └─ ffmpeg pcm 16k mono ─ silero-VAD ─ speech segment ─ Type 2 ────┤
                                                                        ▼
                                                          AURA TCP :12346
                                                                        │
                                      Type 8 text / Type 9 PCM / Type 10 ASR
                                                                        │
                                                          FDBAggregator │
                                                                        ▼
                               outputs/<id>/ (output.wav / wav_transcript.json
                                              / responses.jsonl / events.jsonl
                                              / model_output.jsonl / .txt
                                              / audio_per_second/*.pcm)
```

AURA 主服务端（`Qwen3_VL_online_streaming_v2_ContextManaged.py`）本身
只接受一路连接（`listen(1)` + `active_connection` 全局），因此批跑器是
严格串行的。

## 用法

### 起 AURA 服务

```bash
cd /path/to/Full-Duplex-Bench-Omni
bash start_all.sh
```

等到日志里出现 `Server listening on port 12346` / `Server started`。

自定义 GPU：

```bash
GPU_ASR=0 GPU_TTS=0 GPU_INFERENCE=1,2,3,4 bash start_all.sh
```

### 单个视频试跑

```bash
python eval/video_file_driver.py \
    --video "data/1q1a/videos/0001.mp4" \
    --output_dir eval_outputs/test
```

常用选项：

- `--trailing_seconds`（默认 8.0）：视频推完之后再继续写 WAV / 等 TTS 收尾的时长。
  如果模型响应比较长，建议调到 `20` 以上以免尾音被截断。
- `--vad_threshold` / `--vad_min_silence_ms` / `--vad_speech_pad_ms`：
  `silero-vad` 参数；默认 `0.5 / 500 / 100` 对应「一般会议级别的人声」。
- `--vad_min_segment_ms`（默认 300）：过滤掉太短的语音段，避免噪点触发 ASR。
- `--no_realtime`：不按 wall-clock 对齐节奏（仅调试用，会把数据一次性灌完）。

### 批跑整个数据集（单实例）

直接把 `--root` 指向 OmniInteract `data/` 下的子集目录即可，
`run_dataset.py` 会自动配对视频与标注（`1q1a` / `1q1a_math` 读
`video_json_map.json`；`1qna` 用 `videos_bench/` 镜像 `annotations/`），
无需再传 `--gt_root`：

```bash
python eval/run_dataset.py --root data/1q1a      --output outputs/aura/1q1a      --skip_existing
python eval/run_dataset.py --root data/1q1a_math --output outputs/aura/1q1a_math --skip_existing
python eval/run_dataset.py --root data/1qna      --output outputs/aura/1qna      --skip_existing
```

> `--gt_root` 仍保留，用于「视频与 GT JSON 完全分开存放」的通用数据集。

可选筛选：`--persons ...`、`--ids ...`、`--limit 5`。

分片并行（配合多实例使用）：

```bash
# 实例 A 跑偶数索引视频
python eval/run_dataset.py --root ... --output ... --num_shards 2 --shard_index 0
# 实例 B 跑奇数索引视频
python eval/run_dataset.py --root ... --output ... --num_shards 2 --shard_index 1
```

### 批跑 250 个视频（推荐：双实例并行）

一键启动 2 个完整 AURA 服务栈 + 并行推理 250 个视频：

```bash
bash launch_batch_aura.sh
```

断点续跑（自动跳过已完成的视频）：

```bash
bash launch_batch_aura.sh --skip_existing
```

**GPU 分配（10 张卡）：**

| 实例 | ASR+TTS | vLLM (TP=4) | 端口 |
|------|---------|-------------|------|
| A    | GPU 0   | GPUs 1,2,3,4 | ASR:8001 TTS:8002 vLLM:12346 |
| B    | GPU 5   | GPUs 6,7,8,9 | ASR:8003 TTS:8004 vLLM:12347 |

**数据集分布：**

| 子集       | 视频数 | 实例 A | 实例 B |
|------------|--------|--------|--------|
| 1q1a       | 150    | 75     | 75     |
| 1q1a_math  | 60     | 30     | 30     |
| 1qna       | 40     | 20     | 20     |
| 合计       | 250    | 125    | 125    |

### 输出结构

```
outputs/aura/
├── 1q1a/
│   ├── 0001/
│   │   ├── output.wav            # 24 kHz mono 16-bit，长度 = 视频时长
│   │   ├── wav_transcript.json   # {text, chunks:[{text, timestamp}]}
│   │   ├── responses.jsonl       # 每行一条 response，含 delta + audio_duration
│   │   ├── model_output.jsonl    # 每秒一行（text / user_transcript / pcm 指针）
│   │   ├── model_output.txt      # 人类可读
│   │   ├── events.jsonl          # 原始事件（含 ASR / VAD / TTS）
│   │   ├── audio_per_second/     # 每秒一份 PCM 切片
│   │   └── gt.json               # 从数据集复制的标注
│   ├── 0002/
│   └── ...                       # 共 150 个目录
├── 1q1a_math/
│   ├── 0001/
│   └── ...                       # 共 60 个目录
├── 1qna/
│   ├── captaincook4d__Blender_Banana_Pancakes-21_44/
│   ├── egoper__coffee_u1_a1_error_001/
│   └── ...                       # 共 40 个目录
└── logs/
    ├── A_asr.log / A_tts.log / A_vllm.log / A_warmup.log
    ├── B_asr.log / B_tts.log / B_vllm.log / B_warmup.log
    ├── A_1q1a.log / A_1q1a_math.log / A_1qna.log
    └── B_1q1a.log / B_1q1a_math.log / B_1qna.log
```

## 与 FDB-Omni 输出格式对齐

- `output.wav`: 24 kHz mono 16-bit，长度严格等于输入视频时长；被用户打断
  的旧 response 尾音会被静音填充（语义对齐 `freeze-omni` 的
  `stop_tts → pending.clear() → muted=True`）。
- `wav_transcript.json`: 通过 `_wav_frame_rids`（每帧记录对应 response_id）
  反查，得到每条 response 在 WAV 中的精确 `[start_sec, end_sec]`。
- `responses.jsonl`: 在 `finalize()` 时统一写出（因为 AURA 的 TTS 音频会
  **晚于** 文本 `is_final` 到达，实时写会漏计 `audio_duration_sec`）。
- `model_output.jsonl` / `.txt`: 按秒聚合 `text` / `user_transcript`。
- `audio_per_second/<sec>.pcm`: 每秒一份 24 kHz PCM。
- `events.jsonl`: 原始 socket 事件（含 ASR / VAD / TTS delta 的结构信息）。

下游可直接对 `eval_outputs/<person>/<id>/` 跑
`Full-Duplex-Bench/v1_v1.5/get_transcript/asr.py` 与
`Full-Duplex-Bench/v1_v1.5/evaluation/evaluate.py`。

## 与 AURA 服务端协议

- **C→S**
  - `Type 1` 视频（mp4 字节，≥1 KB）—— 每秒封装 2 帧。
  - `Type 2` 音频（WAV 字节）—— VAD 切出的语音段，AURA 会送 ASR。
  - `Type 4` CLEAR_CONTEXT、`Type 6` START_CAMERA —— 每个视频启动前重置会话。
- **S→C**
  - `Type 8` 流式 token（JSON：`response_id / token / is_start / is_final / is_silent / query?`）。
  - `Type 9` 流式 PCM（header：`response_id_len + response_id + sentence_idx + chunk_idx + sample_rate + is_final`）。
  - `Type 10` ASR 转写回显（JSON：`type="asr_query" / query`）。
  - `Type 5` 会被兜底地解码成 PCM 喂回聚合器。

> 若需要跨视频复用连接，把 `Qwen3_VL_online_streaming_v2_ContextManaged.py`
> 里的 `server_sock.listen(1)` 改成 `listen(8)` 即可；当前实现为每个视频
> **新建** 一条连接，所以不强依赖该修改。

## 已知坑

- `cv2.VideoWriter` 需要 mp4v fourcc；若 opencv 没带该编解码，会报 open failed。
  本仓库要求 `opencv-python-headless>=4.12`。
- `silero-vad` 第一次加载会下载 JIT 模型并占用 ~100 MB 内存。
- AURA 主服务默认只接受**单连接**，单实例批跑必然是串行的。
  `launch_batch_aura.sh` 通过启动 2 个独立实例（不同端口）来双路并行。
  如果后续改成多连接，记得同步改 `active_connection` 的语义。
- `data/1q1a` 和 `data/1qna/videos_bench` 是符号链接。Python 的
  `pathlib.rglob()` 在 3.13 以下**不跟随符号链接**，所以 `_discover_videos`
  使用 `os.walk(followlinks=True)` 代替。
