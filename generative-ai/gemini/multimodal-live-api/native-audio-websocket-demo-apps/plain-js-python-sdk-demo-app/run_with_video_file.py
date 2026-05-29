"""Stream a local video file (video stream + audio stream) into Gemini Live.

The transport layer embeds gemini_live.py's session management directly
(send_audio / send_video / receive_loop + event_queue), so this script
exposes 100% the same multi-turn behavior as the main.py front-end demo.

What it does:
  1. Pulls video frames + audio PCM from an MP4 in lockstep and feeds them
     to the Gemini Live API at real-time cadence.
  2. Aggregates the model's text transcription and audio output one
     second at a time and writes them to disk.
  3. Writes a 30 ms-frame-aligned WAV in real time, matching the
     freeze-omni output format (output duration == input duration).

Output directory layout (--output_dir; defaults to ``outputs/<video_name>/``):

    outputs/<video_name>/
    +-- model_output.jsonl       # one line per second; text/audio/user transcript
    +-- model_output.txt         # text-only preview for humans
    +-- responses.jsonl          # one line per turn; delta-level timestamp chunks
    +-- audio_per_second/        # per-second PCM (24 kHz, mono, 16-bit)
    +-- output.wav               # output audio WAV (24 kHz, mono, 16-bit, aligned to input duration)
    +-- events.jsonl             # raw events (debug)

Usage:
    # With Vertex AI (requires gcloud auth application-default login):
    python run_with_video_file.py \\
        --video "/abs/path/to/video1.mp4" \\
        --use_vertex --project_id YOUR_PROJECT --location us-central1

    # With API key:
    export GEMINI_API_KEY=xxx
    python run_with_video_file.py --video "/abs/path/to/video1.mp4"

Dependencies:
    pip install google-genai opencv-python python-dotenv
    ffmpeg / ffprobe must be on PATH.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# =====================================================================
#  Constants
# =====================================================================
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH = 2

OUTPUT_AUDIO_SAMPLE_RATE = 24000
OUTPUT_FRAME_MS = 30
OUTPUT_BYTES_PER_FRAME = (
    OUTPUT_AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH
    * OUTPUT_FRAME_MS // 1000
)  # 1440

# feeder chunk ~ 85 ms, emulating the browser AudioWorklet (4096 @ 48 kHz -> 16 kHz downsample)
FEEDER_CHUNK_SAMPLES = 1365
FEEDER_CHUNK_BYTES = FEEDER_CHUNK_SAMPLES * AUDIO_SAMPLE_WIDTH  # 2730
FEEDER_CHUNK_DUR = FEEDER_CHUNK_SAMPLES / AUDIO_SAMPLE_RATE     # 0.0853 s

DEFAULT_VIDEO_FPS_TO_SEND = 2.0
DEFAULT_JPEG_QUALITY = 80
DEFAULT_MAX_FRAME_EDGE = 640


# =====================================================================
#  Helpers
# =====================================================================
def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found; please install ffmpeg and add it to PATH.")


def get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


# =====================================================================
#  GeminiAggregator -- collects events, buckets output by second, aligns WAV
# =====================================================================
class GeminiAggregator:
    """Aggregate model text + audio output into one bucket per second.

    Also writes WAV in real time at the 30 ms frame cadence, matching the
    freeze-omni output format:
    - For each chunk of input audio fed in, write the matching number of
      output frames (silence-padded as needed).
    - Output duration == input duration.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.audio_seconds_dir = output_dir / "audio_per_second"
        self.audio_seconds_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = output_dir / "events.jsonl"
        self._events_fp = open(self.events_path, "w", encoding="utf-8")
        self._lock = threading.Lock()

        self._current_second: int = 0

        self.text_by_second: Dict[int, List[str]] = defaultdict(list)
        self.asr_by_second: Dict[int, List[str]] = defaultdict(list)
        self.audio_by_second: Dict[int, bytearray] = defaultdict(bytearray)

        # pending buffer + WAV
        self.pending_audio = bytearray()
        self._pending_tids: List[Optional[int]] = []  # turn_id for each pending byte
        self.wav_path = output_dir / "output.wav"
        self._wav_fp: Optional[wave.Wave_write] = None
        self.frames_written = 0
        self.total_frames_expected = 0
        self._muted = False
        self._wav_frame_tids: List[Optional[int]] = []  # turn_id for each WAV frame

        # delta-level timestamps
        self._responses: Dict[int, dict] = {}
        self._current_turn_id: int = 0
        self._in_turn = False
        self._responses_path = output_dir / "responses.jsonl"
        self._responses_fp = open(self._responses_path, "w", encoding="utf-8")

    # --- WAV ---
    def init_wav(self) -> None:
        self._wav_fp = wave.open(str(self.wav_path), "wb")
        self._wav_fp.setnchannels(AUDIO_CHANNELS)
        self._wav_fp.setsampwidth(AUDIO_SAMPLE_WIDTH)
        self._wav_fp.setframerate(OUTPUT_AUDIO_SAMPLE_RATE)

    def set_total_frames(self, total: int) -> None:
        self.total_frames_expected = total

    def write_frame(self, chunk: bytes, tid: Optional[int] = None) -> None:
        if self._wav_fp is not None and self.frames_written < self.total_frames_expected:
            self._wav_fp.writeframes(chunk)
            self._wav_frame_tids.append(tid)
            self.frames_written += 1

    # --- Time ---
    def set_current_second(self, sec: int) -> None:
        self._current_second = sec

    def get_current_second(self) -> int:
        return self._current_second

    # --- pending ---
    def consume_pending_audio(self, max_bytes: int) -> tuple:
        """Return (bytes, dominant_tid)."""
        if self._muted:
            return b"", None
        chunk = bytes(self.pending_audio[:max_bytes])
        tids = self._pending_tids[:len(chunk)]
        self.pending_audio = self.pending_audio[len(chunk):]
        self._pending_tids = self._pending_tids[len(chunk):]
        if tids:
            dominant_tid = Counter(tids).most_common(1)[0][0]
        else:
            dominant_tid = None
        return chunk, dominant_tid

    # --- Event handling ---
    def _ensure_turn(self) -> None:
        """If there's no active turn, create one.

        Works around Gemini races where a text delta arrives before audio.
        """
        if not self._in_turn:
            self.on_turn_start()

    def on_audio(self, pcm: bytes) -> None:
        if self._muted:
            self._muted = False
        self._ensure_turn()
        sec = self._current_second
        tid = self._current_turn_id
        self.audio_by_second[sec].extend(pcm)
        self.pending_audio.extend(pcm)
        self._pending_tids.extend([tid] * len(pcm))
        rec = self._responses.get(tid)
        if rec is not None:
            rec["audio_bytes_total"] += len(pcm)

    def on_output_transcript(self, text: str) -> None:
        self._ensure_turn()
        sec = self._current_second
        self.text_by_second[sec].append(text)
        rec = self._responses.get(self._current_turn_id)
        if rec is not None:
            rec["deltas"].append({
                "text": text,
                "t_arrival_wall": time.time(),
                "audio_bytes_so_far": rec["audio_bytes_total"],
            })
        print(f"[Text @ {sec}s] {text}")

    def on_input_transcript(self, text: str) -> None:
        sec = self._current_second
        self.asr_by_second[sec].append(text)
        print(f"[ASR @ {sec}s] {text}")

    def on_turn_start(self) -> None:
        if self._in_turn:
            return
        self._in_turn = True
        sec = self._current_second
        self._current_turn_id += 1
        tid = self._current_turn_id
        self._responses[tid] = {
            "turn_id": tid,
            "t_created_wall": time.time(),
            "triggered_at_second": sec,
            "deltas": [],
            "audio_bytes_total": 0,
        }
        print(f"[Gemini @ {sec}s] turn started (id={tid})")

    def on_interrupted(self) -> None:
        sec = self._current_second
        print(f"[VAD @ {sec}s] ======INTERRUPTED======")
        self._muted = True
        self.pending_audio.clear()
        self._pending_tids.clear()
        if self._in_turn:
            self._finalize_response(self._current_turn_id, done_second=sec)
            self._in_turn = False

    def on_turn_complete(self) -> None:
        sec = self._current_second
        tid = self._current_turn_id
        print(f"[Gemini @ {sec}s] turn complete (id={tid})")
        self._finalize_response(tid, done_second=sec)
        self._in_turn = False

    def dump_event(self, event: dict) -> None:
        with self._lock:
            self._events_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._events_fp.flush()

    def _finalize_response(self, tid: int, done_second: int) -> None:
        rec = self._responses.pop(tid, None)
        if rec is None:
            return
        deltas = rec["deltas"]
        audio_bytes_total = rec["audio_bytes_total"]
        bytes_per_sec = OUTPUT_AUDIO_SAMPLE_RATE * 2
        audio_duration_sec = (
            round(audio_bytes_total / bytes_per_sec, 6) if bytes_per_sec else 0.0
        )
        weights = [max(len(d["text"]), 1) for d in deltas]
        total_w = sum(weights) if weights else 0
        cursor = 0.0
        chunks: List[dict] = []
        for i, d in enumerate(deltas):
            seg_len = (audio_duration_sec * weights[i] / total_w
                       if total_w > 0 and audio_duration_sec > 0 else 0.0)
            start = cursor
            end = audio_duration_sec if i == len(deltas) - 1 else cursor + seg_len
            cursor = end
            chunks.append({
                "text": d["text"],
                "timestamp": [round(start, 3), round(end, 3)],
                "t_arrival_ms_from_turn_start": round(
                    (d["t_arrival_wall"] - rec["t_created_wall"]) * 1000, 1
                ),
            })
        out = {
            "turn_id": tid,
            "triggered_at_second": rec["triggered_at_second"],
            "done_at_second": done_second,
            "audio_duration_sec": round(audio_duration_sec, 3),
            "text": "".join(d["text"] for d in deltas),
            "chunks": chunks,
            "timestamp_method": "proportional_by_char_count",
        }
        self._responses_fp.write(json.dumps(out, ensure_ascii=False) + "\n")
        self._responses_fp.flush()

    def finalize(self) -> None:
        for tid in list(self._responses.keys()):
            self._finalize_response(tid, done_second=self._current_second)
        self._events_fp.close()
        self._responses_fp.close()

        if self._wav_fp is not None:
            while self.frames_written < self.total_frames_expected:
                self._wav_fp.writeframes(b"\x00" * OUTPUT_BYTES_PER_FRAME)
                self._wav_frame_tids.append(None)
                self.frames_written += 1
            self._wav_fp.close()
            self._wav_fp = None

        self._write_wav_transcript()

        jsonl_path = self.output_dir / "model_output.jsonl"
        txt_path = self.output_dir / "model_output.txt"
        all_seconds = sorted(
            set(self.text_by_second)
            | set(self.asr_by_second)
            | set(self.audio_by_second)
        )
        with open(jsonl_path, "w", encoding="utf-8") as jf, \
             open(txt_path, "w", encoding="utf-8") as tf:
            for sec in all_seconds:
                text = "".join(self.text_by_second.get(sec, []))
                asr = " ".join(self.asr_by_second.get(sec, []))
                audio_bytes = bytes(self.audio_by_second.get(sec, b""))
                audio_rel: Optional[str] = None
                if audio_bytes:
                    audio_fp = self.audio_seconds_dir / f"{sec:04d}.pcm"
                    audio_fp.write_bytes(audio_bytes)
                    audio_rel = str(audio_fp.relative_to(self.output_dir))
                record = {
                    "second": sec,
                    "text": text,
                    "user_transcript": asr,
                    "audio_pcm_path": audio_rel,
                    "audio_sample_rate": OUTPUT_AUDIO_SAMPLE_RATE,
                    "audio_channels": 1,
                    "audio_sample_width": 2,
                    "audio_bytes": len(audio_bytes),
                }
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
                if text or asr:
                    tf.write(f"[{sec:04d}s] TEXT: {text}\n")
                    if asr:
                        tf.write(f"[{sec:04d}s] ASR : {asr}\n")

        wav_duration = self.frames_written * OUTPUT_FRAME_MS / 1000.0
        print(
            "[Finalize] outputs saved:\n"
            f"  - {jsonl_path}\n"
            f"  - {txt_path}\n"
            f"  - {self._responses_path}\n"
            f"  - {self.wav_path}    # WAV duration {wav_duration:.2f}s\n"
            f"  - {self.audio_seconds_dir}/*.pcm\n"
            f"  - {self.output_dir / 'wav_transcript.json'}    # WAV-aligned native text"
        )

    def _write_wav_transcript(self) -> None:
        """Build wav_transcript.json from _wav_frame_tids (asr.py output.json-compatible)."""
        frame_sec = OUTPUT_FRAME_MS / 1000.0

        tid_to_text: Dict[int, str] = {}
        try:
            with open(self._responses_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    tid = rec.get("turn_id")
                    if tid is not None:
                        tid_to_text[tid] = rec.get("text", "")
        except Exception as exc:
            print(f"[WARN] failed to read responses.jsonl; wav_transcript text will be empty: {exc}")

        raw_segments: List[dict] = []
        prev_tid: Optional[int] = None
        seg_start_frame = 0
        for i, tid in enumerate(self._wav_frame_tids):
            if tid != prev_tid:
                if prev_tid is not None:
                    raw_segments.append({
                        "turn_id": prev_tid,
                        "start": seg_start_frame,
                        "end": i,
                    })
                if tid is not None:
                    seg_start_frame = i
                prev_tid = tid
        if prev_tid is not None:
            raw_segments.append({
                "turn_id": prev_tid,
                "start": seg_start_frame,
                "end": len(self._wav_frame_tids),
            })

        # Merge adjacent segments sharing a turn_id (a few None frames may sit between them when pending runs dry)
        segments: List[dict] = []
        for seg in raw_segments:
            if segments and segments[-1]["turn_id"] == seg["turn_id"]:
                segments[-1]["end"] = seg["end"]
            else:
                segments.append(dict(seg))

        for seg in segments:
            seg["text"] = tid_to_text.get(seg["turn_id"], "")
            seg["timestamp"] = [
                round(seg["start"] * frame_sec, 3),
                round(seg["end"] * frame_sec, 3),
            ]

        full_text = " ".join(seg["text"] for seg in segments if seg["text"])
        output = {
            "text": full_text,
            "chunks": [
                {"text": seg["text"], "timestamp": seg["timestamp"]}
                for seg in segments if seg["text"]
            ],
        }
        out_path = self.output_dir / "wav_transcript.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)


# =====================================================================
#  VideoFrameProducer -- background-thread frame extractor
# =====================================================================
class VideoFrameProducer:
    def __init__(
        self, video_path: Path, target_fps: float,
        max_frame_edge: int, jpeg_quality: int,
    ) -> None:
        self.video_path = video_path
        self.target_fps = target_fps
        self.max_frame_edge = max_frame_edge
        self.jpeg_quality = jpeg_quality
        self.frames: "queue.Queue[tuple[float, bytes]]" = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.duration_sec: float = 0.0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _put(self, item: tuple) -> bool:
        while not self._stop.is_set():
            try:
                self.frames.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"[Video] cannot open: {self.video_path}")
            self._put((-1.0, b""))
            return
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration_sec = total / src_fps if src_fps > 0 else 0.0
        step = max(1, int(round(src_fps / max(0.1, self.target_fps))))
        print(
            f"[Video] fps={src_fps:.2f}, frames={total}, "
            f"duration={self.duration_sec:.1f}s, send_every={step}frames "
            f"(~{src_fps / step:.2f} fps)"
        )
        idx = 0
        enc_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        try:
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    h, w = frame.shape[:2]
                    long_edge = max(h, w)
                    if long_edge > self.max_frame_edge:
                        scale = self.max_frame_edge / long_edge
                        frame = cv2.resize(
                            frame, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                    ok, buf = cv2.imencode(".jpg", frame, enc_params)
                    if ok:
                        if not self._put((idx / src_fps, buf.tobytes())):
                            break
                idx += 1
        finally:
            cap.release()
            self._put((-1.0, b""))


# =====================================================================
#  Main
# =====================================================================
async def stream_video_file(
    video_path: Path,
    output_dir: Path,
    model: str,
    voice: str,
    target_video_fps: float,
    max_frame_edge: int,
    jpeg_quality: int,
    realtime: bool,
    use_vertex: bool,
    project_id: Optional[str],
    location: str,
    system_prompt: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregator = GeminiAggregator(output_dir=output_dir)

    # ---- Create the Gemini client ----
    if use_vertex:
        client = genai.Client(vertexai=True, project=project_id, location=location)
    else:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("please set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        client = genai.Client(api_key=api_key)

    # ---- config: identical shape to gemini_live.py ----
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice,
                )
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part(text=system_prompt)]
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        proactivity=types.ProactivityConfig(proactive_audio=True),
    )

    # ---- Video frame producer ----
    video_producer = VideoFrameProducer(
        video_path=video_path,
        target_fps=target_video_fps,
        max_frame_edge=max_frame_edge,
        jpeg_quality=jpeg_quality,
    )
    video_producer.start()

    # ---- Pre-read all audio ----
    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-loglevel", "error",
            "-i", str(video_path),
            "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", str(AUDIO_CHANNELS),
            "-ar", str(AUDIO_SAMPLE_RATE),
            "pipe:1",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert ffmpeg_proc.stdout is not None
    loop = asyncio.get_event_loop()
    raw_audio = await loop.run_in_executor(None, ffmpeg_proc.stdout.read)
    ffmpeg_proc.wait()
    pad = (-len(raw_audio)) % FEEDER_CHUNK_BYTES
    if pad:
        raw_audio += b"\x00" * pad
    audio_chunks = [
        raw_audio[i: i + FEEDER_CHUNK_BYTES]
        for i in range(0, len(raw_audio), FEEDER_CHUNK_BYTES)
    ]

    video_duration = get_video_duration(video_path)
    total_frames_expected = int(video_duration * 1000 / OUTPUT_FRAME_MS)
    aggregator.set_total_frames(total_frames_expected)
    aggregator.init_wav()
    print(
        f"[Main] video duration: {video_duration:.1f}s, "
        f"audio chunks: {len(audio_chunks)} x {FEEDER_CHUNK_DUR*1000:.0f}ms, "
        f"expected output frames: {total_frames_expected}"
    )

    # ---- asyncio queues (same as gemini_live.py) ----
    audio_queue: asyncio.Queue = asyncio.Queue()
    video_queue: asyncio.Queue = asyncio.Queue()
    text_queue: asyncio.Queue = asyncio.Queue()
    event_queue: asyncio.Queue = asyncio.Queue()

    stop_flag = False
    original_sigint = signal.getsignal(signal.SIGINT)

    def _signal_handler(_sig, _frame):
        nonlocal stop_flag
        print("\n[Main] Ctrl+C; shutting down ...")
        stop_flag = True

    signal.signal(signal.SIGINT, _signal_handler)
    start_time = time.time()

    EXACT_FRAMES_PER_CHUNK = FEEDER_CHUNK_DUR / (OUTPUT_FRAME_MS / 1000.0)

    # ================================================================
    #  feeder: push into audio_queue / video_queue at real-time cadence
    # ================================================================
    async def feeder():
        pending_video: Optional[tuple[float, bytes]] = None
        video_done = False
        fractional_frame_acc = 0.0

        for i, chunk in enumerate(audio_chunks):
            if stop_flag:
                break
            media_time = (i + 1) * FEEDER_CHUNK_DUR
            aggregator.set_current_second(int(media_time))

            if realtime:
                target_t = start_time + i * FEEDER_CHUNK_DUR
                delay = target_t - time.time()
                if delay > 0:
                    await asyncio.sleep(delay)

            await audio_queue.put(chunk)

            # WAV-aligned writing (exact cumulative count, no drift)
            fractional_frame_acc += EXACT_FRAMES_PER_CHUNK
            frames_this_chunk = int(fractional_frame_acc) - aggregator.frames_written
            for _ in range(frames_this_chunk):
                if aggregator.frames_written >= total_frames_expected:
                    break
                out, tid = aggregator.consume_pending_audio(OUTPUT_BYTES_PER_FRAME)
                if len(out) < OUTPUT_BYTES_PER_FRAME:
                    out += b"\x00" * (OUTPUT_BYTES_PER_FRAME - len(out))
                aggregator.write_frame(out, tid=tid)

            # Pump video frames
            while not video_done:
                if pending_video is None:
                    try:
                        pending_video = video_producer.frames.get_nowait()
                    except queue.Empty:
                        break
                t_sec, jpeg_bytes = pending_video
                if t_sec < 0:
                    pending_video = None
                    video_done = True
                    print("[Main] video frame stream ended")
                    break
                if t_sec <= media_time:
                    await video_queue.put(jpeg_bytes)
                    pending_video = None
                else:
                    break

            if (i + 1) % 100 == 0:
                print(
                    f"[Main] media_time={media_time:.1f}s, "
                    f"wall={time.time() - start_time:.1f}s"
                )

        print(f"[Main] audio feed complete, wall={time.time() - start_time:.1f}s")
        # Push a sentinel so send_audio knows the audio stream is over
        await audio_queue.put(None)

    # ================================================================
    #  Gemini session -- behavior identical to gemini_live.py
    # ================================================================
    try:
        async with client.aio.live.connect(model=model, config=config) as session:
            print(f"[Main] Gemini Live session established, model={model}")

            # ---- send_audio: based on gemini_live.py, plus a sentinel-on-end ----
            audio_finished = asyncio.Event()

            async def send_audio():
                try:
                    while True:
                        chunk = await audio_queue.get()
                        if chunk is None:
                            try:
                                await session.send_realtime_input(
                                    audio_stream_end=True
                                )
                            except Exception as exc:
                                print(f"[send_audio] audio_stream_end not supported, relying on timeout: {exc}")
                            audio_finished.set()
                            return
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=chunk,
                                mime_type=f"audio/pcm;rate={AUDIO_SAMPLE_RATE}",
                            )
                        )
                except asyncio.CancelledError:
                    pass

            # ---- send_video: line-for-line copy of gemini_live.py ----
            async def send_video():
                try:
                    while True:
                        chunk = await video_queue.get()
                        await session.send_realtime_input(
                            video=types.Blob(
                                data=chunk, mime_type="image/jpeg",
                            )
                        )
                except asyncio.CancelledError:
                    pass

            # ---- send_text: line-for-line copy of gemini_live.py ----
            async def send_text():
                try:
                    while True:
                        text = await text_queue.get()
                        await session.send(input=text, end_of_turn=True)
                except asyncio.CancelledError:
                    pass

            # ---- receive_loop: line-for-line copy of gemini_live.py ----
            async def receive_loop():
                try:
                    while True:
                        async for response in session.receive():
                            server_content = response.server_content
                            if not server_content:
                                continue

                            # model audio
                            if server_content.model_turn:
                                for part in server_content.model_turn.parts:
                                    if part.inline_data:
                                        aggregator.on_turn_start()
                                        aggregator.on_audio(part.inline_data.data)

                            # user ASR
                            if (server_content.input_transcription
                                    and server_content.input_transcription.text):
                                await event_queue.put({
                                    "type": "user",
                                    "text": server_content.input_transcription.text,
                                })

                            # model text
                            if (server_content.output_transcription
                                    and server_content.output_transcription.text):
                                await event_queue.put({
                                    "type": "gemini",
                                    "text": server_content.output_transcription.text,
                                })

                            # turn_complete
                            if server_content.turn_complete:
                                await event_queue.put({"type": "turn_complete"})

                            # interrupted
                            if server_content.interrupted:
                                aggregator.on_interrupted()
                                await event_queue.put({"type": "interrupted"})

                except Exception as e:
                    await event_queue.put({"type": "error", "error": str(e)})
                finally:
                    await event_queue.put(None)

            # ---- Shutdown timer: wait for the final model turn after audio ends ----
            async def shutdown_timer():
                await audio_finished.wait()
                print("[Main] audio stream ended; waiting up to 10s for the final model reply ...")
                await asyncio.sleep(10.0)
                await event_queue.put(None)

            # ---- Start all tasks ----
            feeder_task = asyncio.create_task(feeder())
            send_audio_task = asyncio.create_task(send_audio())
            send_video_task = asyncio.create_task(send_video())
            send_text_task = asyncio.create_task(send_text())
            receive_task = asyncio.create_task(receive_loop())
            shutdown_task = asyncio.create_task(shutdown_timer())

            # ---- Main loop: consume event_queue ----
            try:
                while True:
                    event = await event_queue.get()
                    if event is None or stop_flag:
                        break

                    etype = event.get("type", "")
                    now = time.time()

                    if etype == "user":
                        aggregator.on_input_transcript(event["text"])
                        aggregator.dump_event({
                            "type": "input_transcription",
                            "text": event["text"], "t": now,
                        })

                    elif etype == "gemini":
                        aggregator.on_output_transcript(event["text"])
                        aggregator.dump_event({
                            "type": "output_transcription",
                            "text": event["text"], "t": now,
                        })

                    elif etype == "turn_complete":
                        aggregator.on_turn_complete()
                        aggregator.dump_event({"type": "turn_complete", "t": now})

                    elif etype == "interrupted":
                        aggregator.dump_event({"type": "interrupted", "t": now})

                    elif etype == "error":
                        print(f"[Error] {event.get('error')}")
                        aggregator.dump_event({
                            "type": "error", "error": event.get("error"),
                            "t": now,
                        })
                        break

            finally:
                feeder_task.cancel()
                send_audio_task.cancel()
                send_video_task.cancel()
                send_text_task.cancel()
                receive_task.cancel()
                shutdown_task.cancel()

        # ---- Trailing flush: drain leftover model audio from pending buffer ----
        trail_start = time.time()
        final_media_time = len(audio_chunks) * FEEDER_CHUNK_DUR
        while time.time() - trail_start < 5.0 and len(aggregator.pending_audio) > 0:
            aggregator.set_current_second(
                int(final_media_time + (time.time() - trail_start))
            )
            out, tid = aggregator.consume_pending_audio(OUTPUT_BYTES_PER_FRAME)
            if len(out) < OUTPUT_BYTES_PER_FRAME:
                out += b"\x00" * (OUTPUT_BYTES_PER_FRAME - len(out))
            aggregator.write_frame(out, tid=tid)
            await asyncio.sleep(OUTPUT_FRAME_MS / 1000.0)

    finally:
        signal.signal(signal.SIGINT, original_sigint)
        print("[Main] closing ffmpeg / video producer ...")
        try:
            ffmpeg_proc.terminate()
        except Exception:
            pass
        video_producer.stop()
        aggregator.finalize()


# =====================================================================
#  CLI
# =====================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--video", required=False,
        default=str(
            Path(__file__).resolve().parents[5]
            / "data/1q1a/videos/0001.mp4"
        ),
        help="input video path (must contain both audio and video tracks)",
    )
    p.add_argument("--output_dir", default=None, help="output directory")
    p.add_argument("--model", default="gemini-live-2.5-flash-native-audio")
    p.add_argument("--voice", default="Puck")
    p.add_argument("--video_fps", type=float, default=DEFAULT_VIDEO_FPS_TO_SEND)
    p.add_argument("--max_frame_edge", type=int, default=DEFAULT_MAX_FRAME_EDGE)
    p.add_argument("--jpeg_quality", type=int, default=DEFAULT_JPEG_QUALITY)
    p.add_argument("--no_realtime", action="store_true",
                    help="don't feed at real-time cadence (debug only)")
    p.add_argument("--use_vertex", action="store_true",
                    help="use the Vertex AI endpoint (requires gcloud auth)")
    p.add_argument("--project_id", default=None, help="GCP project ID")
    p.add_argument("--location", default="us-central1")
    p.add_argument(
        "--system_prompt",
        default=(
            "You are a helpful and friendly AI assistant. "
            "RESPOND IN THE SAME LANGUAGE AS THE USER. "
            "YOU MUST RESPOND UNMISTAKABLY IN THE USER'S LANGUAGE."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    check_ffmpeg()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        print(f"video file not found: {video_path}")
        sys.exit(1)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path(__file__).resolve().parent / "outputs" / video_path.stem)
    )
    print(f"[Main] video:  {video_path}")
    print(f"[Main] output: {output_dir}")
    asyncio.run(stream_video_file(
        video_path=video_path, output_dir=output_dir,
        model=args.model, voice=args.voice,
        target_video_fps=args.video_fps,
        max_frame_edge=args.max_frame_edge,
        jpeg_quality=args.jpeg_quality,
        realtime=not args.no_realtime,
        use_vertex=args.use_vertex,
        project_id=args.project_id,
        location=args.location,
        system_prompt=args.system_prompt,
    ))


if __name__ == "__main__":
    main()
