"""Stream a local video file (video stream + audio stream) into Qwen-Omni Realtime.

What it does:
  1. Pulls video frames + audio PCM from an MP4 in lockstep and feeds them
     to DashScope's Qwen-Omni Realtime API at real-time cadence.
  2. Aggregates the model's ``response.audio_transcript.delta`` text,
     ``response.audio.delta`` audio, and the user-side speech transcript
     into one-second buckets on disk.
  3. Writes a 30 ms-frame-aligned WAV in real time, matching the
     freeze-omni output format (output duration == input duration).

Output directory layout (--output_dir; defaults to ``outputs/<video_name>/``):

    outputs/<video_name>/
    +-- model_output.jsonl       # one line per second; text / audio / user transcript
    +-- model_output.txt         # text-only preview for humans
    +-- responses.jsonl          # one line per response; delta-level timestamp chunks
    +-- audio_per_second/        # per-second PCM (24 kHz, mono, 16-bit)
    +-- output.wav               # output audio WAV (24 kHz, mono, 16-bit, aligned to input duration)
    +-- events.jsonl             # raw events (debug)

Usage:
    export DASHSCOPE_API_KEY=sk-xxx
    python run_with_video_file.py \
        --video "/abs/path/to/video1.mp4" \
        --output_dir outputs/video1

Dependencies:
    pip install dashscope opencv-python
    ffmpeg / ffprobe must be on PATH.
"""

from __future__ import annotations

import argparse
import base64
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
import dashscope
from dashscope.audio.qwen_omni import (
    AudioFormat,
    MultiModality,
    OmniRealtimeCallback,
    OmniRealtimeConversation,
)


AUDIO_SAMPLE_RATE = 16000           # input audio sample rate sent to the model
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH = 2              # 16-bit PCM
AUDIO_CHUNK_MS = 30                 # send 30 ms of audio per chunk (aligned with freeze-omni)
AUDIO_CHUNK_BYTES = (
    AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH * AUDIO_CHUNK_MS // 1000
)
OUTPUT_AUDIO_SAMPLE_RATE = 24000    # sample rate of the audio the model returns
OUTPUT_FRAME_MS = 30                # output frame duration, aligned with freeze-omni
OUTPUT_BYTES_PER_FRAME = (
    OUTPUT_AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH * OUTPUT_FRAME_MS // 1000
)  # 24000 * 2 * 30 / 1000 = 1440 bytes

DEFAULT_VIDEO_FPS_TO_SEND = 2.0     # default video frame send rate (2 fps)
DEFAULT_JPEG_QUALITY = 80           # JPEG encoding quality
DEFAULT_MAX_FRAME_EDGE = 640        # max edge length for video frames (pixels)


# ---------- Helpers ----------
def init_dashscope_api_key() -> None:
    if "DASHSCOPE_API_KEY" in os.environ:
        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    else:
        raise RuntimeError(
            "please set the DASHSCOPE_API_KEY environment variable, or hard-code it inside init_dashscope_api_key."
        )


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found; please install ffmpeg and add it to PATH.")


def get_video_duration(video_path: Path) -> float:
    """Return the video duration in seconds (uses ffprobe)."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


# ---------- Callback: collect events, bucket output per second ----------
class OmniAggregator(OmniRealtimeCallback):
    """Aggregate model text + audio output into one bucket per second.

    The "current second" is updated by the main (sender) thread using wall
    clock starting from the first audio chunk.

    Also writes the WAV file in real time at the 30 ms frame cadence,
    matching the freeze-omni output format:
    - One output frame per input frame (silence-padded as needed).
    - Output duration == input duration.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.audio_seconds_dir = output_dir / "audio_per_second"
        self.audio_seconds_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = output_dir / "events.jsonl"
        self._events_fp = open(self.events_path, "w", encoding="utf-8")
        self._events_lock = threading.Lock()

        self._current_second: int = 0
        self._second_lock = threading.Lock()

        self.text_by_second: Dict[int, List[str]] = defaultdict(list)
        self.asr_by_second: Dict[int, List[str]] = defaultdict(list)
        self.audio_by_second: Dict[int, bytearray] = defaultdict(bytearray)
        self.language_by_second: Dict[int, List[str]] = defaultdict(list)
        self._bucket_lock = threading.Lock()

        # === Freeze-omni style: pending buffer + real-time WAV writing ===
        self.pending_audio = bytearray()  # received-audio buffer
        self._pending_rids: List[Optional[str]] = []  # response_id per byte
        self._pending_lock = threading.Lock()
        self.wav_path = output_dir / "output.wav"  # name matches freeze-omni
        self._wav_fp: Optional[wave.Wave_write] = None
        self._wav_lock = threading.Lock()
        self.frames_written = 0  # frames already written
        self.total_frames_expected = 0  # total frames (set by the main loop)
        # Set on speech_started (mirrors freeze-omni stop_tts -> muted=True);
        # cleared when the first audio chunk of the next response arrives
        # (mirrors freeze-omni _on_audio -> muted=False).
        self._muted = False
        # On speech_started, record the interrupted response_id so we can
        # drop its in-flight tail packets.
        self._interrupted_response_id: Optional[str] = None

        # WAV-frame -> response mapping: [(frame_idx, rid_or_None), ...]
        # Used at finalize time to compute each response's WAV time range.
        self._wav_frame_rids: List[Optional[str]] = []

        self.session_id: Optional[str] = None
        self.connected_event = threading.Event()
        self.closed_event = threading.Event()

        # The server requires every turn's first message to be audio.
        # Set after response.done; the main loop clears it after the next
        # successful append_audio. While set, the main loop won't send video.
        self._needs_audio_first = threading.Event()

        # Delta-level timestamp records: one chunk list per response.
        # self._responses is keyed by response_id; value is dict(meta + deltas).
        self._responses: Dict[str, dict] = {}
        # The currently in-flight response_id (set by response.created,
        # cleared by response.done).
        self._current_response_id: Optional[str] = None
        # Output file handle; one record per response.done.
        self._responses_path = output_dir / "responses.jsonl"
        self._responses_fp = open(self._responses_path, "w", encoding="utf-8")

    def init_wav(self) -> None:
        """Initialize the WAV file (called by the main loop at startup)."""
        with self._wav_lock:
            self._wav_fp = wave.open(str(self.wav_path), "wb")
            self._wav_fp.setnchannels(AUDIO_CHANNELS)
            self._wav_fp.setsampwidth(AUDIO_SAMPLE_WIDTH)
            self._wav_fp.setframerate(OUTPUT_AUDIO_SAMPLE_RATE)

    def set_total_frames(self, total: int) -> None:
        """Set the expected total frame count (derived from input duration)."""
        self.total_frames_expected = total

    def write_frame(self, chunk: bytes, rid: Optional[str] = None) -> None:
        """Write one audio frame to the WAV file and record its response_id."""
        with self._wav_lock:
            if self._wav_fp is not None and self.frames_written < self.total_frames_expected:
                self._wav_fp.writeframes(chunk)
                self._wav_frame_rids.append(rid)
                self.frames_written += 1

    # --- Public: update the current second ---
    def set_current_second(self, sec: int) -> None:
        with self._second_lock:
            self._current_second = sec

    def get_current_second(self) -> int:
        with self._second_lock:
            return self._current_second

    def consume_pending_audio(self, max_bytes: int) -> tuple:
        """Pull up to max_bytes audio bytes from the pending buffer.

        Returns (bytes, dominant_rid):
          - bytes: PCM data (may be empty)
          - dominant_rid: response_id that contributed the most bytes
            (None means silence).
        """
        if self._muted:
            return b"", None
        with self._pending_lock:
            chunk = bytes(self.pending_audio[:max_bytes])
            rids = self._pending_rids[:len(chunk)]
            self.pending_audio = self.pending_audio[len(chunk):]
            self._pending_rids = self._pending_rids[len(chunk):]
            # Pick the rid that contributed the most bytes in this chunk
            if rids:
                dominant_rid = Counter(rids).most_common(1)[0][0]
            else:
                dominant_rid = None
            return chunk, dominant_rid

    # --- OmniRealtimeCallback interface ---
    def on_open(self) -> None:
        print("[Omni] connection opened")
        self.connected_event.set()

    def on_close(self, close_status_code, close_msg) -> None:
        print(
            f"[Omni] connection closed, code={close_status_code}, msg={close_msg}"
        )
        self.closed_event.set()

    def on_event(self, response: dict) -> None:
        try:
            self._dump_event(response)
            ev_type = response.get("type", "")
            sec = self.get_current_second()

            if ev_type == "session.created":
                self.session_id = response["session"]["id"]
                print(f"[Omni] session created: {self.session_id}")

            elif ev_type == "conversation.item.input_audio_transcription.completed":
                transcript = response.get("transcript", "")
                lang = response.get("language") or response.get("lang")
                with self._bucket_lock:
                    if transcript:
                        self.asr_by_second[sec].append(transcript)
                    if lang:
                        self.language_by_second[sec].append(lang)
                print(f"[ASR @ {sec}s] {transcript} (lang={lang})")

            elif ev_type == "response.created":
                rid = response.get("response", {}).get("id", "")
                if rid:
                    self._current_response_id = rid
                    with self._bucket_lock:
                        self._responses[rid] = {
                            "response_id": rid,
                            "t_created_wall": time.time(),
                            "triggered_at_second": sec,
                            "deltas": [],  # [{text, t_arrival_wall, audio_bytes_so_far}]
                            "audio_bytes_total": 0,
                        }
                    print(f"[Omni @ {sec}s] response created: {rid}")

            elif ev_type == "response.audio_transcript.delta":
                delta = response.get("delta", "")
                if delta:
                    rid = response.get("response_id") or self._current_response_id
                    with self._bucket_lock:
                        self.text_by_second[sec].append(delta)
                        rec = self._responses.get(rid) if rid else None
                        if rec is not None:
                            rec["deltas"].append({
                                "text": delta,
                                "t_arrival_wall": time.time(),
                                "audio_bytes_so_far": rec["audio_bytes_total"],
                            })
                    print(f"[Text @ {sec}s] {delta}")

            elif ev_type == "response.audio.delta":
                b64 = response.get("delta", "")
                if b64:
                    pcm = base64.b64decode(b64)
                    rid = response.get("response_id") or self._current_response_id
                    # Tail packets from the interrupted response are dropped
                    # (not appended to pending). Mirrors freeze-omni's
                    # pending.clear() after stop_tts: old audio inside the
                    # interruption window must not land in output.wav.
                    if (
                        self._muted
                        and self._interrupted_response_id is not None
                        and rid == self._interrupted_response_id
                    ):
                        return
                    # First packet of the next response -> unmute.
                    # Mirrors freeze-omni _on_audio: muted=False on first
                    # new audio chunk (not on response.done).
                    if self._muted:
                        self._muted = False
                    with self._bucket_lock:
                        # Still bucket per second (used by model_output.jsonl)
                        self.audio_by_second[sec].extend(pcm)
                    # Append to pending buffer (for real-time WAV writes) and tag rid
                    with self._pending_lock:
                        self.pending_audio.extend(pcm)
                        self._pending_rids.extend([rid] * len(pcm))
                        rec = self._responses.get(rid) if rid else None
                        if rec is not None:
                            rec["audio_bytes_total"] += len(pcm)

            elif ev_type == "input_audio_buffer.speech_started":
                print(f"[VAD @ {sec}s] speech started ======VAD Speech Start======")
                # Mirrors run_with_camera.py b64_player.cancel_playing()
                # Mirrors freeze-omni _on_stop_tts: pending.clear() + muted=True
                self._interrupted_response_id = self._current_response_id
                self._muted = True
                with self._pending_lock:
                    self.pending_audio.clear()
                    self._pending_rids.clear()

            elif ev_type == "input_audio_buffer.speech_stopped":
                print(f"[VAD @ {sec}s] speech stopped")
                # Stay muted; we unmute when the next response's first audio packet arrives

            elif ev_type == "response.done":
                rid = (
                    response.get("response", {}).get("id")
                    or response.get("response_id")
                    or self._current_response_id
                )
                print(f"[Omni @ {sec}s] response done (id={rid})")
                self._finalize_response(rid, done_second=sec)
                self._current_response_id = None
                # Don't unmute here on response.done. Unmute happens when
                # the first response.audio.delta of the next response arrives
                # (mirrors freeze-omni _on_audio -> muted=False).
                # Each new turn must receive audio before it can receive video.
                self._needs_audio_first.set()

            elif ev_type == "error":
                err_msg = response.get("error", {}).get("message", "")
                if "append image before append audio" in err_msg:
                    # Benign race (our video send overlaps with response.done).
                    # Set needs_audio_first so the next frame waits for audio.
                    print(f"[Omni WARN] (benign race) {err_msg}; auto-recovered")
                    self._needs_audio_first.set()
                else:
                    print(f"[Omni ERROR] {response}")

        except Exception as exc:  # noqa: BLE001
            print(f"[Callback Error] {exc}")

    # --- Internal ---
    def _finalize_response(self, rid: Optional[str], done_second: int) -> None:
        """Convert one response's deltas into [start, end] chunks and write
        a record to responses.jsonl.

        Qwen-Omni Realtime ships every text delta before every audio delta,
        so we can't align using "audio_bytes_so_far at text arrival".
        Instead we distribute the chunk durations proportionally to text
        character count, which guarantees:
          - each chunk's [start, end] is monotonically increasing and
            non-overlapping
          - the last chunk's end == audio_duration_sec
          - total coverage == actual synthesized audio duration
        Accuracy is good enough for token-level alignment in the
        full-duplex benchmark. The real wall-clock arrival delay for each
        delta is preserved separately in t_arrival_ms_from_response_created.
        """
        if not rid:
            return
        with self._bucket_lock:
            rec = self._responses.pop(rid, None)
            if rec is None:
                return
            deltas = rec["deltas"]
            audio_bytes_total = rec["audio_bytes_total"]
            bytes_per_sec = OUTPUT_AUDIO_SAMPLE_RATE * 2  # 24kHz * 16-bit
            audio_duration_sec = (
                round(audio_bytes_total / bytes_per_sec, 6) if bytes_per_sec else 0.0
            )

            # Allocate timestamps proportionally to char count (empty text gets weight 1)
            weights = [max(len(d["text"]), 1) for d in deltas]
            total_w = sum(weights) if weights else 0
            cursor = 0.0
            chunks: List[dict] = []
            for i, d in enumerate(deltas):
                if total_w > 0 and audio_duration_sec > 0:
                    seg_len = audio_duration_sec * weights[i] / total_w
                else:
                    seg_len = 0.0
                start = cursor
                # Cap the last segment at audio_duration to avoid float drift
                end = (
                    audio_duration_sec
                    if i == len(deltas) - 1
                    else cursor + seg_len
                )
                cursor = end
                chunks.append({
                    "text": d["text"],
                    "timestamp": [round(start, 3), round(end, 3)],
                    "t_arrival_ms_from_response_created": round(
                        (d["t_arrival_wall"] - rec["t_created_wall"]) * 1000, 1
                    ),
                })
            out = {
                "response_id": rid,
                "triggered_at_second": rec["triggered_at_second"],
                "done_at_second": done_second,
                "audio_duration_sec": round(audio_duration_sec, 3),
                "text": "".join(d["text"] for d in deltas),
                "chunks": chunks,
                "timestamp_method": "proportional_by_char_count",
            }
            self._responses_fp.write(json.dumps(out, ensure_ascii=False) + "\n")
            self._responses_fp.flush()

    def _dump_event(self, event: dict) -> None:
        with self._events_lock:
            # Size cap: drop base64 from audio deltas (events.jsonl keeps only structural info)
            if event.get("type") == "response.audio.delta" and "delta" in event:
                event = {**event, "delta": f"<{len(event['delta'])} b64 chars>"}
            self._events_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._events_fp.flush()

    # --- Wrap-up: flush per-second buckets to disk ---
    def finalize(self) -> None:
        # Force-flush any responses that never received response.done
        with self._bucket_lock:
            unfinished_ids = list(self._responses.keys())
        for rid in unfinished_ids:
            self._finalize_response(rid, done_second=self.get_current_second())

        with self._events_lock:
            self._events_fp.close()
        self._responses_fp.close()

        # Pad remaining frames with silence and close the WAV
        with self._wav_lock:
            if self._wav_fp is not None:
                while self.frames_written < self.total_frames_expected:
                    self._wav_fp.writeframes(b"\x00" * OUTPUT_BYTES_PER_FRAME)
                    self._wav_frame_rids.append(None)
                    self.frames_written += 1
                self._wav_fp.close()
                self._wav_fp = None

        # === Build wav_transcript.json: precise time per speech segment + native text ===
        self._write_wav_transcript()

        jsonl_path = self.output_dir / "model_output.jsonl"
        txt_path = self.output_dir / "model_output.txt"

        with self._bucket_lock:
            all_seconds = sorted(
                set(self.text_by_second)
                | set(self.asr_by_second)
                | set(self.audio_by_second)
                | set(self.language_by_second)
            )

            with open(jsonl_path, "w", encoding="utf-8") as jf, open(
                txt_path, "w", encoding="utf-8"
            ) as tf:
                for sec in all_seconds:
                    text = "".join(self.text_by_second.get(sec, []))
                    asr = " ".join(self.asr_by_second.get(sec, []))
                    audio_bytes = bytes(self.audio_by_second.get(sec, b""))
                    lang = (
                        self.language_by_second.get(sec, [None])[0]
                        if self.language_by_second.get(sec)
                        else None
                    )

                    audio_rel: Optional[str] = None
                    if audio_bytes:
                        audio_fp = self.audio_seconds_dir / f"{sec:04d}.pcm"
                        audio_fp.write_bytes(audio_bytes)
                        audio_rel = str(audio_fp.relative_to(self.output_dir))

                    record = {
                        "second": sec,
                        "text": text,
                        "user_transcript": asr,
                        "language": lang,
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
            f"  - {self._responses_path}    # one record per response, with delta-level timestamps\n"
            f"  - {self.wav_path}    # output WAV (duration {wav_duration:.2f}s, aligned to input)\n"
            f"  - {self.audio_seconds_dir}/*.pcm\n"
            f"  - {self.output_dir / 'wav_transcript.json'}    # WAV-aligned native text"
        )

    def _write_wav_transcript(self) -> None:
        """Build wav_transcript.json from _wav_frame_rids.

        The format is asr.py output.json-compatible:
        {
          "text": "full text",
          "chunks": [{"text": "hello", "timestamp": [3.21, 4.53]}, ...]
        }

        Each chunk corresponds to one contiguous response segment in the WAV;
        text comes from the model's native output (not ASR), so it's 100% accurate.
        """
        frame_sec = OUTPUT_FRAME_MS / 1000.0

        # Read the finalized responses.jsonl to get the response_id -> text mapping
        rid_to_text: Dict[str, str] = {}
        try:
            with open(self._responses_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rid = rec.get("response_id", "")
                    if rid:
                        rid_to_text[rid] = rec.get("text", "")
        except Exception as exc:
            print(f"[WARN] failed to read responses.jsonl; wav_transcript text will be empty: {exc}")

        # Scan _wav_frame_rids and collapse runs of identical rids into segments
        raw_segments: List[dict] = []
        prev_rid: Optional[str] = None
        seg_start_frame = 0
        for i, rid in enumerate(self._wav_frame_rids):
            if rid != prev_rid:
                if prev_rid is not None:
                    raw_segments.append({
                        "response_id": prev_rid,
                        "start": seg_start_frame,
                        "end": i,
                    })
                if rid is not None:
                    seg_start_frame = i
                prev_rid = rid
        if prev_rid is not None:
            raw_segments.append({
                "response_id": prev_rid,
                "start": seg_start_frame,
                "end": len(self._wav_frame_rids),
            })

        # Merge adjacent segments sharing a response_id (a few None frames may sit between them when pending runs dry)
        segments: List[dict] = []
        for seg in raw_segments:
            if segments and segments[-1]["response_id"] == seg["response_id"]:
                segments[-1]["end"] = seg["end"]
            else:
                segments.append(dict(seg))

        for seg in segments:
            seg["text"] = rid_to_text.get(seg["response_id"], "")
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


# ---------- Audio extraction: ffmpeg subprocess pipe ----------
def spawn_ffmpeg_audio_pipe(video_path: Path) -> subprocess.Popen:
    """Decode the video's audio track to 16 kHz mono 16-bit PCM on a binary pipe."""
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vn",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", str(AUDIO_CHANNELS),
        "-ar", str(AUDIO_SAMPLE_RATE),
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# ---------- Video frame extraction: background thread ----------
class VideoFrameProducer:
    """Pull frames from the video at a target fps (lower than source fps),
    JPEG-encode them, base64-encode them, and put them on a queue.

    Each produced frame carries ``t_seconds`` (its video timestamp); the
    main loop ships frames out using wall-clock matching.
    """

    def __init__(
        self,
        video_path: Path,
        target_fps: float,
        max_frame_edge: int,
        jpeg_quality: int,
    ) -> None:
        self.video_path = video_path
        self.target_fps = target_fps
        self.max_frame_edge = max_frame_edge
        self.jpeg_quality = jpeg_quality

        self.frames: "queue.Queue[tuple[float, str]]" = queue.Queue(maxsize=64)
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.total_frames: int = 0
        self.src_fps: float = 0.0
        self.duration_sec: float = 0.0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        self._thread.join(timeout=2)

    def _put(self, item: tuple) -> bool:
        """stop_flag-aware put; never blocks forever when the queue is full. Returns False if stopped."""
        while not self._stop_flag.is_set():
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
            self._put((-1.0, ""))  # sentinel
            return

        self.src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.total_frames = total
        self.duration_sec = total / self.src_fps if self.src_fps > 0 else 0.0
        step = max(1, int(round(self.src_fps / max(0.1, self.target_fps))))
        print(
            f"[Video] fps={self.src_fps:.2f}, frames={total}, "
            f"duration={self.duration_sec:.1f}s, send_every={step}frames "
            f"(~{self.src_fps/step:.2f} fps)"
        )

        idx = 0
        enc_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        try:
            while not self._stop_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    h, w = frame.shape[:2]
                    long_edge = max(h, w)
                    if long_edge > self.max_frame_edge:
                        scale = self.max_frame_edge / long_edge
                        frame = cv2.resize(
                            frame,
                            (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                    ok, buf = cv2.imencode(".jpg", frame, enc_params)
                    if ok:
                        t_seconds = idx / self.src_fps
                        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                        if not self._put((t_seconds, b64)):
                            break
                idx += 1
        finally:
            cap.release()
            self._put((-1.0, ""))  # sentinel


# ---------- Main ----------
DEFAULT_PROBE_INSTRUCTIONS = (
    "Based on what you can currently see and hear, describe in one short "
    "sentence (under 15 words) what is happening right now."
)


def stream_video_file(
    video_path: Path,
    output_dir: Path,
    model: str,
    voice: str,
    target_video_fps: float,
    max_frame_edge: int,
    jpeg_quality: int,
    realtime: bool,
    turn_mode: str,                    # 'server_vad' or 'manual'
    probe_every_sec: float,            # how often (sec) to fire create_response in manual mode
    probe_instructions: str,           # instructions sent in manual-mode probes
    probe_cancel_previous: bool,       # cancel the previous response before triggering a new one (manual mode)
    probe_first_delay_sec: float,      # first probe delay in manual mode
    turn_silence_ms: int,              # silence threshold for server_vad
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregator = OmniAggregator(output_dir=output_dir)
    conversation = OmniRealtimeConversation(model=model, callback=aggregator)
    conversation.connect()

    session_kwargs = dict(
        output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
        voice=voice,
        input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
        output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        enable_input_audio_transcription=True,
        input_audio_transcription_model="gummy-realtime-v1",
    )
    if turn_mode == "manual":
        session_kwargs["enable_turn_detection"] = False
        print(
            f"[Main] manual mode, create_response every {probe_every_sec:.2f}s, "
            f"cancel_previous={probe_cancel_previous}"
        )
    else:
        session_kwargs["enable_turn_detection"] = True
        session_kwargs["turn_detection_type"] = "server_vad"
        session_kwargs["turn_detection_silence_duration_ms"] = int(turn_silence_ms)
        print(
            f"[Main] server_vad mode, silence_ms={turn_silence_ms}"
        )
    conversation.update_session(**session_kwargs)

    video_producer = VideoFrameProducer(
        video_path=video_path,
        target_fps=target_video_fps,
        max_frame_edge=max_frame_edge,
        jpeg_quality=jpeg_quality,
    )
    video_producer.start()

    ffmpeg_proc = spawn_ffmpeg_audio_pipe(video_path)
    assert ffmpeg_proc.stdout is not None

    stop_flag = threading.Event()

    def _signal_handler(_sig, _frame):
        print("[Main] Ctrl+C; shutting down ...")
        stop_flag.set()

    signal.signal(signal.SIGINT, _signal_handler)

    start_time = time.time()
    total_sent_audio_bytes = 0
    chunk_idx = 0
    pending_video: Optional[tuple[float, str]] = None
    video_done = False

    # manual-probe-mode state
    next_probe_media_time = probe_first_delay_sec

    def _elapsed() -> float:
        return time.time() - start_time

    def _issue_probe(media_time: float) -> None:
        # Cancel the previous response first; if there was none, the server
        # may ignore or error -- swallow either.
        if probe_cancel_previous:
            try:
                conversation.cancel_response()
            except Exception:  # noqa: BLE001
                pass
        try:
            conversation.commit()
        except Exception as exc:  # noqa: BLE001
            # commit usually fails when no new audio arrived since the last
            # commit -- safe to skip.
            print(f"[Probe @ {media_time:.1f}s] commit failed (ignored): {exc}")
            return
        try:
            conversation.create_response(
                instructions=probe_instructions,
                output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
            )
            print(f"[Probe @ {media_time:.1f}s] commit + create_response issued")
        except Exception as exc:  # noqa: BLE001
            print(f"[Probe @ {media_time:.1f}s] create_response failed: {exc}")

    print(f"[Main] starting A/V streaming ... (realtime={realtime})")

    # Get video duration and derive the expected total output frame count
    video_duration = get_video_duration(video_path)
    total_frames_expected = int(video_duration * 1000 / OUTPUT_FRAME_MS)
    aggregator.set_total_frames(total_frames_expected)
    aggregator.init_wav()
    print(f"[Main] video duration: {video_duration:.1f}s, expected output frames: {total_frames_expected}")

    frame_dur = OUTPUT_FRAME_MS / 1000.0

    try:
        while not stop_flag.is_set():
            audio_chunk = ffmpeg_proc.stdout.read(AUDIO_CHUNK_BYTES)
            if not audio_chunk:
                print("[Main] end of audio stream")
                break
            # Only the file tail can short-read; zero-pad and the next read returns b''
            if len(audio_chunk) < AUDIO_CHUNK_BYTES:
                audio_chunk = audio_chunk + b"\x00" * (AUDIO_CHUNK_BYTES - len(audio_chunk))

            chunk_idx += 1
            total_sent_audio_bytes += AUDIO_CHUNK_BYTES
            media_time = (
                total_sent_audio_bytes
                / (AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH)
            )
            aggregator.set_current_second(int(media_time))

            try:
                conversation.append_audio(base64.b64encode(audio_chunk).decode("ascii"))
            except Exception as exc:  # noqa: BLE001
                print(f"[Main] append_audio failed: {exc}")
                break
            # First audio of the new turn went through; OK to send video again
            aggregator._needs_audio_first.clear()

            # === Freeze-omni style: one output frame per input frame ===
            chunk, rid = aggregator.consume_pending_audio(OUTPUT_BYTES_PER_FRAME)
            if len(chunk) < OUTPUT_BYTES_PER_FRAME:
                chunk += b"\x00" * (OUTPUT_BYTES_PER_FRAME - len(chunk))
            aggregator.write_frame(chunk, rid=rid)

            # manual mode: fire create_response when the schedule allows
            if turn_mode == "manual" and media_time >= next_probe_media_time:
                _issue_probe(media_time)
                next_probe_media_time += probe_every_sec
                # response.done also clears the previous in_progress flag (in the callback)

            # Ship video frames against the video timeline: flush when media_time
            # catches up to the frame timestamp.
            # Note: the first message of every turn must be audio. If the flag is
            # set, this iteration's audio is still in flight or just crossed
            # response.done; skip video for this round.
            while not video_done:
                if aggregator._needs_audio_first.is_set():
                    break
                if pending_video is None:
                    try:
                        pending_video = video_producer.frames.get_nowait()
                    except queue.Empty:
                        break
                t_sec, b64 = pending_video
                if t_sec < 0:  # sentinel
                    pending_video = None
                    video_done = True
                    print("[Main] video frame stream ended")
                    break
                if t_sec <= media_time:
                    try:
                        conversation.append_video(b64)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[Main] append_video failed: {exc}")
                    pending_video = None
                    # Re-check: if response.done arrived during append_video and
                    # set the flag, stop sending video this iteration so the next
                    # round leads with audio.
                    if aggregator._needs_audio_first.is_set():
                        break
                else:
                    break

            if chunk_idx % 50 == 0:
                qsize = video_producer.frames.qsize()
                print(
                    f"[Main] media_time={media_time:.1f}s, "
                    f"wall={_elapsed():.1f}s, pending_frames={qsize}"
                )

            if realtime:
                # Wall-clock alignment: if we got ahead, sleep
                drift = media_time - _elapsed()
                if drift > 0:
                    time.sleep(drift)

        # After we finish sending, wait for the model to drain its remaining
        # reply; keep writing frames in the meantime (flush pending buffer).
        trailing_sec = 5.0
        print(f"[Main] feed complete; draining {trailing_sec:.0f}s of trailing model output ...")
        final_media_time = total_sent_audio_bytes / (
            AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH
        )
        trail_start = time.time()
        while (
            time.time() - trail_start < trailing_sec
            and len(aggregator.pending_audio) > 0
            and not stop_flag.is_set()
        ):
            aggregator.set_current_second(
                int(final_media_time + (time.time() - trail_start))
            )
            # Keep writing frames (matches freeze-omni's flush logic)
            chunk, rid = aggregator.consume_pending_audio(OUTPUT_BYTES_PER_FRAME)
            if len(chunk) < OUTPUT_BYTES_PER_FRAME:
                chunk += b"\x00" * (OUTPUT_BYTES_PER_FRAME - len(chunk))
            aggregator.write_frame(chunk, rid=rid)
            time.sleep(frame_dur)

    finally:
        print("[Main] closing conversation / ffmpeg / video producer ...")
        try:
            conversation.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            ffmpeg_proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        video_producer.stop()
        aggregator.closed_event.wait(timeout=3.0)
        aggregator.finalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        required=False,
        default=str(
            Path(__file__).resolve().parents[5]
            / "data/1q1a/videos/0001.mp4"
        ),
        help="input video path (must contain both audio and video tracks)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="output directory (default: outputs/<video_stem>/)",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5-omni-flash-realtime",
        help="Omni Realtime model name, e.g. qwen3.5-omni-flash-realtime / qwen-omni-turbo-realtime-latest",
    )
    parser.add_argument("--voice", default="Tina")
    parser.add_argument(
        "--video_fps",
        type=float,
        default=DEFAULT_VIDEO_FPS_TO_SEND,
        help="target fps for sending video frames (default: 2 fps)",
    )
    parser.add_argument(
        "--max_frame_edge",
        type=int,
        default=DEFAULT_MAX_FRAME_EDGE,
        help="max video frame edge in pixels; larger frames are scaled down proportionally",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG encoding quality (1-100)",
    )
    parser.add_argument(
        "--no_realtime",
        action="store_true",
        help="don't feed at real-time cadence (debug only; the production model expects real time)",
    )

    # === Turn detection / full-duplex probe mode ===
    parser.add_argument(
        "--turn_mode",
        choices=["server_vad", "manual"],
        default="server_vad",
        help=(
            "server_vad: wait for the user to finish a sentence before replying "
            "(default, fits typical chat); "
            "manual: disable VAD and fire create_response on a fixed schedule "
            "(--probe_every_sec), best for full-duplex / continuous-output benchmarks."
        ),
    )
    parser.add_argument(
        "--probe_every_sec",
        type=float,
        default=1.0,
        help="how often (sec) to fire create_response in manual mode (default 1 = once per second)",
    )
    parser.add_argument(
        "--probe_first_delay_sec",
        type=float,
        default=1.0,
        help="delay of the first probe in manual mode (avoids running on too little data; default 1s)",
    )
    parser.add_argument(
        "--probe_instructions",
        default=DEFAULT_PROBE_INSTRUCTIONS,
        help="instructions sent with each create_response in manual mode (controls what the model says)",
    )
    parser.add_argument(
        "--probe_no_cancel",
        action="store_true",
        help=(
            "By default we cancel the previous response before each probe so we get "
            "one segment per second; pass this flag to skip the cancel (let the "
            "previous segment finish naturally; multiple segments may queue up)."
        ),
    )
    parser.add_argument(
        "--turn_silence_ms",
        type=int,
        default=800,
        help=(
            "silence threshold (ms) for server_vad; default 800. "
            "Lower (e.g. 200) makes the model reply more often but may false-trigger."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_dashscope_api_key()
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

    stream_video_file(
        video_path=video_path,
        output_dir=output_dir,
        model=args.model,
        voice=args.voice,
        target_video_fps=args.video_fps,
        max_frame_edge=args.max_frame_edge,
        jpeg_quality=args.jpeg_quality,
        realtime=not args.no_realtime,
        turn_mode=args.turn_mode,
        probe_every_sec=args.probe_every_sec,
        probe_instructions=args.probe_instructions,
        probe_cancel_previous=not args.probe_no_cancel,
        probe_first_delay_sec=args.probe_first_delay_sec,
        turn_silence_ms=args.turn_silence_ms,
    )


if __name__ == "__main__":
    main()
