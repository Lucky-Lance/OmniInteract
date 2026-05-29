"""Video-file driver for AURA full-duplex evaluation.

Pretends to be the AURA browser frontend: opens a local MP4, streams its
video at 2 fps and its audio (silero-VAD-segmented) over the existing AURA
TCP protocol, and writes responses to disk in the FDB-Omni-compatible
output format (``output.wav`` + ``wav_transcript.json`` + per-second
aggregates + raw event log).

Intended usage from the command line:

    python eval/video_file_driver.py \
        --video "/path/to/video.mp4" \
        --output_dir eval_outputs/<id>

Or programmatically via ``run_video()`` (used by ``run_dataset.py``).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.signal import resample_poly

# --- silero-VAD streaming helper -------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.silero_vad_helper import SpeechSegmenter  # noqa: E402


# ----------------------------------------------------------------------
# Constants (mirror Full-Duplex-Bench-Omni for the output side)
# ----------------------------------------------------------------------
INPUT_AUDIO_SR = 16000
INPUT_AUDIO_CHANNELS = 1
INPUT_AUDIO_SAMPLE_WIDTH = 2

OUTPUT_AUDIO_SR = 24000          # WAV output sample rate
OUTPUT_FRAME_MS = 30             # WAV frame granularity (FDB-Omni convention)
OUTPUT_BYTES_PER_FRAME = (
    OUTPUT_AUDIO_SR * INPUT_AUDIO_CHANNELS * INPUT_AUDIO_SAMPLE_WIDTH
    * OUTPUT_FRAME_MS // 1000
)  # 24000 * 2 * 30 / 1000 = 1440 bytes

VIDEO_TARGET_FPS = 2.0           # Send 2 frames per second
VIDEO_CHUNK_SECONDS = 1.0        # Pack 1 second's worth of frames per Type 1 message
DEFAULT_MAX_FRAME_EDGE = 640
DEFAULT_JPEG_QUALITY = 80
MIN_VIDEO_CHUNK_BYTES = 1024     # AURA server requires >=1KB

# ----------------------------------------------------------------------
# AURA TCP protocol constants
# ----------------------------------------------------------------------
TYPE_VIDEO = 1
TYPE_AUDIO = 2
TYPE_CLEAR_CONTEXT = 4
TYPE_TTS_AUDIO_FULL = 5          # full WAV per sentence (legacy, unused here)
TYPE_START_CAMERA = 6
TYPE_STREAMING_TOKEN = 8
TYPE_TTS_AUDIO_CHUNK = 9
TYPE_ASR_QUERY_ECHO = 10
TYPE_USER_SPEECH_STARTED = 11       # VAD interrupt: stop generation + TTS

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 12346


# ====================================================================== #
# Aggregator (FDB-Omni style output writer)
# ====================================================================== #
class FDBAggregator:
    """Collects model output and writes the FDB-Omni-compatible bundle.

    Conventions kept identical to ``OmniAggregator`` in
    ``Full-Duplex-Bench-Omni/.../run_with_video_file.py`` so that the
    downstream evaluation scripts can score this output unchanged.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
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
        self._bucket_lock = threading.Lock()

        # Pending output audio (already at OUTPUT_AUDIO_SR), with per-byte
        # response_id annotation.
        self.pending_audio = bytearray()
        self._pending_rids: List[Optional[str]] = []
        self._pending_lock = threading.Lock()

        self.wav_path = output_dir / "output.wav"
        self._wav_fp: Optional[wave.Wave_write] = None
        self._wav_lock = threading.Lock()
        self.frames_written = 0
        self.total_frames_expected = 0
        self._wav_frame_rids: List[Optional[str]] = []

        # speech-interrupt state
        self._muted = False
        self._interrupted_response_ids: set[str] = set()

        # Per-response bookkeeping. ``responses.jsonl`` is not written
        # incrementally - TTS audio arrives asynchronously after the text
        # ``is_final`` fires, so we keep records around and dump them at the
        # very end.  ``_response_done_at`` marks the wall-clock second when
        # Type 8 ``is_final`` arrived.
        self._responses: Dict[str, dict] = {}
        self._response_done_at: Dict[str, int] = {}
        self._current_response_id: Optional[str] = None
        self._response_order: List[str] = []
        self._responses_path = output_dir / "responses.jsonl"

    # ------------------------------------------------------------------
    def init_wav(self) -> None:
        with self._wav_lock:
            self._wav_fp = wave.open(str(self.wav_path), "wb")
            self._wav_fp.setnchannels(INPUT_AUDIO_CHANNELS)
            self._wav_fp.setsampwidth(INPUT_AUDIO_SAMPLE_WIDTH)
            self._wav_fp.setframerate(OUTPUT_AUDIO_SR)

    def set_total_frames(self, total: int) -> None:
        self.total_frames_expected = total

    # ------------------------------------------------------------------
    def set_current_second(self, sec: int) -> None:
        with self._second_lock:
            self._current_second = sec

    def get_current_second(self) -> int:
        with self._second_lock:
            return self._current_second

    # ------------------------------------------------------------------
    def write_frame(self, chunk: bytes, rid: Optional[str] = None) -> None:
        with self._wav_lock:
            if self._wav_fp is None:
                return
            if self.frames_written >= self.total_frames_expected:
                return
            self._wav_fp.writeframes(chunk)
            self._wav_frame_rids.append(rid)
            self.frames_written += 1

    def consume_pending_audio(self, max_bytes: int) -> tuple:
        """Pull up to ``max_bytes`` of PCM from pending; return ``(bytes, rid)``."""
        if self._muted:
            return b"", None
        with self._pending_lock:
            chunk = bytes(self.pending_audio[:max_bytes])
            rids = self._pending_rids[:len(chunk)]
            self.pending_audio = self.pending_audio[len(chunk):]
            self._pending_rids = self._pending_rids[len(chunk):]
            if rids:
                rid = Counter(rids).most_common(1)[0][0]
            else:
                rid = None
            return chunk, rid

    # ------------------------------------------------------------------
    # External event handlers (called by RX thread / VAD thread)
    # ------------------------------------------------------------------
    def on_user_speech_started(self, sec: int) -> None:
        """Called by VAD when user starts speaking (mute output, drop pending)."""
        self._dump_event({"type": "input_audio_buffer.speech_started",
                          "second": sec})
        # Mark all known responses as interrupted so their tail audio is dropped.
        with self._bucket_lock:
            self._interrupted_response_ids.update(self._responses.keys())
        self._muted = True
        with self._pending_lock:
            self.pending_audio.clear()
            self._pending_rids.clear()
        print(f"[VAD @ {sec}s] speech started (mute output)")

    def on_user_speech_stopped(self, sec: int) -> None:
        self._dump_event({"type": "input_audio_buffer.speech_stopped",
                          "second": sec})
        # Stay muted until the next response's first audio chunk arrives.

    def on_asr_query(self, query: str) -> None:
        sec = self.get_current_second()
        self._dump_event({"type": "conversation.item.input_audio_transcription.completed",
                          "transcript": query, "second": sec})
        with self._bucket_lock:
            self.asr_by_second[sec].append(query)
        print(f"[ASR @ {sec}s] {query}")

    def on_streaming_token(self, payload: dict) -> None:
        """Handle Type 8: streaming text token from the model."""
        sec = self.get_current_second()
        rid = payload.get("response_id", "") or self._current_response_id or ""
        token = payload.get("token", "")
        is_start = bool(payload.get("is_start", False))
        is_final = bool(payload.get("is_final", False))
        is_silent = bool(payload.get("is_silent", False))
        query = payload.get("query")

        self._dump_event({"type": "response.audio_transcript.delta",
                          "response_id": rid, "delta": token,
                          "is_start": is_start, "is_final": is_final,
                          "is_silent": is_silent, "second": sec})

        if is_start:
            self._open_response(rid, sec, query=query, is_silent=is_silent)

        if token and not is_silent:
            with self._bucket_lock:
                self.text_by_second[sec].append(token)
                rec = self._responses.get(rid)
                if rec is not None:
                    rec["deltas"].append({
                        "text": token,
                        "t_arrival_wall": time.time(),
                        "audio_bytes_so_far": rec["audio_bytes_total"],
                    })

        if is_final:
            # Just mark the text stream as done; actual finalization (with
            # up-to-date audio_bytes_total) happens in :meth:`finalize`.
            self._response_done_at[rid] = sec
            if self._current_response_id == rid:
                self._current_response_id = None

    def on_audio_chunk(self, payload: dict) -> None:
        """Handle Type 9: streaming PCM chunk from TTS.

        ``payload`` contains: response_id, sentence_idx, chunk_idx,
        sample_rate, is_final, pcm_data (raw int16 LE).
        """
        sec = self.get_current_second()
        rid = payload.get("response_id", "") or self._current_response_id or ""
        is_final = bool(payload.get("is_final", False))
        sample_rate = int(payload.get("sample_rate", OUTPUT_AUDIO_SR))
        pcm = payload.get("pcm_data", b"") or b""
        self._dump_event({"type": "response.audio.delta",
                          "response_id": rid,
                          "sample_rate": sample_rate,
                          "pcm_len": len(pcm),
                          "is_final": is_final,
                          "second": sec})

        if not pcm:
            return

        # Drop in-flight tail audio for the response that was interrupted.
        if self._muted and (not rid or rid in self._interrupted_response_ids):
            return

        # First audio chunk of a fresh response unmutes the pipe.
        if self._muted:
            self._muted = False

        # Resample to OUTPUT_AUDIO_SR if needed.
        pcm_24k = _resample_pcm_int16(pcm, sample_rate, OUTPUT_AUDIO_SR)

        with self._bucket_lock:
            self.audio_by_second[sec].extend(pcm_24k)
            rec = self._responses.get(rid)
            if rec is not None:
                rec["audio_bytes_total"] += len(pcm_24k)

        with self._pending_lock:
            self.pending_audio.extend(pcm_24k)
            self._pending_rids.extend([rid] * len(pcm_24k))

    # ------------------------------------------------------------------
    def _open_response(self, rid: str, sec: int, query: Optional[str] = None,
                       is_silent: bool = False) -> None:
        if not rid:
            return
        with self._bucket_lock:
            if rid not in self._responses:
                self._responses[rid] = {
                    "response_id": rid,
                    "t_created_wall": time.time(),
                    "triggered_at_second": sec,
                    "deltas": [],
                    "audio_bytes_total": 0,
                    "query": query or "",
                    "is_silent": is_silent,
                }
                self._response_order.append(rid)
        self._current_response_id = rid
        print(f"[Resp @ {sec}s] open response {rid} silent={is_silent} "
              f"query={(query or '')[:40]}")

    def _build_response_record(self, rid: str, done_second: int) -> Optional[dict]:
        """Build a fully-populated response record from current state."""
        rec = self._responses.get(rid)
        if rec is None:
            return None
        deltas = rec["deltas"]
        audio_bytes_total = rec["audio_bytes_total"]
        bytes_per_sec = OUTPUT_AUDIO_SR * INPUT_AUDIO_SAMPLE_WIDTH
        audio_duration_sec = (
            round(audio_bytes_total / bytes_per_sec, 6)
            if bytes_per_sec else 0.0
        )

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
        return {
            "response_id": rid,
            "triggered_at_second": rec["triggered_at_second"],
            "done_at_second": done_second,
            "audio_duration_sec": round(audio_duration_sec, 3),
            "text": "".join(d["text"] for d in deltas),
            "query": rec.get("query", ""),
            "is_silent": rec.get("is_silent", False),
            "chunks": chunks,
            "timestamp_method": "proportional_by_char_count",
        }

    def _dump_event(self, event: dict) -> None:
        with self._events_lock:
            self._events_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._events_fp.flush()

    # ------------------------------------------------------------------
    def finalize(self) -> None:
        """Close all output files; pad WAV to expected length; write transcript."""
        with self._bucket_lock:
            ordered_ids = list(self._response_order)
            # Tail-catch any response that ``on_streaming_token`` didn't finish
            # (e.g. connection cut mid-response).
            for rid in self._responses.keys():
                if rid not in ordered_ids:
                    ordered_ids.append(rid)
            default_done = self.get_current_second()
            records: List[dict] = []
            for rid in ordered_ids:
                done_sec = self._response_done_at.get(rid, default_done)
                rec = self._build_response_record(rid, done_sec)
                if rec is not None:
                    records.append(rec)
        with open(self._responses_path, "w", encoding="utf-8") as fp:
            for r in records:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")

        with self._events_lock:
            self._events_fp.close()

        with self._wav_lock:
            if self._wav_fp is not None:
                while self.frames_written < self.total_frames_expected:
                    self._wav_fp.writeframes(b"\x00" * OUTPUT_BYTES_PER_FRAME)
                    self._wav_frame_rids.append(None)
                    self.frames_written += 1
                self._wav_fp.close()
                self._wav_fp = None

        self._write_wav_transcript()

        jsonl_path = self.output_dir / "model_output.jsonl"
        txt_path = self.output_dir / "model_output.txt"

        with self._bucket_lock:
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
                        fp = self.audio_seconds_dir / f"{sec:04d}.pcm"
                        fp.write_bytes(audio_bytes)
                        audio_rel = str(fp.relative_to(self.output_dir))

                    record = {
                        "second": sec,
                        "text": text,
                        "user_transcript": asr,
                        "language": None,
                        "audio_pcm_path": audio_rel,
                        "audio_sample_rate": OUTPUT_AUDIO_SR,
                        "audio_channels": 1,
                        "audio_sample_width": 2,
                        "audio_bytes": len(audio_bytes),
                    }
                    jf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if text or asr:
                        tf.write(f"[{sec:04d}s] TEXT: {text}\n")
                        if asr:
                            tf.write(f"[{sec:04d}s] ASR : {asr}\n")

        wav_dur = self.frames_written * OUTPUT_FRAME_MS / 1000.0
        print("[Finalize] outputs written:")
        print(f"  - {jsonl_path}")
        print(f"  - {txt_path}")
        print(f"  - {self._responses_path}")
        print(f"  - {self.wav_path}  ({wav_dur:.2f}s)")
        print(f"  - {self.audio_seconds_dir}/*.pcm")
        print(f"  - {self.output_dir / 'wav_transcript.json'}")

    # ------------------------------------------------------------------
    def _write_wav_transcript(self) -> None:
        frame_sec = OUTPUT_FRAME_MS / 1000.0

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
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] read responses.jsonl failed: {exc}")

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
        out = {
            "text": full_text,
            "chunks": [
                {"text": seg["text"], "timestamp": seg["timestamp"]}
                for seg in segments if seg["text"]
            ],
        }
        out_path = self.output_dir / "wav_transcript.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


# ====================================================================== #
# Helpers
# ====================================================================== #
def _resample_pcm_int16(pcm_bytes: bytes, src_sr: int, dst_sr: int) -> bytes:
    if not pcm_bytes:
        return b""
    if src_sr == dst_sr:
        return pcm_bytes
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    if arr.size == 0:
        return b""
    from math import gcd
    g = gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    resampled = resample_poly(arr.astype(np.float32), up, down)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def _wrap_pcm_as_wav(pcm_bytes: bytes, sample_rate: int = INPUT_AUDIO_SR) -> bytes:
    """Wrap raw int16 mono PCM in a RIFF WAV container.

    The AURA ASR endpoint accepts MP3 / WAV / etc; sending WAV avoids needing
    an MP3 encoder and is loss-less.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def get_video_duration(video_path: Path) -> float:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ])
        return float(out.strip())
    except Exception:
        return 0.0


# ====================================================================== #
# Video chunk producer (encodes 2-frame mp4 chunks at 2 fps)
# ====================================================================== #
class VideoChunkProducer(threading.Thread):
    """Reads source video at real-time, emits one mp4 chunk per second."""

    def __init__(
        self,
        video_path: Path,
        sock: socket.socket,
        sock_lock: threading.Lock,
        stop_event: threading.Event,
        max_frame_edge: int = DEFAULT_MAX_FRAME_EDGE,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        target_fps: float = VIDEO_TARGET_FPS,
        chunk_seconds: float = VIDEO_CHUNK_SECONDS,
        realtime: bool = True,
        t0: Optional[float] = None,
    ) -> None:
        super().__init__(daemon=True, name="VideoChunkProducer")
        self.video_path = video_path
        self.sock = sock
        self.sock_lock = sock_lock
        self.stop_event = stop_event
        self.max_frame_edge = max_frame_edge
        self.jpeg_quality = jpeg_quality
        self.target_fps = target_fps
        self.chunk_seconds = chunk_seconds
        self.realtime = realtime
        self._t0 = t0 if t0 is not None else time.time()

    # ------------------------------------------------------------------
    def run(self) -> None:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"[Video] cannot open {self.video_path}")
            return
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / src_fps if src_fps > 0 else 0.0
        step = max(1, int(round(src_fps / max(0.1, self.target_fps))))
        print(
            f"[Video] src_fps={src_fps:.2f}, frames={total_frames}, "
            f"duration={duration:.1f}s, step={step} "
            f"(~{src_fps/step:.2f} fps)"
        )

        chunk_idx = 0
        chunk_buf: List[np.ndarray] = []
        next_chunk_deadline = self._t0 + self.chunk_seconds
        idx = 0

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    proc = self._maybe_resize(frame)
                    chunk_buf.append(proc)
                idx += 1

                # Use the source-video timestamp to decide when to flush.
                t_src = idx / src_fps
                if t_src >= (chunk_idx + 1) * self.chunk_seconds and chunk_buf:
                    self._flush_chunk(chunk_buf, chunk_idx)
                    chunk_buf = []
                    chunk_idx += 1
                    if self.realtime:
                        # Sleep until wall-clock catches up so the server
                        # really sees data at 1 chunk/sec.
                        drift = next_chunk_deadline - time.time()
                        if drift > 0:
                            time.sleep(min(drift, self.chunk_seconds))
                        next_chunk_deadline += self.chunk_seconds

            # Tail flush
            if chunk_buf and not self.stop_event.is_set():
                self._flush_chunk(chunk_buf, chunk_idx)
        finally:
            cap.release()
            print("[Video] producer finished")

    # ------------------------------------------------------------------
    def _maybe_resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        long_edge = max(h, w)
        if long_edge > self.max_frame_edge:
            scale = self.max_frame_edge / long_edge
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    # ------------------------------------------------------------------
    def _flush_chunk(self, frames: List[np.ndarray], chunk_idx: int) -> None:
        # Need at least 2 frames so the AURA server's "min 2 frames" rule
        # is satisfied even after server-side downsampling.
        if len(frames) == 1:
            frames = [frames[0], frames[0]]
        mp4_bytes = _encode_frames_to_mp4(frames, self.target_fps,
                                          chunk_idx=chunk_idx)
        if mp4_bytes is None:
            print(f"[Video] chunk {chunk_idx}: encode failed, skipping")
            return
        # Pad up to MIN_VIDEO_CHUNK_BYTES (defensive; mp4 with 2 frames is
        # usually a few KB already).
        if len(mp4_bytes) < MIN_VIDEO_CHUNK_BYTES:
            mp4_bytes = mp4_bytes + b"\x00" * (
                MIN_VIDEO_CHUNK_BYTES - len(mp4_bytes)
            )
        header = struct.pack(">BQ", TYPE_VIDEO, len(mp4_bytes))
        try:
            with self.sock_lock:
                self.sock.sendall(header + mp4_bytes)
            print(f"[Video] sent chunk {chunk_idx}: {len(frames)} frames, "
                  f"{len(mp4_bytes)} bytes")
        except Exception as exc:  # noqa: BLE001
            print(f"[Video] send failed: {exc}")
            self.stop_event.set()


def _encode_frames_to_mp4(
    frames: List[np.ndarray], fps: float, chunk_idx: int = 0
) -> Optional[bytes]:
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4",
                                    prefix=f"aura_chunk_{chunk_idx}_")
    os.close(fd)
    try:
        # mp4v fourcc is the most reliable mp4 encoder bundled with opencv.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            print("[Video] VideoWriter failed to open (mp4v fourcc)")
            return None
        for f in frames:
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(f)
        writer.release()
        with open(tmp_path, "rb") as fp:
            return fp.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ====================================================================== #
# Audio + VAD producer
# ====================================================================== #
class AudioVADProducer(threading.Thread):
    """Reads ffmpeg PCM pipe in real time, runs silero-VAD, sends Type 2."""

    def __init__(
        self,
        video_path: Path,
        sock: socket.socket,
        sock_lock: threading.Lock,
        aggregator: FDBAggregator,
        stop_event: threading.Event,
        vad_threshold: float = 0.5,
        min_silence_ms: int = 500,
        speech_pad_ms: int = 100,
        min_segment_ms: int = 300,
        chunk_ms: int = 30,
        realtime: bool = True,
        t0: Optional[float] = None,
    ) -> None:
        super().__init__(daemon=True, name="AudioVADProducer")
        self.video_path = video_path
        self.sock = sock
        self.sock_lock = sock_lock
        self.aggregator = aggregator
        self.stop_event = stop_event
        self.realtime = realtime
        self.chunk_ms = chunk_ms
        self.min_segment_ms = min_segment_ms
        self.segmenter = SpeechSegmenter(
            threshold=vad_threshold,
            sampling_rate=INPUT_AUDIO_SR,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self._t0 = t0 if t0 is not None else time.time()

    # ------------------------------------------------------------------
    def run(self) -> None:
        chunk_bytes = INPUT_AUDIO_SR * INPUT_AUDIO_SAMPLE_WIDTH * self.chunk_ms // 1000
        proc = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel", "error",
                "-i", str(self.video_path),
                "-vn",
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ac", str(INPUT_AUDIO_CHANNELS),
                "-ar", str(INPUT_AUDIO_SR),
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.stdout is None:
            print("[Audio] ffmpeg pipe failed")
            return

        self._n_starts = 0
        self._n_segments = 0
        try:
            chunk_idx = 0
            while not self.stop_event.is_set():
                pcm = proc.stdout.read(chunk_bytes)
                if not pcm:
                    break
                if len(pcm) < chunk_bytes:
                    pcm = pcm + b"\x00" * (chunk_bytes - len(pcm))

                events = self.segmenter.feed(pcm)
                self._handle_events(events)

                chunk_idx += 1
                if self.realtime:
                    expected = self._t0 + chunk_idx * (self.chunk_ms / 1000.0)
                    drift = expected - time.time()
                    if drift > 0:
                        time.sleep(drift)
            tail_events = self.segmenter.flush()
            self._handle_events(tail_events)
        finally:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
            print(f"[Audio] producer finished "
                  f"(VAD: {self._n_starts} starts, "
                  f"{self._n_segments} segments sent)")

    # ------------------------------------------------------------------
    def _handle_events(self, events) -> None:
        for evt in events:
            kind = evt[0]
            if kind == "start":
                start_sec = evt[1]
                self._n_starts += 1
                self.aggregator.on_user_speech_started(int(start_sec))
                # Fire an immediate hard-interrupt to the server so it
                # cancels the in-flight vLLM generation + TTS synthesis.
                # Payload is a simple int64 timestamp (ms) for logging.
                try:
                    ts_ms = int(start_sec * 1000)
                    payload = struct.pack(">Q", ts_ms)
                    header = struct.pack(">BQ",
                                         TYPE_USER_SPEECH_STARTED,
                                         len(payload))
                    with self.sock_lock:
                        self.sock.sendall(header + payload)
                    print(f"[Audio] sent USER_SPEECH_STARTED @ {start_sec:.2f}s")
                except Exception as exc:  # noqa: BLE001
                    print(f"[Audio] failed to send interrupt: {exc}")
            elif kind == "segment":
                start_sec, end_sec, pcm_seg = evt[1], evt[2], evt[3]
                self.aggregator.on_user_speech_stopped(int(end_sec))
                duration_ms = (end_sec - start_sec) * 1000.0
                if duration_ms < self.min_segment_ms:
                    print(
                        f"[Audio] discarding short segment "
                        f"[{start_sec:.2f}, {end_sec:.2f}] "
                        f"({duration_ms:.0f}ms < {self.min_segment_ms}ms)"
                    )
                    continue
                wav_bytes = _wrap_pcm_as_wav(pcm_seg, sample_rate=INPUT_AUDIO_SR)
                header = struct.pack(">BQ", TYPE_AUDIO, len(wav_bytes))
                try:
                    with self.sock_lock:
                        self.sock.sendall(header + wav_bytes)
                    self._n_segments += 1
                    print(f"[Audio] sent VAD segment "
                          f"[{start_sec:.2f}, {end_sec:.2f}] "
                          f"({len(wav_bytes)} bytes WAV)")
                except Exception as exc:  # noqa: BLE001
                    print(f"[Audio] send failed: {exc}")
                    self.stop_event.set()


# ====================================================================== #
# RX thread (parses server messages)
# ====================================================================== #
class RXThread(threading.Thread):
    def __init__(
        self,
        sock: socket.socket,
        aggregator: FDBAggregator,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name="RXThread")
        self.sock = sock
        self.aggregator = aggregator
        self.stop_event = stop_event

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                header = self._recv_exact(9)
                if header is None:
                    break
                msg_type, msg_len = struct.unpack(">BQ", header)
                if msg_len > 100 * 1024 * 1024:
                    print(f"[RX] suspicious len {msg_len}, breaking")
                    break
                payload = self._recv_exact(msg_len) if msg_len else b""
                if payload is None:
                    break
                self._dispatch(msg_type, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[RX] error: {exc}")
        finally:
            print("[RX] finished")

    # ------------------------------------------------------------------
    def _recv_exact(self, n: int) -> Optional[bytes]:
        data = bytearray()
        while len(data) < n:
            try:
                chunk = self.sock.recv(min(n - len(data), 65536))
            except socket.timeout:
                if self.stop_event.is_set():
                    return None
                continue
            except OSError:
                return None
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    # ------------------------------------------------------------------
    def _dispatch(self, msg_type: int, payload: bytes) -> None:
        if msg_type == TYPE_STREAMING_TOKEN:
            try:
                obj = json.loads(payload.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"[RX] type=8 decode error: {exc}")
                return
            self.aggregator.on_streaming_token(obj)
        elif msg_type == TYPE_TTS_AUDIO_CHUNK:
            try:
                response_id_len = payload[0]
                response_id = payload[1:1 + response_id_len].decode("utf-8")
                offset = 1 + response_id_len
                sentence_idx, chunk_idx, sample_rate, is_final = struct.unpack(
                    ">HHIB", payload[offset:offset + 9]
                )
                pcm_data = payload[offset + 9:]
                self.aggregator.on_audio_chunk({
                    "response_id": response_id,
                    "sentence_idx": sentence_idx,
                    "chunk_idx": chunk_idx,
                    "sample_rate": sample_rate,
                    "is_final": bool(is_final),
                    "pcm_data": pcm_data,
                })
            except Exception as exc:  # noqa: BLE001
                print(f"[RX] type=9 parse error: {exc}")
        elif msg_type == TYPE_ASR_QUERY_ECHO:
            try:
                obj = json.loads(payload.decode("utf-8"))
                query = obj.get("query", "")
            except Exception:
                query = payload.decode("utf-8", errors="ignore")
            if query:
                self.aggregator.on_asr_query(query)
        elif msg_type == TYPE_TTS_AUDIO_FULL:
            # Full-WAV-per-sentence path; also funnel into pending buffer
            # so we don't drop audio if the server happens to send it.
            try:
                response_id_len = payload[0]
                response_id = payload[1:1 + response_id_len].decode("utf-8")
                offset = 1 + response_id_len
                _sidx, _tot = struct.unpack(">HH", payload[offset:offset + 4])
                wav_bytes = payload[offset + 4:]
                # Decode WAV → PCM @ its native SR, then resample.
                with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                    sr = wf.getframerate()
                    pcm = wf.readframes(wf.getnframes())
                self.aggregator.on_audio_chunk({
                    "response_id": response_id,
                    "sentence_idx": _sidx,
                    "chunk_idx": 0,
                    "sample_rate": sr,
                    "is_final": True,
                    "pcm_data": pcm,
                })
            except Exception as exc:  # noqa: BLE001
                print(f"[RX] type=5 parse error: {exc}")
        else:
            print(f"[RX] ignoring unknown msg_type={msg_type} "
                  f"len={len(payload)}")


# ====================================================================== #
# WAV writer thread (30ms tick)
# ====================================================================== #
class WavWriterThread(threading.Thread):
    def __init__(
        self,
        aggregator: FDBAggregator,
        stop_event: threading.Event,
        total_seconds: float,
        t0: float,
    ) -> None:
        super().__init__(daemon=True, name="WavWriterThread")
        self.aggregator = aggregator
        self.stop_event = stop_event
        self.total_seconds = total_seconds
        self.t0 = t0

    def run(self) -> None:
        frame_dur = OUTPUT_FRAME_MS / 1000.0
        idx = 0
        while not self.stop_event.is_set():
            elapsed = time.time() - self.t0
            self.aggregator.set_current_second(int(elapsed))
            if idx >= self.aggregator.total_frames_expected:
                break
            chunk, rid = self.aggregator.consume_pending_audio(
                OUTPUT_BYTES_PER_FRAME
            )
            if len(chunk) < OUTPUT_BYTES_PER_FRAME:
                chunk = chunk + b"\x00" * (OUTPUT_BYTES_PER_FRAME - len(chunk))
            self.aggregator.write_frame(chunk, rid=rid)
            idx += 1
            target = self.t0 + idx * frame_dur
            drift = target - time.time()
            if drift > 0:
                time.sleep(drift)
        print(f"[Wav] writer finished after {idx} frames")


# ====================================================================== #
# Top-level driver
# ====================================================================== #
def _connect_to_server(host: str, port: int, timeout: float = 30.0) -> socket.socket:
    deadline = time.time() + timeout
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=10.0)
            sock.settimeout(1.0)  # short read timeout for RX-thread polling
            return sock
        except OSError as exc:
            last_err = exc
            time.sleep(1.0)
    raise RuntimeError(f"cannot connect to {host}:{port}: {last_err}")


def _send_control(sock: socket.socket, sock_lock: threading.Lock,
                  msg_type: int, payload: bytes = b"") -> None:
    header = struct.pack(">BQ", msg_type, len(payload))
    with sock_lock:
        sock.sendall(header + payload)


def run_video(
    video_path: Path,
    output_dir: Path,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    realtime: bool = True,
    trailing_seconds: float = 8.0,
    vad_threshold: float = 0.5,
    vad_min_silence_ms: int = 500,
    vad_speech_pad_ms: int = 100,
    vad_min_segment_ms: int = 300,
    max_frame_edge: int = DEFAULT_MAX_FRAME_EDGE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    extra_video_seconds: float = 0.0,
) -> Path:
    """Run the full driver for a single video. Returns ``output_dir``."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    duration = get_video_duration(video_path)
    if duration <= 0.0:
        raise RuntimeError(f"could not probe duration of {video_path}")
    total_frames = int(round((duration + extra_video_seconds) * 1000.0
                              / OUTPUT_FRAME_MS))

    print(f"[Driver] video: {video_path}")
    print(f"[Driver] output: {output_dir}")
    print(f"[Driver] duration={duration:.2f}s, total wav frames={total_frames}")

    aggregator = FDBAggregator(output_dir)
    aggregator.set_total_frames(total_frames)
    aggregator.init_wav()

    sock = _connect_to_server(host, port)
    print(f"[Driver] connected to {host}:{port}")
    sock_lock = threading.Lock()
    stop_event = threading.Event()

    # Reset server-side session
    _send_control(sock, sock_lock, TYPE_START_CAMERA)
    time.sleep(0.05)
    _send_control(sock, sock_lock, TYPE_CLEAR_CONTEXT)
    time.sleep(0.05)

    t0 = time.time()
    rx = RXThread(sock, aggregator, stop_event)
    wav_writer = WavWriterThread(aggregator, stop_event,
                                 total_seconds=duration + trailing_seconds,
                                 t0=t0)
    audio_thread = AudioVADProducer(
        video_path, sock, sock_lock, aggregator, stop_event,
        vad_threshold=vad_threshold,
        min_silence_ms=vad_min_silence_ms,
        speech_pad_ms=vad_speech_pad_ms,
        min_segment_ms=vad_min_segment_ms,
        realtime=realtime, t0=t0,
    )
    video_thread = VideoChunkProducer(
        video_path, sock, sock_lock, stop_event,
        max_frame_edge=max_frame_edge,
        jpeg_quality=jpeg_quality,
        realtime=realtime, t0=t0,
    )

    rx.start()
    wav_writer.start()
    audio_thread.start()
    video_thread.start()

    try:
        # Wait until both producers finish (i.e. video stream consumed).
        audio_thread.join()
        video_thread.join()
        # Let the model flush trailing responses for `trailing_seconds`.
        deadline = time.time() + trailing_seconds
        while time.time() < deadline and not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[Driver] Ctrl+C, shutting down")
    finally:
        stop_event.set()
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()
        rx.join(timeout=3.0)
        wav_writer.join(timeout=3.0)
        aggregator.finalize()

    return output_dir


# ====================================================================== #
# CLI
# ====================================================================== #
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="path to mp4")
    parser.add_argument("--output_dir", required=True, help="output directory")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no_realtime", action="store_true",
                        help="don't pace to wall clock (debug only)")
    parser.add_argument("--trailing_seconds", type=float, default=8.0,
                        help="how long to wait for the model to flush "
                             "trailing responses after the video ends")
    parser.add_argument("--vad_threshold", type=float, default=0.7)
    parser.add_argument("--vad_min_silence_ms", type=int, default=500)
    parser.add_argument("--vad_speech_pad_ms", type=int, default=100)
    parser.add_argument("--vad_min_segment_ms", type=int, default=300)
    parser.add_argument("--max_frame_edge", type=int,
                        default=DEFAULT_MAX_FRAME_EDGE)
    parser.add_argument("--jpeg_quality", type=int,
                        default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--extra_video_seconds", type=float, default=0.0,
                        help="extra seconds to keep writing wav frames "
                             "beyond the video duration")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ffmpeg/ffprobe must be on PATH")
        sys.exit(1)
    run_video(
        video_path=Path(args.video),
        output_dir=Path(args.output_dir),
        host=args.host,
        port=args.port,
        realtime=not args.no_realtime,
        trailing_seconds=args.trailing_seconds,
        vad_threshold=args.vad_threshold,
        vad_min_silence_ms=args.vad_min_silence_ms,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
        vad_min_segment_ms=args.vad_min_segment_ms,
        max_frame_edge=args.max_frame_edge,
        jpeg_quality=args.jpeg_quality,
        extra_video_seconds=args.extra_video_seconds,
    )


if __name__ == "__main__":
    main()
