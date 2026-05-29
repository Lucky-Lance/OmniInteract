"""Drive MiniCPM-o 4.5 full-duplex inference from a local video file and
emit a WAV (same duration as the video) plus per-second JSONL records.

What it does:
  1. Pulls video frames (1 fps) + audio PCM (1-second chunks) from an MP4
     and steps the model once per second: prefill -> generate -> finalize.
  2. Enables the sliding window (basic mode) so 5-minute and longer videos
     don't blow up the KV cache.
  3. Produces the same output layout as the Qwen-Omni adapter, so the
     results plug straight into the cross-model benchmark.

Output directory layout:

    outputs/<video_name>/
    +-- model_output.jsonl       # one line per second; text/state for that second
    +-- model_output.txt         # text-only preview for humans
    +-- responses.jsonl          # one record per response (adjacent speak chunks merged)
    +-- output.wav               # 24 kHz, mono, 16-bit; duration == input duration
    +-- wav_transcript.json      # WAV-aligned native text (ASR-compatible format)

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_with_video_file.py \
        --video /path/to/video.mp4 \
        --model_path /path/to/MiniCPM-o-4_5 \
        --output_dir outputs/video1

Dependencies:
    pip install librosa opencv-python soundfile Pillow
    ffmpeg / ffprobe must be on PATH.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

# -- Constants ---------------------------------------------------------------
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
OUTPUT_CHANNELS = 1
OUTPUT_SAMPLE_WIDTH = 2            # 16-bit PCM
SAMPLES_PER_CHUNK = OUTPUT_SAMPLE_RATE  # 1 second = 24000 samples

DEFAULT_MAX_FRAME_EDGE = 640


# -- Helpers -----------------------------------------------------------------
def get_video_duration(video_path: str) -> float:
    if shutil.which("ffprobe"):
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            pass
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n / fps if fps > 0 else 0.0


def extract_audio_chunks(video_path: str, num_chunks: int) -> List[np.ndarray]:
    """Extract audio with ffmpeg and slice into 1-second chunks (16 kHz mono float32)."""
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", video_path, "-vn",
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(INPUT_SAMPLE_RATE),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    raw = proc.stdout
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    chunk_samples = INPUT_SAMPLE_RATE  # 1 second
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        if start >= len(pcm):
            chunks.append(np.zeros(chunk_samples, dtype=np.float32))
        elif end > len(pcm):
            c = np.zeros(chunk_samples, dtype=np.float32)
            c[:len(pcm) - start] = pcm[start:]
            chunks.append(c)
        else:
            chunks.append(pcm[start:end])
    return chunks


def extract_video_frames(
    video_path: str,
    num_chunks: int,
    max_frame_edge: int = DEFAULT_MAX_FRAME_EDGE,
) -> List[Optional[Image.Image]]:
    """Sample one frame per second from the video; returns a list of PIL Images."""
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: List[Optional[Image.Image]] = []

    for chunk_idx in range(num_chunks):
        target_frame = int(chunk_idx * src_fps + src_fps * 0.5)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            frames.append(None)
            continue
        h, w = frame.shape[:2]
        long_edge = max(h, w)
        if long_edge > max_frame_edge:
            scale = max_frame_edge / long_edge
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


# -- Main --------------------------------------------------------------------
def run_inference(
    video_path: Path,
    model_path: Path,
    output_dir: Path,
    ref_audio_path: Optional[Path],
    system_prompt: str,
    attn_implementation: str,
    sliding_window_mode: str,
    sw_high_tokens: int,
    sw_low_tokens: int,
    max_slice_nums: int,
    length_penalty: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_seconds_dir = output_dir / "audio_per_second"
    audio_seconds_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "output.wav"
    jsonl_path = output_dir / "model_output.jsonl"
    txt_path = output_dir / "model_output.txt"
    responses_path = output_dir / "responses.jsonl"
    wav_transcript_path = output_dir / "wav_transcript.json"

    # -- Video metadata --
    video_duration = get_video_duration(str(video_path))
    num_chunks = int(math.ceil(video_duration))
    total_output_samples = int(video_duration * OUTPUT_SAMPLE_RATE)
    print(f"[Info] video duration: {video_duration:.2f}s, chunks: {num_chunks}")

    # -- Extract audio + frames --
    print("[Info] extracting audio ...")
    audio_chunks = extract_audio_chunks(str(video_path), num_chunks)
    print("[Info] extracting video frames ...")
    video_frames = extract_video_frames(str(video_path), num_chunks)
    print(f"[Info] audio chunks: {len(audio_chunks)}, video frames: {len(video_frames)}")

    # -- Load the model (UnifiedProcessor) --
    print("[Info] loading model ...")
    from core.processors.unified import UnifiedProcessor

    actual_ref_audio = ref_audio_path or (model_path / "assets" / "HT_ref_audio.wav")
    if not actual_ref_audio.exists():
        alt = model_path / "assets" / "ref_audio" / "ref_minicpm_signature.wav"
        if alt.exists():
            actual_ref_audio = alt
    print(f"[Info] ref_audio: {actual_ref_audio}")

    processor = UnifiedProcessor(
        model_path=str(model_path),
        ref_audio_path=str(actual_ref_audio),
        attn_implementation=attn_implementation,
    )

    # -- Enable duplex mode + configure the sliding window --
    duplex = processor.set_duplex_mode()

    # Push the window config down to the underlying DuplexCapability
    model = processor.model
    if model.duplex is not None:
        from MiniCPMO45.utils import DuplexWindowConfig
        model.duplex.decoder.set_window_config(DuplexWindowConfig(
            sliding_window_mode=sliding_window_mode,
            basic_window_high_tokens=sw_high_tokens,
            basic_window_low_tokens=sw_low_tokens,
        ))
        window_enabled = sliding_window_mode != "off"
        model.duplex.decoder.set_window_enabled(window_enabled)
        print(
            f"[Info] sliding window: mode={sliding_window_mode}, "
            f"high={sw_high_tokens}, low={sw_low_tokens}, enabled={window_enabled}"
        )

    # -- Tweak DuplexConfig (keeps every other default in place) --
    duplex.config.length_penalty = length_penalty

    # -- Prepare the duplex session --
    duplex.prepare(
        system_prompt_text=system_prompt,
        ref_audio_path=str(actual_ref_audio),
        prompt_wav_path=str(actual_ref_audio),
    )
    print("[Info] duplex session ready")

    # -- Inference loop --
    results_log: List[dict] = []
    audio_by_chunk: Dict[int, np.ndarray] = {}

    t_start = time.time()

    for chunk_idx in range(num_chunks):
        t_chunk = time.time()

        audio_chunk = audio_chunks[chunk_idx]
        frame = video_frames[chunk_idx] if chunk_idx < len(video_frames) else None
        frame_list = [frame] if frame is not None else []

        # prefill
        prefill_result = duplex.prefill(
            audio_waveform=audio_chunk,
            frame_list=frame_list if frame_list else None,
            max_slice_nums=max_slice_nums,
        )

        # generate
        result = duplex.generate()

        # finalize (important: this triggers the sliding window + EOS feed)
        duplex.finalize()

        # Collect audio (only when speaking; listen chunks are silence)
        audio_waveform = None
        if not result.is_listen and result.audio_data is not None:
            audio_bytes = base64.b64decode(result.audio_data)
            audio_waveform = np.frombuffer(audio_bytes, dtype=np.float32)
            if len(audio_waveform) > 0:
                audio_by_chunk[chunk_idx] = audio_waveform

        kv_len = model.duplex.decoder.get_cache_length() if model.duplex else 0
        chunk_ms = (time.time() - t_chunk) * 1000

        chunk_result = {
            "second": chunk_idx,
            "is_listen": result.is_listen,
            "text": result.text,
            "end_of_turn": result.end_of_turn,
            "current_time": result.current_time,
            "audio_samples": len(audio_waveform) if audio_waveform is not None else 0,
            "kv_cache_length": kv_len,
            "cost_llm_ms": result.cost_llm_ms,
            "cost_tts_ms": result.cost_tts_ms,
            "cost_token2wav_ms": result.cost_token2wav_ms,
            "cost_all_ms": result.cost_all_ms,
            "wall_clock_ms": round(chunk_ms, 1),
        }
        results_log.append(chunk_result)

        status = "listen..." if result.is_listen else f"speak> {result.text}"
        if result.end_of_turn:
            status += " [END_OF_TURN]"
        print(f"[{chunk_idx:04d}/{num_chunks}] kv={kv_len:5d} {chunk_ms:6.0f}ms {status}")

    total_time = time.time() - t_start
    print(f"\n[Info] inference done: {total_time:.1f}s (RTF={total_time / video_duration:.2f}x)")

    # -- Stop the session --
    try:
        duplex.stop()
    except Exception:
        pass

    # -- Write per-second PCM + model_output.jsonl / .txt --
    with open(jsonl_path, "w", encoding="utf-8") as jf, \
         open(txt_path, "w", encoding="utf-8") as tf:
        for r in results_log:
            idx = r["second"]
            audio_pcm_path = None
            audio_bytes_len = 0
            waveform = audio_by_chunk.get(idx)
            if waveform is not None and len(waveform) > 0:
                pcm_int16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
                pcm_bytes = pcm_int16.tobytes()
                audio_bytes_len = len(pcm_bytes)
                pcm_fp = audio_seconds_dir / f"{idx:04d}.pcm"
                pcm_fp.write_bytes(pcm_bytes)
                audio_pcm_path = str(pcm_fp.relative_to(output_dir))

            record = {
                "second": idx,
                "text": r["text"],
                "is_listen": r["is_listen"],
                "end_of_turn": r["end_of_turn"],
                "audio_pcm_path": audio_pcm_path,
                "audio_sample_rate": OUTPUT_SAMPLE_RATE,
                "audio_channels": OUTPUT_CHANNELS,
                "audio_sample_width": OUTPUT_SAMPLE_WIDTH,
                "audio_bytes": audio_bytes_len,
                "kv_cache_length": r["kv_cache_length"],
                "cost_llm_ms": r["cost_llm_ms"],
                "cost_tts_ms": r["cost_tts_ms"],
                "cost_token2wav_ms": r["cost_token2wav_ms"],
                "cost_all_ms": r["cost_all_ms"],
                "wall_clock_ms": r["wall_clock_ms"],
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")

            text = r["text"]
            if text:
                tf.write(f"[{idx:04d}s] TEXT: {text}\n")
            elif r["is_listen"]:
                tf.write(f"[{idx:04d}s] listen...\n")
            elif r["end_of_turn"]:
                tf.write(f"[{idx:04d}s] [END_OF_TURN]\n")
            else:
                tf.write(f"[{idx:04d}s] listen...\n")

    # -- Write responses.jsonl (merge adjacent speak chunks into one response) --
    responses: List[dict] = []
    current_response = None
    for r in results_log:
        if not r["is_listen"] and r["text"]:
            if current_response is None:
                current_response = {
                    "response_id": f"resp_{r['second']:04d}",
                    "triggered_at_second": r["second"],
                    "done_at_second": r["second"],
                    "text": r["text"],
                    "chunks": [{
                        "text": r["text"],
                        "timestamp": [float(r["second"]), float(r["second"] + 1)],
                    }],
                }
            else:
                current_response["done_at_second"] = r["second"]
                current_response["text"] += r["text"]
                current_response["chunks"].append({
                    "text": r["text"],
                    "timestamp": [float(r["second"]), float(r["second"] + 1)],
                })
        else:
            if current_response is not None:
                responses.append(current_response)
                current_response = None
    if current_response is not None:
        responses.append(current_response)

    with open(responses_path, "w", encoding="utf-8") as f:
        for resp in responses:
            audio_sec = sum(
                len(audio_by_chunk.get(int(c["timestamp"][0]), np.array([]))) / OUTPUT_SAMPLE_RATE
                for c in resp["chunks"]
            )
            resp["audio_duration_sec"] = round(audio_sec, 3)
            f.write(json.dumps(resp, ensure_ascii=False) + "\n")

    # -- Synthesize output.wav --
    wav_fp = wave.open(str(wav_path), "wb")
    wav_fp.setnchannels(OUTPUT_CHANNELS)
    wav_fp.setsampwidth(OUTPUT_SAMPLE_WIDTH)
    wav_fp.setframerate(OUTPUT_SAMPLE_RATE)

    samples_written = 0
    target_bytes_per_chunk = SAMPLES_PER_CHUNK * OUTPUT_SAMPLE_WIDTH
    try:
        for chunk_idx in range(num_chunks):
            waveform = audio_by_chunk.get(chunk_idx)
            if waveform is not None and len(waveform) > 0:
                pcm_int16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
                pcm_bytes = pcm_int16.tobytes()
                if len(pcm_bytes) >= target_bytes_per_chunk:
                    wav_fp.writeframes(pcm_bytes[:target_bytes_per_chunk])
                else:
                    wav_fp.writeframes(pcm_bytes + b"\x00" * (target_bytes_per_chunk - len(pcm_bytes)))
            else:
                wav_fp.writeframes(b"\x00" * target_bytes_per_chunk)
            samples_written += SAMPLES_PER_CHUNK

        if samples_written < total_output_samples:
            pad = total_output_samples - samples_written
            wav_fp.writeframes(b"\x00" * (pad * OUTPUT_SAMPLE_WIDTH))
            samples_written += pad
    finally:
        wav_fp.close()

    # -- Build wav_transcript.json --
    segments = []
    for r in results_log:
        idx = r["second"]
        if not r["is_listen"] and r["text"]:
            has_audio = idx in audio_by_chunk and len(audio_by_chunk[idx]) > 0
            if has_audio:
                segments.append({
                    "text": r["text"],
                    "timestamp": [float(idx), float(idx + 1)],
                })

    merged: List[dict] = []
    for seg in segments:
        if merged and seg["timestamp"][0] == merged[-1]["timestamp"][1]:
            merged[-1]["text"] += seg["text"]
            merged[-1]["timestamp"][1] = seg["timestamp"][1]
        else:
            merged.append(dict(seg))

    full_text = " ".join(seg["text"] for seg in merged if seg["text"])
    wav_transcript = {
        "text": full_text,
        "chunks": [
            {"text": seg["text"], "timestamp": seg["timestamp"]}
            for seg in merged if seg["text"]
        ],
    }
    with open(wav_transcript_path, "w", encoding="utf-8") as f:
        json.dump(wav_transcript, f, ensure_ascii=False, indent=2)

    output_duration = samples_written / OUTPUT_SAMPLE_RATE
    print(
        f"\n[Done] outputs saved:\n"
        f"  output.wav:          {wav_path} ({output_duration:.2f}s)\n"
        f"  model_output.jsonl:  {jsonl_path}\n"
        f"  model_output.txt:    {txt_path}\n"
        f"  responses.jsonl:     {responses_path}\n"
        f"  audio_per_second/:   {audio_seconds_dir}/*.pcm\n"
        f"  wav_transcript.json: {wav_transcript_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="input video path")
    parser.add_argument("--model_path", required=True, help="path to the MiniCPM-o 4.5 model")
    parser.add_argument("--output_dir", default=None, help="output directory (default: outputs/<video_stem>/)")
    parser.add_argument("--ref_audio", default=None, help="reference audio path (defaults to the one shipped with the model)")
    parser.add_argument(
        "--system_prompt", default="Streaming Omni Conversation.",
        help="system prompt",
    )
    parser.add_argument(
        "--attn_implementation", default="sdpa",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
    )
    parser.add_argument(
        "--sliding_window_mode", default="basic",
        choices=["off", "basic", "context"],
        help="sliding window mode: off=no window (stop when KV full), basic=basic window, context=window with context summary",
    )
    parser.add_argument(
        "--sw_high_tokens", type=int, default=7000,
        help="window high watermark (evict when KV cache exceeds this many tokens)",
    )
    parser.add_argument(
        "--sw_low_tokens", type=int, default=5000,
        help="window low watermark (token count retained after eviction)",
    )
    parser.add_argument(
        "--max_slice_nums", type=int, default=1,
        help="HD video frame slice count (1=standard 64 tok/frame, >1=HD with more tok/frame)",
    )
    parser.add_argument(
        "--length_penalty", type=float, default=1.1,
        help="generation length penalty (>1 encourages longer replies)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        print(f"video file not found: {video_path}")
        sys.exit(1)

    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path("outputs") / video_path.stem
    )
    ref_audio_path = Path(args.ref_audio).resolve() if args.ref_audio else None

    print(f"[Config] video:      {video_path}")
    print(f"[Config] model:      {model_path}")
    print(f"[Config] output:     {output_dir}")
    print(f"[Config] slide_mode: {args.sliding_window_mode}")
    print(f"[Config] sw_high:    {args.sw_high_tokens}")
    print(f"[Config] sw_low:     {args.sw_low_tokens}")

    run_inference(
        video_path=video_path,
        model_path=model_path,
        output_dir=output_dir,
        ref_audio_path=ref_audio_path,
        system_prompt=args.system_prompt,
        attn_implementation=args.attn_implementation,
        sliding_window_mode=args.sliding_window_mode,
        sw_high_tokens=args.sw_high_tokens,
        sw_low_tokens=args.sw_low_tokens,
        max_slice_nums=args.max_slice_nums,
        length_penalty=args.length_penalty,
    )


if __name__ == "__main__":
    main()
