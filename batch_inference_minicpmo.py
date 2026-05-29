"""MiniCPM-o 4.5 full-duplex batch inference (single-GPU worker).

Each worker loads the model once and walks through its assigned video list
sequentially. launch_batch_minicpmo.sh shards the workload across N GPUs.

Usage:
    CUDA_VISIBLE_DEVICES=3 python batch_inference_minicpmo.py \
        --video_list videos_gpu3.txt \
        --model_path /path/to/MiniCPM-o-4_5 \
        --output_root outputs/minicpmo
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import wave
from pathlib import Path
from typing import Dict, List, Optional

try:
    import numpy as np
except ImportError:
    np = None  # collect_all_videos does not need numpy

SUBSETS_1Q1A = ("1q1a", "1q1a_math")

# -- Constants ---------------------------------------------------------------
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
OUTPUT_CHANNELS = 1
OUTPUT_SAMPLE_WIDTH = 2
SAMPLES_PER_CHUNK = OUTPUT_SAMPLE_RATE

DEFAULT_MAX_FRAME_EDGE = 640


# -- Helpers (mirrors run_with_video_file.py) --------------------------------
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
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n / fps if fps > 0 else 0.0


def extract_audio_chunks(video_path: str, num_chunks: int) -> List[np.ndarray]:
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

    chunk_samples = INPUT_SAMPLE_RATE
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        if start >= len(pcm):
            chunks.append(np.zeros(chunk_samples, dtype=np.float32))
        elif end > len(pcm):
            c = np.zeros(chunk_samples, dtype=np.float32)
            c[: len(pcm) - start] = pcm[start:]
            chunks.append(c)
        else:
            chunks.append(pcm[start:end])
    return chunks


def extract_video_frames(
    video_path: str,
    num_chunks: int,
    max_frame_edge: int = DEFAULT_MAX_FRAME_EDGE,
) -> list:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list = []

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
            frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


# -- Per-video inference -----------------------------------------------------
def run_single_video(
    video_path: Path,
    output_dir: Path,
    processor,
    model,
    ref_audio_path: Path,
    system_prompt: str,
    sliding_window_mode: str,
    sw_high_tokens: int,
    sw_low_tokens: int,
    max_slice_nums: int,
    length_penalty: float,
) -> dict:
    """Run full-duplex inference on a single video, returning a summary dict."""
    tmp_dir = output_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    audio_seconds_dir = output_dir / "audio_per_second"
    audio_seconds_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "output.wav"
    jsonl_path = output_dir / "model_output.jsonl"
    txt_path = output_dir / "model_output.txt"
    responses_path = output_dir / "responses.jsonl"
    wav_transcript_path = output_dir / "wav_transcript.json"
    done_marker = output_dir / ".done"

    video_duration = get_video_duration(str(video_path))
    num_chunks = int(math.ceil(video_duration))
    total_output_samples = int(video_duration * OUTPUT_SAMPLE_RATE)

    audio_chunks = extract_audio_chunks(str(video_path), num_chunks)
    video_frames = extract_video_frames(str(video_path), num_chunks)

    # -- Enable duplex mode --
    duplex = processor.set_duplex_mode()

    if model.duplex is not None:
        from MiniCPMO45.utils import DuplexWindowConfig
        model.duplex.decoder.set_window_config(DuplexWindowConfig(
            sliding_window_mode=sliding_window_mode,
            basic_window_high_tokens=sw_high_tokens,
            basic_window_low_tokens=sw_low_tokens,
        ))
        window_enabled = sliding_window_mode != "off"
        model.duplex.decoder.set_window_enabled(window_enabled)

    duplex.config.length_penalty = length_penalty

    duplex.prepare(
        system_prompt_text=system_prompt,
        ref_audio_path=str(ref_audio_path),
        prompt_wav_path=str(ref_audio_path),
    )

    # -- Inference loop --
    results_log: List[dict] = []
    audio_by_chunk: Dict[int, np.ndarray] = {}

    t_start = time.time()

    for chunk_idx in range(num_chunks):
        t_chunk = time.time()

        audio_chunk = audio_chunks[chunk_idx]
        frame = video_frames[chunk_idx] if chunk_idx < len(video_frames) else None
        frame_list = [frame] if frame is not None else []

        prefill_result = duplex.prefill(
            audio_waveform=audio_chunk,
            frame_list=frame_list if frame_list else None,
            max_slice_nums=max_slice_nums,
        )

        result = duplex.generate()
        duplex.finalize()

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

    total_time = time.time() - t_start

    try:
        duplex.stop()
    except Exception:
        pass

    # -- Write per-second PCM + model_output.jsonl / .txt --
    # Stage to .tmp first; rename only after every write succeeds (atomic).
    tmp_jsonl = tmp_dir / "model_output.jsonl"
    tmp_txt = tmp_dir / "model_output.txt"
    tmp_responses = tmp_dir / "responses.jsonl"
    tmp_wav_transcript = tmp_dir / "wav_transcript.json"

    with open(tmp_jsonl, "w", encoding="utf-8") as jf, \
         open(tmp_txt, "w", encoding="utf-8") as tf:
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

    # -- responses.jsonl --
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
                    "chunks": [{"text": r["text"], "timestamp": [float(r["second"]), float(r["second"] + 1)]}],
                }
            else:
                current_response["done_at_second"] = r["second"]
                current_response["text"] += r["text"]
                current_response["chunks"].append(
                    {"text": r["text"], "timestamp": [float(r["second"]), float(r["second"] + 1)]}
                )
        else:
            if current_response is not None:
                responses.append(current_response)
                current_response = None
    if current_response is not None:
        responses.append(current_response)

    with open(tmp_responses, "w", encoding="utf-8") as f:
        for resp in responses:
            audio_sec = sum(
                len(audio_by_chunk.get(int(c["timestamp"][0]), np.array([]))) / OUTPUT_SAMPLE_RATE
                for c in resp["chunks"]
            )
            resp["audio_duration_sec"] = round(audio_sec, 3)
            f.write(json.dumps(resp, ensure_ascii=False) + "\n")

    # -- output.wav --
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

    # -- wav_transcript.json --
    segments = []
    for r in results_log:
        idx = r["second"]
        if not r["is_listen"] and r["text"]:
            has_audio = idx in audio_by_chunk and len(audio_by_chunk[idx]) > 0
            if has_audio:
                segments.append({"text": r["text"], "timestamp": [float(idx), float(idx + 1)]})

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
        "chunks": [{"text": seg["text"], "timestamp": seg["timestamp"]} for seg in merged if seg["text"]],
    }
    with open(tmp_wav_transcript, "w", encoding="utf-8") as f:
        json.dump(wav_transcript, f, ensure_ascii=False, indent=2)

    # Atomic move: tmp -> final, then drop the .done marker
    for src, dst in [
        (tmp_jsonl, jsonl_path),
        (tmp_txt, txt_path),
        (tmp_responses, responses_path),
        (tmp_wav_transcript, wav_transcript_path),
    ]:
        shutil.move(str(src), str(dst))

    done_marker.write_text(json.dumps({
        "video": str(video_path),
        "duration_sec": round(video_duration, 2),
        "inference_sec": round(total_time, 2),
    }))

    # Clean up tmp dir
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    return {
        "video": str(video_path),
        "output_dir": str(output_dir),
        "duration_sec": round(video_duration, 2),
        "num_chunks": num_chunks,
        "inference_sec": round(total_time, 2),
        "rtf": round(total_time / max(video_duration, 0.1), 2),
        "num_responses": len(responses),
        "status": "ok",
    }


# -- Collect all videos ------------------------------------------------------
def collect_all_videos(data_root: Path) -> List[dict]:
    """Scan data/ for 1q1a, 1q1a_math, and 1qna videos.

    Returns a list of {"video_path", "output_name", "subset"} dicts.
    """
    videos = []

    # 1q1a + 1q1a_math: read from each subset's video_json_map.json
    for subset in SUBSETS_1Q1A:
        map_file = data_root / subset / "video_json_map.json"
        if not map_file.exists():
            continue
        with open(map_file) as f:
            mapping = json.load(f)
        for entry in mapping["entries"]:
            vp = data_root / subset / entry["video"]
            if vp.exists():
                output_name = entry["video"].replace("/", "__").replace(".mp4", "")
                videos.append({
                    "video_path": str(vp),
                    "output_name": f"{subset}/{output_name}",
                    "subset": subset,
                })

    # 1qna: walk every mp4 below videos_bench/
    bench_dir = data_root / "1qna" / "videos_bench"
    if bench_dir.exists():
        for mp4 in sorted(bench_dir.rglob("*.mp4")):
            rel = mp4.relative_to(bench_dir)
            output_name = str(rel).replace("/", "__").replace(".mp4", "")
            videos.append({
                "video_path": str(mp4),
                "output_name": f"1qna/{output_name}",
                "subset": "1qna",
            })

    return videos


# -- Main --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MiniCPM-o batch inference worker (single GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video_list", required=True,
                        help="JSONL video list file; one {video_path, output_name} per line")
    parser.add_argument("--model_path", required=True, help="path to the MiniCPM-o 4.5 model")
    parser.add_argument("--output_root", default="outputs/minicpmo",
                        help="output root directory")
    parser.add_argument("--ref_audio", default=None, help="reference audio path")
    parser.add_argument("--system_prompt", default="Streaming Omni Conversation.")
    parser.add_argument("--attn_implementation", default="sdpa",
                        choices=["auto", "sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--sliding_window_mode", default="basic",
                        choices=["off", "basic", "context"])
    parser.add_argument("--sw_high_tokens", type=int, default=7000)
    parser.add_argument("--sw_low_tokens", type=int, default=5000)
    parser.add_argument("--max_slice_nums", type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=1.1)
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

    # Read this worker's slice of the video list
    video_list_path = Path(args.video_list)
    tasks = []
    with open(video_list_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))

    print(f"[GPU {gpu_id}] assigned {len(tasks)} videos")
    if not tasks:
        print(f"[GPU {gpu_id}] no tasks; exiting")
        return

    # -- Load model (once) --
    print(f"[GPU {gpu_id}] loading model {model_path} ...")
    sys.path.insert(0, str(Path(__file__).parent / "MiniCPM-o-Demo"))
    from core.processors.unified import UnifiedProcessor

    ref_audio_path = Path(args.ref_audio) if args.ref_audio else (model_path / "assets" / "HT_ref_audio.wav")
    if not ref_audio_path.exists():
        alt = model_path / "assets" / "ref_audio" / "ref_minicpm_signature.wav"
        if alt.exists():
            ref_audio_path = alt
    print(f"[GPU {gpu_id}] ref_audio: {ref_audio_path}")

    processor = UnifiedProcessor(
        model_path=str(model_path),
        ref_audio_path=str(ref_audio_path),
        attn_implementation=args.attn_implementation,
    )
    model = processor.model
    print(f"[GPU {gpu_id}] model loaded")

    # -- Process videos one by one --
    summary_path = output_root / f"summary_gpu{gpu_id}.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    failed = 0
    t_all_start = time.time()

    # append mode: preserve prior records on resume
    with open(summary_path, "a", encoding="utf-8") as sf:
        for i, task in enumerate(tasks):
            video_path = Path(task["video_path"])
            output_dir = output_root / task["output_name"]

            # Skip videos already marked .done (jsonl alone is not enough)
            done_flag = output_dir / ".done"
            if done_flag.exists():
                print(f"[GPU {gpu_id}] [{i+1}/{len(tasks)}] skipped (already done): {video_path.name}")
                skipped += 1
                continue

            # Wipe any partial output left over from a previous crash
            tmp_dir = output_dir / ".tmp"
            if tmp_dir.exists():
                shutil.rmtree(str(tmp_dir), ignore_errors=True)

            print(f"[GPU {gpu_id}] [{i+1}/{len(tasks)}] start: {video_path.name}")

            try:
                result = run_single_video(
                    video_path=video_path,
                    output_dir=output_dir,
                    processor=processor,
                    model=model,
                    ref_audio_path=ref_audio_path,
                    system_prompt=args.system_prompt,
                    sliding_window_mode=args.sliding_window_mode,
                    sw_high_tokens=args.sw_high_tokens,
                    sw_low_tokens=args.sw_low_tokens,
                    max_slice_nums=args.max_slice_nums,
                    length_penalty=args.length_penalty,
                )
                sf.write(json.dumps(result, ensure_ascii=False) + "\n")
                sf.flush()
                done += 1
                print(
                    f"[GPU {gpu_id}] [{i+1}/{len(tasks)}] done: {video_path.name} "
                    f"({result['duration_sec']}s, RTF={result['rtf']}x, {result['num_responses']} responses)"
                )
            except Exception as e:
                failed += 1
                err_result = {
                    "video": str(video_path),
                    "output_dir": str(output_dir),
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                sf.write(json.dumps(err_result, ensure_ascii=False) + "\n")
                sf.flush()
                print(f"[GPU {gpu_id}] [{i+1}/{len(tasks)}] failed: {video_path.name} -> {e}")
            finally:
                # Release CUDA fragmentation after each video
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    total_time = time.time() - t_all_start
    print(
        f"\n[GPU {gpu_id}] all done: {done} ok, {skipped} skipped, {failed} failed, "
        f"elapsed {total_time/60:.1f} min"
    )


if __name__ == "__main__":
    main()
