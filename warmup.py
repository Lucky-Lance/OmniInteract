#!/usr/bin/env python3
"""End-to-end warmup for the AURA full-duplex eval pipeline.

Called automatically by ``start_all.sh`` after the ASR (``:8001``), TTS
(``:8002``) and vLLM streaming server (``:12345``) are up. Exercises every
cold-start path so the first real request does not pay the
``torch.compile`` / CUDA-graph-record cost that caused the silent
``output.wav`` issue in the Huang-Zheng run (see ``logs/vllm.log:4275``).

Phases (each phase soft-fails: a broken warmup never blocks normal usage):

1. **ASR** - ``POST /asr`` with ``shuhan.mp3`` → prime Qwen3-ASR encoder.
2. **TTS** - ``POST /v1/tts/stream`` with several different-length Chinese
   sentences → prime Qwen3-TTS Inductor compile + CUDA graphs across shapes.
   This is the single biggest win.
3. **End-to-end** - Stream a short demo clip through
   :func:`eval.video_file_driver.run_video` → exercises the real AURA TCP
   protocol, ContextManager, VAD interrupt path, and ASR→vLLM→TTS handoff.

Usage::

    python warmup.py                          # defaults
    python warmup.py --skip-e2e              # only ASR + TTS
    python warmup.py --e2e-seconds 6         # shorter E2E clip
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import requests


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

_ASR_PORT = int(os.environ.get("WARMUP_ASR_PORT", "8001"))
_TTS_PORT = int(os.environ.get("WARMUP_TTS_PORT", "8002"))
_VLLM_PORT = int(os.environ.get("WARMUP_VLLM_PORT", "12346"))
ASR_URL = f"http://localhost:{_ASR_PORT}"
TTS_URL = f"http://localhost:{_TTS_PORT}"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = _VLLM_PORT

# Representative Chinese sentence lengths. Covers 6-55 chars; real eval
# sentences typically land in 15-60 chars, so each "bucket" has at least one
# warmup pass before real traffic arrives.
TTS_WARMUP_TEXTS: List[str] = [
    "好的。",
    "知道了，我这就帮你盯着。",
    "画面里出现了一个蓝色的水杯，旁边还有一支笔。",
    "收到，我已经注意到了桌上新出现的物品，是一本摊开的笔记本。",
    "已经看到了，画面右上角出现的那个东西应该是遥控器，它旁边还有一台很小的保温杯和一些文具。",
]


def _pretty(s: str, width: int = 30) -> str:
    return s if len(s) <= width else s[:width] + "…"


# ---------------------------------------------------------------------------
# Readiness checks
# ---------------------------------------------------------------------------
def wait_tcp(host: str, port: int, timeout: float, label: str) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            time.sleep(1.0)
    print(f"✗ {label} ({host}:{port}) not reachable within {timeout:.0f}s")
    return False


def wait_http(url: str, timeout: float, label: str) -> bool:
    deadline = time.time() + timeout
    last_err = ""
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2.0)
            if r.ok:
                return True
            last_err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(1.0)
    print(f"✗ {label} ({url}) not ready within {timeout:.0f}s: {last_err}")
    return False


# ---------------------------------------------------------------------------
# Phase 1 - ASR
# ---------------------------------------------------------------------------
def warm_asr(audio_path: Path) -> str:
    print(f"\n🎙  [1/3] ASR warmup with {audio_path.name} ...")
    if not audio_path.exists():
        print(f"    ⚠️  audio file missing: {audio_path}")
        return ""
    t0 = time.time()
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{ASR_URL}/asr",
            files={"file": (audio_path.name, f, "audio/mpeg")},
            params={"language": "Chinese"},
            timeout=120,
        )
    resp.raise_for_status()
    text = (resp.json().get("text") or "").strip()
    print(f"    ✓ ASR ok in {time.time() - t0:.2f}s → {_pretty(text, 40)!r}")
    return text


# ---------------------------------------------------------------------------
# Phase 2 - TTS (the main fix)
# ---------------------------------------------------------------------------
def warm_tts_once(text: str) -> Tuple[int, int, float, float]:
    """Call TTS stream once and drain all PCM chunks.

    Returns ``(n_chunks, pcm_bytes, first_chunk_seconds, total_seconds)``.
    """
    t0 = time.time()
    resp = requests.post(
        f"{TTS_URL}/v1/tts/stream",
        json={"text": text, "language": "Chinese", "speaker": "Vivian"},
        stream=True,
        timeout=(5, 180),  # Generous read timeout only for the warmup path.
    )
    resp.raise_for_status()

    buf = b""
    pcm_bytes = 0
    chunks = 0
    t_first: float | None = None
    for raw in resp.iter_content(chunk_size=8192):
        if not raw:
            continue
        buf += raw
        while len(buf) >= 8:
            _sr, pcm_len = struct.unpack(">II", buf[:8])
            if len(buf) < 8 + pcm_len:
                break
            if t_first is None:
                t_first = time.time() - t0
            pcm_bytes += pcm_len
            chunks += 1
            buf = buf[8 + pcm_len:]
    total = time.time() - t0
    return chunks, pcm_bytes, t_first if t_first is not None else total, total


def warm_tts() -> None:
    n = len(TTS_WARMUP_TEXTS)
    print(f"\n🔊 [2/3] TTS warmup with {n} varied-length sentences ...")
    for i, text in enumerate(TTS_WARMUP_TEXTS, 1):
        try:
            chunks, pcm, ttfb, total = warm_tts_once(text)
            audio_sec = pcm / 2 / 24000  # int16 mono 24kHz
            print(
                f"    [{i}/{n}] {len(text):2d} chars | "
                f"first_chunk={ttfb * 1000:7.1f}ms total={total:5.2f}s | "
                f"{chunks:3d} chunks / {audio_sec:.2f}s audio | "
                f"{_pretty(text)!r}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"    ⚠️  [{i}/{n}] failed: {e}")


# ---------------------------------------------------------------------------
# Phase 3 - End-to-end via the real driver
# ---------------------------------------------------------------------------
def _trim_video(src: Path, dst: Path, seconds: float) -> bool:
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(src),
                "-t", f"{seconds:.2f}",
                "-c:v", "copy",
                "-c:a", "aac",       # re-encode audio for clean stream boundary
                str(dst),
            ],
            check=True,
            timeout=60,
        )
        return dst.exists() and dst.stat().st_size > 1024
    except Exception as e:  # noqa: BLE001
        print(f"    ⚠️  ffmpeg trim failed: {e}")
        return False


def warm_e2e(video_path: Path, seconds: float) -> None:
    print(f"\n🎬 [3/3] End-to-end warmup via {video_path.name} "
          f"(first {seconds:.0f}s) ...")
    if not video_path.exists():
        print(f"    ⚠️  demo video missing: {video_path}")
        return

    try:
        from eval.video_file_driver import run_video  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"    ⚠️  cannot import video_file_driver: {e}")
        return

    tmpdir = Path(tempfile.mkdtemp(prefix="aura_warmup_"))
    clip = tmpdir / "clip.mp4"
    outdir = tmpdir / "out"

    try:
        if not _trim_video(video_path, clip, seconds):
            print("    ⚠️  using full source video (fallback)")
            clip = video_path

        t0 = time.time()
        run_video(
            video_path=clip,
            output_dir=outdir,
            host=VLLM_HOST,
            port=VLLM_PORT,
            realtime=True,
            trailing_seconds=3.0,
            # Slightly relaxed VAD so short clips still trigger at least one
            # ASR segment, ensuring the ASR→vLLM→TTS loop is exercised.
            vad_threshold=0.4,
            vad_min_segment_ms=150,
        )
        print(f"    ✓ E2E warmup done in {time.time() - t0:.2f}s")
    except Exception as e:  # noqa: BLE001
        print(f"    ⚠️  E2E warmup failed: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref-audio", default=str(ROOT / "shuhan.mp3"),
                   help="audio file for ASR warmup (default: shuhan.mp3)")
    p.add_argument("--demo-video",
                   default=str(ROOT / "demos" / "我刚才关灯了吗.mp4"),
                   help="demo clip for E2E warmup (default: shortest demo)")
    p.add_argument("--e2e-seconds", type=float, default=10.0,
                   help="seconds of the demo video to stream (default 10)")
    p.add_argument("--skip-e2e", action="store_true",
                   help="run only ASR + TTS warmup")
    p.add_argument("--wait-timeout", type=float, default=300.0,
                   help="how long to wait for services to come up")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("🔎 Checking services ...")
    ok = (
        wait_http(f"{ASR_URL}/docs",
                  timeout=args.wait_timeout, label="ASR")
        and wait_http(f"{TTS_URL}/v1/tts/health",
                      timeout=args.wait_timeout, label="TTS")
        and wait_tcp(VLLM_HOST, VLLM_PORT,
                     timeout=args.wait_timeout, label="vLLM")
    )
    if not ok:
        print("❌ one or more services not ready; skipping warmup")
        return 1

    t_total = time.time()

    try:
        warm_asr(Path(args.ref_audio))
    except Exception as e:  # noqa: BLE001
        print(f"    ⚠️  ASR warmup failed: {e}")

    try:
        warm_tts()
    except Exception as e:  # noqa: BLE001
        print(f"    ⚠️  TTS warmup failed: {e}")

    if not args.skip_e2e:
        try:
            warm_e2e(Path(args.demo_video), seconds=args.e2e_seconds)
        except Exception as e:  # noqa: BLE001
            print(f"    ⚠️  E2E warmup failed: {e}")

    print(f"\n✅ Warmup finished in {time.time() - t_total:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
