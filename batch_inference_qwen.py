"""Qwen-Omni Realtime full-duplex batch inference script.

Spawns concurrent subprocesses that call run_with_video_file.py, with
resume-on-restart support. Qwen runs over the DashScope API (no GPU needed);
the bottleneck is concurrent connection count and real-time wall clock.

Usage:
    export DASHSCOPE_API_KEY=sk-xxx
    python batch_inference_qwen.py \
        --workers 10 \
        --output_root outputs/qwen

Each video takes ~300s of real-time processing. 10 workers x 250 videos ~= 2 hours.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List


SUBSETS_1Q1A = ("1q1a", "1q1a_math")


# -- Video collection ---------------------------------------------------------
def collect_all_videos(data_root: Path) -> List[dict]:
    """Scan data/ for 1q1a, 1q1a_math, and 1qna videos.

    Returns a list of {"video_path", "output_name", "subset"} dicts.
    """
    data_root = data_root.resolve()
    videos = []

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
                    "video_path": str(vp.resolve()),
                    "output_name": f"{subset}/{output_name}",
                    "subset": subset,
                })

    bench_dir = data_root / "1qna" / "videos_bench"
    if bench_dir.exists():
        for mp4 in sorted(bench_dir.rglob("*.mp4")):
            rel = mp4.relative_to(bench_dir)
            output_name = str(rel).replace("/", "__").replace(".mp4", "")
            videos.append({
                "video_path": str(mp4.resolve()),
                "output_name": f"1qna/{output_name}",
                "subset": "1qna",
            })

    return videos


# -- Per-video worker (subprocess shell-out to the adapter) -------------------
QWEN_SCRIPT = (
    Path(__file__).resolve().parent
    / "alibabacloud-bailian-speech-demo"
    / "samples"
    / "conversation"
    / "omni"
    / "python"
    / "run_with_video_file.py"
)


def run_single_video(
    video_path: str,
    output_dir: str,
    model: str,
    voice: str,
    turn_mode: str,
    probe_every_sec: float,
    turn_silence_ms: int,
    video_fps: float,
    max_frame_edge: int,
    jpeg_quality: int,
    no_realtime: bool,
) -> dict:
    """Call run_with_video_file.py in a subprocess and return a result summary."""
    t_start = time.time()

    cmd = [
        sys.executable, str(QWEN_SCRIPT),
        "--video", video_path,
        "--output_dir", output_dir,
        "--model", model,
        "--voice", voice,
        "--turn_mode", turn_mode,
        "--probe_every_sec", str(probe_every_sec),
        "--turn_silence_ms", str(turn_silence_ms),
        "--video_fps", str(video_fps),
        "--max_frame_edge", str(max_frame_edge),
        "--jpeg_quality", str(jpeg_quality),
    ]
    if no_realtime:
        cmd.append("--no_realtime")

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=900,  # 15 min timeout (5 min video + slack)
    )

    elapsed = time.time() - t_start
    out_dir = Path(output_dir)

    if proc.returncode == 0 and (out_dir / "model_output.jsonl").exists():
        return {
            "video": video_path,
            "output_dir": output_dir,
            "inference_sec": round(elapsed, 2),
            "status": "ok",
        }
    else:
        log_path = out_dir / "subprocess.log"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"=== STDOUT ===\n{proc.stdout}\n\n=== STDERR ===\n{proc.stderr}\n",
            encoding="utf-8",
        )
        return {
            "video": video_path,
            "output_dir": output_dir,
            "inference_sec": round(elapsed, 2),
            "status": "error",
            "error": f"exit_code={proc.returncode}",
            "log": str(log_path),
        }


def _worker(args_tuple):
    """ProcessPoolExecutor worker."""
    task, common_args = args_tuple
    video_path = task["video_path"]
    output_dir = str(Path(common_args["output_root"]) / task["output_name"])

    done_flag = Path(output_dir) / ".done"
    if done_flag.exists():
        return {"video": video_path, "output_dir": output_dir, "status": "skipped"}

    try:
        result = run_single_video(
            video_path=video_path,
            output_dir=output_dir,
            model=common_args["model"],
            voice=common_args["voice"],
            turn_mode=common_args["turn_mode"],
            probe_every_sec=common_args["probe_every_sec"],
            turn_silence_ms=common_args["turn_silence_ms"],
            video_fps=common_args["video_fps"],
            max_frame_edge=common_args["max_frame_edge"],
            jpeg_quality=common_args["jpeg_quality"],
            no_realtime=common_args["no_realtime"],
        )

        if result["status"] == "ok":
            done_flag.parent.mkdir(parents=True, exist_ok=True)
            done_flag.write_text(json.dumps({
                "video": video_path,
                "inference_sec": result["inference_sec"],
            }))

        return result

    except subprocess.TimeoutExpired as e:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "subprocess.log"
        log_path.write_text(
            f"=== TIMEOUT after 900s ===\n"
            f"=== STDOUT ===\n{e.stdout or ''}\n\n"
            f"=== STDERR ===\n{e.stderr or ''}\n",
            encoding="utf-8",
        )
        return {
            "video": video_path,
            "output_dir": output_dir,
            "status": "error",
            "error": "timeout (900s)",
            "log": str(log_path),
        }
    except Exception as e:
        return {
            "video": video_path,
            "output_dir": output_dir,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# -- Main ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Qwen-Omni batch inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workers", type=int, default=10,
                        help="number of concurrent workers (concurrent API connections)")
    parser.add_argument("--output_root", default="outputs/qwen")
    parser.add_argument("--model", default="qwen3.5-omni-flash-realtime")
    parser.add_argument("--voice", default="Tina")
    parser.add_argument("--turn_mode", default="server_vad",
                        choices=["server_vad", "manual"])
    parser.add_argument("--probe_every_sec", type=float, default=1.0)
    parser.add_argument("--turn_silence_ms", type=int, default=800)
    parser.add_argument("--video_fps", type=float, default=2.0)
    parser.add_argument("--max_frame_edge", type=int, default=640)
    parser.add_argument("--jpeg_quality", type=int, default=80)
    parser.add_argument("--no_realtime", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("error: please set the DASHSCOPE_API_KEY environment variable first")
        sys.exit(1)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    videos = collect_all_videos(Path("data"))
    print(f"Found {len(videos)} videos, using {args.workers} concurrent workers")

    already_done = sum(
        1 for v in videos
        if (output_root / v["output_name"] / ".done").exists()
    )
    remaining = len(videos) - already_done
    print(f"Done: {already_done}, remaining: {remaining}")

    if remaining == 0:
        print("All videos already done; nothing to do")
        _write_summary(output_root, videos)
        return

    # Submit only outstanding tasks instead of pushing all 250 onto the pool.
    pending_tasks = [
        v for v in videos
        if not (output_root / v["output_name"] / ".done").exists()
    ]
    print(f"Submitting {len(pending_tasks)} tasks to the worker pool")

    common_args = {
        "output_root": str(output_root),
        "model": args.model,
        "voice": args.voice,
        "turn_mode": args.turn_mode,
        "probe_every_sec": args.probe_every_sec,
        "turn_silence_ms": args.turn_silence_ms,
        "video_fps": args.video_fps,
        "max_frame_edge": args.max_frame_edge,
        "jpeg_quality": args.jpeg_quality,
        "no_realtime": args.no_realtime,
    }

    summary_path = output_root / "batch_progress.jsonl"
    ok_count = already_done
    failed_count = 0
    processed = 0
    t_all_start = time.time()

    with open(summary_path, "a", encoding="utf-8") as sf:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_worker, (task, common_args)): task
                for task in pending_tasks
            }

            try:
                for future in as_completed(futures):
                    task = futures[future]
                    processed += 1
                    try:
                        result = future.result()
                    except Exception as e:
                        result = {
                            "video": task["video_path"],
                            "status": "error",
                            "error": str(e),
                        }

                    status = result.get("status", "?")
                    vname = Path(result.get("video", "")).name

                    if status == "ok":
                        ok_count += 1
                        sf.write(json.dumps(result, ensure_ascii=False) + "\n")
                        sf.flush()
                        elapsed = result.get("inference_sec", 0)
                        print(
                            f"[{processed}/{len(pending_tasks)}] "
                            f"done: {vname} ({elapsed:.0f}s) "
                            f"[overall {ok_count}/{len(videos)}]"
                        )
                    elif status == "skipped":
                        ok_count += 1
                    elif status == "error":
                        failed_count += 1
                        sf.write(json.dumps(result, ensure_ascii=False) + "\n")
                        sf.flush()
                        err = result.get("error", "unknown")
                        print(
                            f"[{processed}/{len(pending_tasks)}] "
                            f"failed: {vname} -> {err}"
                        )
            except KeyboardInterrupt:
                print("\nCtrl+C received, cancelling remaining tasks ...")
                for f in futures:
                    f.cancel()
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    pool.shutdown(wait=False)
                print("Cancelled. Re-run to resume from where we left off.")
                _write_summary(output_root, videos)
                sys.exit(130)

    total_time = time.time() - t_all_start
    print(
        f"\nAll done: {ok_count} ok, {failed_count} failed, "
        f"elapsed {total_time/60:.1f} min"
    )

    _write_summary(output_root, videos)


def _write_summary(output_root: Path, videos: list):
    """Merge the final summary."""
    results = []
    for v in videos:
        out_dir = output_root / v["output_name"]
        done_flag = out_dir / ".done"
        if done_flag.exists():
            results.append({
                "video": v["video_path"],
                "output_dir": str(out_dir),
                "status": "ok",
            })
        else:
            results.append({
                "video": v["video_path"],
                "output_dir": str(out_dir),
                "status": "incomplete",
            })

    ok = sum(1 for r in results if r["status"] == "ok")
    summary = {
        "total": len(results),
        "success": ok,
        "incomplete": len(results) - ok,
        "results": results,
    }

    out = output_root / "batch_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary written: {out} (ok: {ok}/{len(results)})")


if __name__ == "__main__":
    main()
