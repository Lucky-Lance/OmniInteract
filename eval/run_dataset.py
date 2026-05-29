"""Batch evaluation runner over a dataset of MP4 + JSON pairs.

Supports the OmniInteract ``data/`` layout out of the box (point ``--root`` at a
subset directory):

    # 1q1a / 1q1a_math  (paired via video_json_map.json)
    <root>/video_json_map.json   -> entries of {"video": "videos/0001.mp4",
                                                 "annotation": "annotations/0001.json"}
    <root>/videos/0001.mp4
    <root>/annotations/0001.json

    # 1qna  (no map; videos_bench/ mirrors annotations/)
    <root>/videos_bench/<cat>/<name>.mp4
    <root>/annotations/<cat>/<name>.json

It also still supports the generic ego-centric layout (one video per leaf dir,
JSON co-located or under ``--gt_root``):

    <root>/<person>/<id>/<id>.mp4
    <root>/<person>/<id>/<id>.json

For every video we:
  1. open a fresh TCP connection to the AURA service,
  2. run :func:`eval.video_file_driver.run_video` (which itself sends
     `START_CAMERA` + `CLEAR_CONTEXT` to reset the server-side session),
  3. copy the matching ``<id>.json`` next to the outputs as ``gt.json``,
  4. write a top-level ``run_summary.jsonl`` with status / wall-clock time.

The AURA server only accepts a single concurrent client connection, so this
runner processes videos strictly sequentially.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.video_file_driver import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_FRAME_EDGE,
    run_video,
)


def _discover_videos(root: Path) -> List[Path]:
    """Find every MP4 under ``root``, following symlinks.

    Uses ``os.walk(followlinks=True)`` instead of ``pathlib.rglob`` because
    the latter does **not** follow symbolic links (even on Python 3.13).
    """
    import os

    out: List[Path] = []
    for dirpath, _dirs, files in os.walk(str(root), followlinks=True):
        for f in sorted(files):
            if f.lower().endswith(".mp4"):
                out.append(Path(dirpath) / f)
    out.sort()
    return out


def _flatten(rel: Path) -> str:
    """``videos/0001.mp4`` / ``egoper/x.mp4`` -> ``0001`` / ``egoper__x``."""
    return str(rel.with_suffix("")).replace("/", "__")


def _strip_media_prefix(rel: Path) -> Path:
    """Drop a leading ``videos/`` or ``videos_bench/`` component for naming."""
    if rel.parts and rel.parts[0] in ("videos", "videos_bench"):
        return Path(*rel.parts[1:])
    return rel


class Task:
    """A single (video, ground-truth, output-name) work item."""

    __slots__ = ("video", "gt", "output_name")

    def __init__(self, video: Path, gt: Optional[Path], output_name: str):
        self.video = video
        self.gt = gt
        self.output_name = output_name


def _collect_tasks(root: Path, gt_root: Optional[Path]) -> List[Task]:
    """Discover (video, gt, output_name) tasks under ``root``.

    Three layouts are recognised, in priority order:

    1. ``root/video_json_map.json`` present (OmniInteract 1q1a / 1q1a_math):
       pair each entry's ``video`` with its ``annotation``.
    2. ``root/videos_bench/`` present (OmniInteract 1qna): mirror
       ``videos_bench/<rel>.mp4`` to ``annotations/<rel>.json``.
    3. Generic: walk for every ``*.mp4``; GT is ``gt_root/<rel>.json`` when
       ``--gt_root`` is given, else co-located ``<video>.json``.
    """
    tasks: List[Task] = []

    map_file = root / "video_json_map.json"
    if map_file.exists():
        with open(map_file) as f:
            mapping = json.load(f)
        for entry in mapping.get("entries", []):
            video = (root / entry["video"]).resolve()
            if not video.exists():
                continue
            ann = entry.get("annotation")
            gt = (root / ann).resolve() if ann else None
            if gt is not None and not gt.exists():
                gt = None
            output_name = _flatten(_strip_media_prefix(Path(entry["video"])))
            tasks.append(Task(video, gt, output_name))
        return tasks

    videos_bench = root / "videos_bench"
    if videos_bench.is_dir():
        ann_dir = root / "annotations"
        for video in _discover_videos(videos_bench):
            rel = video.relative_to(videos_bench)
            gt = (ann_dir / rel).with_suffix(".json")
            tasks.append(Task(video, gt if gt.exists() else None, _flatten(rel)))
        return tasks

    for video in _discover_videos(root):
        rel = video.relative_to(root)
        if gt_root is not None:
            gt = (gt_root / rel).with_suffix(".json")
        else:
            gt = video.with_suffix(".json")
        tasks.append(Task(video, gt if gt.exists() else None, _flatten(rel)))
    return tasks


def _copy_gt(gt_src: Optional[Path], out_dir: Path) -> Optional[Path]:
    """Copy the ground-truth JSON next to the outputs as ``gt.json``."""
    if gt_src is None or not gt_src.exists():
        return None
    gt_dst = out_dir / "gt.json"
    shutil.copyfile(gt_src, gt_dst)
    return gt_dst


def _is_complete(out_dir: Path) -> bool:
    """Heuristic: the run produced its core artefacts."""
    return (out_dir / "output.wav").exists() and \
           (out_dir / "wav_transcript.json").exists() and \
           (out_dir / "responses.jsonl").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                        help="dataset root, e.g. /scratch/.../ego_dataset_collection")
    parser.add_argument("--output", required=True,
                        help="output root, e.g. eval_outputs/")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--limit", type=int, default=0,
                        help="if >0, only process this many videos")
    parser.add_argument("--skip_existing", action="store_true",
                        help="skip videos whose output dir already has output.wav")
    parser.add_argument("--persons", nargs="*",
                        help="optional list of person folders to include")
    parser.add_argument("--ids", nargs="*",
                        help="optional list of sample ids to include")
    parser.add_argument("--no_realtime", action="store_true")
    parser.add_argument("--trailing_seconds", type=float, default=8.0)
    parser.add_argument("--vad_threshold", type=float, default=0.5)
    parser.add_argument("--vad_min_silence_ms", type=int, default=500)
    parser.add_argument("--vad_speech_pad_ms", type=int, default=100)
    parser.add_argument("--vad_min_segment_ms", type=int, default=300)
    parser.add_argument("--max_frame_edge", type=int,
                        default=DEFAULT_MAX_FRAME_EDGE)
    parser.add_argument("--jpeg_quality", type=int,
                        default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--extra_video_seconds", type=float, default=0.0)
    parser.add_argument("--gt_root", default=None,
                        help="alternative root for ground-truth JSON files "
                             "(when not co-located with videos)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="total number of shards for parallel runs")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="this worker's shard index (0-based)")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    gt_root = Path(args.gt_root).expanduser().resolve() if args.gt_root else None
    output_root.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"[Batcher] dataset root not found: {root}")
        sys.exit(1)

    tasks = _collect_tasks(root, gt_root)
    if args.persons:
        wanted = set(args.persons)
        tasks = [t for t in tasks if t.video.parent.parent.name in wanted]
    if args.ids:
        wanted = set(args.ids)
        tasks = [t for t in tasks if t.video.parent.name in wanted]
    if args.num_shards > 1:
        tasks = tasks[args.shard_index :: args.num_shards]
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    print(f"[Batcher] discovered {len(tasks)} videos under {root}")

    summary_path = output_root / "run_summary.jsonl"
    summary_fp = open(summary_path, "a", encoding="utf-8")

    n_ok = n_fail = n_skip = 0
    for i, task in enumerate(tasks, 1):
        video = task.video
        out_dir = output_root / task.output_name
        out_dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "index": i,
            "video": str(video),
            "output_dir": str(out_dir),
            "person": video.parent.parent.name,
            "id": video.parent.name,
        }
        if args.skip_existing and _is_complete(out_dir):
            print(f"[Batcher][{i}/{len(tasks)}] SKIP existing {out_dir}")
            rec.update({"status": "skipped", "elapsed_sec": 0.0})
            summary_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            summary_fp.flush()
            n_skip += 1
            continue

        print(f"\n[Batcher][{i}/{len(tasks)}] {video}")
        t_start = time.time()
        try:
            run_video(
                video_path=video,
                output_dir=out_dir,
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
            _copy_gt(task.gt, out_dir)
            rec.update({"status": "ok",
                        "elapsed_sec": round(time.time() - t_start, 2)})
            n_ok += 1
        except KeyboardInterrupt:
            print("[Batcher] interrupted by user")
            rec.update({"status": "interrupted",
                        "elapsed_sec": round(time.time() - t_start, 2)})
            summary_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            summary_fp.flush()
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[Batcher] FAILED on {video}: {exc}")
            traceback.print_exc()
            rec.update({"status": "failed",
                        "error": str(exc),
                        "elapsed_sec": round(time.time() - t_start, 2)})
            n_fail += 1
        finally:
            summary_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            summary_fp.flush()
            # Brief pause so the AURA server can release the previous
            # connection (its accept loop only handles one client at a
            # time).
            time.sleep(1.0)

    summary_fp.close()
    print(
        f"\n[Batcher] done: ok={n_ok}, failed={n_fail}, "
        f"skipped={n_skip}, summary={summary_path}"
    )


if __name__ == "__main__":
    main()
