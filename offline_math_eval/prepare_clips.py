"""Step 1: Extract the first and last 1-minute clip from each 5-minute video
and emit an evaluation manifest.

Output:
    clips/                  extracted video clips
    eval_manifest.json      eval manifest, list of
                            {clip_path, question, ground_truth, ...}

Usage:
    python offline_math_eval/prepare_clips.py \
        --data_root data/1q1a_math \
        --output_dir offline_math_eval/clips
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import List


# Real Q&A entries always use an MM:SS or HH:MM:SS timestamp. The first entry
# in every annotation file is a template placeholder with a free-form
# description in question_time; this regex filters it out without depending
# on the placeholder's exact text.
TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")


def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def clip_video(src: str, dst: str, start_sec: float, duration_sec: float):
    """Cut a video clip with ffmpeg (no re-encode)."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start_sec),
        "-i", src,
        "-t", str(duration_sec),
        "-c", "copy",
        dst,
    ]
    subprocess.run(cmd, check=True)


def parse_annotation(json_path: str) -> List[dict]:
    """Parse an annotation JSON, dropping the template placeholder entry."""
    with open(json_path, encoding="utf-8") as f:
        entries = json.load(f)
    return [
        e for e in entries
        if "question_time" in e and TIMESTAMP_RE.match(str(e["question_time"]))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/1q1a_math",
                        help="root of the 1q1a_math subset")
    parser.add_argument("--output_dir", default="offline_math_eval/clips",
                        help="output directory for the extracted clips")
    parser.add_argument("--clip_duration", type=float, default=60.0,
                        help="clip length in seconds")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    map_file = data_root / "video_json_map.json"
    with open(map_file) as f:
        mapping = json.load(f)

    entries = mapping["entries"]
    print(f"Found {len(entries)} videos under {data_root}")

    manifest: List[dict] = []
    clip_dur = args.clip_duration

    for idx, entry in enumerate(entries):
        video_path = data_root / entry["video"]
        json_path = data_root / entry["annotation"]

        if not video_path.exists():
            print(f"[{idx+1}/{len(entries)}] skip (video missing): {entry['video']}")
            continue
        if not json_path.exists():
            print(f"[{idx+1}/{len(entries)}] skip (annotation missing): {entry['annotation']}")
            continue

        qas = parse_annotation(str(json_path))
        if len(qas) < 2:
            print(f"[{idx+1}/{len(entries)}] skip (fewer than 2 questions): {entry['annotation']}")
            continue

        duration = get_video_duration(str(video_path))
        video_stem = video_path.stem

        # Q1 -> first clip_dur seconds
        clip1_name = f"{video_stem}__q1.mp4"
        clip1_path = output_dir / clip1_name
        if not clip1_path.exists():
            clip_video(str(video_path), str(clip1_path),
                       start_sec=0, duration_sec=clip_dur)

        manifest.append({
            "clip_path": str(clip1_path),
            "clip_name": clip1_name,
            "source_video": entry["video"],
            "question_idx": 0,
            "question_time": qas[0].get("question_time", ""),
            "question_text": qas[0]["question_text"],
            "answer_time": qas[0].get("answer_time", ""),
            "ground_truth": qas[0]["answer_text"],
            "clip_start_sec": 0,
            "clip_duration_sec": clip_dur,
            "video_duration_sec": round(duration, 2),
        })

        # Q2 -> last clip_dur seconds
        clip2_start = max(0, duration - clip_dur)
        clip2_name = f"{video_stem}__q2.mp4"
        clip2_path = output_dir / clip2_name
        if not clip2_path.exists():
            clip_video(str(video_path), str(clip2_path),
                       start_sec=clip2_start, duration_sec=clip_dur)

        manifest.append({
            "clip_path": str(clip2_path),
            "clip_name": clip2_name,
            "source_video": entry["video"],
            "question_idx": 1,
            "question_time": qas[1].get("question_time", ""),
            "question_text": qas[1]["question_text"],
            "answer_time": qas[1].get("answer_time", ""),
            "ground_truth": qas[1]["answer_text"],
            "clip_start_sec": round(clip2_start, 2),
            "clip_duration_sec": clip_dur,
            "video_duration_sec": round(duration, 2),
        })

        print(f"[{idx+1}/{len(entries)}] done: {entry['video']} "
              f"(duration={duration:.1f}s, 2 clips)")

    manifest_path = output_dir.parent / "eval_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nExtraction done; {len(manifest)} eval samples")
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
