"""
Merge the source video with the model's response audio (output.wav) into a
single video file.

output.wav has the same duration as the video and the model responses are
time-aligned to the source. The merged file plays back the user (source video
audio track) and the model's reply (output.wav) simultaneously.

Usage:
    python merge_video_response.py --video data/video3.mp4 \\
        --wav alibabacloud-bailian-speech-demo/samples/conversation/omni/python/outputs/video3/output.wav \\
        --output merged_video3_qwen.mp4

    # Use only the model audio (don't mix the source track):
    python merge_video_response.py --video data/video3.mp4 \\
        --wav outputs/video3/output.wav \\
        --output merged.mp4 --no-mix
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def has_audio_stream(video_path: str) -> bool:
    """Return True if the video file has at least one audio stream."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            video_path,
        ],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def merge(video: Path, wav: Path, output: Path, mix: bool, vol_video: float, vol_model: float) -> None:
    video_has_audio = has_audio_stream(str(video))

    if mix and video_has_audio:
        # Mix: source audio track + model reply track
        filter_complex = (
            f"[0:a]volume={vol_video}[a_user];"
            f"[1:a]volume={vol_model}[a_model];"
            f"[a_user][a_model]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            "-i", str(wav),
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output),
        ]
        print(f"[mix] video_vol={vol_video}, model_vol={vol_model}")
    else:
        # Use only the model audio (replaces or fills in an empty source track)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            "-i", str(wav),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(output),
        ]
        print("[model-only] source audio not mixed in")

    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("[ERROR] ffmpeg failed", file=sys.stderr)
        sys.exit(1)
    print(f"\nDone. Output: {output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="source input video path")
    p.add_argument("--wav", required=True, help="model reply audio path (output.wav)")
    p.add_argument("--output", required=True, help="output video path (mp4)")
    p.add_argument(
        "--no-mix", dest="mix", action="store_false", default=True,
        help="don't mix in the source audio; use only the model audio",
    )
    p.add_argument(
        "--vol-video", type=float, default=1.0,
        help="source audio volume multiplier (default: 1.0)",
    )
    p.add_argument(
        "--vol-model", type=float, default=1.5,
        help="model reply volume multiplier (default: 1.5; mildly boosted for clarity)",
    )
    args = p.parse_args()

    video = Path(args.video)
    wav = Path(args.wav)
    output = Path(args.output)

    if not video.exists():
        print(f"[ERROR] video missing: {video}", file=sys.stderr)
        sys.exit(1)
    if not wav.exists():
        print(f"[ERROR] WAV missing: {wav}", file=sys.stderr)
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    merge(video, wav, output, args.mix, args.vol_video, args.vol_model)


if __name__ == "__main__":
    main()
