#!/usr/bin/env python3
"""One-shot public entry point for OmniInteract evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model-output matching, LLM judging, slot scoring, and aggregate metric summarization."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="JSON/JSONL manifest with sample_id, gt_json, model_json, scene_type.")
    source.add_argument("--batch_summary_json", help="Inference batch_summary.json produced by model runners.")
    parser.add_argument("--output_root", default="", help="Optional model-output root used with --batch_summary_json.")
    parser.add_argument("--model_json_name", default="", help="Force a model output filename, e.g. wav_transcript_aligned.json.")
    parser.add_argument("--out_dir", required=True, help="Directory for *.unified_match.json, *.unified_eval.json, and summary.")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--last_slot_tail_sec", type=float, default=60.0)
    parser.add_argument("--w_ack", type=float, default=0.2)
    parser.add_argument("--decay_alpha", type=float, default=1.0)
    parser.add_argument("--decay_gamma", type=float, default=1.0)
    parser.add_argument("--core_quality_threshold", type=float, default=0.5)
    parser.add_argument("--partial_quality_threshold", type=float, default=0.5)
    parser.add_argument("--no_interrupted_output_eval", action="store_true")
    parser.add_argument("--no_unmatched_fp", action="store_true")
    parser.add_argument("--judge_api_url", default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--judge_api_model", default="gpt-4o-2024-08-06")
    parser.add_argument("--judge_api_key", default="")
    parser.add_argument("--judge_api_timeout_sec", type=float, default=60.0)
    parser.add_argument("--judge_max_tokens", type=int, default=512)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    return parser.parse_args()


def add_arg(cmd: list[str], flag: str, value: object) -> None:
    if value not in (None, ""):
        cmd.extend([flag, str(value)])


def add_bool(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def main() -> None:
    args = parse_args()
    root = repo_root()
    runner = root / "eval" / "evaluation" / "run_batch_unified_eval.py"
    cmd = [sys.executable, str(runner)]
    add_arg(cmd, "--manifest", args.manifest)
    add_arg(cmd, "--batch_summary_json", args.batch_summary_json)
    add_arg(cmd, "--output_root", args.output_root)
    add_arg(cmd, "--model_json_name", args.model_json_name)
    add_arg(cmd, "--out_dir", args.out_dir)
    add_arg(cmd, "--num_workers", args.num_workers)
    add_arg(cmd, "--limit", args.limit)
    add_arg(cmd, "--last_slot_tail_sec", args.last_slot_tail_sec)
    add_arg(cmd, "--w_ack", args.w_ack)
    add_arg(cmd, "--decay_alpha", args.decay_alpha)
    add_arg(cmd, "--decay_gamma", args.decay_gamma)
    add_arg(cmd, "--core_quality_threshold", args.core_quality_threshold)
    add_arg(cmd, "--partial_quality_threshold", args.partial_quality_threshold)
    add_arg(cmd, "--judge_api_url", args.judge_api_url)
    add_arg(cmd, "--judge_api_model", args.judge_api_model)
    add_arg(cmd, "--judge_api_key", args.judge_api_key)
    add_arg(cmd, "--judge_api_timeout_sec", args.judge_api_timeout_sec)
    add_arg(cmd, "--judge_max_tokens", args.judge_max_tokens)
    add_arg(cmd, "--judge_temperature", args.judge_temperature)
    add_bool(cmd, "--skip_existing", args.skip_existing)
    add_bool(cmd, "--dry_run", args.dry_run)
    add_bool(cmd, "--no_interrupted_output_eval", args.no_interrupted_output_eval)
    add_bool(cmd, "--no_unmatched_fp", args.no_unmatched_fp)
    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
