#!/usr/bin/env python3
"""Score MiniCPM-o offline math results with the existing LLM judge API.

Input:  offline_math_eval/all_results.json from merge_results.py
Output: offline_math_eval/offline_quality_summary.json
        offline_math_eval/offline_quality_details.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval" / "evaluation"
sys.path.insert(0, str(EVAL_DIR))

from common import clamp, save_json  # noqa: E402
from llm_judge import APIJudge, JudgeAPICallError  # noqa: E402


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_slot(record: Dict[str, Any], slot_id: int) -> Dict[str, Any]:
    return {
        "slot_id": slot_id,
        "scene_type": "offline_math",
        "turn_index": int(record.get("question_idx", 0)) + 1,
        "step_index": None,
        "boundary_type": "Offline",
        "is_interrupted": False,
        "question_text": str(record.get("question_text", "") or ""),
        "gt_answer": str(record.get("ground_truth", "") or ""),
    }


def score_record(judge: APIJudge, record: Dict[str, Any], slot_id: int) -> Dict[str, Any]:
    answer = str(record.get("model_answer", "") or "").strip()
    if record.get("status") != "ok" or not answer:
        return {
            "clip_name": record.get("clip_name", ""),
            "status": record.get("status", "error"),
            "score": 0.0,
            "rationale": "Missing or failed offline inference output.",
            "parse_source": "missing_output",
        }

    slot = make_slot(record, slot_id)
    score, meta = judge.judge_core(
        slot=slot,
        full_context=answer,
        actual_text=answer,
        future_answers="",
    )
    return {
        "clip_name": record.get("clip_name", ""),
        "source_video": record.get("source_video", ""),
        "question_idx": record.get("question_idx"),
        "status": "ok",
        "score": clamp(score),
        "rationale": meta.get("rationale", ""),
        "parse_source": meta.get("parse_source", ""),
        "raw": meta.get("raw", ""),
    }


def score_record_with_retries(
    judge: APIJudge,
    record: Dict[str, Any],
    slot_id: int,
    max_retries: int,
) -> Dict[str, Any]:
    clip_name = str(record.get("clip_name", "") or "")
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            return score_record(judge, record, slot_id)
        except JudgeAPICallError as exc:
            last_error = str(exc)
            wait = min(2 ** attempt, 30)
            print(f"[retry {attempt}/{max_retries}] {clip_name}: {last_error[:200]}")
            time.sleep(wait)
    return {
        "clip_name": clip_name,
        "source_video": record.get("source_video", ""),
        "question_idx": record.get("question_idx"),
        "status": "judge_error",
        "score": 0.0,
        "rationale": last_error,
        "parse_source": "judge_error",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score offline MiniCPM-o math answers with an LLM judge.")
    parser.add_argument("--input", type=Path, default=Path("offline_math_eval/all_results.json"))
    parser.add_argument("--details", type=Path, default=Path("offline_math_eval/offline_quality_details.json"))
    parser.add_argument("--output", type=Path, default=Path("offline_math_eval/offline_quality_summary.json"))
    parser.add_argument("--judge_api_url", default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--judge_api_key", default=os.getenv("JUDGE_API_KEY", ""))
    parser.add_argument("--judge_api_model", default="gpt-4o-2024-08-06")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--timeout_sec", type=float, default=60.0)
    parser.add_argument("--sleep_sec", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    records: List[Dict[str, Any]] = load_json(args.input)
    done: Dict[str, Dict[str, Any]] = {}
    if args.details.exists():
        for item in load_json(args.details):
            clip_name = str(item.get("clip_name", "") or "")
            if clip_name:
                done[clip_name] = item

    judge = APIJudge(
        api_url=args.judge_api_url,
        api_key=args.judge_api_key,
        model=args.judge_api_model,
        max_tokens=args.max_tokens,
        temperature=0.0,
        timeout_sec=args.timeout_sec,
    )

    details: List[Dict[str, Any]] = list(done.values())
    total = len(records)
    pending = [
        (idx, record)
        for idx, record in enumerate(records, start=1)
        if str(record.get("clip_name", "") or "") not in done
    ]
    for idx, record in enumerate(records, start=1):
        clip_name = str(record.get("clip_name", "") or "")
        if clip_name in done:
            print(f"[{idx}/{total}] skip {clip_name} score={done[clip_name].get('score')}")

    workers = max(1, int(args.num_workers))
    if pending:
        print(f"Scoring {len(pending)} pending items with {workers} worker(s)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(score_record_with_retries, judge, record, idx, args.max_retries): (idx, record)
            for idx, record in pending
        }
        for future in concurrent.futures.as_completed(futures):
            idx, record = futures[future]
            clip_name = str(record.get("clip_name", "") or "")
            item = future.result()
            done[clip_name] = item
            details = list(done.values())
            save_json(args.details, details)
            print(f"[{idx}/{total}] scored {clip_name} status={item.get('status')} score={float(item.get('score', 0.0)):.4f}")
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    ok_items = [x for x in details if x.get("status") == "ok"]
    avg = sum(float(x.get("score", 0.0) or 0.0) for x in ok_items) / len(ok_items) if ok_items else 0.0
    summary = {
        "avg_quality_score": avg,
        "scored_items": len(ok_items),
        "total_ok_items": sum(1 for r in records if r.get("status") == "ok"),
        "total_items": len(records),
        "judge_error_items": sum(1 for x in details if x.get("status") == "judge_error"),
        "details_path": str(args.details),
        "input_path": str(args.input),
    }
    save_json(args.output, summary)
    print(f"Saved summary to {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
