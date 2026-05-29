#!/usr/bin/env python3
"""Evaluate interrupted slots that already have model output.

This diagnostic script does not change Global IA-QTF1. It reads existing
*.unified_eval.json and sibling *.unified_match.json files, then asks an LLM
judge to score the usefulness of the content that was already spoken in
interrupted slots. Spill duration and no-output rates are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from common import clamp, f1_from_pr, safe_div, save_json, to_bool
from llm_judge import APIJudge, JudgeAPICallError, extract_first_json_object


MODEL_ORDER = ("aura", "gemini", "minicpmo", "qwen")


PARTIAL_QUALITY_SYSTEM_PROMPT = """You are a strict evaluator for interrupted voice-assistant answers.
Judge only from the provided text. Return valid JSON only, with no extra text."""


PARTIAL_QUALITY_USER_TEMPLATE = """Evaluate the quality of the assistant output that was already spoken before or around an interruption.

[Task]
The user asked a question, but then interrupted or changed the topic. The assistant was not required to complete the full original answer. Your job is to score whether the content that the assistant already spoke is relevant, correct, and useful for the current ground-truth answer.

[Question]
{question}

[Ground Truth Answer]
{gt_answer}

[Assistant Output Already Spoken]
{actual_text}

[Scoring Rules]
1. Score from 0 to 1.
2. Do NOT penalize the output merely because it is incomplete. The user interrupted, so a partial answer can receive a high score if the spoken part is correct and useful.
3. Score high when the spoken content substantially overlaps with, paraphrases, or correctly conveys useful parts of the ground-truth answer.
4. Score low when the output is only an acknowledgement or preface, such as "OK, I will read it", without substantive answer content.
5. Score low when the output answers the wrong question, is irrelevant, or is mostly generic filler.
6. Mark hallucination=true if the output contains clear incorrect facts, wrong target content, or content unsupported by the ground truth.
7. Ignore interruption overflow duration when assigning quality; overflow is measured separately.
8. Example: If the user asks the assistant to read kitchen rules, output that only says "OK, I will read the rules" should receive a low score because it does not satisfy the request. Output that correctly reads several rules should receive a high score even if it does not read every rule. Output that reads many correct rules but continues after the interruption should still receive a high quality score; the overflow will be penalized separately.

Return JSON in this exact shape:
{{"score": <float 0..1>, "hallucination": <true|false>, "rationale": "<one concise sentence>"}}"""


class PartialQualityJudge:
    def __init__(self, api: APIJudge) -> None:
        self.api = api

    def judge(self, slot: Dict[str, Any], actual_text: str) -> Dict[str, Any]:
        raw = self.api._generate(
            PARTIAL_QUALITY_SYSTEM_PROMPT,
            PARTIAL_QUALITY_USER_TEMPLATE.format(
                question=str(slot.get("question_text", "") or "").strip(),
                gt_answer=str(slot.get("gt_answer", "") or "").strip(),
                actual_text=str(actual_text or "").strip(),
            ),
        )
        obj = extract_first_json_object(raw) or {}
        try:
            score = clamp(float(obj.get("score", 0.0)))
        except Exception:
            score = 0.0
        return {
            "score": score,
            "hallucination": to_bool(obj.get("hallucination", False)),
            "rationale": str(obj.get("rationale", "") or "").strip(),
            "raw": raw,
            "parse_source": "llm_json" if obj else "llm_parse_failed",
        }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def model_sort_key(path: Path) -> Tuple[int, str]:
    try:
        return (MODEL_ORDER.index(path.name), path.name)
    except ValueError:
        return (len(MODEL_ORDER), path.name)


def eval_dirs_from_args(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    if args.eval_dir:
        p = args.eval_dir.resolve()
        return [(args.model_name or p.parent.name or p.name, p)]
    out: List[Tuple[str, Path]] = []
    for model_dir in sorted(args.outputs_dir.iterdir(), key=model_sort_key):
        cand = model_dir / args.eval_subdir
        if cand.is_dir():
            out.append((model_dir.name, cand.resolve()))
    return out


def sibling_match_path(eval_path: Path) -> Path:
    return eval_path.with_name(eval_path.name.replace(".unified_eval.json", ".unified_match.json"))


def sample_id_from_path(path: Path) -> str:
    return path.name.replace(".unified_eval.json", "")


def concat_chunk_text(chunks: Iterable[Dict[str, Any]]) -> str:
    return "".join(str(c.get("text", "") or "") for c in chunks if isinstance(c, dict)).strip()


def chunk_bounds(chunks: Iterable[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    starts: List[float] = []
    ends: List[float] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        try:
            if chunk.get("start") is not None:
                starts.append(float(chunk.get("start")))
            if chunk.get("end") is not None:
                ends.append(float(chunk.get("end")))
        except Exception:
            continue
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def load_match_slot_by_id(eval_path: Path) -> Dict[str, Dict[str, Any]]:
    match_path = sibling_match_path(eval_path)
    if not match_path.exists():
        return {}
    root = load_json(match_path)
    out: Dict[str, Dict[str, Any]] = {}
    for slot in root.get("slots", []) if isinstance(root, dict) else []:
        if isinstance(slot, dict):
            out[str(slot.get("slot_id"))] = slot
    return out


def fallback_text_from_eval_slot(slot: Dict[str, Any]) -> str:
    early = slot.get("stage_early", {}) if isinstance(slot.get("stage_early"), dict) else {}
    core = slot.get("stage_core", {}) if isinstance(slot.get("stage_core"), dict) else {}
    return (str(early.get("text_early", "") or "") + str(core.get("text_core", "") or "")).strip()


def interrupted_actual_output(eval_path: Path, eval_slot: Dict[str, Any], match_slots: Dict[str, Dict[str, Any]]) -> Tuple[str, int, Optional[float], Optional[float]]:
    match_slot = match_slots.get(str(eval_slot.get("slot_id"))) or {}
    chunks = list(match_slot.get("all_chunks", []) or [])
    text = concat_chunk_text(chunks)
    start, end = chunk_bounds(chunks)
    num_chunks = len(chunks)
    if not text:
        text = fallback_text_from_eval_slot(eval_slot)
        num_chunks = int(eval_slot.get("num_chunks") or 0)
        max_chunk_end = (eval_slot.get("spill_diagnostics", {}) or {}).get("max_chunk_end")
        if max_chunk_end is not None:
            try:
                end = float(max_chunk_end)
            except Exception:
                end = None
    return text, num_chunks, start, end


def spill_seconds(slot: Dict[str, Any]) -> float:
    return float((slot.get("spill_diagnostics", {}) or {}).get("spill_seconds") or 0.0)


def precision_recall_f1(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {"Precision": precision, "Recall": recall, "F1": f1_from_pr(precision, recall)}


def read_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    root = load_json(path)
    return root if isinstance(root, dict) else {}


def write_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    save_json(path, cache)


def evaluate_dir(
    model: str,
    eval_dir: Path,
    judge: PartialQualityJudge,
    args: argparse.Namespace,
    cache: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    total_interrupted = 0
    no_output = 0
    judge_errors = 0

    for eval_path in sorted(eval_dir.glob(args.pattern)):
        try:
            root = load_json(eval_path)
            match_slots = load_match_slot_by_id(eval_path)
        except Exception as exc:
            rows.append({"model": model, "sample_id": sample_id_from_path(eval_path), "status": "load_error", "error": str(exc)})
            continue

        for slot in root.get("slots", []) if isinstance(root, dict) else []:
            if not isinstance(slot, dict) or not to_bool(slot.get("is_interrupted", False)):
                continue
            total_interrupted += 1
            actual_text, num_chunks, response_start, response_end = interrupted_actual_output(eval_path, slot, match_slots)
            has_output = bool(num_chunks > 0 or actual_text)
            sp = spill_seconds(slot)
            if not has_output:
                no_output += 1
                rows.append(
                    {
                        "model": model,
                        "sample_id": sample_id_from_path(eval_path),
                        "slot_id": slot.get("slot_id"),
                        "status": "no_output",
                        "has_output": False,
                        "num_chunks": num_chunks,
                        "spill_seconds": sp,
                        "metric_json": str(eval_path),
                        "match_json": str(sibling_match_path(eval_path)),
                    }
                )
                continue

            cache_key = f"{eval_path.resolve()}::{slot.get('slot_id')}"
            quality = cache.get(cache_key)
            if quality is None or args.refresh_cache:
                try:
                    quality = judge.judge(slot, actual_text)
                    cache[cache_key] = quality
                except JudgeAPICallError as exc:
                    judge_errors += 1
                    quality = {
                        "score": 0.0,
                        "hallucination": False,
                        "rationale": "",
                        "raw": "",
                        "parse_source": "api_error",
                        "error": str(exc),
                    }
                    if not args.continue_on_judge_error:
                        raise

            q = float(quality.get("score", 0.0) or 0.0)
            hallucination = to_bool(quality.get("hallucination", False))
            low_quality = q < float(args.quality_threshold)
            tp = 0.0 if low_quality else q
            fp = (1 if low_quality else 0) + (1 if sp > 0 else 0) + (1 if hallucination else 0)
            fn = 1 if low_quality else 0
            prf = precision_recall_f1(tp, fp, fn)

            rows.append(
                {
                    "model": model,
                    "sample_id": sample_id_from_path(eval_path),
                    "slot_id": slot.get("slot_id"),
                    "status": "ok",
                    "has_output": True,
                    "num_chunks": num_chunks,
                    "text_len": len(actual_text),
                    "response_start": response_start,
                    "response_end": response_end,
                    "slot_end": slot.get("end"),
                    "spill_seconds": sp,
                    "spill_positive": sp > 0,
                    "partial_quality": q,
                    "low_quality": low_quality,
                    "hallucination": hallucination,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    **prf,
                    "rationale": quality.get("rationale", ""),
                    "parse_source": quality.get("parse_source", ""),
                    "judge_error": quality.get("error", ""),
                    "metric_json": str(eval_path),
                    "match_json": str(sibling_match_path(eval_path)),
                }
            )

    output_rows = [r for r in rows if r.get("status") == "ok" and r.get("has_output")]
    spill_rows = [r for r in output_rows if float(r.get("spill_seconds") or 0.0) > 0]
    tp_sum = sum(float(r.get("TP") or 0.0) for r in output_rows)
    fp_sum = sum(float(r.get("FP") or 0.0) for r in output_rows)
    fn_sum = sum(float(r.get("FN") or 0.0) for r in output_rows)
    prf = precision_recall_f1(tp_sum, fp_sum, fn_sum)
    summary = {
        "model": model,
        "eval_dir": str(eval_dir),
        "total_interrupted_slots": total_interrupted,
        "interrupted_no_output_count": no_output,
        "interrupted_no_output_rate": safe_div(no_output, total_interrupted),
        "interrupted_with_output_count": len(output_rows),
        "judge_error_count": judge_errors,
        "quality_threshold": float(args.quality_threshold),
        "partial_quality_avg": safe_div(sum(float(r.get("partial_quality") or 0.0) for r in output_rows), len(output_rows)),
        "low_quality_count": sum(1 for r in output_rows if r.get("low_quality")),
        "hallucination_count": sum(1 for r in output_rows if r.get("hallucination")),
        "spill_positive_count": len(spill_rows),
        "spill_positive_rate_in_output_slots": safe_div(len(spill_rows), len(output_rows)),
        "total_spill_seconds_in_output_slots": sum(float(r.get("spill_seconds") or 0.0) for r in output_rows),
        "avg_spill_seconds_in_output_slots": safe_div(sum(float(r.get("spill_seconds") or 0.0) for r in output_rows), len(output_rows)),
        "TP": tp_sum,
        "FP": fp_sum,
        "FN": fn_sum,
        **prf,
    }
    return summary, rows


def build_markdown(summaries: List[Dict[str, Any]]) -> str:
    lines = [
        "# Interrupted Output Diagnostic",
        "",
        "This report does not modify Global IA-QTF1. It evaluates only interrupted slots where the model produced output.",
        "Quality is judged as partial usefulness: the spoken content can be incomplete, but it should be relevant, correct, and useful with respect to the current GT answer. Spill duration is reported separately.",
        "",
        "| model | interrupted | no-output rate | with output | avg quality | low quality | hallucination | spill rate | avg spill(s) | TP | FP | FN | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['model']} | {s['total_interrupted_slots']} | {s['interrupted_no_output_rate']:.4f} | "
            f"{s['interrupted_with_output_count']} | {s['partial_quality_avg']:.4f} | "
            f"{s['low_quality_count']} | {s['hallucination_count']} | "
            f"{s['spill_positive_rate_in_output_slots']:.4f} | {s['avg_spill_seconds_in_output_slots']:.4f} | "
            f"{s['TP']:.4f} | {s['FP']:.0f} | {s['FN']:.0f} | {s['F1']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate output-bearing interrupted slots in gpt4o_eval2 folders.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--outputs-dir", type=Path, default=Path("outputs"), help="Root containing outputs/<model>/<eval-subdir>.")
    src.add_argument("--eval-dir", type=Path, help="Single gpt4o_eval2 directory to evaluate.")
    parser.add_argument("--model-name", default="", help="Model name to use with --eval-dir.")
    parser.add_argument("--eval-subdir", default="gpt4o_eval2")
    parser.add_argument("--pattern", default="*.unified_eval.json")
    parser.add_argument("--out-dir", type=Path, default=Path("unified_eval_4models_report_gpt4o_eval2_summaries") / "interrupted_output_quality")
    parser.add_argument("--quality-threshold", type=float, default=0.5, help="Below this partial-quality score, count low-quality FP and FN.")
    parser.add_argument("--judge-api-url", default=os.getenv("JUDGE_API_URL", ""))
    parser.add_argument("--judge-api-key", default=os.getenv("JUDGE_API_KEY", ""))
    parser.add_argument("--judge-api-model", default=os.getenv("JUDGE_API_MODEL", "gpt-4o-2024-08-06"))
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-timeout-sec", type=float, default=60.0)
    parser.add_argument("--cache-json", type=Path, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--continue-on-judge-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.judge_api_url:
        raise SystemExit("--judge-api-url or JUDGE_API_URL is required.")
    api = APIJudge(
        api_url=args.judge_api_url,
        api_key=args.judge_api_key,
        model=args.judge_api_model,
        max_tokens=args.judge_max_tokens,
        temperature=args.judge_temperature,
        timeout_sec=args.judge_timeout_sec,
    )
    judge = PartialQualityJudge(api)
    cache_path = args.cache_json or (args.out_dir / "partial_quality_cache.json")
    cache = read_cache(cache_path)

    summaries: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for model, eval_dir in eval_dirs_from_args(args):
        summary, rows = evaluate_dir(model, eval_dir, judge, args, cache)
        summaries.append(summary)
        all_rows.extend(rows)
        write_cache(cache_path, cache)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.out_dir / "interrupted_output_quality_summary.json", {"summaries": summaries, "rows": all_rows})
    write_csv(
        args.out_dir / "interrupted_output_quality_by_model.csv",
        summaries,
        [
            "model",
            "eval_dir",
            "total_interrupted_slots",
            "interrupted_no_output_count",
            "interrupted_no_output_rate",
            "interrupted_with_output_count",
            "partial_quality_avg",
            "low_quality_count",
            "hallucination_count",
            "spill_positive_count",
            "spill_positive_rate_in_output_slots",
            "total_spill_seconds_in_output_slots",
            "avg_spill_seconds_in_output_slots",
            "TP",
            "FP",
            "FN",
            "Precision",
            "Recall",
            "F1",
            "judge_error_count",
        ],
    )
    write_csv(
        args.out_dir / "interrupted_output_quality_slots.csv",
        all_rows,
        [
            "model",
            "sample_id",
            "slot_id",
            "status",
            "has_output",
            "num_chunks",
            "text_len",
            "response_start",
            "response_end",
            "slot_end",
            "spill_seconds",
            "spill_positive",
            "partial_quality",
            "low_quality",
            "hallucination",
            "TP",
            "FP",
            "FN",
            "Precision",
            "Recall",
            "F1",
            "rationale",
            "parse_source",
            "judge_error",
            "metric_json",
            "match_json",
        ],
    )
    report = build_markdown(summaries)
    (args.out_dir / "interrupted_output_quality_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
