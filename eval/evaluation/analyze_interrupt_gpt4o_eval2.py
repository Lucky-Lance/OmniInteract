#!/usr/bin/env python3
"""Analyze interrupted slots in outputs/*/gpt4o_eval2.

The script recomputes interrupted-only IA-QTF1 from slot-level
*.unified_eval.json files, then reads sibling *.unified_match.json files to
separate no-spill cases into:

  - no output before the interrupt boundary
  - output that ended near the interrupt boundary
  - output that finished before the interrupt boundary
  - short output that likely avoided spill because it was brief
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MODEL_ORDER = ("aura", "gemini", "minicpmo", "qwen")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def safe_div(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if denom > 0 else 0.0


def f1_from_pr(precision: float, recall: float) -> float:
    return 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "t"}


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def model_sort_key(path: Path) -> Tuple[int, str]:
    name = path.name
    try:
        return (MODEL_ORDER.index(name), name)
    except ValueError:
        return (len(MODEL_ORDER), name)


def eval_files_for_model(model_dir: Path, eval_subdir: str, pattern: str) -> List[Path]:
    eval_dir = model_dir / eval_subdir
    if not eval_dir.is_dir():
        return []
    return sorted(eval_dir.glob(pattern))


def sibling_match_path(eval_path: Path) -> Path:
    return eval_path.with_name(eval_path.name.replace(".unified_eval.json", ".unified_match.json"))


def load_match_slots(eval_path: Path) -> Dict[str, Dict[str, Any]]:
    path = sibling_match_path(eval_path)
    if not path.exists():
        return {}
    root = load_json(path)
    out: Dict[str, Dict[str, Any]] = {}
    for slot in root.get("slots", []) if isinstance(root, dict) else []:
        if isinstance(slot, dict):
            out[str(slot.get("slot_id"))] = slot
    return out


def chunk_text_len(chunks: Iterable[Dict[str, Any]]) -> int:
    return sum(len(str(chunk.get("text", "") or "").strip()) for chunk in chunks)


def chunk_time_bounds(chunks: Iterable[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    starts: List[float] = []
    ends: List[float] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if chunk.get("start") is not None:
            starts.append(safe_float(chunk.get("start")))
        if chunk.get("end") is not None:
            ends.append(safe_float(chunk.get("end")))
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def fallback_text_len(slot: Dict[str, Any]) -> int:
    stage_early = slot.get("stage_early", {}) if isinstance(slot.get("stage_early"), dict) else {}
    stage_core = slot.get("stage_core", {}) if isinstance(slot.get("stage_core"), dict) else {}
    return len(str(stage_early.get("text_early", "") or "")) + len(str(stage_core.get("text_core", "") or ""))


def slot_sample_id(path: Path) -> str:
    return path.name.replace(".unified_eval.json", "")


def classify_slot(
    slot: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    near_boundary_sec: float,
    short_duration_sec: float,
    short_text_chars: int,
) -> Dict[str, Any]:
    spill_seconds = safe_float((slot.get("spill_diagnostics", {}) or {}).get("spill_seconds"), 0.0)
    slot_end = safe_float(slot.get("end"), 0.0)
    num_chunks = len(chunks) if chunks else int(safe_float(slot.get("num_chunks"), 0.0))
    text_len = chunk_text_len(chunks) if chunks else fallback_text_len(slot)
    response_start, response_end = chunk_time_bounds(chunks)

    if response_end is None:
        response_end = (slot.get("spill_diagnostics", {}) or {}).get("max_chunk_end")
        response_end = None if response_end is None else safe_float(response_end)

    response_duration = None
    last_output_gap = None
    if response_start is not None and response_end is not None:
        response_duration = max(0.0, response_end - response_start)
    if response_end is not None:
        last_output_gap = slot_end - response_end

    is_spill = spill_seconds > 0
    has_output = num_chunks > 0 or text_len > 0
    is_short = bool(has_output and (text_len <= short_text_chars or (response_duration is not None and response_duration <= short_duration_sec)))
    is_near_interrupt = bool(has_output and last_output_gap is not None and last_output_gap <= near_boundary_sec)
    finished_before_interrupt = bool(has_output and not is_spill and last_output_gap is not None and last_output_gap > near_boundary_sec)

    if is_spill:
        label = "overflow"
    elif not has_output:
        label = "no_output"
    elif is_near_interrupt and is_short:
        label = "short_near_interrupt"
    elif is_near_interrupt:
        label = "stopped_near_interrupt"
    elif is_short:
        label = "short_finished_before_interrupt"
    else:
        label = "finished_before_interrupt"

    return {
        "spill_seconds": spill_seconds,
        "num_chunks": num_chunks,
        "text_len": text_len,
        "response_start": response_start,
        "response_end": response_end,
        "response_duration": response_duration,
        "last_output_gap": last_output_gap,
        "has_output": has_output,
        "is_spill": is_spill,
        "is_short": is_short,
        "is_near_interrupt": is_near_interrupt,
        "finished_before_interrupt": finished_before_interrupt,
        "label": label,
    }


def metric_row(tp: float, fp: float, fn: float) -> Dict[str, Any]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "Global_TP": tp,
        "Global_FP": int(round(fp)),
        "Global_FN": int(round(fn)),
        "Precision": precision,
        "Recall": recall,
        "IA_QTF1": f1_from_pr(precision, recall),
    }


def summarize_values(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"avg": None, "median": None, "p90": None, "max": None}
    return {
        "avg": statistics.mean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.9),
        "max": max(values),
    }


def analyze_model(
    model_dir: Path,
    eval_subdir: str,
    pattern: str,
    near_boundary_sec: float,
    short_duration_sec: float,
    short_text_chars: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    files = eval_files_for_model(model_dir, eval_subdir, pattern)
    slot_rows: List[Dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    tp = fp = fn = 0.0
    load_errors: List[Dict[str, str]] = []

    for eval_path in files:
        try:
            root = load_json(eval_path)
            match_slots = load_match_slots(eval_path)
        except Exception as exc:
            load_errors.append({"path": str(eval_path), "error": str(exc)})
            continue

        for slot in root.get("slots", []) if isinstance(root, dict) else []:
            if not isinstance(slot, dict) or not to_bool(slot.get("is_interrupted", False)):
                continue
            chunks = list((match_slots.get(str(slot.get("slot_id"))) or {}).get("all_chunks", []) or [])
            cls = classify_slot(slot, chunks, near_boundary_sec, short_duration_sec, short_text_chars)
            label_counts[cls["label"]] += 1
            tp += safe_float(slot.get("TP_n"), 0.0)
            fp += safe_float(slot.get("FP_delta"), 0.0)
            fn += safe_float(slot.get("FN_delta"), 0.0)

            slot_rows.append(
                {
                    "model": model_dir.name,
                    "sample_id": slot_sample_id(eval_path),
                    "slot_id": slot.get("slot_id"),
                    "scene_type": slot.get("scene_type", ""),
                    "question_type": slot.get("question_type", ""),
                    "start": safe_float(slot.get("start"), 0.0),
                    "t_a": safe_float(slot.get("t_a"), 0.0),
                    "end": safe_float(slot.get("end"), 0.0),
                    "TP_n": safe_float(slot.get("TP_n"), 0.0),
                    "FP_delta": safe_float(slot.get("FP_delta"), 0.0),
                    "FN_delta": safe_float(slot.get("FN_delta"), 0.0),
                    **cls,
                    "metric_json": str(eval_path),
                    "match_json": str(sibling_match_path(eval_path)),
                }
            )

    n = len(slot_rows)
    no_spill_rows = [r for r in slot_rows if not r["is_spill"]]
    no_spill_output_rows = [r for r in no_spill_rows if r["has_output"]]
    spill_rows = [r for r in slot_rows if r["is_spill"]]

    no_spill_count = len(no_spill_rows)
    no_spill_output_count = len(no_spill_output_rows)
    no_output_count = sum(1 for r in no_spill_rows if not r["has_output"])
    no_spill_near_count = sum(1 for r in no_spill_output_rows if r["is_near_interrupt"])
    no_spill_finished_count = sum(1 for r in no_spill_output_rows if r["finished_before_interrupt"])
    no_spill_short_count = sum(1 for r in no_spill_output_rows if r["is_short"])
    short_finished_count = sum(1 for r in no_spill_output_rows if r["is_short"] and r["finished_before_interrupt"])
    substantial_near_count = sum(1 for r in no_spill_output_rows if r["is_near_interrupt"] and not r["is_short"])

    summary: Dict[str, Any] = {
        "model": model_dir.name,
        "eval_files": len(files),
        "load_error_count": len(load_errors),
        "interrupted_slot_count": n,
        "thresholds": {
            "near_boundary_sec": near_boundary_sec,
            "short_duration_sec": short_duration_sec,
            "short_text_chars": short_text_chars,
        },
        "metrics": metric_row(tp, fp, fn),
        "spill": {
            "positive_count": len(spill_rows),
            "positive_rate": safe_div(len(spill_rows), n),
            "total_seconds": sum(r["spill_seconds"] for r in spill_rows),
            **summarize_values([r["spill_seconds"] for r in spill_rows]),
        },
        "no_spill": {
            "count": no_spill_count,
            "rate": safe_div(no_spill_count, n),
            "no_output_count": no_output_count,
            "no_output_rate_in_no_spill": safe_div(no_output_count, no_spill_count),
            "with_output_count": no_spill_output_count,
            "with_output_rate_in_no_spill": safe_div(no_spill_output_count, no_spill_count),
            "near_interrupt_count": no_spill_near_count,
            "near_interrupt_rate_in_with_output": safe_div(no_spill_near_count, no_spill_output_count),
            "finished_before_interrupt_count": no_spill_finished_count,
            "finished_before_interrupt_rate_in_with_output": safe_div(no_spill_finished_count, no_spill_output_count),
            "short_answer_count": no_spill_short_count,
            "short_answer_rate_in_with_output": safe_div(no_spill_short_count, no_spill_output_count),
            "short_finished_before_interrupt_count": short_finished_count,
            "substantial_near_interrupt_count": substantial_near_count,
            "last_output_gap_seconds": summarize_values(
                [r["last_output_gap"] for r in no_spill_output_rows if r["last_output_gap"] is not None]
            ),
            "response_duration_seconds": summarize_values(
                [r["response_duration"] for r in no_spill_output_rows if r["response_duration"] is not None]
            ),
            "text_len_chars": summarize_values([float(r["text_len"]) for r in no_spill_output_rows]),
        },
        "labels": dict(sorted(label_counts.items())),
        "load_errors": load_errors,
    }
    return summary, slot_rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_model_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics = summary["metrics"]
    spill = summary["spill"]
    no_spill = summary["no_spill"]
    return {
        "model": summary["model"],
        "eval_files": summary["eval_files"],
        "interrupted_slot_count": summary["interrupted_slot_count"],
        "Global_TP": metrics["Global_TP"],
        "Global_FP": metrics["Global_FP"],
        "Global_FN": metrics["Global_FN"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "IA_QTF1": metrics["IA_QTF1"],
        "spill_positive_count": spill["positive_count"],
        "spill_positive_rate": spill["positive_rate"],
        "spill_total_seconds": spill["total_seconds"],
        "spill_avg_seconds": spill["avg"],
        "spill_p90_seconds": spill["p90"],
        "spill_max_seconds": spill["max"],
        "no_spill_count": no_spill["count"],
        "no_spill_no_output_count": no_spill["no_output_count"],
        "no_spill_with_output_count": no_spill["with_output_count"],
        "no_spill_near_interrupt_count": no_spill["near_interrupt_count"],
        "no_spill_finished_before_interrupt_count": no_spill["finished_before_interrupt_count"],
        "no_spill_short_answer_count": no_spill["short_answer_count"],
        "no_spill_short_finished_before_interrupt_count": no_spill["short_finished_before_interrupt_count"],
        "no_spill_substantial_near_interrupt_count": no_spill["substantial_near_interrupt_count"],
        "no_spill_gap_avg_seconds": no_spill["last_output_gap_seconds"]["avg"],
        "no_spill_gap_median_seconds": no_spill["last_output_gap_seconds"]["median"],
        "no_spill_duration_avg_seconds": no_spill["response_duration_seconds"]["avg"],
        "no_spill_text_len_avg": no_spill["text_len_chars"]["avg"],
    }


def build_markdown_report(summaries: List[Dict[str, Any]]) -> str:
    if summaries:
        thresholds = summaries[0]["thresholds"]
    else:
        thresholds = {"near_boundary_sec": 0.5, "short_duration_sec": 2.0, "short_text_chars": 20}

    lines: List[str] = []
    lines.append("# gpt4o_eval2 interrupted-slot analysis")
    lines.append("")
    lines.append(
        "Scope: only `is_interrupted=true` slots from `outputs/<model>/gpt4o_eval2/*.unified_eval.json`; "
        "`FN` follows the unified definition (0 for interrupted slots)."
    )
    lines.append(
        f"No-spill split thresholds: last output within <= {fmt(thresholds['near_boundary_sec'], 2)}s of the interrupt boundary counts as near interrupt; "
        f"text <= {int(thresholds['short_text_chars'])} chars or duration <= {fmt(thresholds['short_duration_sec'], 2)}s counts as a short answer."
    )
    lines.append("")
    lines.append("## Model summary")
    lines.append("")
    lines.append(
        "| model | interrupted | IA_QTF1 | TP | FP | spill>0 | spill rate | avg spill(s) | "
        "no spill | no output | no-spill with output | near interrupt | short answer | finished before interrupt |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        m = s["metrics"]
        spill = s["spill"]
        ns = s["no_spill"]
        lines.append(
            f"| {s['model']} | {s['interrupted_slot_count']} | {fmt(m['IA_QTF1'])} | "
            f"{fmt(m['Global_TP'])} | {m['Global_FP']} | {spill['positive_count']} | "
            f"{fmt(spill['positive_rate'])} | {fmt(spill['avg'])} | {ns['count']} | "
            f"{ns['no_output_count']} | {ns['with_output_count']} | {ns['near_interrupt_count']} | "
            f"{ns['short_answer_count']} | {ns['finished_before_interrupt_count']} |"
        )
    lines.append("")
    lines.append("## No-spill breakdown")
    lines.append("")
    lines.append(
        "| model | no-spill output | near interrupt rate | short answer rate | short+finished count | substantial near count | "
        "median gap(s) | avg text len | avg duration(s) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        ns = s["no_spill"]
        lines.append(
            f"| {s['model']} | {ns['with_output_count']} | {fmt(ns['near_interrupt_rate_in_with_output'])} | "
            f"{fmt(ns['short_answer_rate_in_with_output'])} | {ns['short_finished_before_interrupt_count']} | "
            f"{ns['substantial_near_interrupt_count']} | {fmt(ns['last_output_gap_seconds']['median'])} | "
            f"{fmt(ns['text_len_chars']['avg'])} | {fmt(ns['response_duration_seconds']['avg'])} |"
        )
    lines.append("")
    lines.append("## How to read the table")
    lines.append("")
    lines.append("- `no output`: the model did not produce anything inside the interrupted-slot window, so the absence of spill does not prove the model actively stopped.")
    lines.append("- `short answer`: very brief output; in particular `short+finished` is more likely a naturally short answer than a clean stop after the interrupt.")
    lines.append("- `near interrupt`: the last output is close to the interrupt boundary; `substantial near` further excludes short answers and is the cleanest evidence of \"actually stopped on interrupt\".")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze interrupted gpt4o_eval2 slots under outputs/*.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--eval-subdir", default="gpt4o_eval2")
    parser.add_argument("--pattern", default="*.unified_eval.json")
    parser.add_argument("--out-dir", type=Path, default=Path("unified_eval_4models_report_gpt4o_eval2_summaries"))
    parser.add_argument("--near-boundary-sec", type=float, default=0.5)
    parser.add_argument("--short-duration-sec", type=float, default=2.0)
    parser.add_argument("--short-text-chars", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dirs = sorted([p for p in args.outputs_dir.iterdir() if p.is_dir()], key=model_sort_key)
    summaries: List[Dict[str, Any]] = []
    all_slot_rows: List[Dict[str, Any]] = []

    for model_dir in model_dirs:
        if not (model_dir / args.eval_subdir).is_dir():
            continue
        summary, slot_rows = analyze_model(
            model_dir=model_dir,
            eval_subdir=args.eval_subdir,
            pattern=args.pattern,
            near_boundary_sec=args.near_boundary_sec,
            short_duration_sec=args.short_duration_sec,
            short_text_chars=args.short_text_chars,
        )
        summaries.append(summary)
        all_slot_rows.extend(slot_rows)

    out_json = {
        "source": str(args.outputs_dir / "*" / args.eval_subdir),
        "summaries": summaries,
        "slot_count": len(all_slot_rows),
    }
    save_json(args.out_dir / "interrupt_gpt4o_eval2_summary.json", out_json)

    model_rows = [flatten_model_summary(s) for s in summaries]
    write_csv(
        args.out_dir / "interrupt_gpt4o_eval2_by_model.csv",
        model_rows,
        list(model_rows[0].keys()) if model_rows else [],
    )

    slot_fields = [
        "model",
        "sample_id",
        "slot_id",
        "scene_type",
        "question_type",
        "start",
        "t_a",
        "end",
        "TP_n",
        "FP_delta",
        "FN_delta",
        "spill_seconds",
        "num_chunks",
        "text_len",
        "response_start",
        "response_end",
        "response_duration",
        "last_output_gap",
        "has_output",
        "is_spill",
        "is_short",
        "is_near_interrupt",
        "finished_before_interrupt",
        "label",
        "metric_json",
        "match_json",
    ]
    write_csv(args.out_dir / "interrupt_gpt4o_eval2_slots.csv", all_slot_rows, slot_fields)

    report = build_markdown_report(summaries)
    report_path = args.out_dir / "interrupt_gpt4o_eval2_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote: {report_path}")


if __name__ == "__main__":
    main()
