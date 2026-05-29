#!/usr/bin/env python3
"""
Compute F1-related metrics for 1Q1A mutually exclusive categories.

Mutually exclusive mapping:
- realtime_exclusive = by_question_type.realtime - nested_by_role.inner
- proactive_exclusive = by_question_type.proactive - nested_by_role.outer
- nested = by_scene_type.nested
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r)
    return p, r, f1


def metric_sub(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    return {
        "Global_TP": safe_float(a.get("Global_TP")) - safe_float(b.get("Global_TP")),
        "Global_FP": safe_float(a.get("Global_FP")) - safe_float(b.get("Global_FP")),
        "Global_FN": safe_float(a.get("Global_FN")) - safe_float(b.get("Global_FN")),
        "num_slots": safe_float(a.get("num_slots")) - safe_float(b.get("num_slots")),
    }


def get_nested(summary: Dict[str, Any]) -> Dict[str, Any]:
    return (summary.get("by_scene_type", {}) or {}).get("nested", {}) or {}


def get_qtype(summary: Dict[str, Any], qtype: str) -> Dict[str, Any]:
    return (summary.get("by_question_type", {}) or {}).get(qtype, {}) or {}


def get_nested_role(summary: Dict[str, Any], role: str) -> Dict[str, Any]:
    return (summary.get("nested_by_role", {}) or {}).get(role, {}) or {}


def compute_one(summary: Dict[str, Any]) -> Dict[str, Any]:
    rt = get_qtype(summary, "realtime")
    pro = get_qtype(summary, "proactive")
    nested = get_nested(summary)
    inner = get_nested_role(summary, "inner")
    outer = get_nested_role(summary, "outer")

    classes: Dict[str, Dict[str, float]] = {
        "realtime_exclusive": metric_sub(rt, inner),
        "proactive_exclusive": metric_sub(pro, outer),
        "nested": {
            "Global_TP": safe_float(nested.get("Global_TP")),
            "Global_FP": safe_float(nested.get("Global_FP")),
            "Global_FN": safe_float(nested.get("Global_FN")),
            "num_slots": safe_float(nested.get("num_slots")),
        },
    }

    for row in classes.values():
        p, r, f1 = prf(row["Global_TP"], row["Global_FP"], row["Global_FN"])
        row["Precision"] = p
        row["Recall"] = r
        row["F1"] = f1

    tp_sum = sum(v["Global_TP"] for v in classes.values())
    fp_sum = sum(v["Global_FP"] for v in classes.values())
    fn_sum = sum(v["Global_FN"] for v in classes.values())
    support_sum = sum(v["num_slots"] for v in classes.values())

    micro_p, micro_r, micro_f1 = prf(tp_sum, fp_sum, fn_sum)
    macro_f1 = safe_div(sum(v["F1"] for v in classes.values()), len(classes))
    weighted_f1 = safe_div(
        sum(v["F1"] * v["num_slots"] for v in classes.values()),
        support_sum,
    )

    return {
        "classes": classes,
        "aggregate": {
            "Global_TP": tp_sum,
            "Global_FP": fp_sum,
            "Global_FN": fn_sum,
            "num_slots": support_sum,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "IA_QTF1_1Q1A_Global": micro_f1,
        },
    }


def iter_summary_files(input_dir: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(input_dir.glob(pattern))


def model_name_from_path(path: Path) -> str:
    # aura.unified_eval_summary.json -> aura
    return path.name.split(".")[0]


def fmt(x: float, digits: int) -> str:
    return f"{x:.{digits}f}"


def print_markdown(results: Dict[str, Dict[str, Any]], digits: int) -> None:
    print("## 1Q1A mutually exclusive F1 metrics")
    print("")
    print("| model | micro_f1(=1Q1A_Global_IA_QTF1) | macro_f1 | weighted_f1 | micro_precision | micro_recall | slots |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for model, row in results.items():
        agg = row["aggregate"]
        print(
            f"| {model} | {fmt(agg['micro_f1'], digits)} | {fmt(agg['macro_f1'], digits)} | "
            f"{fmt(agg['weighted_f1'], digits)} | {fmt(agg['micro_precision'], digits)} | "
            f"{fmt(agg['micro_recall'], digits)} | {int(round(agg['num_slots']))} |"
        )

    print("")
    print("## Per-class breakdown")
    print("")
    print("| model | class | TP | FP | FN | Precision | Recall | F1 | slots |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    class_order = ["realtime_exclusive", "proactive_exclusive", "nested"]
    for model, row in results.items():
        for cls in class_order:
            c = row["classes"][cls]
            print(
                f"| {model} | {cls} | {fmt(c['Global_TP'], digits)} | {int(round(c['Global_FP']))} | "
                f"{int(round(c['Global_FN']))} | {fmt(c['Precision'], digits)} | {fmt(c['Recall'], digits)} | "
                f"{fmt(c['F1'], digits)} | {int(round(c['num_slots']))} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute 1Q1A mutually exclusive F1-related metrics from unified_eval summary JSON files."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("unified_eval_4models_report_gpt4o_eval2_summaries"),
        help="Directory containing *.unified_eval_summary.json files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.unified_eval_summary.json",
        help="Glob pattern for summary files.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path to save all computed metrics as JSON.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="Decimal digits for terminal markdown output.",
    )
    args = parser.parse_args()

    files = list(iter_summary_files(args.input_dir, args.pattern))
    if not files:
        raise SystemExit(f"No files matched: {args.input_dir / args.pattern}")

    results: Dict[str, Dict[str, Any]] = {}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data.get("summary", {}) or {}
        model = model_name_from_path(path)
        results[model] = compute_one(summary)

    print_markdown(results, args.digits)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
