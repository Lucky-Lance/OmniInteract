#!/usr/bin/env python3
"""Summarize Unified-IA-QTF1 from existing *.unified_eval.json details.

This is the canonical aggregate-metric entry point. It does not call the LLM
judge. All metrics are recomputed from slot-level details in unified_eval JSONs.

FN policy:
  - interrupted slots: FN = 0
  - non-interrupted slots: FN = 1 when Score_core <= 0
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
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
        return float(value)
    except Exception:
        return float(default)


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "t"}


def safe_div(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if denom > 0 else 0.0


def f1_from_pr(precision: float, recall: float) -> float:
    return 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)


def metric_row(tp: float, fp: float, fn: float, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    out: Dict[str, Any] = {
        "Global_TP": float(tp),
        "Global_FP": int(fp),
        "Global_FN": int(fn),
        "Precision": precision,
        "Recall": recall,
        "IA_QTF1": f1_from_pr(precision, recall),
    }
    if extra:
        out = {**extra, **out}
    return out


def metric_sub(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    tp = safe_float(a.get("Global_TP")) - safe_float(b.get("Global_TP"))
    fp = safe_float(a.get("Global_FP")) - safe_float(b.get("Global_FP"))
    fn = safe_float(a.get("Global_FN")) - safe_float(b.get("Global_FN"))
    num_slots = safe_float(a.get("num_slots")) - safe_float(b.get("num_slots"))
    return metric_row(tp, max(0.0, fp), max(0.0, fn), {"num_slots": int(max(0.0, num_slots))})


def build_paper_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    by_scene = summary.get("by_scene_type", {}) or {}
    by_qtype = summary.get("by_question_type", {}) or {}
    nested_by_role = summary.get("nested_by_role", {}) or {}
    interrupted = summary.get("interrupted", {}) or {}
    nested_chain = summary.get("nested_chain_completion", {}) or summary.get("nested_ira", {}) or {}

    realtime_exclusive = metric_sub(by_qtype.get("realtime", {}) or {}, nested_by_role.get("inner", {}) or {})
    proactive_exclusive = metric_sub(by_qtype.get("proactive", {}) or {}, nested_by_role.get("outer", {}) or {})
    nested = by_scene.get("nested", {}) or {}
    one_qna = by_scene.get("1QnA", {}) or {}

    one_q1a_tp = (
        safe_float(realtime_exclusive.get("Global_TP"))
        + safe_float(proactive_exclusive.get("Global_TP"))
        + safe_float(nested.get("Global_TP"))
    )
    one_q1a_fp = (
        safe_float(realtime_exclusive.get("Global_FP"))
        + safe_float(proactive_exclusive.get("Global_FP"))
        + safe_float(nested.get("Global_FP"))
    )
    one_q1a_fn = (
        safe_float(realtime_exclusive.get("Global_FN"))
        + safe_float(proactive_exclusive.get("Global_FN"))
        + safe_float(nested.get("Global_FN"))
    )
    one_q1a_slots = (
        int(realtime_exclusive.get("num_slots", 0) or 0)
        + int(proactive_exclusive.get("num_slots", 0) or 0)
        + int(nested.get("num_slots", 0) or 0)
    )

    return {
        "exp_f1": {
            "realtime": realtime_exclusive,
            "proactive": proactive_exclusive,
            "nested": nested,
            "one_q1a_global": metric_row(one_q1a_tp, one_q1a_fp, one_q1a_fn, {"num_slots": one_q1a_slots}),
            "one_qna": one_qna,
            "all_global": {
                "num_slots": summary.get("num_slots", 0),
                "Global_TP": summary.get("Global_TP", 0.0),
                "Global_FP": summary.get("Global_FP", 0),
                "Global_FN": summary.get("Global_FN", 0),
                "Precision": summary.get("Precision", 0.0),
                "Recall": summary.get("Recall", 0.0),
                "IA_QTF1": summary.get("IA_QTF1", 0.0),
            },
            "definition": "Matches paper/tabs/exp_f1.tex: realtime/proactive exclude nested inner/outer slots; 1Q1A Global is recomputed from realtime, proactive, and nested TP/FP/FN.",
        },
        "exp_interruption": {
            "NOR": interrupted.get("interrupted_no_output_rate", 0.0),
            "PAQ": interrupted.get("partial_quality_avg", 0.0),
            "CSM_SR": interrupted.get("spill_positive_rate_in_output_slots", interrupted.get("spill_positive_rate", 0.0)),
            "CSM_AS_seconds": interrupted.get("avg_spill_seconds_in_output_slots", interrupted.get("avg_spill_seconds", 0.0)),
            "interrupted_slot_count": interrupted.get("interrupted_slot_count", 0),
            "interrupted_with_output_count": interrupted.get("interrupted_with_output_count", 0),
            "definition": "Matches paper/tabs/exp_interruption.tex: NOR, PAQ, conditional spill rate, and conditional average spill over interrupted slots with output.",
        },
        "exp_nested": {
            "NCCS": nested_chain.get("NCCS", nested_chain.get("Nested_IRA", 0.0)),
            "inner_IA_QTF1": (nested_by_role.get("inner", {}) or {}).get("IA_QTF1", 0.0),
            "outer_IA_QTF1": (nested_by_role.get("outer", {}) or {}).get("IA_QTF1", 0.0),
            "missed_outer": nested_chain.get("missing_q1_count", 0),
            "num_pairs": nested_chain.get("num_pairs", 0),
            "definition": "Matches paper/tabs/exp_nested.tex: NCCS plus inner/outer IA-QTF1 and missed outer count.",
        },
    }


def new_metric_acc() -> Dict[str, float]:
    return {"num_slots": 0.0, "Global_TP": 0.0, "Global_FP": 0.0, "Global_FN": 0.0}


def add_metric(acc: Dict[str, float], tp: float, fp: float, fn: float, num_slots: float = 1.0) -> None:
    acc["num_slots"] += float(num_slots)
    acc["Global_TP"] += float(tp)
    acc["Global_FP"] += float(fp)
    acc["Global_FN"] += float(fn)


def add_group(groups: Dict[str, Dict[str, float]], key: str, tp: float, fp: float, fn: float) -> None:
    add_metric(groups.setdefault(str(key or "unknown"), new_metric_acc()), tp, fp, fn)


def summarize_metric_acc(acc: Dict[str, float], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row_extra = {"num_slots": int(acc.get("num_slots", 0.0))}
    if extra:
        row_extra.update(extra)
    return metric_row(acc.get("Global_TP", 0.0), acc.get("Global_FP", 0.0), acc.get("Global_FN", 0.0), row_extra)


def summarize_groups(groups: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    return {k: summarize_metric_acc(v) for k, v in sorted(groups.items())}


def new_score_acc() -> Dict[str, float]:
    return {"count": 0.0, "quality_sum": 0.0, "timeliness_sum": 0.0}


def add_score(acc: Dict[str, float], quality: float, timeliness: float) -> None:
    acc["count"] += 1.0
    acc["quality_sum"] += float(quality)
    acc["timeliness_sum"] += float(timeliness)


def summarize_score_acc(acc: Dict[str, float]) -> Dict[str, Any]:
    count = int(acc.get("count", 0.0))
    return {
        "num_slots": count,
        "pure_quality_avg": safe_div(acc.get("quality_sum", 0.0), count),
        "pure_timeliness_avg": safe_div(acc.get("timeliness_sum", 0.0), count),
    }


def slot_has_output(slot: Dict[str, Any]) -> bool:
    if int(safe_float(slot.get("num_chunks"), 0)) > 0:
        return True
    if int(safe_float(slot.get("num_early"), 0)) > 0 or int(safe_float(slot.get("num_core"), 0)) > 0:
        return True
    early = slot.get("stage_early", {}) if isinstance(slot.get("stage_early", {}), dict) else {}
    core = slot.get("stage_core", {}) if isinstance(slot.get("stage_core", {}), dict) else {}
    return bool(str(early.get("text_early", "") or "").strip() or str(core.get("text_core", "") or "").strip())


def sample_id_from_path(path: Path) -> str:
    return path.name.replace(".unified_eval.json", "")


def is_math_sample(item_info: Optional[Dict[str, Any]], metric_path: Path) -> bool:
    """A sample belongs to the math subset iff its source paths live under data/1q1a_math/."""
    candidates = [str(metric_path)]
    for key in ("gt_json", "model_json", "video", "match_json"):
        value = (item_info or {}).get(key, "")
        if value:
            candidates.append(str(value))
    return any("/1q1a_math/" in c or "\\1q1a_math\\" in c for c in candidates)


def infer_nested_role(slot: Dict[str, Any]) -> Optional[str]:
    if str(slot.get("scene_type", "") or "") != "nested":
        return None
    role = str(slot.get("nested_role", "") or "").strip().lower()
    if role in {"inner", "outer"}:
        return role
    qtype = str(slot.get("question_type", "") or "").strip().lower()
    if qtype == "proactive":
        return "outer"
    if qtype == "realtime":
        return "inner"
    try:
        slot_id = int(slot.get("slot_id"))
        return "outer" if slot_id % 2 == 1 else "inner"
    except Exception:
        return "unknown"


def core_start_time(slot: Dict[str, Any]) -> Optional[float]:
    stage_core = slot.get("stage_core", {}) if isinstance(slot.get("stage_core", {}), dict) else {}
    for key in ("T_start_true_effective", "T_start", "T_start_fallback", "T_end"):
        value = stage_core.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def nested_pair_key(slot: Dict[str, Any]) -> str:
    step = slot.get("step_index")
    if step not in (None, ""):
        return f"step_{step}"
    try:
        slot_id = int(slot.get("slot_id"))
        return f"slot_pair_{(slot_id + 1) // 2}"
    except Exception:
        return f"time_pair_{safe_float(slot.get('start'), 0.0):.3f}"


def slot_order_key(slot: Dict[str, Any]) -> Tuple[float, int]:
    try:
        slot_id = int(slot.get("slot_id"))
    except Exception:
        slot_id = 10**9
    return (safe_float(slot.get("start"), 0.0), slot_id)


def summarize_nested_ira_from_slots(slots: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        role = infer_nested_role(slot)
        if role not in {"outer", "inner"}:
            continue
        key = nested_pair_key(slot)
        groups.setdefault(key, {}).setdefault(role, []).append(slot)

    num_pairs = 0
    success_pairs = 0
    missing_pair_count = 0
    missing_q1_count = 0
    missing_q2_count = 0
    order_error_count = 0
    ira_sum = 0.0

    for group in groups.values():
        outers = sorted(group.get("outer", []), key=slot_order_key)
        inners = sorted(group.get("inner", []), key=slot_order_key)
        if not outers and not inners:
            continue
        num_pairs += 1
        if not outers or not inners:
            missing_pair_count += 1
            continue

        outer = outers[0]
        inner = inners[0]
        q1 = max(0.0, safe_float(outer.get("Score_core"), 0.0))
        q2 = max(0.0, safe_float(inner.get("Score_core"), 0.0))
        q1_ok = q1 > 0
        q2_ok = q2 > 0
        if not q1_ok:
            missing_q1_count += 1
        if not q2_ok:
            missing_q2_count += 1

        q1_time = core_start_time(outer)
        q2_time = core_start_time(inner)
        order_ok = True
        if q1_ok and q2_ok and q1_time is not None and q2_time is not None:
            order_ok = q2_time <= q1_time
        if q1_ok and q2_ok and not order_ok:
            order_error_count += 1

        if q1_ok and q2_ok and order_ok:
            ira = math.sqrt(q1 * q2)
            ira_sum += ira
            success_pairs += 1

    nccs = safe_div(ira_sum, num_pairs)
    return {
        "metric_name": "NCCS",
        "full_name": "Nested Chain Completion Score",
        "num_pairs": num_pairs,
        "success_pairs": success_pairs,
        "success_rate": safe_div(success_pairs, num_pairs),
        "NCCS": nccs,
        "Nested_Chain_Completion_Score": nccs,
        "Nested_IRA": nccs,
        "score_sum": ira_sum,
        "ira_sum": ira_sum,
        "missing_pair_count": missing_pair_count,
        "missing_q1_count": missing_q1_count,
        "missing_q2_count": missing_q2_count,
        "order_error_count": order_error_count,
        "definition": "mean(sqrt(Score_core_Q1 * Score_core_Q2)) when Q2 and Q1 both have Score_core > 0 and Q2 starts no later than Q1; otherwise 0",
    }


def collect_metric_files(eval_dir: Optional[Path], summary_json: Optional[Path], pattern: str) -> Tuple[List[Path], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    files: List[Path] = []
    item_by_metric: Dict[str, Dict[str, Any]] = {}
    passthrough_items: List[Dict[str, Any]] = []

    if summary_json is not None and summary_json.exists():
        root = load_json(summary_json)
        for item in root.get("items", []) if isinstance(root, dict) else []:
            if not isinstance(item, dict):
                continue
            if str(item.get("status", "")) != "ok":
                passthrough_items.append(item)
                continue
            metric = item.get("metric_json")
            if metric:
                p = Path(str(metric)).resolve()
                files.append(p)
                item_by_metric[str(p)] = item

    if eval_dir is not None:
        files.extend(sorted(eval_dir.resolve().glob(pattern)))

    uniq: List[Path] = []
    seen = set()
    for path in files:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(path.resolve())
    return uniq, item_by_metric, passthrough_items


def summarize_one_file(path: Path, previous_item: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    root = load_json(path)
    slots = list(root.get("slots", []) or []) if isinstance(root, dict) else []
    unmatched = list(root.get("unmatched_chunks", []) or []) if isinstance(root, dict) else []
    config = root.get("config", {}) if isinstance(root, dict) else {}
    count_unmatched_as_fp = bool(config.get("count_unmatched_as_fp", True))
    unmatched_fp = len(unmatched) if count_unmatched_as_fp and str(root.get("status", "")) == "ok" else 0

    global_acc = new_metric_acc()
    by_scene: Dict[str, Dict[str, float]] = {}
    by_qtype: Dict[str, Dict[str, float]] = {}
    score_all = new_score_acc()
    score_non_interrupted = new_score_acc()
    score_by_scene: Dict[str, Dict[str, float]] = {}
    score_by_qtype: Dict[str, Dict[str, float]] = {}
    math_acc = new_metric_acc()
    math_score = new_score_acc()
    nested_by_role: Dict[str, Dict[str, float]] = {}
    nested_score_by_role: Dict[str, Dict[str, float]] = {}
    interrupted_acc = new_metric_acc()
    interrupted_score = new_score_acc()

    interrupted_count = 0
    interrupted_no_output_count = 0
    interrupted_with_output_count = 0
    interrupted_spill_positive = 0
    interrupted_spill_seconds = 0.0
    interrupted_partial_quality_sum = 0.0
    interrupted_partial_quality_count = 0
    interrupted_low_quality_count = 0
    interrupted_hallucination_count = 0
    interrupted_diag_acc = new_metric_acc()
    fn_slot_ids: List[Any] = []
    sample_id = str((previous_item or {}).get("sample_id") or sample_id_from_path(path))
    sample_is_math = is_math_sample(previous_item, path)

    for slot in slots:
        if not isinstance(slot, dict):
            continue
        tp = safe_float(slot.get("TP_n"), 0.0)
        fp = safe_float(slot.get("FP_delta"), 0.0)
        score_core = safe_float(slot.get("Score_core"), 0.0)
        interrupted = to_bool(slot.get("is_interrupted", False))
        fn = 0 if interrupted else (1 if score_core <= 0 else 0)
        if fn:
            fn_slot_ids.append(slot.get("slot_id"))

        scene = str(slot.get("scene_type", "unknown") or "unknown")
        qtype = str(slot.get("question_type", "unknown") or "unknown")
        stage_core = slot.get("stage_core", {}) if isinstance(slot.get("stage_core", {}), dict) else {}
        s_core = safe_float(stage_core.get("S_core"), 0.0)
        t_core = safe_float(stage_core.get("T_core"), 0.0)

        add_metric(global_acc, tp, fp, fn)
        add_group(by_scene, scene, tp, fp, fn)
        add_group(by_qtype, qtype, tp, fp, fn)
        add_score(score_all, s_core, t_core)
        add_score(score_by_scene.setdefault(scene, new_score_acc()), s_core, t_core)
        add_score(score_by_qtype.setdefault(qtype, new_score_acc()), s_core, t_core)
        nested_role = infer_nested_role(slot)
        if nested_role:
            add_metric(nested_by_role.setdefault(nested_role, new_metric_acc()), tp, fp, fn)
            add_score(nested_score_by_role.setdefault(nested_role, new_score_acc()), s_core, t_core)
        if not interrupted:
            add_score(score_non_interrupted, s_core, t_core)
        else:
            interrupted_count += 1
            add_metric(interrupted_acc, tp, fp, fn)
            add_score(interrupted_score, s_core, t_core)
            spill = safe_float((slot.get("spill_diagnostics", {}) or {}).get("spill_seconds"), 0.0)
            if slot_has_output(slot):
                interrupted_with_output_count += 1
                interrupted_spill_seconds += spill
                if spill > 0:
                    interrupted_spill_positive += 1
                diag = slot.get("interruption_diagnostic", {}) if isinstance(slot.get("interruption_diagnostic", {}), dict) else {}
                if diag.get("status") == "ok":
                    interrupted_partial_quality_count += 1
                    interrupted_partial_quality_sum += safe_float(diag.get("partial_quality"), 0.0)
                    if to_bool(diag.get("low_quality", False)):
                        interrupted_low_quality_count += 1
                    if to_bool(diag.get("hallucination", False)):
                        interrupted_hallucination_count += 1
                    add_metric(
                        interrupted_diag_acc,
                        safe_float(diag.get("TP"), 0.0),
                        safe_float(diag.get("FP"), 0.0),
                        safe_float(diag.get("FN"), 0.0),
                    )
            else:
                interrupted_no_output_count += 1

        if sample_is_math:
            add_metric(math_acc, tp, fp, fn)
            add_score(math_score, s_core, t_core)

    global_acc["Global_FP"] += unmatched_fp
    summary = summarize_metric_acc(
        global_acc,
        {
            "num_processed_slots": len(slots),
            "num_unmatched_chunks": len(unmatched),
            "fn_policy": "strict_core_fn_non_interrupted_score_core_le_0",
        },
    )
    summary["by_scene_type"] = summarize_groups(by_scene)
    summary["by_question_type"] = summarize_groups(by_qtype)
    summary["nested_by_role"] = summarize_groups(nested_by_role)

    pure_scores = {
        "overall": summarize_score_acc(score_all),
        "non_interrupted": summarize_score_acc(score_non_interrupted),
        "by_scene_type": {k: summarize_score_acc(v) for k, v in sorted(score_by_scene.items())},
        "by_question_type": {k: summarize_score_acc(v) for k, v in sorted(score_by_qtype.items())},
        "nested_by_role": {k: summarize_score_acc(v) for k, v in sorted(nested_score_by_role.items())},
        "interrupted": summarize_score_acc(interrupted_score),
    }
    interrupted_summary = {
        "interrupted_slot_count": interrupted_count,
        "interrupted_no_output_count": interrupted_no_output_count,
        "interrupted_no_output_rate": safe_div(interrupted_no_output_count, interrupted_count),
        "interrupted_with_output_count": interrupted_with_output_count,
        "partial_quality_count": interrupted_partial_quality_count,
        "partial_quality_avg": safe_div(interrupted_partial_quality_sum, interrupted_partial_quality_count),
        "low_quality_count": interrupted_low_quality_count,
        "hallucination_count": interrupted_hallucination_count,
        "interrupted_with_positive_spill_count": interrupted_spill_positive,
        "spill_positive_rate": safe_div(interrupted_spill_positive, interrupted_with_output_count),
        "spill_positive_rate_in_output_slots": safe_div(interrupted_spill_positive, interrupted_with_output_count),
        "total_spill_seconds": interrupted_spill_seconds,
        "total_spill_seconds_in_output_slots": interrupted_spill_seconds,
        "avg_spill_seconds": safe_div(interrupted_spill_seconds, interrupted_with_output_count),
        "avg_spill_seconds_in_output_slots": safe_div(interrupted_spill_seconds, interrupted_with_output_count),
        "metrics": summarize_metric_acc(interrupted_acc),
        "diagnostic_metrics": summarize_metric_acc(interrupted_diag_acc),
    }

    out = {
        "status": str(root.get("status", "ok")) if isinstance(root, dict) else "invalid",
        "sample_id": sample_id,
        "metric_json": str(path),
        "summary": summary,
        "pure_scores": pure_scores,
        "interrupted_summary": interrupted_summary,
        "fn_slot_ids": fn_slot_ids,
        "math_summary": summarize_metric_acc(math_acc) if math_acc["num_slots"] else None,
        "math_pure_scores": summarize_score_acc(math_score) if math_score["count"] else None,
        "nested_ira": summarize_nested_ira_from_slots(slots),
    }
    for key in ("video", "scene_type", "gt_json", "model_json", "match_json", "item_index"):
        if previous_item and key in previous_item:
            out[key] = previous_item[key]
    if "scene_type" not in out:
        out["scene_type"] = (root.get("meta", {}) or {}).get("scene_type", "")
    return out


def aggregate_items(items: List[Dict[str, Any]], passthrough_items: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    global_acc = new_metric_acc()
    by_scene: Dict[str, Dict[str, float]] = {}
    by_qtype: Dict[str, Dict[str, float]] = {}
    score_all = new_score_acc()
    score_non_interrupted = new_score_acc()
    score_by_scene: Dict[str, Dict[str, float]] = {}
    score_by_qtype: Dict[str, Dict[str, float]] = {}
    math_acc = new_metric_acc()
    math_score = new_score_acc()
    nested_by_role: Dict[str, Dict[str, float]] = {}
    nested_score_by_role: Dict[str, Dict[str, float]] = {}
    interrupted_acc = new_metric_acc()
    interrupted_score = new_score_acc()
    nested_ira_acc: Dict[str, float] = {
        "num_pairs": 0.0,
        "success_pairs": 0.0,
        "ira_sum": 0.0,
        "missing_pair_count": 0.0,
        "missing_q1_count": 0.0,
        "missing_q2_count": 0.0,
        "order_error_count": 0.0,
    }

    interrupted_count = 0
    interrupted_no_output = 0
    interrupted_with_output = 0
    interrupted_positive = 0
    interrupted_spill = 0.0
    interrupted_partial_quality_sum = 0.0
    interrupted_partial_quality_count = 0
    interrupted_low_quality_count = 0
    interrupted_hallucination_count = 0
    interrupted_diag_acc = new_metric_acc()
    unmatched = 0

    for item in items:
        s = item.get("summary", {}) or {}
        add_metric(global_acc, s.get("Global_TP", 0), s.get("Global_FP", 0), s.get("Global_FN", 0), s.get("num_slots", 0))
        unmatched += int(safe_float(s.get("num_unmatched_chunks"), 0))
        for key, row in (s.get("by_scene_type", {}) or {}).items():
            add_metric(by_scene.setdefault(key, new_metric_acc()), row.get("Global_TP", 0), row.get("Global_FP", 0), row.get("Global_FN", 0), row.get("num_slots", 0))
        for key, row in (s.get("by_question_type", {}) or {}).items():
            add_metric(by_qtype.setdefault(key, new_metric_acc()), row.get("Global_TP", 0), row.get("Global_FP", 0), row.get("Global_FN", 0), row.get("num_slots", 0))
        for key, row in (s.get("nested_by_role", {}) or {}).items():
            add_metric(nested_by_role.setdefault(key, new_metric_acc()), row.get("Global_TP", 0), row.get("Global_FP", 0), row.get("Global_FN", 0), row.get("num_slots", 0))

        ps = item.get("pure_scores", {}) or {}
        for src, dst in ((ps.get("overall", {}), score_all), (ps.get("non_interrupted", {}), score_non_interrupted)):
            count = safe_float(src.get("num_slots"), 0)
            dst["count"] += count
            dst["quality_sum"] += safe_float(src.get("pure_quality_avg"), 0) * count
            dst["timeliness_sum"] += safe_float(src.get("pure_timeliness_avg"), 0) * count
        for key, row in (ps.get("by_scene_type", {}) or {}).items():
            count = safe_float(row.get("num_slots"), 0)
            acc = score_by_scene.setdefault(key, new_score_acc())
            acc["count"] += count
            acc["quality_sum"] += safe_float(row.get("pure_quality_avg"), 0) * count
            acc["timeliness_sum"] += safe_float(row.get("pure_timeliness_avg"), 0) * count
        for key, row in (ps.get("by_question_type", {}) or {}).items():
            count = safe_float(row.get("num_slots"), 0)
            acc = score_by_qtype.setdefault(key, new_score_acc())
            acc["count"] += count
            acc["quality_sum"] += safe_float(row.get("pure_quality_avg"), 0) * count
            acc["timeliness_sum"] += safe_float(row.get("pure_timeliness_avg"), 0) * count
        for key, row in (ps.get("nested_by_role", {}) or {}).items():
            count = safe_float(row.get("num_slots"), 0)
            acc = nested_score_by_role.setdefault(key, new_score_acc())
            acc["count"] += count
            acc["quality_sum"] += safe_float(row.get("pure_quality_avg"), 0) * count
            acc["timeliness_sum"] += safe_float(row.get("pure_timeliness_avg"), 0) * count

        intr = item.get("interrupted_summary", {}) or {}
        interrupted_count += int(safe_float(intr.get("interrupted_slot_count"), 0))
        interrupted_no_output += int(safe_float(intr.get("interrupted_no_output_count"), 0))
        interrupted_with_output += int(safe_float(intr.get("interrupted_with_output_count"), 0))
        pq_count = int(safe_float(intr.get("partial_quality_count", intr.get("interrupted_with_output_count", 0)), 0))
        interrupted_partial_quality_count += pq_count
        interrupted_partial_quality_sum += safe_float(intr.get("partial_quality_avg"), 0) * pq_count
        interrupted_low_quality_count += int(safe_float(intr.get("low_quality_count"), 0))
        interrupted_hallucination_count += int(safe_float(intr.get("hallucination_count"), 0))
        interrupted_positive += int(safe_float(intr.get("interrupted_with_positive_spill_count"), 0))
        interrupted_spill += safe_float(intr.get("total_spill_seconds_in_output_slots", intr.get("total_spill_seconds", 0)), 0)
        intr_metrics = intr.get("metrics", {}) or {}
        add_metric(
            interrupted_acc,
            intr_metrics.get("Global_TP", 0),
            intr_metrics.get("Global_FP", 0),
            intr_metrics.get("Global_FN", 0),
            intr_metrics.get("num_slots", 0),
        )
        intr_score = (ps.get("interrupted", {}) or {})
        intr_count = safe_float(intr_score.get("num_slots"), 0)
        interrupted_score["count"] += intr_count
        interrupted_score["quality_sum"] += safe_float(intr_score.get("pure_quality_avg"), 0) * intr_count
        interrupted_score["timeliness_sum"] += safe_float(intr_score.get("pure_timeliness_avg"), 0) * intr_count
        intr_diag = intr.get("diagnostic_metrics", {}) or {}
        add_metric(
            interrupted_diag_acc,
            intr_diag.get("Global_TP", intr.get("TP", 0)),
            intr_diag.get("Global_FP", intr.get("FP", 0)),
            intr_diag.get("Global_FN", intr.get("FN", 0)),
            intr_diag.get("num_slots", pq_count),
        )

        if item.get("math_summary"):
            ms = item["math_summary"]
            add_metric(math_acc, ms.get("Global_TP", 0), ms.get("Global_FP", 0), ms.get("Global_FN", 0), ms.get("num_slots", 0))
        if item.get("math_pure_scores"):
            mps = item["math_pure_scores"]
            count = safe_float(mps.get("num_slots"), 0)
            math_score["count"] += count
            math_score["quality_sum"] += safe_float(mps.get("pure_quality_avg"), 0) * count
            math_score["timeliness_sum"] += safe_float(mps.get("pure_timeliness_avg"), 0) * count

        ni = item.get("nested_ira", {}) or {}
        nested_ira_acc["num_pairs"] += safe_float(ni.get("num_pairs"), 0)
        nested_ira_acc["success_pairs"] += safe_float(ni.get("success_pairs"), 0)
        nested_ira_acc["ira_sum"] += safe_float(ni.get("score_sum", ni.get("ira_sum", 0)), 0)
        nested_ira_acc["missing_pair_count"] += safe_float(ni.get("missing_pair_count"), 0)
        nested_ira_acc["missing_q1_count"] += safe_float(ni.get("missing_q1_count"), 0)
        nested_ira_acc["missing_q2_count"] += safe_float(ni.get("missing_q2_count"), 0)
        nested_ira_acc["order_error_count"] += safe_float(ni.get("order_error_count"), 0)

    summary = summarize_metric_acc(
        global_acc,
        {
            "num_items": len(items) + len(passthrough_items or []),
            "success": len(items),
            "failed_or_skipped": len(passthrough_items or []),
            "num_unmatched_chunks": unmatched,
            "fn_policy": "strict_core_fn_non_interrupted_score_core_le_0",
        },
    )
    summary["by_scene_type"] = summarize_groups(by_scene)
    summary["by_question_type"] = summarize_groups(by_qtype)
    summary["nested_by_role"] = summarize_groups(nested_by_role)
    summary["pure_scores"] = {
        "overall": summarize_score_acc(score_all),
        "non_interrupted": summarize_score_acc(score_non_interrupted),
        "by_scene_type": {k: summarize_score_acc(v) for k, v in sorted(score_by_scene.items())},
        "by_question_type": {k: summarize_score_acc(v) for k, v in sorted(score_by_qtype.items())},
        "nested_by_role": {k: summarize_score_acc(v) for k, v in sorted(nested_score_by_role.items())},
        "interrupted": summarize_score_acc(interrupted_score),
    }
    summary["interrupted"] = {
        "interrupted_slot_count": interrupted_count,
        "interrupted_no_output_count": interrupted_no_output,
        "interrupted_no_output_rate": safe_div(interrupted_no_output, interrupted_count),
        "interrupted_with_output_count": interrupted_with_output,
        "partial_quality_count": interrupted_partial_quality_count,
        "partial_quality_avg": safe_div(interrupted_partial_quality_sum, interrupted_partial_quality_count),
        "low_quality_count": interrupted_low_quality_count,
        "hallucination_count": interrupted_hallucination_count,
        "interrupted_with_positive_spill_count": interrupted_positive,
        "spill_positive_rate": safe_div(interrupted_positive, interrupted_with_output),
        "spill_positive_rate_in_output_slots": safe_div(interrupted_positive, interrupted_with_output),
        "total_spill_seconds": interrupted_spill,
        "total_spill_seconds_in_output_slots": interrupted_spill,
        "avg_spill_seconds": safe_div(interrupted_spill, interrupted_with_output),
        "avg_spill_seconds_in_output_slots": safe_div(interrupted_spill, interrupted_with_output),
        "metrics": summarize_metric_acc(interrupted_acc),
        "diagnostic_metrics": summarize_metric_acc(interrupted_diag_acc),
    }
    summary["math"] = summarize_metric_acc(math_acc) if math_acc["num_slots"] else {"num_slots": 0}
    summary["math_pure_scores"] = {
        "overall": summarize_score_acc(math_score),
    }
    num_nested_pairs = int(nested_ira_acc["num_pairs"])
    success_nested_pairs = int(nested_ira_acc["success_pairs"])
    nccs = safe_div(nested_ira_acc["ira_sum"], num_nested_pairs)
    nested_chain_completion = {
        "metric_name": "NCCS",
        "full_name": "Nested Chain Completion Score",
        "num_pairs": num_nested_pairs,
        "success_pairs": success_nested_pairs,
        "success_rate": safe_div(success_nested_pairs, num_nested_pairs),
        "NCCS": nccs,
        "Nested_Chain_Completion_Score": nccs,
        "Nested_IRA": nccs,
        "score_sum": nested_ira_acc["ira_sum"],
        "ira_sum": nested_ira_acc["ira_sum"],
        "missing_pair_count": int(nested_ira_acc["missing_pair_count"]),
        "missing_q1_count": int(nested_ira_acc["missing_q1_count"]),
        "missing_q2_count": int(nested_ira_acc["missing_q2_count"]),
        "order_error_count": int(nested_ira_acc["order_error_count"]),
        "definition": "mean(sqrt(Score_core_Q1 * Score_core_Q2)) when Q2 and Q1 both have Score_core > 0 and Q2 starts no later than Q1; otherwise 0",
    }
    summary["nested_chain_completion"] = nested_chain_completion
    summary["nested_ira"] = nested_chain_completion
    summary["paper_metrics"] = build_paper_metrics(summary)
    return summary


def build_summary(eval_dir: Path, summary_json: Optional[Path] = None, pattern: str = "*.unified_eval.json") -> Dict[str, Any]:
    files, previous_by_metric, passthrough = collect_metric_files(eval_dir, summary_json, pattern)
    items: List[Dict[str, Any]] = []
    missing_files: List[str] = []
    for path in files:
        if not path.exists():
            missing_files.append(str(path))
            continue
        previous = previous_by_metric.get(str(path.resolve()))
        items.append(summarize_one_file(path, previous))
    items.sort(
        key=lambda x: (
            0,
            int(x.get("item_index", 10**9)),
        )
        if str(x.get("item_index", "")).isdigit()
        else (1, str(x.get("sample_id", "")))
    )
    return {
        "meta": {
            "source": "slot_details_from_unified_eval_json",
            "fn_policy": "strict_core_fn_non_interrupted_score_core_le_0",
            "eval_dir": str(eval_dir.resolve()),
            "summary_json_input": str(summary_json.resolve()) if summary_json else None,
            "num_metric_files_loaded": len(items),
            "missing_metric_files": missing_files,
        },
        "items": items + passthrough,
        "summary": aggregate_items(items, passthrough),
    }


def fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "0.0000"


def get_qtype(summary: Dict[str, Any], qtype: str) -> Dict[str, Any]:
    return (summary.get("by_question_type", {}) or {}).get(qtype, {})


def metric_table_row(model: str, metric: Dict[str, Any], score: Dict[str, Any]) -> str:
    return (
        f"| {model} | {fmt(metric.get('Global_TP'))} | {int(metric.get('Global_FP', 0))} | "
        f"{int(metric.get('Global_FN', 0))} | {fmt(metric.get('Precision'))} | "
        f"{fmt(metric.get('Recall'))} | {fmt(metric.get('IA_QTF1'))} | "
        f"{fmt(score.get('pure_quality_avg'))} | {fmt(score.get('pure_timeliness_avg'))} | "
        f"{int(metric.get('num_slots', score.get('num_slots', 0)) or 0)} |"
    )


def best_model_by_metric(model_summaries: Dict[str, Dict[str, Any]], metric_getter, metric_name: str = "IA_QTF1") -> str:
    return max(MODEL_ORDER, key=lambda m: safe_float(metric_getter(model_summaries[m]["summary"]).get(metric_name), 0.0))


def nested_chain_score(summary: Dict[str, Any]) -> Dict[str, Any]:
    return (
        summary.get("nested_chain_completion", {})
        or summary.get("nested_ira", {})
        or {}
    )


def append_type_analysis(lines: List[str], label: str, model_summaries: Dict[str, Dict[str, Any]], metric_getter, score_getter) -> None:
    best_f1 = best_model_by_metric(model_summaries, metric_getter)
    best_quality = best_model_by_metric(model_summaries, lambda s: score_getter(s), "pure_quality_avg")
    best_timeliness = best_model_by_metric(model_summaries, lambda s: score_getter(s), "pure_timeliness_avg")
    lines.append(
        f"Notes: in `{label}`, the highest IA_QTF1 is `{best_f1}` "
        f"({fmt(metric_getter(model_summaries[best_f1]['summary']).get('IA_QTF1'))}); "
        f"best pure quality is `{best_quality}`, best pure timeliness is `{best_timeliness}`."
    )
    lines.append("")


def append_metric_section(lines: List[str], title: str, model_summaries: Dict[str, Dict[str, Any]], metric_getter, score_getter, label: Optional[str] = None) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| model | Global_TP | Global_FP | Global_FN | Precision | Recall | IA_QTF1 | pure_quality | pure_timeliness | num_slots |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODEL_ORDER:
        lines.append(metric_table_row(model, metric_getter(model_summaries[model]["summary"]), score_getter(model_summaries[model]["summary"])))
    lines.append("")
    append_type_analysis(lines, label or title, model_summaries, metric_getter, score_getter)


def build_report(model_summaries: Dict[str, Dict[str, Any]], offline_math_path: Optional[Path] = None) -> str:
    lines: List[str] = []
    lines.append("# Unified evaluation report (4 systems)")
    lines.append("")
    lines.append("- Source: slot-level details in `outputs/{aura,gemini,minicpmo,qwen}/unified_eval/*.unified_eval.json`.")
    lines.append("- Aggregator: `eval/evaluation/summarize_unified_eval.py`.")
    lines.append("- Sections: realtime (incl. math), proactive, 1QnA, nested, interrupted.")
    lines.append("")
    lines.append("## Headline result: global IA-QTF1")
    lines.append("")
    lines.append("IA-QTF1 is the headline metric, combining answer reward (TP), error cost (FP), and missed-answer cost (FN).")
    lines.append("")
    lines.append("| model | Global_TP | Global_FP | Global_FN | Precision | Recall | IA_QTF1 | num_slots |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODEL_ORDER:
        s = model_summaries[model]["summary"]
        lines.append(f"| {model} | {fmt(s['Global_TP'])} | {int(s['Global_FP'])} | {int(s['Global_FN'])} | {fmt(s['Precision'])} | {fmt(s['Recall'])} | {fmt(s['IA_QTF1'])} | {int(s.get('num_slots', 0))} |")
    lines.append("")
    best_global = best_model_by_metric(model_summaries, lambda s: s)
    lines.append(f"Overall, `{best_global}` reaches the highest global IA_QTF1 ({fmt(model_summaries[best_global]['summary']['IA_QTF1'])}).")
    lines.append("")
    lines.append("## Metric design")
    lines.append("")
    lines.append("This evaluation measures two core capabilities of a real-time voice assistant: whether the model responds at the right time, and whether the response is actually useful. Each question is converted to a unified slot `slot=[start, t_a, end)`: `start` is when the user's question begins, `t_a` is when a core answer becomes permissible, and `end` closes the answering window for that question. All subsequent metrics are computed over this time window.")
    lines.append("")
    lines.append("### 1. TP: reward useful output, but separate acknowledgement from the core answer")
    lines.append("")
    lines.append("- `Score_ack`: interaction acknowledgement between `start` and `t_a` (confirm, wait, brief feedback). It reflects real-time interactivity but does not substitute for answering.")
    lines.append("- `Score_core`: content quality and timeliness of the core answer after `t_a`. Content quality is judged by an LLM judge; timeliness is computed from the semantic anchor relative to `t_a/end`.")
    lines.append("- Per-slot soft TP: `TP_n = Score_ack + Score_core`. This rewards good real-time interaction while keeping most of the credit on actually answering.")
    lines.append("")
    lines.append("### 2. FP: penalize undesirable output")
    lines.append("")
    lines.append("- `Early Hallucination`: substantive guessing or fabricated answers before `t_a`.")
    lines.append("- `Low Quality / Spoilers`: core quality below threshold, or leaking future steps in 1QnA.")
    lines.append("- `Spillover`: for hard-boundary questions, response continues into the next slot.")
    lines.append("- `Unmatched`: chunks not landing in any slot window.")
    lines.append("")
    lines.append("### 3. FN: only a useful core answer counts as completed")
    lines.append("")
    lines.append("- Non-interrupted slot with `Score_core <= 0` -> `FN=1`. Pure acknowledgements (\"ok\", \"let me check\") do not count as answering.")
    lines.append("- Interrupted slots do not contribute FN, because the user cut the model off; spill is still tracked separately.")
    lines.append("")
    lines.append("### 4. Aggregates: reward, error cost, and miss in one number")
    lines.append("")
    lines.append("- `Global_TP = sum(TP_n)`: total soft credit across slots.")
    lines.append("- `Global_FP = sum(per-slot FP)`: early guesses, low-quality, spill, unmatched outputs.")
    lines.append("- `Global_FN = sum(per-slot FN)`: questions without a useful core answer.")
    lines.append("- `Precision = TP/(TP+FP)`: higher when fewer mistakes per output.")
    lines.append("- `Recall = TP/(TP+FN)`: higher when more questions receive a useful core answer.")
    lines.append("- `IA_QTF1`: F1 of precision and recall.")
    lines.append("- `pure_quality`: content quality of core answers only (`S_core`), ignoring timing.")
    lines.append("- `pure_timeliness`: timing of core answers only (`T_core`), ignoring content.")
    lines.append("")
    lines.append("### 5. Per-type slices: different splits stress different capabilities")
    lines.append("")
    lines.append("- `realtime`: respond when explicitly asked; the math split is also realtime and gets an additional online/offline quality comparison for MiniCPM-o.")
    lines.append("- `proactive`: spot the moment and speak up without rushing.")
    lines.append("- `1QnA`: one long task broken into steps; each step must arrive on time without spoiling later steps.")
    lines.append("- `nested`: a second question is inserted inside the answer window of the first. We report inner/outer slots separately and add `NCCS` (Nested Chain Completion Score) for whether the model answers Q2 first and then recovers Q1.")
    lines.append("- `interrupted`: after a user interrupt, the model is not required to finish; we track whether it keeps talking (spillover).")
    lines.append("")

    append_metric_section(
        lines,
        "1. realtime (incl. math)",
        model_summaries,
        lambda s: get_qtype(s, "realtime"),
        lambda s: ((s.get("pure_scores", {}) or {}).get("by_question_type", {}) or {}).get("realtime", {}),
        "realtime",
    )
    lines.append("### Math split")
    lines.append("")
    lines.append("| model | Global_TP | Global_FP | Global_FN | Precision | Recall | IA_QTF1 | pure_quality | pure_timeliness | num_slots |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODEL_ORDER:
        s = model_summaries[model]["summary"]
        metric = s.get("math", {}) or {}
        score = ((s.get("math_pure_scores", {}) or {}).get("overall", {}) or {})
        lines.append(metric_table_row(model, metric, score))
    lines.append("")
    append_type_analysis(
        lines,
        "math split",
        model_summaries,
        lambda s: s.get("math", {}) or {},
        lambda s: ((s.get("math_pure_scores", {}) or {}).get("overall", {}) or {}),
    )
    lines.append("### MiniCPM-o math: offline vs. online quality")
    lines.append("")
    mini_math = (model_summaries.get("minicpmo", {}).get("summary", {}).get("math_pure_scores", {}) or {}).get("overall", {}) or {}
    offline = load_json(offline_math_path) if offline_math_path and offline_math_path.exists() else {}
    offline_avg = safe_float(offline.get("avg_quality_score"), 0.0)
    online_avg = safe_float(mini_math.get("pure_quality_avg"), 0.0)
    lines.append("| metric | online (unified_eval) | offline (all_results) | offline - online |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| avg_quality_score | {fmt(online_avg)} | {fmt(offline_avg)} | {fmt(offline_avg - online_avg)} |")
    lines.append(f"| scored_items | {int(mini_math.get('num_slots', 0))} | {int(offline.get('scored_items', 0) or 0)} | - |")
    lines.append(f"| total_ok_items | {int(mini_math.get('num_slots', 0))} | {int(offline.get('total_ok_items', 0) or 0)} | - |")
    lines.append("")
    lines.append(f"Notes: MiniCPM-o math is {fmt(offline_avg - online_avg)} better offline than online, indicating that offline text reasoning does not fully transfer to the real-time speech setting.")
    lines.append("")

    append_metric_section(
        lines,
        "2. proactive",
        model_summaries,
        lambda s: get_qtype(s, "proactive"),
        lambda s: ((s.get("pure_scores", {}) or {}).get("by_question_type", {}) or {}).get("proactive", {}),
        "proactive",
    )

    append_metric_section(
        lines,
        "3. 1QnA",
        model_summaries,
        lambda s: ((s.get("by_scene_type", {}) or {}).get("1QnA", {}) or {}),
        lambda s: ((s.get("pure_scores", {}) or {}).get("by_scene_type", {}) or {}).get("1QnA", {}),
        "1QnA",
    )

    lines.append("## 4. nested")
    lines.append("")
    lines.append("`NCCS` (Nested Chain Completion Score) scores each nested pair as a unit: only when the model answers the inner Q2 inside its window and then resumes and answers the outer Q1, the pair earns `sqrt(Score_core_Q1 * Score_core_Q2)`; any missing answer or wrong ordering scores the pair 0.")
    lines.append("")
    lines.append("### 4.0 Whole pair: NCCS")
    lines.append("")
    lines.append("| model | NCCS | success_pairs | num_pairs | success_rate | missing_Q1 | missing_Q2 | order_error |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODEL_ORDER:
        ni = nested_chain_score(model_summaries[model]["summary"])
        lines.append(
            f"| {model} | {fmt(ni.get('NCCS', ni.get('Nested_IRA')))} | {int(ni.get('success_pairs', 0))} | "
            f"{int(ni.get('num_pairs', 0))} | {fmt(ni.get('success_rate'))} | "
            f"{int(ni.get('missing_q1_count', 0))} | {int(ni.get('missing_q2_count', 0))} | "
            f"{int(ni.get('order_error_count', 0))} |"
        )
    lines.append("")
    best_nccs = max(MODEL_ORDER, key=lambda m: safe_float(nested_chain_score(model_summaries[m]["summary"]).get("NCCS", nested_chain_score(model_summaries[m]["summary"]).get("Nested_IRA")), 0.0))
    lines.append(f"Notes: highest NCCS is `{best_nccs}` ({fmt(nested_chain_score(model_summaries[best_nccs]['summary']).get('NCCS', nested_chain_score(model_summaries[best_nccs]['summary']).get('Nested_IRA')))}), i.e. it best handles the \"answer Q2 first, then resume Q1\" chain.")
    lines.append("")
    lines.append("### 4.1 Outer slots")
    lines.append("")
    lines.append("| model | Global_TP | Global_FP | Global_FN | Precision | Recall | IA_QTF1 | pure_quality | pure_timeliness | num_slots |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODEL_ORDER:
        s = model_summaries[model]["summary"]
        metric = ((s.get("nested_by_role", {}) or {}).get("outer", {}) or {})
        score = (((s.get("pure_scores", {}) or {}).get("nested_by_role", {}) or {}).get("outer", {}) or {})
        lines.append(metric_table_row(model, metric, score))
    lines.append("")
    append_type_analysis(
        lines,
        "nested outer",
        model_summaries,
        lambda s: ((s.get("nested_by_role", {}) or {}).get("outer", {}) or {}),
        lambda s: (((s.get("pure_scores", {}) or {}).get("nested_by_role", {}) or {}).get("outer", {}) or {}),
    )
    lines.append("### 4.2 Inner slots")
    lines.append("")
    lines.append("| model | Global_TP | Global_FP | Global_FN | Precision | Recall | IA_QTF1 | pure_quality | pure_timeliness | num_slots |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODEL_ORDER:
        s = model_summaries[model]["summary"]
        metric = ((s.get("nested_by_role", {}) or {}).get("inner", {}) or {})
        score = (((s.get("pure_scores", {}) or {}).get("nested_by_role", {}) or {}).get("inner", {}) or {})
        lines.append(metric_table_row(model, metric, score))
    lines.append("")
    append_type_analysis(
        lines,
        "nested inner",
        model_summaries,
        lambda s: ((s.get("nested_by_role", {}) or {}).get("inner", {}) or {}),
        lambda s: (((s.get("pure_scores", {}) or {}).get("nested_by_role", {}) or {}).get("inner", {}) or {}),
    )

    lines.append("## 5. interrupted")
    lines.append("")
    lines.append("Interrupted slots still report IA-QTF1: FN is fixed at 0 (the user cut the model off, so it is not expected to finish); FP captures \"keeps talking after interrupt\" and other unwanted output.")
    lines.append("")
    lines.append("| model | Global_TP | Global_FP | Global_FN | Precision | Recall | IA_QTF1 | pure_quality | pure_timeliness | num_slots |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODEL_ORDER:
        s = model_summaries[model]["summary"]
        metric = ((s.get("interrupted", {}) or {}).get("metrics", {}) or {})
        score = ((s.get("pure_scores", {}) or {}).get("interrupted", {}) or {})
        lines.append(metric_table_row(model, metric, score))
    lines.append("")
    best_interrupt_f1 = best_model_by_metric(model_summaries, lambda s: ((s.get("interrupted", {}) or {}).get("metrics", {}) or {}))
    lines.append(
        f"Notes: highest interrupted IA_QTF1 is `{best_interrupt_f1}` "
        f"({fmt(((model_summaries[best_interrupt_f1]['summary'].get('interrupted', {}) or {}).get('metrics', {}) or {}).get('IA_QTF1'))})."
        " For this split the goal is to stop quickly after the interrupt, so also read the spill rate and spill seconds below."
    )
    lines.append("")
    lines.append("### Interrupt spill control")
    lines.append("")
    lines.append("| model | interrupted_slot_count | no-output rate | with output | PAQ | spill>0 (output slots) | CSM-SR | total_spill_seconds(output) | CSM-AS(s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODEL_ORDER:
        intr = model_summaries[model]["summary"].get("interrupted", {}) or {}
        lines.append(
            f"| {model} | {int(intr.get('interrupted_slot_count', 0))} | "
            f"{fmt(intr.get('interrupted_no_output_rate'))} | "
            f"{int(intr.get('interrupted_with_output_count', 0))} | "
            f"{fmt(intr.get('partial_quality_avg'))} | "
            f"{int(intr.get('interrupted_with_positive_spill_count', 0))} | "
            f"{fmt(intr.get('spill_positive_rate_in_output_slots', intr.get('spill_positive_rate')))} | "
            f"{fmt(intr.get('total_spill_seconds_in_output_slots', intr.get('total_spill_seconds')))} | "
            f"{fmt(intr.get('avg_spill_seconds_in_output_slots', intr.get('avg_spill_seconds')))} |"
        )
    lines.append("")
    best_spill = min(
        MODEL_ORDER,
        key=lambda m: (
            (model_summaries[m]["summary"].get("interrupted", {}) or {}).get("avg_spill_seconds_in_output_slots", 0.0)
            if (model_summaries[m]["summary"].get("interrupted", {}) or {}).get("interrupted_with_output_count", 0)
            else math.inf
        ),
    )
    lines.append(f"Notes: `{best_spill}` has the smallest conditional avg spill ({fmt((model_summaries[best_spill]['summary'].get('interrupted', {}) or {}).get('avg_spill_seconds_in_output_slots', (model_summaries[best_spill]['summary'].get('interrupted', {}) or {}).get('avg_spill_seconds')))}); this value is computed only over interrupted slots that produced output.")
    lines.append("")

    lines.append("## 6. Summary")
    lines.append("")
    best_realtime = max(MODEL_ORDER, key=lambda m: get_qtype(model_summaries[m]["summary"], "realtime").get("IA_QTF1", 0.0))
    best_math = max(MODEL_ORDER, key=lambda m: ((model_summaries[m]["summary"].get("math", {}) or {}).get("IA_QTF1", 0.0)))
    best_proactive = max(MODEL_ORDER, key=lambda m: get_qtype(model_summaries[m]["summary"], "proactive").get("IA_QTF1", 0.0))
    best_1qna = max(MODEL_ORDER, key=lambda m: ((model_summaries[m]["summary"].get("by_scene_type", {}) or {}).get("1QnA", {}) or {}).get("IA_QTF1", 0.0))
    best_nested_outer = max(MODEL_ORDER, key=lambda m: ((model_summaries[m]["summary"].get("nested_by_role", {}) or {}).get("outer", {}) or {}).get("IA_QTF1", 0.0))
    best_nested_inner = max(MODEL_ORDER, key=lambda m: ((model_summaries[m]["summary"].get("nested_by_role", {}) or {}).get("inner", {}) or {}).get("IA_QTF1", 0.0))
    best_nccs = max(MODEL_ORDER, key=lambda m: safe_float(nested_chain_score(model_summaries[m]["summary"]).get("NCCS", nested_chain_score(model_summaries[m]["summary"]).get("Nested_IRA")), 0.0))
    lines.append(f"- Best global IA_QTF1: `{best_global}` ({fmt(model_summaries[best_global]['summary']['IA_QTF1'])}).")
    lines.append(f"- Best realtime IA_QTF1: `{best_realtime}` ({fmt(get_qtype(model_summaries[best_realtime]['summary'], 'realtime').get('IA_QTF1'))}).")
    best_math_value = ((model_summaries[best_math]["summary"].get("math", {}) or {}).get("IA_QTF1"))
    lines.append(f"- Best math IA_QTF1: `{best_math}` ({fmt(best_math_value)}).")
    lines.append(f"- Best proactive IA_QTF1: `{best_proactive}` ({fmt(get_qtype(model_summaries[best_proactive]['summary'], 'proactive').get('IA_QTF1'))}).")
    lines.append(f"- Best 1QnA IA_QTF1: `{best_1qna}` ({fmt(((model_summaries[best_1qna]['summary'].get('by_scene_type', {}) or {}).get('1QnA', {}) or {}).get('IA_QTF1'))}).")
    best_nccs_value = nested_chain_score(model_summaries[best_nccs]["summary"]).get("NCCS", nested_chain_score(model_summaries[best_nccs]["summary"]).get("Nested_IRA"))
    lines.append(f"- Best nested NCCS: `{best_nccs}` ({fmt(best_nccs_value)}).")
    lines.append(f"- Best nested outer IA_QTF1: `{best_nested_outer}` ({fmt(((model_summaries[best_nested_outer]['summary'].get('nested_by_role', {}) or {}).get('outer', {}) or {}).get('IA_QTF1'))}).")
    lines.append(f"- Best nested inner IA_QTF1: `{best_nested_inner}` ({fmt(((model_summaries[best_nested_inner]['summary'].get('nested_by_role', {}) or {}).get('inner', {}) or {}).get('IA_QTF1'))}).")
    lines.append(f"- Best interrupt spill control (lowest CSM-AS, over interrupted slots with output): `{best_spill}` ({fmt((model_summaries[best_spill]['summary'].get('interrupted', {}) or {}).get('avg_spill_seconds_in_output_slots', (model_summaries[best_spill]['summary'].get('interrupted', {}) or {}).get('avg_spill_seconds')))}).")
    lines.append(f"- `minicpmo` offline math quality exceeds online by {fmt(offline_avg - online_avg)}.")
    lines.append("")
    return "\n".join(lines)


def parse_model_eval_dir(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        p = Path(value)
        return p.parent.name, p
    name, path = value.split("=", 1)
    return name.strip(), Path(path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize existing unified_eval JSON details with strict core-based FN.")
    parser.add_argument("--eval_dir", type=Path, default=None, help="Directory with *.unified_eval.json for one model.")
    parser.add_argument("--summary_json", type=Path, default=None, help="Optional existing summary to preserve item metadata.")
    parser.add_argument("--out_json", type=Path, default=None, help="Output summary JSON. Default: <eval_dir>/unified_eval_summary.json")
    parser.add_argument("--pattern", default="*.unified_eval.json")
    parser.add_argument("--model_eval_dir", action="append", default=[], help="For report/update mode: name=path/to/unified_eval.")
    parser.add_argument("--report_md", type=Path, default=None)
    parser.add_argument("--offline_math_json", type=Path, default=None)
    parser.add_argument("--copy_parent_summary", action="store_true", help="Also copy <eval_dir>/unified_eval_summary.json to <model_root>/unified_eval_summary.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_summaries: Dict[str, Dict[str, Any]] = {}

    if args.eval_dir is not None:
        out_json = args.out_json.resolve() if args.out_json else args.eval_dir.resolve() / "unified_eval_summary.json"
        summary = build_summary(args.eval_dir.resolve(), args.summary_json.resolve() if args.summary_json else None, args.pattern)
        save_json(out_json, summary)
        if args.copy_parent_summary:
            parent_summary = args.eval_dir.resolve().parent / "unified_eval_summary.json"
            shutil.copyfile(out_json, parent_summary)
        print(f"Saved summary to: {out_json}")
        s = summary["summary"]
        print(f"TP={s['Global_TP']:.6f} FP={s['Global_FP']} FN={s['Global_FN']} P={s['Precision']:.6f} R={s['Recall']:.6f} F1={s['IA_QTF1']:.6f}")

    for spec in args.model_eval_dir:
        name, eval_dir = parse_model_eval_dir(spec)
        summary = build_summary(eval_dir, eval_dir / "unified_eval_summary.json", args.pattern)
        out_json = eval_dir / "unified_eval_summary.json"
        save_json(out_json, summary)
        shutil.copyfile(out_json, eval_dir.parent / "unified_eval_summary.json")
        model_summaries[name] = summary
        s = summary["summary"]
        print(f"{name}: TP={s['Global_TP']:.6f} FP={s['Global_FP']} FN={s['Global_FN']} F1={s['IA_QTF1']:.6f}")

    if args.report_md:
        if not model_summaries:
            raise ValueError("--report_md requires at least one --model_eval_dir.")
        missing = [m for m in MODEL_ORDER if m not in model_summaries]
        if missing:
            raise ValueError(f"Missing model summaries for report: {missing}")
        report = build_report(model_summaries, args.offline_math_json.resolve() if args.offline_math_json else None)
        args.report_md.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.report_md.resolve().write_text(report, encoding="utf-8")
        print(f"Saved report to: {args.report_md.resolve()}")


if __name__ == "__main__":
    main()
