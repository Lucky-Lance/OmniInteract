#!/usr/bin/env python3
"""Compute unified IA-QTF1 with one logic path for multi-turn and 1QnA."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from build_slots import build_slots
from common import (
    HARD,
    SOFT,
    calculate_decay,
    clamp,
    concat_chunk_text,
    contains_cjk,
    f1_from_pr,
    load_json,
    normalize_chunks,
    safe_div,
    save_json,
    to_bool,
)
from llm_judge import APIJudge, JudgeAPICallError
from match_slots import build_matching


def _sort_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(chunks, key=lambda c: (float(c.get("start", 0.0)), float(c.get("end", 0.0))))


def _chunk_uid(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("source_chunk_id", chunk.get("chunk_id", "")))


def _chunk_full_text(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("full_text", "") or chunk.get("text", "") or "").strip()


def _stage_bounds(chunks: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    if not chunks:
        return None, None
    sorted_chunks = _sort_chunks(chunks)
    return float(sorted_chunks[0].get("start", 0.0)), float(sorted_chunks[-1].get("end", 0.0))


def _build_stage_full_context(stage_chunks: List[Dict[str, Any]], all_chunks: List[Dict[str, Any]]) -> str:
    if not stage_chunks:
        return ""
    wanted = {_chunk_uid(c) for c in stage_chunks}
    texts: List[str] = []
    seen = set()
    for c in _sort_chunks(all_chunks):
        uid = _chunk_uid(c)
        if uid in wanted and uid not in seen:
            txt = _chunk_full_text(c)
            if txt:
                texts.append(txt)
                seen.add(uid)
    if not texts:
        for c in _sort_chunks(stage_chunks):
            txt = _chunk_full_text(c)
            if txt:
                texts.append(txt)
    return "".join(texts).strip()


def _core_effective_start(first_core_chunk: Dict[str, Any], t_a: float) -> float:
    raw = float(first_core_chunk.get("start", t_a))
    hint = first_core_chunk.get("effective_start_hint")
    if hint is None:
        return max(raw, t_a)
    try:
        return max(min(raw, float(hint)), t_a)
    except Exception:
        return max(raw, t_a)


def _chunk_char_start_times(chunk: Dict[str, Any]) -> List[Optional[float]]:
    text = str(chunk.get("text", "") or "")
    words = [w for w in list(chunk.get("aligned_words", []) or []) if isinstance(w, dict)]
    if not text:
        return []
    if not words:
        return [None] * len(text)
    tokens: List[str] = []
    starts: List[float] = []
    for w in words:
        wt = str(w.get("text", "") or "")
        if not wt:
            continue
        try:
            ws = float(w.get("start"))
        except Exception:
            continue
        tokens.append(wt)
        starts.append(ws)
    if not tokens:
        return [None] * len(text)
    sep = "" if contains_cjk("".join(tokens)) else " "
    built_chars: List[str] = []
    built_times: List[Optional[float]] = []
    for i, (tok, st) in enumerate(zip(tokens, starts)):
        for ch in tok:
            built_chars.append(ch)
            built_times.append(float(st))
        if sep and i + 1 < len(tokens):
            built_chars.append(sep)
            built_times.append(None)
    built = "".join(built_chars).strip()
    if built == text:
        return built_times[: len(text)]
    pos = text.find(built)
    out: List[Optional[float]] = [None] * len(text)
    if pos >= 0:
        for i, t in enumerate(built_times):
            if pos + i < len(out):
                out[pos + i] = t
    return out


def _build_core_char_time_map(core_chunks: List[Dict[str, Any]]) -> Tuple[str, List[Optional[float]]]:
    pieces: List[str] = []
    times: List[Optional[float]] = []
    for c in _sort_chunks(core_chunks):
        pieces.append(str(c.get("text", "") or ""))
        times.extend(_chunk_char_start_times(c))
    return "".join(pieces), times


def _find_trigger_start_time(actual_text: str, core_chunks: List[Dict[str, Any]], trigger_phrase: str) -> Optional[float]:
    phrase = str(trigger_phrase or "").strip().strip("\"'“”‘’")
    if not phrase:
        return None
    text, times = _build_core_char_time_map(core_chunks)
    hay = text or str(actual_text or "")
    idx = hay.find(phrase)
    if idx < 0:
        idx = hay.lower().find(phrase.lower())
    if idx < 0 or not times:
        return None
    for i in range(idx, min(len(hay), idx + len(phrase))):
        if i < len(times) and times[i] is not None:
            return float(times[i])
    return None


def _future_answers(slots: List[Dict[str, Any]], index: int) -> str:
    if index + 1 >= len(slots):
        return "(none)"
    rows = []
    for row in slots[index + 1 :]:
        if row.get("scene_type") == slots[index].get("scene_type"):
            rows.append(f"- slot {row.get('slot_id')}: {str(row.get('gt_answer', '')).strip()}")
    return "\n".join(rows) if rows else "(none)"


def _effective_boundary_type(slot: Dict[str, Any]) -> str:
    if to_bool(slot.get("is_interrupted", False)):
        return HARD
    raw = str(slot.get("boundary_type", HARD) or HARD)
    return SOFT if raw.lower() == "soft" else HARD


def _has_hard_spill(slot: Dict[str, Any], all_chunks: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, float]]:
    if _effective_boundary_type(slot) != HARD or not all_chunks:
        return False, {"spill_seconds": 0.0, "max_chunk_end": None}
    slot_end = float(slot.get("end", 0.0))
    max_end = max(float(c.get("end", slot_end)) for c in all_chunks)
    spill_seconds = max(0.0, max_end - slot_end)
    return spill_seconds > 0, {"spill_seconds": spill_seconds, "max_chunk_end": max_end}


def _precision_recall_f1(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {"Precision": precision, "Recall": recall, "F1": f1_from_pr(precision, recall)}


def _build_interruption_diagnostic(
    slot: Dict[str, Any],
    all_chunks: List[Dict[str, Any]],
    has_spill: bool,
    spill_diag: Dict[str, float],
    judge: APIJudge,
    partial_quality_threshold: float,
    evaluate_interrupted_outputs: bool,
) -> Dict[str, Any]:
    actual_text = concat_chunk_text(all_chunks)
    has_output = bool(all_chunks or actual_text)
    base: Dict[str, Any] = {
        "is_interrupted": True,
        "has_output": has_output,
        "num_chunks": len(all_chunks),
        "actual_text": actual_text,
        "spill_positive": bool(has_spill),
        "spill_seconds": float(spill_diag.get("spill_seconds") or 0.0),
        "quality_threshold": float(partial_quality_threshold),
    }
    if not has_output:
        return {
            **base,
            "status": "no_output",
            "partial_quality": None,
            "low_quality": False,
            "hallucination": False,
            "TP": 0.0,
            "FP": 0,
            "FN": 0,
            **_precision_recall_f1(0.0, 0.0, 0.0),
        }
    if not evaluate_interrupted_outputs:
        return {**base, "status": "skipped"}

    quality = judge.judge_interrupted_partial(slot, actual_text)
    q = float(quality.get("score", 0.0) or 0.0)
    hallucination = to_bool(quality.get("hallucination", False))
    low_quality = q < float(partial_quality_threshold)
    tp = 0.0 if low_quality else q
    fp = (1 if low_quality else 0) + (1 if has_spill else 0) + (1 if hallucination else 0)
    fn = 1 if low_quality else 0
    return {
        **base,
        "status": "ok",
        "partial_quality": q,
        "low_quality": low_quality,
        "hallucination": hallucination,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        **_precision_recall_f1(tp, fp, fn),
        "judge_partial": quality,
    }


def score_one_slot(
    slot: Dict[str, Any],
    slots: List[Dict[str, Any]],
    slot_index: int,
    judge: APIJudge,
    w_ack: float,
    decay_alpha: float,
    decay_gamma: float,
    core_quality_threshold: float,
    evaluate_interrupted_outputs: bool,
    partial_quality_threshold: float,
) -> Tuple[Dict[str, Any], float, int, int]:
    start = float(slot.get("start", 0.0))
    t_a = float(slot.get("t_a", start))
    end = float(slot.get("end", t_a))
    scene_type = str(slot.get("scene_type", "") or "")
    is_interrupted = to_bool(slot.get("is_interrupted", False))
    effective_boundary_type = _effective_boundary_type(slot)
    all_chunks = _sort_chunks(list(slot.get("all_chunks", []) or []))
    early_chunks = _sort_chunks(list(slot.get("early_chunks", []) or []))
    core_chunks = _sort_chunks(list(slot.get("core_chunks", []) or []))

    # 1QnA step 1 keeps the normal interaction window. Later steps are built
    # with start == t_a, so they normally have no early phase and no ack TP.
    local_w_ack = float(w_ack)
    if scene_type == "1QnA" and int(slot.get("step_index", slot.get("slot_id", 1)) or 1) > 1:
        local_w_ack = 0.0
    local_w_core = 1.0 - local_w_ack

    fp_delta = 0
    tp_ack = 0.0
    text_early = concat_chunk_text(early_chunks)
    full_context_early = _build_stage_full_context(early_chunks, all_chunks)
    early_start, early_end = _stage_bounds(early_chunks)
    if early_chunks:
        t_ack = calculate_decay(
            float(early_chunks[0].get("start", start)),
            peak=start,
            end=t_a,
            alpha=decay_alpha,
            gamma=decay_gamma,
        )
        category, s_ack, early_judge = judge.judge_early(slot, full_context_early, text_early)
        if category == "hallucination":
            fp_delta += 1
            s_ack = 0.0
        elif category == "neutral":
            tp_ack = t_ack * s_ack * local_w_ack
        early_diag = {
            "category": category,
            "answer_start": early_start,
            "answer_end": early_end,
            "actual_text": text_early,
            "quality_score": s_ack,
            "timeliness_score": t_ack,
            "tp_contribution": tp_ack,
            "judge_rationale": str(early_judge.get("rationale", "") or ""),
            "T_ack": t_ack,
            "S_ack": s_ack,
            "W_ack_used": local_w_ack,
            "diagnostics": early_judge,
            "full_context": full_context_early,
            "text_early": text_early,
            "chunks": early_chunks,
        }
    else:
        early_diag = {
            "category": "none",
            "answer_start": None,
            "answer_end": None,
            "actual_text": "",
            "quality_score": 0.0,
            "timeliness_score": 0.0,
            "tp_contribution": 0.0,
            "judge_rationale": "",
            "T_ack": 0.0,
            "S_ack": 0.0,
            "W_ack_used": local_w_ack,
            "diagnostics": {},
            "full_context": "",
            "text_early": "",
            "chunks": [],
        }

    tp_core = 0.0
    text_core = concat_chunk_text(core_chunks)
    full_context_core = _build_stage_full_context(core_chunks, all_chunks)
    core_start_observed, core_end_observed = _stage_bounds(core_chunks)
    core_diag: Dict[str, Any]
    if is_interrupted:
        core_diag = {
            "skipped_interrupted": True,
            "answer_start": core_start_observed,
            "answer_end": core_end_observed,
            "actual_text": text_core,
            "quality_score": 0.0,
            "timeliness_score": 0.0,
            "tp_contribution": 0.0,
            "judge_rationale": "",
            "T_start": None,
            "T_end": float(core_chunks[-1].get("end", t_a)) if core_chunks else None,
            "T_core": 0.0,
            "S_core": 0.0,
            "quality_threshold": float(core_quality_threshold),
            "low_quality_fp": False,
            "judge_core": {},
            "full_context": full_context_core,
            "text_core": text_core,
            "chunks": core_chunks,
        }
    elif core_chunks:
        t_start_fallback = _core_effective_start(core_chunks[0], t_a)
        t_end_actual = float(core_chunks[-1].get("end", t_start_fallback))
        s_core, core_judge = judge.judge_core(slot, full_context_core, text_core, _future_answers(slots, slot_index))
        trigger = str(core_judge.get("trigger_phrase", "") or "")
        t_anchor = _find_trigger_start_time(text_core, core_chunks, trigger) if s_core > 0 else None
        t_start = max(t_anchor if t_anchor is not None else t_start_fallback, t_a)
        t_core = calculate_decay(t_start, peak=t_a, end=end, alpha=decay_alpha, gamma=decay_gamma)
        low_quality = s_core < float(core_quality_threshold)
        if low_quality:
            fp_delta += 1
            tp_core = 0.0
        else:
            tp_core = t_core * s_core * local_w_core
        core_diag = {
            "skipped_interrupted": False,
            "answer_start": t_start,
            "answer_start_observed": core_start_observed,
            "answer_end": t_end_actual,
            "actual_text": text_core,
            "quality_score": s_core,
            "timeliness_score": t_core,
            "tp_contribution": tp_core,
            "judge_rationale": str(core_judge.get("rationale", "") or ""),
            "T_start": t_start,
            "T_start_fallback": t_start_fallback,
            "T_start_true_effective": t_anchor,
            "T_end": t_end_actual,
            "T_core": t_core,
            "S_core": s_core,
            "W_core_used": local_w_core,
            "quality_threshold": float(core_quality_threshold),
            "low_quality_fp": low_quality,
            "judge_core": core_judge,
            "full_context": full_context_core,
            "text_core": text_core,
            "chunks": core_chunks,
        }
    else:
        core_diag = {
            "skipped_interrupted": False,
            "answer_start": None,
            "answer_end": None,
            "actual_text": "",
            "quality_score": 0.0,
            "timeliness_score": 0.0,
            "tp_contribution": 0.0,
            "judge_rationale": "",
            "T_start": None,
            "T_end": None,
            "T_core": 0.0,
            "S_core": 0.0,
            "W_core_used": local_w_core,
            "quality_threshold": float(core_quality_threshold),
            "low_quality_fp": False,
            "judge_core": {},
            "full_context": "",
            "text_core": "",
            "chunks": [],
        }

    has_spill, spill_diag = _has_hard_spill(slot, all_chunks)
    if has_spill:
        fp_delta += 1

    interruption_diag: Dict[str, Any] = {}
    if is_interrupted:
        interruption_diag = _build_interruption_diagnostic(
            slot=slot,
            all_chunks=all_chunks,
            has_spill=has_spill,
            spill_diag=spill_diag,
            judge=judge,
            partial_quality_threshold=float(partial_quality_threshold),
            evaluate_interrupted_outputs=bool(evaluate_interrupted_outputs),
        )

    tp_n = clamp(tp_ack + tp_core, 0.0, 1.0)
    # A neutral/acknowledgement-only response may earn Score_ack, but it is not
    # an effective answer. Non-interrupted slots are counted as FN unless the
    # core answer receives positive credit.
    fn_delta = 0 if is_interrupted else (1 if tp_core <= 0 else 0)

    out = {
        "slot_id": slot.get("slot_id"),
        "scene_type": scene_type,
        "question_type": slot.get("question_type", "unknown"),
        "step_index": slot.get("step_index"),
        "turn_index": slot.get("turn_index"),
        "label": slot.get("label", ""),
        "nested_group_id": slot.get("nested_group_id"),
        "nested_role": slot.get("nested_role"),
        "boundary_type": slot.get("boundary_type"),
        "effective_boundary_type": effective_boundary_type,
        "is_interrupted": is_interrupted,
        "start": start,
        "t_a": t_a,
        "end": end,
        "question_text": slot.get("question_text", ""),
        "gt_answer": slot.get("gt_answer", ""),
        "num_chunks": len(all_chunks),
        "num_early": len(early_chunks),
        "num_core": len(core_chunks),
        "Score_ack": tp_ack,
        "Score_core": tp_core,
        "TP_ack": tp_ack,
        "TP_core": tp_core,
        "TP_n": tp_n,
        "FP_delta": fp_delta,
        "FN_delta": fn_delta,
        "all_chunks": all_chunks,
        "stage_early": early_diag,
        "stage_core": core_diag,
        "spill": has_spill,
        "spill_diagnostics": spill_diag,
        "interruption_diagnostic": interruption_diag,
    }
    return out, tp_n, fp_delta, fn_delta


def _accumulate(by_key: Dict[str, Dict[str, float]], key: str, tp: float, fp: int, fn: int) -> None:
    k = str(key or "unknown")
    if k not in by_key:
        by_key[k] = {"num_slots": 0.0, "Global_TP": 0.0, "Global_FP": 0.0, "Global_FN": 0.0}
    by_key[k]["num_slots"] += 1.0
    by_key[k]["Global_TP"] += float(tp)
    by_key[k]["Global_FP"] += float(fp)
    by_key[k]["Global_FN"] += float(fn)


def _summarize_groups(groups: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, row in groups.items():
        tp = float(row["Global_TP"])
        fp = int(row["Global_FP"])
        fn = int(row["Global_FN"])
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        out[key] = {
            "num_slots": int(row["num_slots"]),
            "Global_TP": tp,
            "Global_FP": fp,
            "Global_FN": fn,
            "Precision": p,
            "Recall": r,
            "IA_QTF1": f1_from_pr(p, r),
        }
    return out


def _summarize_interruption_diagnostics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    interrupted = [r for r in rows if to_bool(r.get("is_interrupted", False))]
    output_rows = []
    tp = fp = fn = 0.0
    quality_sum = 0.0
    low_quality = 0
    hallucination = 0
    spill_positive = 0
    spill_seconds = 0.0
    for row in interrupted:
        diag = row.get("interruption_diagnostic", {}) if isinstance(row.get("interruption_diagnostic"), dict) else {}
        if not diag.get("has_output"):
            continue
        output_rows.append(row)
        if diag.get("status") == "ok":
            q = float(diag.get("partial_quality") or 0.0)
            quality_sum += q
            tp += float(diag.get("TP") or 0.0)
            fp += float(diag.get("FP") or 0.0)
            fn += float(diag.get("FN") or 0.0)
            if diag.get("low_quality"):
                low_quality += 1
            if diag.get("hallucination"):
                hallucination += 1
        if diag.get("spill_positive"):
            spill_positive += 1
        spill_seconds += float(diag.get("spill_seconds") or 0.0)

    prf = _precision_recall_f1(tp, fp, fn)
    return {
        "interrupted_slot_count": len(interrupted),
        "interrupted_no_output_count": len(interrupted) - len(output_rows),
        "interrupted_no_output_rate": safe_div(len(interrupted) - len(output_rows), len(interrupted)),
        "interrupted_with_output_count": len(output_rows),
        "partial_quality_avg": safe_div(quality_sum, len(output_rows)),
        "low_quality_count": low_quality,
        "hallucination_count": hallucination,
        "spill_positive_count": spill_positive,
        "spill_positive_rate_in_output_slots": safe_div(spill_positive, len(output_rows)),
        "total_spill_seconds_in_output_slots": spill_seconds,
        "avg_spill_seconds_in_output_slots": safe_div(spill_seconds, len(output_rows)),
        "TP": tp,
        "FP": int(fp),
        "FN": int(fn),
        **prf,
    }


def compute_unified_eval(
    match_root: Dict[str, Any],
    judge: APIJudge,
    w_ack: float,
    decay_alpha: float,
    decay_gamma: float,
    core_quality_threshold: float,
    count_unmatched_as_fp: bool,
    evaluate_interrupted_outputs: bool = True,
    partial_quality_threshold: float = 0.5,
) -> Dict[str, Any]:
    slots = list(match_root.get("slots", []) or [])
    unmatched = list(match_root.get("unmatched_chunks", []) or [])
    slot_rows: List[Dict[str, Any]] = []
    global_tp = 0.0
    global_fp = 0
    global_fn = 0
    by_scene: Dict[str, Dict[str, float]] = {}
    by_type: Dict[str, Dict[str, float]] = {}
    failed: Optional[Dict[str, Any]] = None

    for idx, slot in enumerate(slots):
        try:
            row, tp, fp, fn = score_one_slot(
                slot=slot,
                slots=slots,
                slot_index=idx,
                judge=judge,
                w_ack=float(w_ack),
                decay_alpha=float(decay_alpha),
                decay_gamma=float(decay_gamma),
                core_quality_threshold=float(core_quality_threshold),
                evaluate_interrupted_outputs=bool(evaluate_interrupted_outputs),
                partial_quality_threshold=float(partial_quality_threshold),
            )
        except JudgeAPICallError as exc:
            failed = {
                "type": "judge_api_error",
                "message": str(exc),
                "failed_slot_index": idx,
                "failed_slot_id": slot.get("slot_id"),
                "processed_slots": len(slot_rows),
            }
            break
        slot_rows.append(row)
        global_tp += tp
        global_fp += fp
        global_fn += fn
        _accumulate(by_scene, str(row.get("scene_type", "unknown")), tp, fp, fn)
        _accumulate(by_type, str(row.get("question_type", "unknown")), tp, fp, fn)

    if count_unmatched_as_fp and failed is None:
        global_fp += len(unmatched)

    precision = safe_div(global_tp, global_tp + global_fp)
    recall = safe_div(global_tp, global_tp + global_fn)
    return {
        "metric": "Unified-IA-QTF1",
        "status": "failed" if failed else "ok",
        "error": failed,
        "config": {
            "W_ack": float(w_ack),
            "W_core_default": 1.0 - float(w_ack),
            "one_qna_later_steps_start_at_t_a": True,
            "one_qna_later_steps_W_ack": 0.0,
            "decay_alpha": float(decay_alpha),
            "decay_gamma": float(decay_gamma),
            "core_quality_threshold": float(core_quality_threshold),
            "evaluate_interrupted_outputs": bool(evaluate_interrupted_outputs),
            "partial_quality_threshold": float(partial_quality_threshold),
            "count_unmatched_as_fp": bool(count_unmatched_as_fp),
        },
        "summary": {
            "num_slots": len(slots),
            "num_processed_slots": len(slot_rows),
            "num_unmatched_chunks": len(unmatched),
            "Global_TP": global_tp,
            "Global_FP": global_fp,
            "Global_FN": global_fn,
            "Precision": precision,
            "Recall": recall,
            "IA_QTF1": f1_from_pr(precision, recall),
            "by_scene_type": _summarize_groups(by_scene),
            "by_question_type": _summarize_groups(by_type),
            "interruption_diagnostics": _summarize_interruption_diagnostics(slot_rows),
        },
        "slots": slot_rows,
        "unmatched_chunks": unmatched,
    }


def _load_or_build_match(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if args.match_json:
        path = Path(args.match_json).resolve()
        return load_json(path), {"match_json": str(path)}
    gt_path = Path(args.gt_json).resolve()
    model_path = Path(args.model_json).resolve()
    gt_root = load_json(gt_path)
    model_root = load_json(model_path)
    slots = [s.to_dict() for s in build_slots(gt_root, args.scene_type, float(args.last_slot_tail_sec))]
    match_root = build_matching(slots, normalize_chunks(model_root))
    match_root["meta"] = {
        "gt_json": str(gt_path),
        "model_json": str(model_path),
        "scene_type": args.scene_type,
        "last_slot_tail_sec": float(args.last_slot_tail_sec),
    }
    return match_root, {"gt_json": str(gt_path), "model_json": str(model_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified IA-QTF1 evaluation with an LLM API judge.")
    parser.add_argument("--match_json", default="", help="Optional precomputed unified match JSON.")
    parser.add_argument("--gt_json", default="", help="GT JSON path when --match_json is not used.")
    parser.add_argument("--model_json", default="", help="Model wav_transcript/precise_truncation JSON when --match_json is not used.")
    parser.add_argument("--scene_type", default="multi_turn", choices=("multi_turn", "1QnA", "nested"))
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--last_slot_tail_sec", type=float, default=60.0)
    parser.add_argument("--w_ack", type=float, default=0.2)
    parser.add_argument("--decay_alpha", type=float, default=1.0)
    parser.add_argument("--decay_gamma", type=float, default=1.0)
    parser.add_argument("--core_quality_threshold", type=float, default=0.5)
    parser.add_argument("--partial_quality_threshold", type=float, default=0.5)
    parser.add_argument("--no_interrupted_output_eval", action="store_true", help="Skip PAQ/CSM diagnostic judging for interrupted slots with output.")
    parser.add_argument("--no_unmatched_fp", action="store_true")
    parser.add_argument("--judge_api_url", default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--judge_api_model", default="gpt-4o-2024-08-06")
    parser.add_argument("--judge_api_key", default="")
    parser.add_argument("--judge_api_timeout_sec", type=float, default=60.0)
    parser.add_argument("--judge_max_tokens", type=int, default=512)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.match_json and (not args.gt_json or not args.model_json):
        raise ValueError("Provide either --match_json, or both --gt_json and --model_json.")

    match_root, meta_source = _load_or_build_match(args)
    judge = APIJudge(
        api_url=str(args.judge_api_url),
        api_key=str(args.judge_api_key or os.getenv("JUDGE_API_KEY", "")),
        model=str(args.judge_api_model),
        max_tokens=int(args.judge_max_tokens),
        temperature=float(args.judge_temperature),
        timeout_sec=float(args.judge_api_timeout_sec),
    )
    result = compute_unified_eval(
        match_root=match_root,
        judge=judge,
        w_ack=float(args.w_ack),
        decay_alpha=float(args.decay_alpha),
        decay_gamma=float(args.decay_gamma),
        core_quality_threshold=float(args.core_quality_threshold),
        count_unmatched_as_fp=not bool(args.no_unmatched_fp),
        evaluate_interrupted_outputs=not bool(args.no_interrupted_output_eval),
        partial_quality_threshold=float(args.partial_quality_threshold),
    )
    result["meta"] = {
        **meta_source,
        "match_meta": match_root.get("meta", {}),
        "judge_backend": "api",
        "judge_api_url": str(args.judge_api_url),
        "judge_api_model": str(args.judge_api_model),
    }
    save_json(Path(args.out_json), result)
    s = result["summary"]
    print(f"Saved unified evaluation to: {Path(args.out_json).resolve()}")
    print(f"TP={s['Global_TP']:.6f} FP={s['Global_FP']} FN={s['Global_FN']} P={s['Precision']:.6f} R={s['Recall']:.6f} F1={s['IA_QTF1']:.6f}")
    if result.get("status") == "failed":
        err = result.get("error", {})
        print(f"Judge API failed at slot_id={err.get('failed_slot_id')}; progress saved.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
