#!/usr/bin/env python3
"""Build unified Slot=[start, t_a, end) rows for all supported tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import HARD, SOFT, Slot, load_json, parse_time, save_json, to_bool


def _collect_qa_rows(gt_root: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    iterable = gt_root if isinstance(gt_root, list) else gt_root.get("items", []) if isinstance(gt_root, dict) else []
    if not isinstance(iterable, list):
        return rows
    for item in iterable:
        if not isinstance(item, dict):
            continue
        q_time = parse_time(item.get("question_time"))
        a_time = parse_time(item.get("answer_time"))
        if q_time is None or a_time is None:
            continue
        rows.append(
            {
                "q_time": float(q_time),
                "a_time": float(a_time),
                "question_text": str(item.get("question_text", "") or ""),
                "answer_text": str(item.get("answer_text", "") or ""),
                "question_type": str(item.get("question_type", "") or "unknown").strip().lower() or "unknown",
                "is_interrupted": to_bool(item.get("is_interrupted", False)),
            }
        )
    rows.sort(key=lambda r: (r["q_time"], r["a_time"]))
    return rows


def build_multi_turn_slots(gt_root: Any, last_slot_tail_sec: float, scene_type: str = "multi_turn") -> List[Slot]:
    rows = _collect_qa_rows(gt_root)
    slots: List[Slot] = []
    for idx, row in enumerate(rows):
        end = rows[idx + 1]["q_time"] if idx + 1 < len(rows) else row["a_time"] + float(last_slot_tail_sec)
        end = max(float(end), float(row["q_time"]))
        slots.append(
            Slot(
                slot_id=idx + 1,
                start=float(row["q_time"]),
                t_a=float(row["a_time"]),
                end=end,
                boundary_type=HARD,
                question_text=row["question_text"],
                gt_answer=row["answer_text"],
                scene_type=scene_type,
                turn_index=idx + 1,
                question_type=row["question_type"],
                is_interrupted=bool(row["is_interrupted"]),
            )
        )
    return slots


def _parse_1qna_answers(gt_root: Dict[str, Any]) -> Tuple[float, str, str, List[Dict[str, Any]]]:
    inferred = str(gt_root.get("inferred_knowledge", "") or "").strip()
    q_time = parse_time(gt_root.get("question_time"))
    question_text = str(gt_root.get("question_text", "") or "")
    answers: List[Dict[str, Any]] = []

    if isinstance(gt_root.get("conversations"), list):
        first_user = next(
            (x for x in gt_root["conversations"] if isinstance(x, dict) and str(x.get("from", "")) == "user"),
            None,
        )
        if first_user is not None:
            q_time = parse_time(first_user.get("timestamp"))
            question_text = str(first_user.get("value", "") or "")
        for item in gt_root["conversations"]:
            if not isinstance(item, dict) or str(item.get("from", "")) != "assistant":
                continue
            t = parse_time(item.get("timestamp"))
            if t is None:
                continue
            answers.append(
                {
                    "answer_time": float(t),
                    "answer_text": str(item.get("value", "") or ""),
                    "is_interrupted": to_bool(item.get("interrupted", False)),
                    "label": str(item.get("label", "") or ""),
                }
            )
        if inferred:
            question_text = (inferred + "\n\n" + question_text).strip()

    elif isinstance(gt_root.get("answers"), list):
        for item in gt_root["answers"]:
            if not isinstance(item, dict):
                continue
            t = parse_time(item.get("answer_time"))
            if t is None:
                continue
            answers.append(
                {
                    "answer_time": float(t),
                    "answer_text": str(item.get("answer_text", "") or ""),
                    "is_interrupted": to_bool(item.get("interrupted", item.get("is_interrupted", False))),
                    "label": str(item.get("label", "") or ""),
                }
            )
    else:
        raise ValueError("1QnA GT must contain conversations[] or answers[].")

    if q_time is None:
        q_time = 0.0
    answers.sort(key=lambda r: r["answer_time"])
    return float(q_time), question_text, inferred, answers


def build_1qna_slots(gt_root: Any, last_slot_tail_sec: float) -> List[Slot]:
    if not isinstance(gt_root, dict):
        raise ValueError("scene_type=1QnA requires GT as a JSON object.")
    q_start, question_text, inferred, answers = _parse_1qna_answers(gt_root)
    slots: List[Slot] = []
    for idx, ans in enumerate(answers):
        t_a = float(ans["answer_time"])
        # In 1QnA, only the first step has a real waiting/ack window from the
        # user's initial request to the first actionable answer. Later steps
        # start at their own action time; premature future-step content remains
        # in the previous slot core and is judged as spoiler/low quality there.
        start = q_start if idx == 0 else t_a
        end = float(answers[idx + 1]["answer_time"]) if idx + 1 < len(answers) else t_a + float(last_slot_tail_sec)
        end = max(end, start)
        slots.append(
            Slot(
                slot_id=idx + 1,
                start=start,
                t_a=t_a,
                end=end,
                boundary_type=SOFT,
                question_text=question_text,
                gt_answer=str(ans["answer_text"]),
                scene_type="1QnA",
                step_index=idx + 1,
                turn_index=1,
                question_type=str(ans.get("label", "") or "step").strip().lower() or "step",
                is_interrupted=bool(ans.get("is_interrupted", False)),
                label=str(ans.get("label", "") or ""),
                inferred_knowledge=inferred,
            )
        )
    return slots


def build_nested_slots(gt_root: Any, last_slot_tail_sec: float) -> List[Slot]:
    rows = _collect_qa_rows(gt_root)
    if len(rows) < 2:
        raise ValueError("scene_type=nested requires at least two QA rows.")

    groups: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    idx = 0
    while idx < len(rows):
        outer = rows[idx]
        inner_idx: Optional[int] = None
        for j in range(idx + 1, len(rows)):
            cand = rows[j]
            if outer["q_time"] < cand["q_time"] < outer["a_time"] and cand["a_time"] <= outer["a_time"]:
                inner_idx = j
                break
        if inner_idx is None:
            raise ValueError("Failed to infer nested pair from QA rows.")
        groups.append((outer, rows[inner_idx]))
        idx = inner_idx + 1

    slots: List[Slot] = []
    slot_id = 1
    for gidx, (outer, inner) in enumerate(groups):
        next_outer_q = groups[gidx + 1][0]["q_time"] if gidx + 1 < len(groups) else outer["a_time"] + float(last_slot_tail_sec)
        next_outer_q = max(float(next_outer_q), float(outer["a_time"]))
        slots.append(
            Slot(
                slot_id=slot_id,
                start=float(outer["q_time"]),
                t_a=float(outer["a_time"]),
                end=next_outer_q,
                boundary_type=HARD,
                question_text=outer["question_text"],
                gt_answer=outer["answer_text"],
                scene_type="nested",
                turn_index=gidx + 1,
                question_type=outer["question_type"],
                is_interrupted=bool(outer["is_interrupted"]),
                nested_group_id=gidx + 1,
                nested_role="outer",
            )
        )
        slot_id += 1
        slots.append(
            Slot(
                slot_id=slot_id,
                start=float(inner["q_time"]),
                t_a=float(inner["a_time"]),
                end=float(outer["a_time"]),
                boundary_type=HARD,
                question_text=inner["question_text"],
                gt_answer=inner["answer_text"],
                scene_type="nested",
                turn_index=gidx + 1,
                question_type=inner["question_type"],
                is_interrupted=bool(inner["is_interrupted"]),
                nested_group_id=gidx + 1,
                nested_role="inner",
            )
        )
        slot_id += 1
    return slots


def build_slots(gt_root: Any, scene_type: str, last_slot_tail_sec: float) -> List[Slot]:
    scene = str(scene_type).strip()
    if scene == "multi_turn":
        return build_multi_turn_slots(gt_root, last_slot_tail_sec, scene)
    if scene == "1QnA":
        return build_1qna_slots(gt_root, last_slot_tail_sec)
    if scene == "nested":
        return build_nested_slots(gt_root, last_slot_tail_sec)
    raise ValueError("scene_type must be one of: multi_turn, 1QnA, nested")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified evaluation slots.")
    parser.add_argument("--gt_json", required=True)
    parser.add_argument("--scene_type", required=True, choices=("multi_turn", "1QnA", "nested"))
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--last_slot_tail_sec", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slots = build_slots(load_json(Path(args.gt_json)), args.scene_type, float(args.last_slot_tail_sec))
    out = {
        "slots": [s.to_dict() for s in slots],
        "summary": {"num_slots": len(slots)},
        "meta": {
            "gt_json": str(Path(args.gt_json).resolve()),
            "scene_type": args.scene_type,
            "last_slot_tail_sec": float(args.last_slot_tail_sec),
        },
    }
    save_json(Path(args.out_json), out)
    print(f"Saved unified slots to: {Path(args.out_json).resolve()}")


if __name__ == "__main__":
    main()
