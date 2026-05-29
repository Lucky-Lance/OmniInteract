#!/usr/bin/env python3
"""Match model output chunks to unified slots and split early/core phases."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import Chunk, contains_cjk, load_json, normalize_chunks, save_json
from build_slots import build_slots


def _sort_dict_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(chunks, key=lambda c: (float(c.get("start", 0.0)), float(c.get("end", 0.0))))


def _join_words(words: List[Dict[str, Any]]) -> str:
    tokens = [str(w.get("text", "") or "") for w in words if str(w.get("text", "") or "")]
    if not tokens:
        return ""
    return ("".join(tokens) if contains_cjk("".join(tokens)) else " ".join(tokens)).strip()


def _split_cross_boundary(chunk: Chunk, boundary: float) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not (chunk.start < boundary < chunk.end) or not chunk.aligned_words:
        return None, None
    early_words = [w for w in chunk.aligned_words if float(w.get("start", boundary)) < boundary]
    core_words = [w for w in chunk.aligned_words if float(w.get("start", boundary)) >= boundary]
    if not early_words and not core_words:
        return None, None

    early_text = _join_words(early_words)
    core_text = _join_words(core_words)
    early = None
    core = None
    if early_text:
        early = {
            "chunk_id": f"{chunk.chunk_id}:early",
            "source_chunk_id": chunk.chunk_id,
            "start": float(early_words[0]["start"]),
            "end": float(early_words[-1]["end"]),
            "text": early_text,
            "aligned_words": early_words,
            "split_part": "early",
            "split_boundary": float(boundary),
            "full_text": chunk.text,
        }
    if core_text:
        core = {
            "chunk_id": f"{chunk.chunk_id}:core",
            "source_chunk_id": chunk.chunk_id,
            "start": float(core_words[0]["start"]),
            "end": float(core_words[-1]["end"]),
            "text": core_text,
            "aligned_words": core_words,
            "split_part": "core",
            "split_boundary": float(boundary),
            "full_text": chunk.text,
            "effective_start_hint": float(boundary),
        }
    return early, core


def _match_slot_index(slots: List[Dict[str, Any]], chunk_start: float) -> Optional[int]:
    candidates = [
        idx
        for idx, slot in enumerate(slots)
        if float(slot.get("start", 0.0)) <= chunk_start < float(slot.get("end", 0.0))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda idx: (float(slots[idx].get("start", 0.0)), int(slots[idx].get("slot_id", 0))))


def build_matching(slots: List[Dict[str, Any]], chunks: List[Chunk]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for slot in slots:
        rows.append({**slot, "all_chunks": [], "early_chunks": [], "core_chunks": []})

    unmatched: List[Dict[str, Any]] = []
    for chunk in chunks:
        cdict = asdict(chunk)
        idx = _match_slot_index(rows, chunk.start)
        if idx is None:
            unmatched.append(cdict)
            continue

        slot = rows[idx]
        boundary = float(slot.get("t_a", slot.get("start", 0.0)))
        phase = "early" if chunk.start < boundary else "core"
        cross = chunk.start < boundary < chunk.end
        spill = chunk.end > float(slot.get("end", boundary))
        tagged = {**cdict, "phase": phase, "cross_boundary": cross, "spill": spill}
        slot["all_chunks"].append(tagged)

        if cross:
            early, core = _split_cross_boundary(chunk, boundary)
            if early is not None:
                slot["early_chunks"].append({**early, "phase": "early", "cross_boundary": True, "spill": early["end"] > slot["end"]})
            if core is not None:
                slot["core_chunks"].append({**core, "phase": "core", "cross_boundary": True, "spill": core["end"] > slot["end"]})
            if early is None and core is None:
                slot["early_chunks" if phase == "early" else "core_chunks"].append(tagged)
            continue

        slot["early_chunks" if phase == "early" else "core_chunks"].append(tagged)

    for slot in rows:
        slot["all_chunks"] = _sort_dict_chunks(slot["all_chunks"])
        slot["early_chunks"] = _sort_dict_chunks(slot["early_chunks"])
        slot["core_chunks"] = _sort_dict_chunks(slot["core_chunks"])
        slot["text_early"] = "".join(str(c.get("text", "") or "") for c in slot["early_chunks"]).strip()
        slot["text_core"] = "".join(str(c.get("text", "") or "") for c in slot["core_chunks"]).strip()
        slot["num_chunks"] = len(slot["all_chunks"])
        slot["num_early"] = len(slot["early_chunks"])
        slot["num_core"] = len(slot["core_chunks"])
        slot["num_spill"] = sum(1 for c in slot["all_chunks"] if c.get("spill"))
        slot["num_cross_boundary_chunks"] = sum(1 for c in slot["all_chunks"] if c.get("cross_boundary"))

    return {
        "slots": rows,
        "unmatched_chunks": unmatched,
        "summary": {
            "num_slots": len(rows),
            "num_chunks": len(chunks),
            "num_matched_chunks": sum(len(r["all_chunks"]) for r in rows),
            "num_unmatched_chunks": len(unmatched),
            "num_spill_chunks": sum(int(r["num_spill"]) for r in rows),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match chunks to unified slots.")
    parser.add_argument("--gt_json", required=True)
    parser.add_argument("--model_json", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--scene_type", required=True, choices=("multi_turn", "1QnA", "nested"))
    parser.add_argument("--last_slot_tail_sec", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path(args.gt_json).resolve()
    model_path = Path(args.model_json).resolve()
    slots = [s.to_dict() for s in build_slots(load_json(gt_path), args.scene_type, float(args.last_slot_tail_sec))]
    chunks = normalize_chunks(load_json(model_path))
    out = build_matching(slots, chunks)
    out["meta"] = {
        "gt_json": str(gt_path),
        "model_json": str(model_path),
        "scene_type": args.scene_type,
        "last_slot_tail_sec": float(args.last_slot_tail_sec),
    }
    save_json(Path(args.out_json), out)
    print(f"Saved unified matching to: {Path(args.out_json).resolve()}")
    print(f"slots={out['summary']['num_slots']} chunks={out['summary']['num_chunks']} matched={out['summary']['num_matched_chunks']}")


if __name__ == "__main__":
    main()
