#!/usr/bin/env python3
"""Common helpers for the unified Full-Duplex evaluation pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


HARD = "Hard"
SOFT = "Soft"


@dataclass
class Chunk:
    chunk_id: int
    start: float
    end: float
    text: str
    aligned_words: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Slot:
    slot_id: int
    start: float
    t_a: float
    end: float
    boundary_type: str
    question_text: str
    gt_answer: str
    scene_type: str
    step_index: Optional[int] = None
    turn_index: Optional[int] = None
    question_type: str = "unknown"
    is_interrupted: bool = False
    label: str = ""
    nested_group_id: Optional[int] = None
    nested_role: Optional[str] = None
    inferred_knowledge: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_time(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    parts = s.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        nums = [float(x) for x in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        minutes, seconds = nums
        return minutes * 60.0 + seconds
    hours, minutes, seconds = nums
    return hours * 3600.0 + minutes * 60.0 + seconds


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value or "").strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def safe_div(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if denom > 0 else 0.0


def f1_from_pr(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def calculate_decay(
    time_value: float,
    peak: float,
    end: float,
    alpha: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """Return a timeliness score in [0, 1]."""
    t = float(time_value)
    p = float(peak)
    e = float(end)
    if e <= p:
        return 1.0 if t <= p else 0.0
    if t <= p:
        return 1.0
    if t >= e:
        return 0.0
    ratio = (t - p) / (e - p)
    return clamp(1.0 - float(alpha) * (ratio ** float(gamma)))


def concat_chunk_text(chunks: List[Dict[str, Any]]) -> str:
    return "".join(str(c.get("text", "") or "") for c in chunks).strip()


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def normalize_chunks(model_root: Any) -> List[Chunk]:
    if not isinstance(model_root, dict) or not isinstance(model_root.get("chunks"), list):
        return []

    chunks: List[Chunk] = []
    for idx, raw in enumerate(model_root.get("chunks", [])):
        if not isinstance(raw, dict):
            continue
        ts = raw.get("timestamp")
        text = str(raw.get("text", "") or "")
        if not (isinstance(ts, list) and len(ts) == 2):
            continue
        start = parse_time(ts[0])
        end = parse_time(ts[1])
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start

        words: List[Dict[str, Any]] = []
        if isinstance(raw.get("aligned_words"), list):
            for w in raw.get("aligned_words", []):
                if not isinstance(w, dict):
                    continue
                ws = parse_time(w.get("start"))
                we = parse_time(w.get("end"))
                wt = str(w.get("text", "") or "")
                if ws is None or we is None:
                    continue
                if we < ws:
                    ws, we = we, ws
                words.append({"text": wt, "start": float(ws), "end": float(we)})
        words.sort(key=lambda x: (float(x["start"]), float(x["end"])))
        chunks.append(Chunk(idx, float(start), float(end), text, words))

    chunks.sort(key=lambda c: (c.start, c.end, c.chunk_id))
    return chunks


def parse_numbered_steps(text: str) -> List[str]:
    import re

    out: List[str] = []
    for line in str(text or "").splitlines():
        line = line.strip()
        m = re.match(r"^\d+[\.)]\s*(.+)$", line)
        if m:
            out.append(m.group(1).strip())
    return out
