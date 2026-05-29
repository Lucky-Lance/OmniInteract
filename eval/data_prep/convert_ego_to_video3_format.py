#!/usr/bin/env python3
"""Convert ego dataset conversation JSON to 1QnA JSON.

Rule:
- question_text = inferred_knowledge + user question
- one assistant message -> one answer item
- only one question block in output
- timestamps keep seconds (numeric), no mm:ss conversion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def to_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp string: {value!r}") from exc
    raise TypeError(f"Unsupported timestamp type: {type(value)}")


def build_question_text(inferred_knowledge: str, user_question: str, separator: str = "\n\n") -> str:
    ik = (inferred_knowledge or "").strip()
    uq = (user_question or "").strip()
    if ik and uq:
        return f"{ik}{separator}{uq}"
    return ik or uq


def convert(src: Dict[str, Any]) -> Dict[str, Any]:
    inferred_knowledge = src.get("inferred_knowledge", "")
    conversations = src.get("conversations", [])
    if not isinstance(conversations, list):
        raise ValueError("Field 'conversations' must be a list")

    first_user = next((m for m in conversations if m.get("from") == "user"), None)
    if first_user is None:
        raise ValueError("No user message found in conversations")

    question_time = to_seconds(first_user.get("timestamp", 0))
    question_text = build_question_text(inferred_knowledge, first_user.get("value", ""))

    answers: List[Dict[str, Any]] = []
    for m in conversations:
        if m.get("from") != "assistant":
            continue
        answers.append(
            {
                "answer_time": to_seconds(m.get("timestamp", 0)),
                "answer_text": (m.get("value") or "").strip(),
            }
        )

    return {
        "question_time": question_time,
        "question_text": question_text,
        "answers": answers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ego conversation JSON to 1QnA format")
    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    src = load_json(in_path)
    out = convert(src)
    dump_json(out_path, out)

    print(f"Converted {in_path} -> {out_path}")
    print(f"Total answers: {len(out['answers'])}")


if __name__ == "__main__":
    main()

