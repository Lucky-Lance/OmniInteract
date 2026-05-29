#!/usr/bin/env python3
"""Force-align wav_transcript.json text without running ASR.

This is intended for MiniCPM-o outputs, where wav_transcript.json already
contains the model-native text aligned to coarse output.wav ranges. The script
keeps that text and only adds word-level timestamps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf
import torch

from qwen_asr import Qwen3ForcedAligner


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def detect_language(text: str) -> str:
    return "Chinese" if contains_cjk(text) else "English"


def iter_batch_output_dirs(batch_summary_json: Path, output_root: Optional[Path], process_non_ok: bool) -> Iterable[Path]:
    root = load_json(batch_summary_json)
    rows = root.get("results", []) if isinstance(root, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not process_non_ok and str(row.get("status", "") or "") != "ok":
            continue
        raw = str(row.get("output_dir", "") or "")
        if not raw:
            continue
        out_dir = resolve_output_dir(raw, output_root)
        if out_dir is not None:
            yield out_dir


_SUBSETS = ("1q1a_math", "1q1a", "1qna")


def resolve_output_dir(raw_output_dir: str, output_root: Optional[Path]) -> Optional[Path]:
    raw = Path(str(raw_output_dir))
    if raw.exists():
        return raw.resolve()
    if output_root is None:
        return None

    raw_parts = [p for p in str(raw).replace("\\", "/").split("/") if p]
    hinted = [s for s in _SUBSETS if s in raw_parts]
    ordered = hinted + [s for s in _SUBSETS if s not in hinted]
    candidates = [output_root / raw.name] + [output_root / s / raw.name for s in ordered]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    matches = list(output_root.rglob(raw.name))
    matches = [p for p in matches if p.is_dir()]
    if len(matches) == 1:
        return matches[0].resolve()
    return None


def collect_sample_dirs(args: argparse.Namespace) -> List[Path]:
    dirs: List[Path] = []
    if args.sample_dir:
        dirs.append(Path(args.sample_dir).resolve())
    if args.wav_dirs_json:
        payload = load_json(Path(args.wav_dirs_json).resolve())
        if not isinstance(payload, list):
            raise ValueError("--wav_dirs_json must point to a JSON list of sample directories.")
        dirs.extend(Path(str(x)).resolve() for x in payload)
    if args.batch_summary_json:
        output_root = Path(args.output_root).resolve() if args.output_root else None
        dirs.extend(iter_batch_output_dirs(Path(args.batch_summary_json).resolve(), output_root, bool(args.process_non_ok)))

    uniq: List[Path] = []
    seen = set()
    for d in dirs:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
    return uniq


def align_one_sample(sample_dir: Path, aligner: Any, out_name: str, skip_existing: bool) -> Dict[str, Any]:
    wav_transcript_path = sample_dir / "wav_transcript.json"
    audio_path = sample_dir / "output.wav"
    out_path = sample_dir / out_name

    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        return {"sample_dir": str(sample_dir), "status": "skipped_existing", "out_json": str(out_path)}
    if not wav_transcript_path.exists() or not audio_path.exists():
        missing = [str(p) for p in (wav_transcript_path, audio_path) if not p.exists()]
        return {"sample_dir": str(sample_dir), "status": "missing_input", "missing": missing}

    wav_transcript = load_json(wav_transcript_path)
    chunks = wav_transcript.get("chunks", []) if isinstance(wav_transcript, dict) else []

    wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    audio_inputs = []
    text_inputs = []
    lang_inputs = []
    valid_refs: List[Dict[str, Any]] = []
    valid_starts: List[float] = []

    for raw in chunks:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "") or "")
        ts = raw.get("timestamp")
        if not text.strip() or not (isinstance(ts, list) and len(ts) == 2):
            continue
        start, end = float(ts[0]), float(ts[1])
        if end <= start:
            continue
        start_idx = max(0, int(start * sr))
        end_idx = min(len(wav), int(end * sr))
        if end_idx <= start_idx:
            continue
        audio_inputs.append((np.asarray(wav[start_idx:end_idx], dtype=np.float32), sr))
        text_inputs.append(text)
        lang_inputs.append(detect_language(text))
        valid_refs.append({"text": text, "timestamp": [start, end]})
        valid_starts.append(start)

    if audio_inputs:
        results = aligner.align(audio=audio_inputs, text=text_inputs, language=lang_inputs)
    else:
        results = []

    out_chunks: List[Dict[str, Any]] = []
    for ref, chunk_start, chunk_result in zip(valid_refs, valid_starts, results):
        aligned_words = [
            {
                "text": item.text,
                "start": round(chunk_start + float(item.start_time), 3),
                "end": round(chunk_start + float(item.end_time), 3),
            }
            for item in chunk_result
        ]
        out_chunks.append({**ref, "aligned_words": aligned_words})

    out = {
        "text": str(wav_transcript.get("text", "") if isinstance(wav_transcript, dict) else ""),
        "chunks": out_chunks,
    }
    save_json(out_path, out)
    return {"sample_dir": str(sample_dir), "status": "ok", "out_json": str(out_path), "num_chunks": len(out_chunks)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add word timestamps to wav_transcript.json without ASR.")
    parser.add_argument("--sample_dir", default="", help="Single sample directory containing output.wav + wav_transcript.json.")
    parser.add_argument("--wav_dirs_json", default="", help="JSON list of sample directories.")
    parser.add_argument("--batch_summary_json", default="", help="Inference batch_summary.json.")
    parser.add_argument("--output_root", default="", help="Fallback root for stale batch_summary output_dir paths.")
    parser.add_argument("--out_name", default="wav_transcript_aligned.json", help="Output filename inside each sample dir.")
    parser.add_argument("--align_model", default="/path/to/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--process_non_ok", action="store_true")
    parser.add_argument("--summary_json", default="", help="Optional path for per-sample alignment summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dirs = collect_sample_dirs(args)
    if not sample_dirs:
        raise ValueError("No sample directories found.")

    aligner = Qwen3ForcedAligner.from_pretrained(
        args.align_model,
        dtype=torch.bfloat16,
        device_map=args.device,
    )

    rows = []
    for idx, sample_dir in enumerate(sample_dirs, start=1):
        print(f"[{idx}/{len(sample_dirs)}] {sample_dir}", flush=True)
        row = align_one_sample(sample_dir, aligner, args.out_name, bool(args.skip_existing))
        rows.append(row)
        print(f"  {row.get('status')} -> {row.get('out_json', '')}", flush=True)

    summary = {
        "total": len(rows),
        "ok": sum(1 for r in rows if r.get("status") == "ok"),
        "skipped_existing": sum(1 for r in rows if r.get("status") == "skipped_existing"),
        "missing_input": sum(1 for r in rows if r.get("status") == "missing_input"),
        "items": rows,
    }
    if args.summary_json:
        save_json(Path(args.summary_json).resolve(), summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
