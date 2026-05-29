#!/usr/bin/env python3
"""Batch data prep for precise_truncation.json.

This replaces the old xargs shell pipeline with a resumable Python runner. It
collects sample directories, assigns workers to GPUs, runs missing ASR output if
needed, and then runs precise truncation in per-sample subprocesses.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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

    matches = [p for p in output_root.rglob(raw.name) if p.is_dir()]
    if len(matches) == 1:
        return matches[0].resolve()
    return None


def iter_batch_output_dirs(batch_summary_json: Path, output_root: Optional[Path], process_non_ok: bool) -> Iterable[Path]:
    root = load_json(batch_summary_json)
    rows = root.get("results", []) if isinstance(root, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not process_non_ok and str(row.get("status", "") or "") != "ok":
            continue
        out_dir = resolve_output_dir(str(row.get("output_dir", "") or ""), output_root)
        if out_dir is not None:
            yield out_dir


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def collect_sample_dirs(args: argparse.Namespace, root: Path) -> List[Path]:
    dirs: List[Path] = []
    output_root = Path(args.output_root).resolve() if args.output_root else root / "outputs"

    for sample_dir in args.sample_dir:
        dirs.append(Path(sample_dir).resolve())

    if args.wav_dirs_json:
        payload = load_json(Path(args.wav_dirs_json).resolve())
        if not isinstance(payload, list):
            raise ValueError("--wav_dirs_json must point to a JSON list of directories.")
        dirs.extend(Path(str(x)).resolve() for x in payload)

    if args.batch_summary_json:
        dirs.extend(
            iter_batch_output_dirs(
                Path(args.batch_summary_json).resolve(),
                output_root,
                bool(args.process_non_ok),
            )
        )

    for model in args.model:
        model_root = output_root / model
        if model_root.exists():
            dirs.extend(p.parent.resolve() for p in model_root.rglob("output.wav"))

    uniq: List[Path] = []
    seen = set()
    for d in dirs:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
    return uniq


def validate_sample_dir(sample_dir: Path) -> Dict[str, Any]:
    audio = sample_dir / "output.wav"
    wav_transcript = sample_dir / "wav_transcript.json"
    asr_output = sample_dir / "output.json"
    precise = sample_dir / "precise_truncation.json"
    missing = [str(p) for p in (audio, wav_transcript) if not p.exists()]
    return {
        "sample_dir": str(sample_dir),
        "audio": str(audio),
        "wav_transcript": str(wav_transcript),
        "asr_output": str(asr_output),
        "precise_output": str(precise),
        "missing": missing,
    }


def run_cmd(cmd: List[str], env: Dict[str, str], dry_run: bool) -> Dict[str, Any]:
    started = time.time()
    if dry_run:
        return {"returncode": 0, "cmd": cmd, "elapsed_sec": 0.0, "stdout_tail": "", "stderr_tail": ""}
    proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "returncode": int(proc.returncode),
        "cmd": cmd,
        "elapsed_sec": round(time.time() - started, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def prep_one_sample(item: Dict[str, Any], args: argparse.Namespace, root: Path, gpu: str) -> Dict[str, Any]:
    sample_dir = Path(str(item["sample_dir"])).resolve()
    audio = sample_dir / "output.wav"
    wav_transcript = sample_dir / "wav_transcript.json"
    asr_output = sample_dir / "output.json"
    precise = sample_dir / "precise_truncation.json"
    row: Dict[str, Any] = {
        "sample_dir": str(sample_dir),
        "gpu": str(gpu),
        "status": "pending",
        "asr_output": str(asr_output),
        "precise_output": str(precise),
    }

    missing = [str(p) for p in (audio, wav_transcript) if not p.exists()]
    if missing:
        row["status"] = "missing_input"
        row["missing"] = missing
        return row

    if precise.exists() and precise.stat().st_size > 0 and not args.force_precise:
        row["status"] = "skipped_existing"
        return row

    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if (not asr_output.exists() or asr_output.stat().st_size == 0 or args.force_asr) and not args.no_asr:
        asr_cmd = [
            sys.executable,
            str(root / "eval/data_prep/ASR.py"),
            "--wav_dir",
            str(sample_dir),
            "--asr_model",
            str(args.asr_model),
            "--align_model",
            str(args.align_model),
            "--device",
            "cuda:0" if gpu else str(args.device),
        ]
        row["asr"] = run_cmd(asr_cmd, env, bool(args.dry_run))
        if row["asr"]["returncode"] != 0:
            row["status"] = "failed_asr"
            return row
    elif not asr_output.exists() or asr_output.stat().st_size == 0:
        row["status"] = "missing_asr_output"
        return row
    else:
        row["asr"] = {"status": "skipped_existing", "cmd": []}

    trunc_cmd = [
        sys.executable,
        str(root / "eval/data_prep/get_precise_truncation.py"),
        "--wav_transcript",
        str(wav_transcript),
        "--asr_output",
        str(asr_output),
        "--audio",
        str(audio),
        "--out",
        str(precise),
        "--align_model",
        str(args.align_model),
        "--device",
        "cuda:0" if gpu else str(args.device),
        "--tolerance_sec",
        str(args.tolerance_sec),
    ]
    row["precise"] = run_cmd(trunc_cmd, env, bool(args.dry_run))
    if row["precise"]["returncode"] != 0:
        row["status"] = "failed_precise"
        return row

    row["status"] = "dry_run" if args.dry_run else "ok"
    return row


def summarize(rows: List[Dict[str, Any]], total: int) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for row in rows:
        st = str(row.get("status", "unknown"))
        counts[st] = counts.get(st, 0) + 1
    failed = sum(v for k, v in counts.items() if k.startswith("failed_"))
    return {
        "total": int(total),
        "finished": len(rows),
        "counts": counts,
        "failed": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-generate precise_truncation.json.")
    parser.add_argument("--sample_dir", action="append", default=[], help="Sample dir. Can be passed multiple times.")
    parser.add_argument("--wav_dirs_json", default="", help="JSON list of sample directories.")
    parser.add_argument("--batch_summary_json", default="", help="Inference batch_summary.json.")
    parser.add_argument("--output_root", default="", help="outputs root. Default: <repo>/outputs.")
    parser.add_argument("--model", action="append", default=[], help="Model under output_root. Can be passed multiple times.")
    parser.add_argument("--num_workers", type=int, default=0, help="Parallel sample workers. Default: number of GPU ids.")
    parser.add_argument("--gpu_ids", default=os.getenv("GPU_IDS", "0,1,2,3,4,5,6,7,8,9"))
    parser.add_argument("--asr_model", default="/path/to/Qwen3-ASR-1.7B")
    parser.add_argument("--align_model", default="/path/to/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--device", default="cuda:0", help="Used only when no GPU ids are provided.")
    parser.add_argument("--tolerance_sec", type=float, default=0.15)
    parser.add_argument("--force_asr", action="store_true")
    parser.add_argument("--force_precise", action="store_true")
    parser.add_argument("--no_asr", action="store_true", help="Do not run ASR; require output.json to exist.")
    parser.add_argument("--process_non_ok", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary_json", default="", help="Default: <output_root>/data_prep_batch_summary.json.")
    parser.add_argument("--fail_fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    output_root = Path(args.output_root).resolve() if args.output_root else root / "outputs"
    if not args.sample_dir and not args.wav_dirs_json and not args.batch_summary_json and not args.model:
        args.model = ["qwen", "gemini", "minicpmo"]

    sample_dirs = collect_sample_dirs(args, root)
    if args.limit and args.limit > 0:
        sample_dirs = sample_dirs[: int(args.limit)]
    if not sample_dirs:
        raise ValueError("No sample directories found.")

    gpu_ids = parse_csv(args.gpu_ids)
    num_workers = int(args.num_workers) if int(args.num_workers) > 0 else max(1, len(gpu_ids) or 1)
    summary_path = Path(args.summary_json).resolve() if args.summary_json else output_root / "data_prep_batch_summary.json"

    rows: List[Dict[str, Any]] = []
    print(f"[info] samples={len(sample_dirs)} workers={num_workers} gpu_ids={','.join(gpu_ids) if gpu_ids else '(none)'}")
    print(f"[info] summary={summary_path}")

    work_items = []
    for idx, sample_dir in enumerate(sample_dirs):
        item = validate_sample_dir(sample_dir)
        item["item_index"] = idx + 1
        item["assigned_gpu"] = gpu_ids[idx % len(gpu_ids)] if gpu_ids else ""
        work_items.append(item)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_item = {
            executor.submit(prep_one_sample, item, args, root, str(item.get("assigned_gpu", ""))): item
            for item in work_items
        }
        for done_count, future in enumerate(as_completed(future_to_item), start=1):
            item = future_to_item[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "sample_dir": str(item.get("sample_dir", "")),
                    "gpu": str(item.get("assigned_gpu", "")),
                    "status": "failed_exception",
                    "error": str(exc),
                }
            row["item_index"] = item.get("item_index")
            rows.append(row)
            print(f"[{done_count}/{len(work_items)}] {row.get('status')} gpu={row.get('gpu')} {row.get('sample_dir')}", flush=True)
            save_json(summary_path, {"items": sorted(rows, key=lambda r: int(r.get("item_index", 10**9))), "summary": summarize(rows, len(work_items))})
            if args.fail_fast and str(row.get("status", "")).startswith("failed_"):
                break

    rows = sorted(rows, key=lambda r: int(r.get("item_index", 10**9)))
    summary = summarize(rows, len(work_items))
    save_json(summary_path, {"items": rows, "summary": summary})
    print(json.dumps(summary, ensure_ascii=False))
    if summary["failed"] > 0 and args.fail_fast:
        sys.exit(1)


if __name__ == "__main__":
    main()
