#!/usr/bin/env python3
"""Batch runner for unified IA-QTF1 evaluation."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from build_slots import build_slots
from common import load_json, normalize_chunks, save_json
from compute_unified_eval import compute_unified_eval
from llm_judge import APIJudge, JudgeAPICallError
from match_slots import build_matching
from summarize_unified_eval import build_summary


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


_SCENE_MAP_CACHE: Dict[str, str] = {}


def _load_scene_map(repo: Path) -> Dict[str, str]:
    """Build {subset-relative-video-path: scene_type} from each subset's
    video_json_map.json.

    The anonymized release flattens speaker/scene structure, so per-video scene
    metadata is carried inside data/<subset>/video_json_map.json (see paper
    Tab.~\\ref{tab:data_statistics}: 638 real-time / 184 proactive / 240 nested
    + 368 monitoring slots).
    """
    if _SCENE_MAP_CACHE:
        return _SCENE_MAP_CACHE
    for subset in ("1q1a", "1q1a_math", "1qna"):
        manifest = repo / "data" / subset / "video_json_map.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        entries = data.get("entries", []) if isinstance(data, dict) else []
        for entry in entries:
            video_rel = str(entry.get("video", "")).replace("\\", "/").lstrip("./")
            scene = str(entry.get("scene_type", "")).strip()
            if not video_rel or not scene:
                continue
            _SCENE_MAP_CACHE[f"{subset}/{video_rel}"] = scene
    return _SCENE_MAP_CACHE


def _scene_from_video(video: str, repo: Optional[Path] = None) -> str:
    v = str(video).replace("\\", "/")
    if "/1qna/" in v.lower() or "videos_bench" in v:
        return "1QnA"
    scene_map = _load_scene_map(repo or _repo_root())
    for subset in ("1q1a_math", "1q1a"):
        marker = f"{subset}/"
        idx = v.rfind(marker)
        if idx >= 0:
            key = v[idx:]
            scene = scene_map.get(key)
            if scene:
                return scene
    if "/nested/" in v or "/nested2/" in v or "__nested__" in v or "__nested2__" in v:
        return "nested"
    return "multi_turn"


def _infer_gt_from_video(repo: Path, video: str, scene_type: str) -> Optional[Path]:
    """Map a model-input video path to its annotation JSON under data/.

    Supported layouts (matches the anonymized release):
      data/1q1a/videos/<stem>.mp4        -> data/1q1a/annotations/<stem>.json
      data/1q1a_math/videos/<stem>.mp4   -> data/1q1a_math/annotations/<stem>.json
      data/1qna/videos_bench/<ds>/<stem>.mp4
                                          -> data/1qna/annotations/<ds>/<stem>.json
    """
    v = str(video).replace("\\", "/")
    parts = v.split("/")

    if scene_type == "1QnA":
        if "videos_bench" in parts:
            idx = parts.index("videos_bench")
            rel = parts[idx + 1 :]
            if rel:
                rel[-1] = Path(rel[-1]).with_suffix(".json").name
                cand = repo / "data" / "1qna" / "annotations" / Path(*rel)
                if cand.exists():
                    return cand
        return None

    for subset in ("1q1a_math", "1q1a"):
        if subset in parts:
            idx = parts.index(subset)
            rel = parts[idx + 1 :]
            if not rel:
                continue
            if rel and rel[0] == "videos":
                rel = rel[1:]
            if not rel:
                continue
            rel[-1] = Path(rel[-1]).with_suffix(".json").name
            cand = repo / "data" / subset / "annotations" / Path(*rel)
            if cand.exists():
                return cand
    return None


_SUBSETS = ("1q1a_math", "1q1a", "1qna")


def _resolve_output_dir(raw_output_dir: str, output_root: Optional[Path]) -> Optional[Path]:
    raw = Path(str(raw_output_dir))
    if raw.exists():
        return raw.resolve()
    if output_root:
        # Prefer the subset whose prefix is already present in the input path
        # (e.g. "outputs/qwen/1q1a_math/videos__0001" -> try 1q1a_math first).
        # This prevents the 0001-name collision between 1q1a and 1q1a_math from
        # silently routing math outputs to the general 1q1a subset.
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


def _model_json_from_output_dir(output_dir: Path, model_json_name: str = "") -> Optional[Path]:
    if model_json_name:
        cand = output_dir / model_json_name
        return cand if cand.exists() else None
    for name in ("precise_truncation.json", "wav_transcript_aligned.json", "wav_transcript.json", "output.json"):
        cand = output_dir / name
        if cand.exists():
            return cand
    return None


def _items_from_batch_summary(path: Path, repo: Path, output_root: Optional[Path], model_json_name: str = "") -> List[Dict[str, Any]]:
    root = load_json(path)
    rows = root.get("results", []) if isinstance(root, dict) else []
    out: List[Dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict) or str(item.get("status", "ok")) != "ok":
            continue
        video = str(item.get("video", "") or "")
        scene = _scene_from_video(video, repo)
        gt = _infer_gt_from_video(repo, video, scene)
        out_dir = _resolve_output_dir(str(item.get("output_dir", "") or ""), output_root)
        model_json = _model_json_from_output_dir(out_dir, model_json_name) if out_dir is not None else None
        if gt is None or model_json is None:
            out.append({"status": "missing_input", "video": video, "scene_type": scene, "gt_json": str(gt) if gt else "", "model_json": str(model_json) if model_json else ""})
            continue
        sample_id = out_dir.name if out_dir is not None else Path(video).stem
        out.append(
            {
                "status": "ready",
                "sample_id": sample_id,
                "video": video,
                "scene_type": scene,
                "gt_json": str(gt.resolve()),
                "model_json": str(model_json.resolve()),
            }
        )
    return out


def _items_from_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return list(_read_jsonl(path))
    root = load_json(path)
    if isinstance(root, list):
        return [x for x in root if isinstance(x, dict)]
    if isinstance(root, dict) and isinstance(root.get("items"), list):
        return [x for x in root["items"] if isinstance(x, dict)]
    raise ValueError("Manifest must be JSONL, JSON list, or JSON object with items[].")


def _eval_one(
    item: Dict[str, Any],
    judge: APIJudge,
    out_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    scene_type = str(item["scene_type"])
    gt_path = Path(str(item["gt_json"])).resolve()
    model_path = Path(str(item["model_json"])).resolve()
    sample_id = str(item.get("sample_id") or gt_path.stem)
    metric_path = out_dir / f"{sample_id}.unified_eval.json"
    match_path = out_dir / f"{sample_id}.unified_match.json"

    gt_root = load_json(gt_path)
    model_root = load_json(model_path)
    slots = [s.to_dict() for s in build_slots(gt_root, scene_type, float(args.last_slot_tail_sec))]
    match_root = build_matching(slots, normalize_chunks(model_root))
    match_root["meta"] = {"gt_json": str(gt_path), "model_json": str(model_path), "scene_type": scene_type}
    if not args.dry_run:
        save_json(match_path, match_root)

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
        "gt_json": str(gt_path),
        "model_json": str(model_path),
        "match_json": str(match_path),
        "scene_type": scene_type,
        "judge_backend": "api",
        "judge_api_url": str(args.judge_api_url),
        "judge_api_model": str(args.judge_api_model),
    }
    if not args.dry_run:
        save_json(metric_path, result)
    return {
        "sample_id": sample_id,
        "status": result.get("status", "ok"),
        "scene_type": scene_type,
        "gt_json": str(gt_path),
        "model_json": str(model_path),
        "match_json": str(match_path),
        "metric_json": str(metric_path),
        "summary": result.get("summary", {}),
        "error": result.get("error"),
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = 0.0
    fp = 0
    fn = 0
    ok = 0
    failed = 0
    for row in rows:
        if row.get("status") != "ok":
            failed += 1
            continue
        s = row.get("summary", {}) or {}
        tp += float(s.get("Global_TP", 0.0))
        fp += int(s.get("Global_FP", 0))
        fn += int(s.get("Global_FN", 0))
        ok += 1
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * p * r / (p + r) if p + r > 0 else 0.0
    return {
        "num_items": len(rows),
        "success": ok,
        "failed_or_skipped": failed,
        "Global_TP": tp,
        "Global_FP": fp,
        "Global_FN": fn,
        "Precision": p,
        "Recall": r,
        "IA_QTF1": f1,
    }


class DryRunJudge:
    def judge_early(self, slot: Dict[str, Any], full_context: str, actual_text: str):
        return "neutral", 0.0, {"dry_run": True}

    def judge_core(self, slot: Dict[str, Any], full_context: str, actual_text: str, future_answers: str):
        return 0.0, {"dry_run": True, "trigger_phrase": ""}

    def judge_interrupted_partial(self, slot: Dict[str, Any], actual_text: str):
        return {"score": 0.0, "hallucination": False, "rationale": "", "dry_run": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch unified IA-QTF1 evaluation.")
    parser.add_argument("--manifest", default="", help="JSONL/JSON list with gt_json, model_json, scene_type.")
    parser.add_argument("--batch_summary_json", default="", help="Existing inference batch_summary.json.")
    parser.add_argument("--output_root", default="", help="Fallback output root containing per-sample directories.")
    parser.add_argument("--model_json_name", default="", help="When using --batch_summary_json, require this filename in each output_dir.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", help="Validate input and scheduling without API calls or writes.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of samples to evaluate concurrently. Each worker uses the same API key.",
    )
    parser.add_argument("--skip_existing", action="store_true")
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
    if not args.manifest and not args.batch_summary_json:
        raise ValueError("Provide --manifest or --batch_summary_json.")
    repo = _repo_root()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root).resolve() if args.output_root else None

    if args.manifest:
        items = _items_from_manifest(Path(args.manifest).resolve())
    else:
        items = _items_from_batch_summary(
            Path(args.batch_summary_json).resolve(),
            repo,
            output_root,
            str(args.model_json_name or ""),
        )
    if args.limit and args.limit > 0:
        items = items[: int(args.limit)]
    for item_index, item in enumerate(items, start=1):
        item["item_index"] = item_index

    if args.dry_run:
        judge = DryRunJudge()
    else:
        judge = APIJudge(
            api_url=str(args.judge_api_url),
            api_key=str(args.judge_api_key or os.getenv("JUDGE_API_KEY", "")),
            model=str(args.judge_api_model),
            max_tokens=int(args.judge_max_tokens),
            temperature=float(args.judge_temperature),
            timeout_sec=float(args.judge_api_timeout_sec),
        )

    rows: List[Dict[str, Any]] = []
    progress_path = out_dir / "unified_eval_progress.json"
    work_items: List[Dict[str, Any]] = []
    for item in items:
        if item.get("status") == "missing_input":
            rows.append(item)
            if not args.dry_run:
                save_json(progress_path, {"items": sorted(rows, key=lambda r: int(r.get("item_index", 10**9))), "summary": _aggregate(rows)})
            continue
        sample_id = str(item.get("sample_id") or Path(str(item.get("gt_json", "sample"))).stem)
        metric_path = out_dir / f"{sample_id}.unified_eval.json"
        if args.skip_existing and metric_path.exists():
            existing = load_json(metric_path)
            rows.append(
                {
                    **item,
                    "sample_id": sample_id,
                    "status": existing.get("status", "ok"),
                    "metric_json": str(metric_path),
                    "summary": existing.get("summary", {}),
                }
            )
            if not args.dry_run:
                save_json(progress_path, {"items": sorted(rows, key=lambda r: int(r.get("item_index", 10**9))), "summary": _aggregate(rows)})
            continue
        work_items.append(item)

    num_workers = max(1, int(args.num_workers))
    if num_workers == 1:
        for idx, item in enumerate(work_items, start=1):
            sample_id = str(item.get("sample_id") or Path(str(item.get("gt_json", "sample"))).stem)
            print(f"[{idx}/{len(work_items)}] {sample_id} ({item.get('scene_type')})")
            try:
                rows.append(_eval_one(item, judge, out_dir, args))
            except JudgeAPICallError as exc:
                rows.append({"sample_id": sample_id, "status": "judge_api_error", "error": str(exc), **item})
                if not args.dry_run:
                    save_json(progress_path, {"items": sorted(rows, key=lambda r: int(r.get("item_index", 10**9))), "summary": _aggregate(rows)})
                raise
            except Exception as exc:
                rows.append({"sample_id": sample_id, "status": "failed", "error": str(exc), **item})
            if not args.dry_run:
                save_json(progress_path, {"items": sorted(rows, key=lambda r: int(r.get("item_index", 10**9))), "summary": _aggregate(rows)})
    else:
        print(f"Running with num_workers={num_workers}, work_items={len(work_items)}")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_item = {executor.submit(_eval_one, item, judge, out_dir, args): item for item in work_items}
            for done_count, future in enumerate(as_completed(future_to_item), start=1):
                item = future_to_item[future]
                sample_id = str(item.get("sample_id") or Path(str(item.get("gt_json", "sample"))).stem)
                try:
                    row = future.result()
                except JudgeAPICallError as exc:
                    row = {"sample_id": sample_id, "status": "judge_api_error", "error": str(exc), **item}
                    rows.append(row)
                    if not args.dry_run:
                        save_json(progress_path, {"items": sorted(rows, key=lambda r: int(r.get("item_index", 10**9))), "summary": _aggregate(rows)})
                    raise
                except Exception as exc:
                    row = {"sample_id": sample_id, "status": "failed", "error": str(exc), **item}
                rows.append(row)
                print(f"[{done_count}/{len(work_items)}] done {sample_id}: {row.get('status')}")
                if not args.dry_run:
                    save_json(progress_path, {"items": sorted(rows, key=lambda r: int(r.get("item_index", 10**9))), "summary": _aggregate(rows)})

    summary = _aggregate(rows)
    rows = sorted(rows, key=lambda r: int(r.get("item_index", 10**9)))
    if not args.dry_run:
        # First save the batch rows as metadata, then rebuild the canonical
        # aggregate from slot-level *.unified_eval.json details. This keeps
        # resumed/skipped runs and fresh runs on the same metric definition.
        save_json(out_dir / "unified_eval_summary.json", {"items": rows, "summary": summary})
        canonical = build_summary(out_dir, out_dir / "unified_eval_summary.json")
        save_json(out_dir / "unified_eval_summary.json", canonical)
        summary = canonical["summary"]
        print(f"Saved batch summary to: {(out_dir / 'unified_eval_summary.json').resolve()}")
    print(f"TP={summary['Global_TP']:.6f} FP={summary['Global_FP']} FN={summary['Global_FN']} F1={summary['IA_QTF1']:.6f}")


if __name__ == "__main__":
    main()
