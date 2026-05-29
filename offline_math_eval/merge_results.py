"""Step 3: Merge per-worker inference results into a single file.

Output: offline_math_eval/all_results.json with one record per sample:
    - clip_name / clip_path
    - source_video
    - question_idx (0 = Q1 first minute, 1 = Q2 last minute)
    - question_text
    - ground_truth
    - model_answer
    - num_frames / inference_sec / status

Usage:
    python offline_math_eval/merge_results.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="offline_math_eval/results")
    parser.add_argument("--manifest", default="offline_math_eval/eval_manifest.json")
    parser.add_argument("--output", default="offline_math_eval/all_results.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Read every worker's results
    all_results = []
    for jsonl_file in sorted(results_dir.glob("results_worker*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_results.append(json.loads(line))

    # Deduplicate by clip_name (resumes may produce duplicates)
    seen = set()
    unique = []
    for r in all_results:
        key = r.get("clip_name", "")
        if key not in seen:
            seen.add(key)
            unique.append(r)
    all_results = unique

    # Pull clip_path metadata from the manifest
    manifest_lookup = {}
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        for item in manifest:
            manifest_lookup[item["clip_name"]] = item

    for r in all_results:
        clip_name = r.get("clip_name", "")
        if clip_name in manifest_lookup:
            r["clip_path"] = manifest_lookup[clip_name].get("clip_path", "")
            r["clip_start_sec"] = manifest_lookup[clip_name].get("clip_start_sec", 0)
            r["clip_duration_sec"] = manifest_lookup[clip_name].get("clip_duration_sec", 60)
            r["video_duration_sec"] = manifest_lookup[clip_name].get("video_duration_sec", 0)

    ok = sum(1 for r in all_results if r.get("status") == "ok")
    err = sum(1 for r in all_results if r.get("status") == "error")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Merged: {len(all_results)} records")
    print(f"  ok:     {ok}")
    print(f"  failed: {err}")

    # Coverage check vs manifest
    if manifest_path.exists():
        total_expected = len(manifest)
        covered = sum(1 for item in manifest if item["clip_name"] in seen)
        missing = [item["clip_name"] for item in manifest if item["clip_name"] not in seen]
        print(f"  coverage: {covered}/{total_expected}")
        if missing:
            print(f"  missing {len(missing)}:")
            for m in missing[:10]:
                print(f"    {m}")
            if len(missing) > 10:
                print(f"    ... {len(missing)} total")

    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
