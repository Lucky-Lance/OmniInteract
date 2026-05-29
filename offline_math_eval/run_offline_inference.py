"""Step 2: Run offline math-question inference using MiniCPM-o 4.5's
half-duplex chat-with-video mode.

Reads eval_manifest.json and, for each (clip, question), runs inference and
saves the model's answer.

Usage (single-GPU worker):
    CUDA_VISIBLE_DEVICES=0 python offline_math_eval/run_offline_inference.py \
        --manifest offline_math_eval/eval_manifest.json \
        --model_path /path/to/MiniCPM-o-4_5 \
        --output_dir offline_math_eval/results \
        --worker_id 0 --num_workers 1

Multi-GPU runs are orchestrated by launch.sh, which sets worker_id / num_workers.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel


def extract_frames_from_clip(video_path: str, max_frames: int = 64) -> list:
    """Extract frames from a clip; returns a list of PIL.Image."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0

    # Sample at 1 fps, capped at max_frames.
    target_fps = 1.0
    step = max(1, int(fps / target_fps))
    frame_indices = list(range(0, total_frames, step))
    if len(frame_indices) > max_frames:
        step2 = max(1, len(frame_indices) // max_frames)
        frame_indices = frame_indices[::step2][:max_frames]

    frames = []
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True,
                        help="path to eval_manifest.json")
    parser.add_argument("--model_path", required=True,
                        help="path to the MiniCPM-o 4.5 model")
    parser.add_argument("--output_dir", default="offline_math_eval/results")
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_frames", type=int, default=64,
                        help="max sampled frames per clip")
    parser.add_argument("--max_slice_nums", type=int, default=1,
                        help="MiniCPM image slice count (use 1 for video)")
    parser.add_argument("--do_sample", action="store_true",
                        help="enable sampling (default: greedy decoding)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="sampling temperature (only used with --do_sample)")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="enable thinking mode")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    my_items = [
        item for i, item in enumerate(manifest)
        if i % args.num_workers == args.worker_id
    ]
    print(f"[Worker {args.worker_id}] assigned {len(my_items)}/{len(manifest)} samples")

    if not my_items:
        print(f"[Worker {args.worker_id}] no tasks; exiting")
        return

    # Vision-only mode (no audio / TTS needed)
    print(f"[Worker {args.worker_id}] loading model {args.model_path} ...")
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=False,
        init_tts=False,
    )
    model.eval().cuda()
    print(f"[Worker {args.worker_id}] model loaded")

    results_path = output_dir / f"results_worker{args.worker_id}.jsonl"
    done_set = set()
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done_set.add(r.get("clip_name", ""))

    processed = 0
    skipped = 0
    failed = 0
    t_all = time.time()

    with open(results_path, "a", encoding="utf-8") as rf:
        for i, item in enumerate(my_items):
            clip_name = item["clip_name"]

            if clip_name in done_set:
                skipped += 1
                continue

            clip_path = item["clip_path"]
            question = item["question_text"]

            print(f"[Worker {args.worker_id}] [{i+1}/{len(my_items)}] {clip_name}")

            try:
                t0 = time.time()

                frames = extract_frames_from_clip(clip_path, max_frames=args.max_frames)
                if not frames:
                    raise RuntimeError(f"No frames extracted from {clip_path}")

                msgs = [{"role": "user", "content": frames + [question]}]

                chat_kwargs = dict(
                    msgs=msgs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    use_image_id=False,
                    max_slice_nums=args.max_slice_nums,
                    use_tts_template=False,
                    enable_thinking=args.enable_thinking,
                )
                if args.do_sample:
                    chat_kwargs["temperature"] = args.temperature

                answer = model.chat(**chat_kwargs)

                elapsed = time.time() - t0

                result = {
                    "clip_name": clip_name,
                    "source_video": item["source_video"],
                    "question_idx": item["question_idx"],
                    "question_text": question,
                    "ground_truth": item["ground_truth"],
                    "model_answer": answer,
                    "num_frames": len(frames),
                    "inference_sec": round(elapsed, 2),
                    "status": "ok",
                }
                rf.write(json.dumps(result, ensure_ascii=False) + "\n")
                rf.flush()
                processed += 1

                print(f"  -> {answer[:120]}... ({elapsed:.1f}s, {len(frames)} frames)")

            except Exception as e:
                failed += 1
                err_result = {
                    "clip_name": clip_name,
                    "source_video": item["source_video"],
                    "question_idx": item["question_idx"],
                    "question_text": question,
                    "ground_truth": item["ground_truth"],
                    "model_answer": "",
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                rf.write(json.dumps(err_result, ensure_ascii=False) + "\n")
                rf.flush()
                print(f"  -> ERROR: {e}")

            finally:
                torch.cuda.empty_cache()

    total_time = time.time() - t_all
    print(
        f"\n[Worker {args.worker_id}] done: {processed} ok, "
        f"{skipped} skipped, {failed} failed, elapsed {total_time/60:.1f} min"
    )


if __name__ == "__main__":
    main()
