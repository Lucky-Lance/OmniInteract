#!/usr/bin/env python3
"""Run ASR on output.wav and save word-level output.json.

The script expects one or more sample directories. Each sample directory should
contain output.wav. The result is written next to it as output.json, using the
same chunks schema as the model transcript files.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from glob import glob
from pathlib import Path
from typing import Iterable, List

from tqdm import tqdm


MODEL_NAME = ""


def _collect_audio_paths_from_dirs(directories: Iterable[str]) -> List[str]:
    audio_paths: List[str] = []
    for data_path in directories:
        audio_paths.extend(sorted(glob(os.path.join(str(data_path), f"{MODEL_NAME}*output.wav"))))

    seen = set()
    uniq: List[str] = []
    for path in audio_paths:
        if path in seen:
            continue
        seen.add(path)
        uniq.append(path)
    return uniq


def get_time_aligned_transcription(
    data_paths: List[str],
    asr_model_path: str,
    align_model_path: str,
    device: str,
) -> None:
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel

    audio_paths = _collect_audio_paths_from_dirs(data_paths)
    if not audio_paths:
        raise FileNotFoundError("No output.wav files found from provided directories.")

    print("Loading Qwen3-ASR and ForcedAligner models...")
    asr_model = Qwen3ASRModel.from_pretrained(
        asr_model_path,
        dtype=torch.bfloat16,
        device_map=device,
        forced_aligner=align_model_path,
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": device},
    )

    for audio_path in tqdm(audio_paths):
        print(f"\nProcessing: {audio_path}")
        waveform, sr = sf.read(audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_name = tmp.name
            sf.write(tmp_name, waveform, sr)

        try:
            results = asr_model.transcribe(
                audio=[tmp_name],
                language=None,
                return_time_stamps=True,
            )
        finally:
            os.unlink(tmp_name)

        result = results[0]
        chunks = []
        if hasattr(result, "time_stamps") and result.time_stamps:
            for item in result.time_stamps:
                word_text = str(getattr(item, "text", "") or "").strip()
                if not word_text:
                    continue
                start_val = float(getattr(item, "start_time", 0.0) or 0.0)
                end_val = float(getattr(item, "end_time", 0.0) or 0.0)
                if end_val > 10000:
                    start_val /= 1000.0
                    end_val /= 1000.0
                chunks.append({"text": word_text, "timestamp": [start_val, end_val]})

        output_dict = {"text": str(getattr(result, "text", "") or ""), "chunks": chunks}
        result_path = str(audio_path).replace(f"{MODEL_NAME}output.wav", "output.json")
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe output.wav and export time-aligned output.json.")
    parser.add_argument("--wav_dir", default="", help="Single sample directory containing output.wav.")
    parser.add_argument("--wav_dirs_json", default="", help="JSON list of sample directories.")
    parser.add_argument("--asr_model", default="/path/to/Qwen3-ASR-1.7B")
    parser.add_argument("--align_model", default="/path/to/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs: List[str] = []
    if args.wav_dir:
        dirs.append(args.wav_dir)
    if args.wav_dirs_json:
        with open(args.wav_dirs_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError("--wav_dirs_json must point to a JSON list of sample directories.")
        dirs.extend(str(x) for x in payload)
    if not dirs:
        raise ValueError("At least one of --wav_dir / --wav_dirs_json is required.")

    get_time_aligned_transcription(dirs, args.asr_model, args.align_model, args.device)


if __name__ == "__main__":
    main()
