#!/usr/bin/env python3
"""Single-script precise char timestamps.

Pipeline in-memory:
1) Read wav_transcript.json + ASR output.json
2) Build precise truncated transcript by prefix matching
3) Forced-align precise chunks against audio
4) Save only final char-level timestamps JSON
"""

import argparse
import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from qwen_asr import Qwen3ForcedAligner


CN_DIGIT_MAP = {
    "零": "0", "〇": "0", "○": "0", "一": "1", "二": "2", "两": "2",
    "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
}
CN_UNIT_MAP = {"十": 10, "百": 100, "千": 1000, "万": 10000}
CN_NUMERAL_CHARS = set(CN_DIGIT_MAP.keys()) | set(CN_UNIT_MAP.keys())


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def detect_language(text: str) -> str:
    return "Chinese" if re.search(r"[\u4e00-\u9fff]", text) else "English"


def normalize_char(ch: str) -> str:
    out = []
    for c in unicodedata.normalize("NFKC", ch):
        if c in CN_DIGIT_MAP:
            out.append(CN_DIGIT_MAP[c])
            continue
        cat = unicodedata.category(c)
        if c.isspace() or cat.startswith("P") or cat.startswith("S"):
            continue
        out.append(c.lower())
    return "".join(out)


def normalize_text(text: str) -> str:
    return "".join(normalize_char(ch) for ch in text)


def normalize_with_mapping(text: str) -> Tuple[str, List[int]]:
    norm_chars: List[str] = []
    mapping: List[int] = []
    for i, ch in enumerate(text):
        norm = normalize_char(ch)
        if not norm:
            continue
        for c in norm:
            norm_chars.append(c)
            mapping.append(i)
    return "".join(norm_chars), mapping


def longest_common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def extend_trailing_nonsemantic(original: str, end_idx: int) -> int:
    i = end_idx
    while i < len(original):
        if normalize_char(original[i]):
            break
        i += 1
    return i


def cn_numeral_to_arabic_str(token: str) -> Optional[str]:
    if not token or any(ch not in CN_NUMERAL_CHARS for ch in token):
        return None
    has_unit = any(ch in CN_UNIT_MAP for ch in token)
    if not has_unit:
        return "".join(CN_DIGIT_MAP[ch] for ch in token)

    total, section, number = 0, 0, 0
    for ch in token:
        if ch in CN_DIGIT_MAP:
            number = int(CN_DIGIT_MAP[ch])
            continue
        unit = CN_UNIT_MAP[ch]
        if unit == 10000:
            section = (section + number) * unit
            total += section
            section = 0
            number = 0
        else:
            if number == 0:
                number = 1
            section += number * unit
            number = 0
    return str(total + section + number)


def normalize_asr_chunks_with_mapping(asr_chunks: List[Dict]) -> List[Dict]:
    normalized: List[Dict] = []
    i = 0
    while i < len(asr_chunks):
        token = str(asr_chunks[i].get("text", "")).strip()
        if token and all(ch in CN_NUMERAL_CHARS for ch in token):
            j = i
            run_tokens: List[str] = []
            run_indices: List[int] = []
            while j < len(asr_chunks):
                nxt = str(asr_chunks[j].get("text", "")).strip()
                if not nxt or any(ch not in CN_NUMERAL_CHARS for ch in nxt):
                    break
                run_tokens.append(nxt)
                run_indices.append(j)
                j += 1
            merged = "".join(run_tokens)
            arabic = cn_numeral_to_arabic_str(merged)
            normalized.append({"text": arabic if arabic is not None else merged, "src_indices": run_indices})
            i = j
            continue

        if token:
            normalized.append({"text": token, "src_indices": [i]})
        i += 1
    return normalized


def collect_asr_chunks_in_window(asr_chunks: List[Dict], start: float, end: float, tol: float) -> List[Dict]:
    out: List[Dict] = []
    for c in asr_chunks:
        ts = c.get("timestamp", [None, None])
        if not isinstance(ts, list) or len(ts) != 2:
            continue
        ws, we = ts
        if ws is None or we is None:
            continue
        if (we >= start - tol) and (ws <= end + tol):
            text = str(c.get("text", "")).strip()
            if text:
                out.append({"text": text, "timestamp": [float(ws), float(we)]})
    return out


def prefix_match_truncated_text(original_text: str, asr_chunks: List[Dict]) -> str:
    if not original_text or not asr_chunks:
        return ""

    original_norm, mapping = normalize_with_mapping(original_text)
    if not original_norm:
        return ""

    pos = 0
    matched_end = 0
    normalized_chunks = normalize_asr_chunks_with_mapping(asr_chunks)

    for ch in normalized_chunks:
        w_norm = normalize_text(ch["text"])
        if not w_norm:
            continue

        idx = original_norm.find(w_norm, pos)
        if idx >= 0:
            pos = idx + len(w_norm)
            matched_end = max(matched_end, pos)
            continue

        lcp = longest_common_prefix_len(original_norm[pos:], w_norm)
        if lcp > 0:
            pos += lcp
            matched_end = max(matched_end, pos)

    if matched_end <= 0:
        return ""

    end_orig = mapping[matched_end - 1] + 1
    end_orig = extend_trailing_nonsemantic(original_text, end_orig)
    return original_text[:end_orig]


def build_precise_chunks(wav_transcript: Dict, asr_output: Dict, tolerance_sec: float) -> Dict:
    wav_chunks = wav_transcript.get("chunks", [])
    asr_chunks = asr_output.get("chunks", [])

    precise_chunks: List[Dict] = []
    for ch in wav_chunks:
        text = ch.get("text", "")
        ts = ch.get("timestamp", [None, None])
        if not isinstance(ts, list) or len(ts) != 2 or ts[0] is None or ts[1] is None:
            continue

        start, end = float(ts[0]), float(ts[1])
        asr_window = collect_asr_chunks_in_window(asr_chunks, start, end, tolerance_sec)
        truncated = prefix_match_truncated_text(text, asr_window)

        if truncated.strip():
            precise_chunks.append({"text": truncated, "timestamp": [start, end]})

    out_text = " ".join([c["text"].strip() for c in precise_chunks if c["text"].strip()]).strip()
    return {"text": out_text, "chunks": precise_chunks}


def force_align_precise_chunks(precise: Dict, audio_path: str, model_path: str, device: str) -> Dict:
    aligner = Qwen3ForcedAligner.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
    )

    wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    audio_inputs = []
    text_inputs = []
    lang_inputs = []
    valid_chunk_starts: List[float] = []
    valid_chunk_refs: List[Dict] = []

    for chunk in precise.get("chunks", []):
        start_time, end_time = chunk["timestamp"]
        text = chunk.get("text", "")
        if not text.strip() or end_time <= start_time:
            continue

        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr)
        wav_slice = np.asarray(wav[start_idx:end_idx], dtype=np.float32)

        audio_inputs.append((wav_slice, sr))
        text_inputs.append(text)
        lang_inputs.append(detect_language(text))
        valid_chunk_starts.append(float(start_time))
        valid_chunk_refs.append(chunk)

    if not audio_inputs:
        return {"text": precise.get("text", ""), "chunks": []}

    results = aligner.align(audio=audio_inputs, text=text_inputs, language=lang_inputs)

    final_chunks: List[Dict] = []
    for chunk_ref, chunk_start_time, chunk_result in zip(valid_chunk_refs, valid_chunk_starts, results):
        aligned_words: List[Dict] = []
        for item in chunk_result:
            aligned_words.append(
                {
                    "text": item.text,
                    "start": round(chunk_start_time + float(item.start_time), 3),
                    "end": round(chunk_start_time + float(item.end_time), 3),
                }
            )
        final_chunks.append(
            {
                "text": chunk_ref["text"],
                "timestamp": chunk_ref["timestamp"],
                "aligned_words": aligned_words,
            }
        )

    return {"text": precise.get("text", ""), "chunks": final_chunks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final precise char-level timestamps in one script")
    parser.add_argument("--wav_transcript", type=str, required=True, help="Path to wav_transcript.json")
    parser.add_argument("--asr_output", type=str, required=True, help="Path to ASR output.json")
    parser.add_argument("--audio", type=str, required=True, help="Path to output.wav")
    parser.add_argument("--out", type=str, required=True, help="Final output JSON path")
    parser.add_argument("--align_model", type=str, default="/path/to/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tolerance_sec", type=float, default=0.15)
    args = parser.parse_args()

    wav_transcript = load_json(args.wav_transcript)
    asr_output = load_json(args.asr_output)

    precise = build_precise_chunks(wav_transcript, asr_output, args.tolerance_sec)
    final = force_align_precise_chunks(precise, args.audio, args.align_model, args.device)

    save_json(args.out, final)
    print(f"[OK] {args.out}")


if __name__ == "__main__":
    main()
