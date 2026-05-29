#!/usr/bin/env python3
"""OpenAI-compatible API judge for unified IA-QTF1 scoring."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from common import clamp, to_bool


class JudgeAPICallError(RuntimeError):
    pass


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    pos = str(text).find("{")
    while pos != -1:
        try:
            obj, _ = decoder.raw_decode(str(text)[pos:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        pos = str(text).find("{", pos + 1)
    return None


def extract_first_float(text: str) -> Optional[float]:
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


# The three prompt templates below are the verbatim English versions of the
# Chinese prompts used in the paper experiments. They match Listings 1-3 in
# OmniInteract Appendix Sec.~\ref{sec:judge_protocol}.

EARLY_SYSTEM_PROMPT = (
    "You are a streaming voice assistant evaluation judge. "
    "Judge only based on the given text. "
    "Output must be parseable JSON with no other text."
)

EARLY_USER_TEMPLATE = """Determine whether the early output between start and t_a is an early hallucination.

[scene_type] {scene_type}
[slot] slot_id={slot_id},
  turn_index={turn_index},
  step_index={step_index},
  boundary_type={boundary_type},
  is_interrupted={is_interrupted}
[question] {question}
[current_gt_answer] {gt_answer}
[full_chunk_context] {full_context}
[early_actual_text] {actual_text}

Rules:
1. Greetings, confirmations, waiting, brief observations, and follow-up phrases -> Neutral.
2. If the model starts substantively answering, guessing unseen info, revealing future steps, or making definitive factual claims -> FP.
3. For 1QnA first step, reciting the full procedure before acting -> FP.
4. score is interaction quality 0-1 when Neutral; 0 when hallucination.

Output JSON:
{{"flag":"Neutral|FP_Hallucination",
 "score":float 0-1,
 "rationale":"one sentence"}}"""


CORE_SYSTEM_PROMPT = (
    "You are a strict streaming voice assistant core-answer evaluation judge. "
    "Judge only based on the given text and reference answer. "
    "Output must be parseable JSON with no other text."
)

CORE_USER_TEMPLATE = """Score the core output after t_a.

[scene_type] {scene_type}
[slot] slot_id={slot_id},
  turn_index={turn_index},
  step_index={step_index},
  boundary_type={boundary_type},
  is_interrupted={is_interrupted}
[question] {question}
[current_gt_answer] {gt_answer}
[future_gt_answers_or_steps]
  {future_answers}
[full_chunk_context] {full_context}
[core_actual_text] {actual_text}

Rules:
1. score 0-1: correctness and coverage of core_actual_text vs gt_answer.
2. Off-topic, factual errors, or missing key answer -> low score.
3. 1QnA: reward only current-step info; penalize spoiling future steps or skipping the current step.
4. If score > 0, extract the earliest contiguous substring from core_actual_text that establishes the answer as trigger_phrase.
5. trigger_phrase must be a verbatim substring; empty if score == 0.

Output JSON:
{{"score":float 0-1,
 "trigger_phrase":"substring or empty",
 "spoiler":true|false,
 "rationale":"one sentence"}}"""


INTERRUPT_PARTIAL_SYSTEM_PROMPT = (
    "You are a strict evaluator for interrupted voice-assistant answers. "
    "Judge only from the provided text. "
    "Return valid JSON only, with no extra text."
)


INTERRUPT_PARTIAL_USER_TEMPLATE = """Evaluate the quality of the assistant output that was already spoken before or around an interruption.

[Task]
The assistant was answering, but the interaction was interrupted before completion. The assistant was not required to complete the full original answer. Score whether the content already spoken is relevant, correct, and useful for the current ground-truth answer.

[Question] {question}
[Ground Truth Answer] {gt_answer}
[Assistant Output Already Spoken]
{actual_text}

Scoring Rules:
1. Score from 0 to 1.
2. Do not penalize incompleteness: a partial answer can receive a high score if the spoken part is correct and useful.
3. Score high when the spoken content overlaps with, paraphrases, or conveys useful parts of the ground truth.
4. Score low for acknowledgments or prefaces without substantive answer content.
5. Score low for wrong-question, irrelevant, or generic-filler output.
6. hallucination=true if the output contains clear incorrect facts, wrong target content, or unsupported content.
7. Ignore overflow duration when scoring quality; spill is measured separately.

Output JSON:
{{"score":float 0-1,
 "hallucination":true|false,
 "rationale":"one sentence"}}"""


class APIJudge:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_sec: float = 60.0,
    ) -> None:
        self.api_url = str(api_url).strip()
        self.api_key = str(api_key or os.getenv("JUDGE_API_KEY", "")).strip()
        self.model = str(model).strip()
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.timeout_sec = float(timeout_sec)
        if not self.api_url:
            raise ValueError("api_url is required.")
        if not self.model:
            raise ValueError("model is required.")
        if not self.api_key:
            raise ValueError("API judge requires --judge_api_key or env JUDGE_API_KEY.")

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        import requests

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "stream": False,
            "max_tokens": self.max_tokens,
        }
        try:
            resp = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False),
                timeout=self.timeout_sec,
            )
        except requests.exceptions.RequestException as exc:
            raise JudgeAPICallError(f"API request error: {exc}") from exc
        if resp.status_code != 200:
            raise JudgeAPICallError(f"API status={resp.status_code}, body={resp.text[:1200]}")
        try:
            obj = resp.json()
            return str(obj["choices"][0]["message"]["content"])
        except Exception as exc:
            raise JudgeAPICallError(f"Unexpected API response: {resp.text[:1200]}") from exc

    def judge_early(self, slot: Dict[str, Any], full_context: str, actual_text: str) -> Tuple[str, float, Dict[str, Any]]:
        raw = self._generate(
            EARLY_SYSTEM_PROMPT,
            EARLY_USER_TEMPLATE.format(
                scene_type=slot.get("scene_type", ""),
                slot_id=slot.get("slot_id"),
                turn_index=slot.get("turn_index"),
                step_index=slot.get("step_index"),
                boundary_type=slot.get("boundary_type", ""),
                is_interrupted=str(bool(slot.get("is_interrupted", False))).lower(),
                question=str(slot.get("question_text", "") or "").strip(),
                gt_answer=str(slot.get("gt_answer", "") or "").strip(),
                full_context=str(full_context or "").strip(),
                actual_text=str(actual_text or "").strip(),
            ),
        )
        obj = extract_first_json_object(raw) or {}
        flag_raw = str(obj.get("flag", "") if obj else raw).strip()
        norm = flag_raw.lower().replace("-", "_").replace(" ", "")
        if "hallucination" in norm:
            category = "hallucination"
            score = 0.0
        elif "neutral" in norm:
            category = "neutral"
            try:
                score = clamp(float(obj.get("score", 1.0)))
            except Exception:
                score = 1.0
        else:
            category = "neutral"
            score = 0.0
        return category, score, {
            "parse_source": "llm_json" if obj else "llm_parse_failed",
            "raw": raw,
            "flag_raw": flag_raw,
            "rationale": str(obj.get("rationale", "") or "") if obj else "",
        }

    def judge_core(
        self,
        slot: Dict[str, Any],
        full_context: str,
        actual_text: str,
        future_answers: str,
    ) -> Tuple[float, Dict[str, Any]]:
        raw = self._generate(
            CORE_SYSTEM_PROMPT,
            CORE_USER_TEMPLATE.format(
                scene_type=slot.get("scene_type", ""),
                slot_id=slot.get("slot_id"),
                turn_index=slot.get("turn_index"),
                step_index=slot.get("step_index"),
                boundary_type=slot.get("boundary_type", ""),
                is_interrupted=str(bool(slot.get("is_interrupted", False))).lower(),
                question=str(slot.get("question_text", "") or "").strip(),
                gt_answer=str(slot.get("gt_answer", "") or "").strip(),
                future_answers=str(future_answers or "").strip(),
                full_context=str(full_context or "").strip(),
                actual_text=str(actual_text or "").strip(),
            ),
        )
        obj = extract_first_json_object(raw) or {}
        score: Optional[float] = None
        parse_source = "llm_parse_failed"
        if obj:
            try:
                score = clamp(float(obj.get("score", 0.0)))
                parse_source = "llm_json"
            except Exception:
                score = None
        if score is None:
            raw_float = extract_first_float(raw)
            score = clamp(raw_float) if raw_float is not None else 0.0
            parse_source = "llm_float_text" if raw_float is not None else "llm_parse_failed"
        return score, {
            "parse_source": parse_source,
            "raw": raw,
            "llm_score_raw": score,
            "trigger_phrase": str(obj.get("trigger_phrase", "") or "").strip() if obj else "",
            "spoiler": to_bool(obj.get("spoiler", False)) if obj else False,
            "rationale": str(obj.get("rationale", "") or "").strip() if obj else "",
        }

    def judge_interrupted_partial(self, slot: Dict[str, Any], actual_text: str) -> Dict[str, Any]:
        raw = self._generate(
            INTERRUPT_PARTIAL_SYSTEM_PROMPT,
            INTERRUPT_PARTIAL_USER_TEMPLATE.format(
                question=str(slot.get("question_text", "") or "").strip(),
                gt_answer=str(slot.get("gt_answer", "") or "").strip(),
                actual_text=str(actual_text or "").strip(),
            ),
        )
        obj = extract_first_json_object(raw) or {}
        try:
            score = clamp(float(obj.get("score", 0.0)))
        except Exception:
            score = 0.0
        return {
            "score": score,
            "hallucination": to_bool(obj.get("hallucination", False)) if obj else False,
            "rationale": str(obj.get("rationale", "") or "").strip() if obj else "",
            "raw": raw,
            "parse_source": "llm_json" if obj else "llm_parse_failed",
        }
