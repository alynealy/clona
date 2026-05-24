from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


DEFAULT_GRAMMATICAL_MODEL = "llama3.2"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
MAX_TEXT_CHARS_FOR_LLM = 10000
DEFAULT_EXPLANATION = (
    "The grammatical analysis did not find a clear signal, so the score remains moderate."
)
VALID_CONFIDENCE_LEVELS = {"low", "medium", "high"}


class GrammaticalResult(TypedDict):
    score: int
    confidence: Literal["low", "medium", "high"]
    reasons_for_rating: list[str]
    lowered_confidence_reasons: list[str]


class GrammaticalAgentState(TypedDict, total=False):
    text: str
    raw_response: str
    parsed_result: GrammaticalResult
    score: int
    confidence: Literal["low", "medium", "high"]
    reasons_for_rating: list[str]
    lowered_confidence_reasons: list[str]
    error: str | None


GRAMMATICAL_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert linguist analyzing text for grammar-based AI authorship signals. "
            "Evaluate only spelling errors, punctuation errors, formatting consistency, and the polish of the text. "
            "AI-generated writing often looks unusually clean, with consistent punctuation, few mistakes, and regular formatting. "
            "Human writing may contain small errors, uneven punctuation, natural variation, or imperfect formatting. "
            "Do not assume that correct grammar automatically means AI-written; the text may be carefully written or edited by a human. "
            "Return valid JSON only. Do not use markdown, code fences, or any extra text. "
            "Every string in your response must be written strictly in English, even when the input text is not English.",
        ),
        (
            "human",
            "Analyze the text below and return exactly this JSON object, with only these four keys:\n"
            "{{"
            "\"score\": 0, "
            "\"confidence\": \"low\", "
            "\"reasons_for_rating\": [\"short English reason\"], "
            "\"lowered_confidence_reasons\": [\"short English reason\"]"
            "}}\n\n"
            "Rules:\n"
            "- score must be an integer from 0 to 100.\n"
            "- A higher score means the grammar and punctuation make the text more likely to be AI-generated.\n"
            "- A lower score means visible errors or natural variation make the text less likely to be AI-generated.\n"
            "- confidence must be exactly one of: low, medium, high.\n"
            "- reasons_for_rating must contain short English bullet-point strings explaining the rating.\n"
            "- lowered_confidence_reasons must contain short English bullet-point strings explaining uncertainty; use an empty list if none apply.\n"
            "- Do not include any Romanian words or non-English explanation text.\n\n"
            "Text:\n{text}",
        ),
    ]
)


@dataclass
class GrammaticalAgent:
    model_name: str = DEFAULT_GRAMMATICAL_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    temperature: float = 0.0

    def __post_init__(self) -> None:
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GrammaticalAgentState)
        graph.add_node("analyze_grammar", self.analyze_grammar)
        graph.add_node("finalize_result", self.finalize_result)

        graph.add_edge(START, "analyze_grammar")
        graph.add_edge("analyze_grammar", "finalize_result")
        graph.add_edge("finalize_result", END)
        return graph.compile()

    def run(self, text: str) -> GrammaticalResult:
        final_state = self.graph.invoke({"text": text})
        return _normalize_result(final_state)

    def analyze_grammar(self, state: GrammaticalAgentState) -> GrammaticalAgentState:
        text = state.get("text", "").strip()
        if not text:
            return {
                "parsed_result": {
                    "score": 0,
                    "confidence": "low",
                    "reasons_for_rating": [
                        "The text is empty, so grammar and punctuation cannot be evaluated."
                    ],
                    "lowered_confidence_reasons": [
                        "There is not enough text to support a reliable grammatical judgment."
                    ],
                },
                "error": None,
            }

        prompt_text = text[:MAX_TEXT_CHARS_FOR_LLM]
        try:
            response = self.llm.invoke(
                GRAMMATICAL_ANALYSIS_PROMPT.format_messages(text=prompt_text)
            )
            raw_response = response.content if isinstance(response.content, str) else str(response.content)
            parsed_result = _parse_llm_json(raw_response)
            return {
                "raw_response": raw_response,
                "parsed_result": parsed_result,
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - depends on local Ollama availability
            return {
                "parsed_result": _fallback_result(text),
                "error": str(exc),
            }

    def finalize_result(self, state: GrammaticalAgentState) -> GrammaticalResult:
        result = state.get("parsed_result") or _fallback_result(state.get("text", ""))
        return _normalize_result(result)


def run_grammatical_agent(text: str) -> GrammaticalResult:
    return GrammaticalAgent().run(text)


def _parse_llm_json(raw_response: str) -> GrammaticalResult:
    payload = raw_response.strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        if not json_match:
            raise ValueError("The model did not return valid JSON.")
        parsed = json.loads(json_match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("The model response is not a JSON object.")

    return _normalize_result(parsed)


def _normalize_result(value: dict[str, Any]) -> GrammaticalResult:
    score = _coerce_score(value.get("score"))
    return {
        "score": score,
        "confidence": _coerce_confidence(value.get("confidence")),
        "reasons_for_rating": _clean_string_list(
            value.get("reasons_for_rating"),
            fallback=[_default_reason_for_score(score)],
        ),
        "lowered_confidence_reasons": _clean_string_list(
            value.get("lowered_confidence_reasons"),
            fallback=[],
        ),
    }


def _coerce_score(value: Any) -> int:
    try:
        score = round(float(value))
    except (TypeError, ValueError):
        score = 50
    return max(0, min(100, score))


def _coerce_confidence(value: Any) -> Literal["low", "medium", "high"]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in VALID_CONFIDENCE_LEVELS:
            return normalized  # type: ignore[return-value]
    return "low"


def _clean_string_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = []

    cleaned_items = []
    for item in raw_items:
        if not isinstance(item, str):
            continue

        cleaned = " ".join(item.strip().split())
        if cleaned:
            cleaned_items.append(cleaned[:240])

    return cleaned_items[:5] if cleaned_items else fallback


def _default_reason_for_score(score: int) -> str:
    if score >= 70:
        return "The text appears unusually clean and consistent from a grammar and punctuation perspective."
    if score <= 40:
        return "The text contains visible grammar or punctuation variation that feels more human."
    return DEFAULT_EXPLANATION


def _fallback_result(text: str) -> GrammaticalResult:
    stripped_text = text.strip()
    if not stripped_text:
        return {
            "score": 0,
            "confidence": "low",
            "reasons_for_rating": [
                "The text is empty, so grammar and punctuation cannot be evaluated."
            ],
            "lowered_confidence_reasons": [
                "There is not enough text to support a reliable grammatical judgment."
            ],
        }

    punctuation_count = sum(1 for char in stripped_text if char in ".,;:!?")
    obvious_spacing_errors = len(re.findall(r"\s+[,.!?;:]|[,.!?;:][^\s\)\]\"']", stripped_text))
    typo_like_patterns = len(re.findall(r"\b\w*(?:aaa|eee|iii|ooo|uuu)\w*\b", stripped_text, re.IGNORECASE))

    if obvious_spacing_errors or typo_like_patterns:
        return {
            "score": 35,
            "confidence": "medium",
            "reasons_for_rating": [
                "The text contains visible spelling or punctuation issues.",
                "These imperfections make the writing less typical of polished AI output.",
            ],
            "lowered_confidence_reasons": [
                "Grammar alone cannot prove whether the author was human or AI."
            ],
        }

    if punctuation_count > 0 and len(stripped_text) > 300:
        return {
            "score": 65,
            "confidence": "low",
            "reasons_for_rating": [
                "The text appears clean and consistently punctuated.",
                "There are few obvious grammar or formatting disruptions.",
            ],
            "lowered_confidence_reasons": [
                "A careful human writer can also produce polished grammar."
            ],
        }

    return {
        "score": 50,
        "confidence": "low",
        "reasons_for_rating": [DEFAULT_EXPLANATION],
        "lowered_confidence_reasons": [
            "The text is too limited for grammar and punctuation to provide a strong signal."
        ],
    }
