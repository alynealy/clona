from __future__ import annotations

import json
import logging
import os
import re
import statistics
from datetime import UTC, datetime
from dataclasses import dataclass, field
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)

WORD_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)
WHITESPACE_PATTERN = re.compile(r"\s+")
BULLET_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+", re.MULTILINE)
QUOTE_PATTERN = re.compile(r'^[\s>"\']+', re.MULTILINE)

DEFAULT_TEXT_MODEL = os.getenv("CHECKWISE_TEXT_MODEL", os.getenv("CHECKWISE_STATS_MODEL", "llama3.2:1b"))
DEFAULT_TEXT_OLLAMA_BASE_URL = os.getenv(
    "CHECKWISE_TEXT_OLLAMA_BASE_URL",
    os.getenv("CHECKWISE_STATS_OLLAMA_BASE_URL", "http://localhost:11434"),
)
DEFAULT_TEXT_OLLAMA_API_KEY = os.getenv("CHECKWISE_TEXT_OLLAMA_API_KEY") or os.getenv(
    "CHECKWISE_STATS_OLLAMA_API_KEY"
) or os.getenv("OLLAMA_API_KEY")

LINKING_WORD_PATTERNS: dict[str, str] = {
    "however": r"\bhowever\b",
    "therefore": r"\btherefore\b",
    "moreover": r"\bmoreover\b",
    "furthermore": r"\bfurthermore\b",
    "thus": r"\bthus\b",
    "in addition": r"\bin addition\b",
    "consequently": r"\bconsequently\b",
    "overall": r"\boverall\b",
    "indeed": r"\bindeed\b",
    "for example": r"\bfor example\b",
}
EXPRESSIVE_REPEAT_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "so", "to", "of", "in", "on", "at", "for", "with",
    "is", "are", "was", "were", "be", "been", "it", "this", "that", "these", "those", "i", "you",
    "he", "she", "we", "they",
}

MIN_WORDS_FOR_STRONG_VERDICT = 35
MIN_SENTENCES_FOR_STRONG_VERDICT = 3
SENTENCE_RANGE_HUMAN_THRESHOLD = 10
MAX_BULLET_POINTS = 5
LLM_TEXT_LIMIT = 2500
MAX_HIGHLIGHT_PHRASE_WORDS = 6
MAX_HIGHLIGHT_PHRASE_CHARS = 48
MAX_HIGHLIGHT_COUNT = 6
GENERIC_RECOVERY_OBSERVATION = "The wording detector returned an unstructured response, so the result was recovered conservatively."

COMMON_ENGLISH_WORDS = {
    "the", "and", "is", "are", "to", "of", "in", "that", "it", "for", "with", "as", "was",
    "were", "be", "on", "this", "by", "an", "or", "from", "at", "which", "but", "not", "have",
    "has", "had", "can", "will", "would", "there", "their", "about", "into", "more", "than", "also",
}

LINGUISTIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Judge the text only on wording, phrase naturalness, expressive style, and semantic coherence. "
            "Return valid JSON only. No markdown. No code fences. "
            "A human_likelihood_score of 1.0 means strongly human-written and 0.0 means strongly AI-like.",
        ),
        (
            "human",
            "Return exactly this JSON object shape with these keys only: "
            "{{\"human_likelihood_score\":0.0,\"natural_wording_score\":0.0,\"expressive_style_score\":0.0,"
            "\"coherence_score\":0.0,\"awkwardness_score\":0.0,\"bullet_points\":[],\"important_phrases\":[]}}.\n\n"
            "Text:\n{text}",
        ),
    ]
)


class LinguisticAssessment(BaseModel):
    human_likelihood_score: float = Field(ge=0.0, le=1.0)
    natural_wording_score: float = Field(ge=0.0, le=1.0)
    expressive_style_score: float = Field(ge=0.0, le=1.0)
    coherence_score: float = Field(ge=0.0, le=1.0)
    awkwardness_score: float = Field(ge=0.0, le=1.0)
    bullet_points: list[str] = Field(default_factory=list)
    important_phrases: list[str] = Field(default_factory=list)


LINGUISTIC_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "human_likelihood_score": (
        "human_likelihood_score",
        "score",
        "human_score",
        "likelihood_score",
        "human_probability",
    ),
    "natural_wording_score": ("natural_wording_score", "wording_score"),
    "expressive_style_score": ("expressive_style_score", "style_score", "expressiveness_score"),
    "coherence_score": ("coherence_score", "semantic_coherence_score"),
    "awkwardness_score": ("awkwardness_score", "awkward_score"),
    "bullet_points": ("bullet_points", "observations", "reasons"),
    "important_phrases": ("important_phrases", "influential_phrases", "phrases"),
}


@dataclass
class SentenceRecord:
    text: str
    start: int
    end: int
    word_count: int


@dataclass
class NormalizedText:
    original_text: str
    analysis_text: str
    character_count: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    sentences: list[SentenceRecord]
    bullet_ratio: float
    quote_ratio: float
    prose_ratio: float


@dataclass
class DetectionContext:
    normalized: NormalizedText
    language: str
    language_valid: bool
    eligibility: dict[str, Any]
    linguistic: dict[str, Any]
    sentence_length: dict[str, Any]
    linking_words: dict[str, Any]
    robustness: dict[str, Any]
    document_assessment: dict[str, Any]
    verdict: str
    percentage: int
    final_label: str
    bullet_points: list[str]
    highlights: list[dict[str, Any]]
    metrics: dict[str, Any]
    detector_details: dict[str, Any]
    why: list[str] = field(default_factory=list)
    weakens: list[str] = field(default_factory=list)
    segment_assessment: dict[str, Any] = field(default_factory=lambda: {"available": False, "spans": []})
    final_user_message: str = ""


class LinguisticDetector:
    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        base_url: str = DEFAULT_TEXT_OLLAMA_BASE_URL,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        config: dict[str, Any] = {
            "model": model_name,
            "base_url": base_url,
            "temperature": temperature,
        }
        if DEFAULT_TEXT_OLLAMA_API_KEY and "ollama.com" in base_url.lower():
            config["client_kwargs"] = {
                "headers": {
                    "Authorization": f"Bearer {DEFAULT_TEXT_OLLAMA_API_KEY}",
                }
            }

        self.llm = ChatOllama(**config)
        self.cache: dict[str, dict[str, Any]] = {}

    def score_text(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            return _fallback_linguistic_assessment("The text is empty.")
        if cleaned in self.cache:
            return self.cache[cleaned]

        try:
            raw_response = self.llm.invoke(LINGUISTIC_PROMPT.format_messages(text=cleaned[:LLM_TEXT_LIMIT]))
            raw_text = raw_response.content if isinstance(raw_response.content, str) else str(raw_response.content)
            LOGGER.debug("Raw linguistic detector output: %s", raw_text)
        except Exception as exc:
            LOGGER.warning("Linguistic detector model call failed.", exc_info=exc)
            payload = _fallback_linguistic_assessment(
                "The wording detector could not return a reliable structured result.",
                diagnostics=_extract_invoke_error_details(
                    exc=exc,
                    provider="ollama",
                    model_name=self.model_name,
                    base_url=self.base_url,
                ),
            )
            self.cache[cleaned] = payload
            return payload

        try:
            result = _parse_linguistic_assessment(raw_text)
            human_likelihood_score = float(max(0.0, min(1.0, result.human_likelihood_score)))
            style_support_score = max(
                0.0,
                min(
                    1.0,
                    (
                        result.natural_wording_score * 0.35
                        + result.expressive_style_score * 0.25
                        + result.coherence_score * 0.25
                        + (1.0 - result.awkwardness_score) * 0.15
                    ),
                ),
            )
            human_style_score = (human_likelihood_score * 0.65) + (style_support_score * 0.35)
            payload = {
                "available": True,
                "status": "ok",
                "human_style_score": round(human_style_score, 3),
                "ai_score": round(1.0 - human_style_score, 3),
                "raw_score": round(human_likelihood_score, 3),
                "natural_wording_score": round(result.natural_wording_score, 3),
                "expressive_style_score": round(result.expressive_style_score, 3),
                "coherence_score": round(result.coherence_score, 3),
                "awkwardness_score": round(result.awkwardness_score, 3),
                "bullet_points": _deduplicate(result.bullet_points)[:3],
                "important_phrases": _sanitize_important_phrases(result.important_phrases),
                "observations": _deduplicate(result.bullet_points)[:3],
                "technical_note": None,
                "weaknesses": [],
            }
        except KeyError as exc:
            LOGGER.warning("Structured linguistic schema mismatch; attempting recovery.", exc_info=exc)
            payload = _recover_linguistic_assessment_from_raw(
                raw_text,
                status="schema_mismatch",
                technical_note="The detector returned JSON-like content, but required fields were missing or renamed. Details were recovered from the raw output.",
            )
        except Exception as exc:
            LOGGER.warning("Structured linguistic parsing failed; attempting recovery.", exc_info=exc)
            payload = _recover_linguistic_assessment_from_raw(raw_text)

        self.cache[cleaned] = payload
        return payload


def normalize_text(text: str) -> NormalizedText:
    original_text = text.replace("\r\n", "\n").strip()
    analysis_text = WHITESPACE_PATTERN.sub(" ", original_text).strip()

    sentences: list[SentenceRecord] = []
    for match in SENTENCE_PATTERN.finditer(original_text):
        sentence_text = match.group(0).strip()
        if not sentence_text:
            continue
        words = WORD_PATTERN.findall(sentence_text)
        if not words:
            continue
        sentences.append(
            SentenceRecord(
                text=sentence_text,
                start=match.start(),
                end=match.end(),
                word_count=len(words),
            )
        )

    paragraphs = [paragraph for paragraph in re.split(r"\n\s*\n", original_text) if paragraph.strip()]
    lines = [line for line in original_text.splitlines() if line.strip()]
    total_lines = max(len(lines), 1)
    bullet_lines = sum(1 for line in lines if BULLET_PATTERN.search(line))
    quote_lines = sum(1 for line in lines if QUOTE_PATTERN.search(line))
    prose_lines = sum(1 for line in lines if len(WORD_PATTERN.findall(line)) >= 5 and not BULLET_PATTERN.search(line))

    return NormalizedText(
        original_text=original_text,
        analysis_text=analysis_text,
        character_count=len(original_text),
        word_count=len(WORD_PATTERN.findall(analysis_text)),
        sentence_count=len(sentences),
        paragraph_count=len(paragraphs) if paragraphs else (1 if original_text else 0),
        sentences=sentences,
        bullet_ratio=bullet_lines / total_lines,
        quote_ratio=quote_lines / total_lines,
        prose_ratio=prose_lines / total_lines,
    )


def detect_language_and_meta(normalized: NormalizedText) -> tuple[str, bool]:
    words = [word.lower() for word in WORD_PATTERN.findall(normalized.analysis_text)]
    if not words:
        return "en", False

    english_hits = sum(1 for word in words if word in COMMON_ENGLISH_WORDS)
    english_ratio = english_hits / max(len(words), 1)
    ascii_alpha_ratio = sum(character.isascii() and character.isalpha() for character in normalized.original_text) / max(
        normalized.character_count,
        1,
    )

    if ascii_alpha_ratio < 0.45 and normalized.word_count >= 20:
        return "en", False
    if english_ratio < 0.015 and normalized.word_count >= 50:
        return "en", False
    return "en", True


def run_eligibility_gate(normalized: NormalizedText, language_valid: bool) -> dict[str, Any]:
    reasons: list[str] = []
    if not language_valid:
        reasons.append("english_validation_failed")
    if normalized.word_count < MIN_WORDS_FOR_STRONG_VERDICT:
        reasons.append("text_too_short")
    if normalized.sentence_count < MIN_SENTENCES_FOR_STRONG_VERDICT:
        reasons.append("insufficient_sentence_count")
    if normalized.bullet_ratio > 0.4:
        reasons.append("mostly_bullet_points")
    if normalized.quote_ratio > 0.5:
        reasons.append("mostly_quotes")
    if normalized.prose_ratio < 0.45:
        reasons.append("insufficient_natural_prose")
    return {
        "strong_verdict_allowed": len(reasons) == 0,
        "reasons": reasons,
    }


def run_sentence_length_analysis(normalized: NormalizedText) -> dict[str, Any]:
    sentence_lengths = [sentence.word_count for sentence in normalized.sentences]
    if not sentence_lengths:
        return {
            "score": 0.5,
            "length_variation_score": 0.5,
            "sentence_lengths": [],
            "range": 0,
            "std_dev": 0.0,
            "consecutive_diff_over_10": False,
            "bullet_points": ["The text does not contain enough sentence structure to judge variation."],
        }

    if len(sentence_lengths) == 1:
        return {
            "score": 0.5,
            "length_variation_score": 0.5,
            "sentence_lengths": sentence_lengths,
            "range": 0,
            "std_dev": 0.0,
            "consecutive_diff_over_10": False,
            "bullet_points": ["The text has only one sentence, so length variation is limited."],
        }

    length_range = max(sentence_lengths) - min(sentence_lengths)
    std_dev = statistics.pstdev(sentence_lengths)
    consecutive_diff_over_10 = any(
        abs(sentence_lengths[index] - sentence_lengths[index - 1]) > SENTENCE_RANGE_HUMAN_THRESHOLD
        for index in range(1, len(sentence_lengths))
    )

    variation_score = (
        _normalize(length_range, 3.0, 16.0) * 0.45
        + _normalize(std_dev, 1.5, 8.0) * 0.35
        + (0.2 if consecutive_diff_over_10 else 0.0)
    )
    variation_score = max(0.0, min(1.0, variation_score))
    ai_score = 1.0 - variation_score

    bullet_points: list[str] = []
    if length_range <= 5 and std_dev <= 2.5:
        bullet_points.append("Most sentences have a very similar length.")
    elif length_range >= SENTENCE_RANGE_HUMAN_THRESHOLD or consecutive_diff_over_10:
        bullet_points.append("Sentence lengths vary clearly, which looks more human.")
    else:
        bullet_points.append("Sentence length variation is present but not especially strong.")

    return {
        "score": round(ai_score, 3),
        "length_variation_score": round(variation_score, 3),
        "sentence_lengths": sentence_lengths,
        "range": int(length_range),
        "std_dev": round(std_dev, 3),
        "consecutive_diff_over_10": consecutive_diff_over_10,
        "bullet_points": bullet_points,
    }


def run_linking_word_analysis(normalized: NormalizedText) -> dict[str, Any]:
    lowered = normalized.analysis_text.lower()
    all_counts = {
        phrase: len(re.findall(pattern, lowered))
        for phrase, pattern in LINKING_WORD_PATTERNS.items()
    }
    repeated_counts = {phrase: count for phrase, count in all_counts.items() if count > 1}
    repeated_total = sum(count - 1 for count in repeated_counts.values())
    repeated_types = len(repeated_counts)
    max_repeat = max(repeated_counts.values(), default=0)
    expressive_repetition = _detect_expressive_repetition(normalized)

    transition_ai_score = (
        _normalize(repeated_total, 0.0, 8.0) * 0.5
        + _normalize(repeated_types, 0.0, 4.0) * 0.3
        + _normalize(max_repeat, 1.0, 5.0) * 0.2
    )
    ai_score = max(0.0, min(1.0, transition_ai_score - expressive_repetition["human_score"] * 0.45))

    bullet_points: list[str] = []
    if repeated_counts:
        top_words = ", ".join(f"{word} ({count})" for word, count in list(repeated_counts.items())[:3])
        bullet_points.append(f"Some linking words repeat often: {top_words}.")
    elif expressive_repetition["human_score"] >= 0.25:
        bullet_points.append("Some repetition looks expressive or lyrical rather than mechanically structured.")
    else:
        bullet_points.append("Linking words are not repeated heavily.")

    if expressive_repetition["human_score"] >= 0.45:
        bullet_points.append("Repeated emotional or stylized wording makes the repetition feel more human.")

    return {
        "score": round(ai_score, 3),
        "repeated_linking_words": repeated_counts,
        "all_linking_words": all_counts,
        "repeated_total": repeated_total,
        "expressive_repetition": expressive_repetition,
        "bullet_points": bullet_points,
    }


def run_linguistic_analysis(detector: LinguisticDetector, normalized: NormalizedText) -> dict[str, Any]:
    return detector.score_text(normalized.analysis_text)


def run_robustness_checks(
    sentence_length: dict[str, Any],
    linking_words: dict[str, Any],
    linguistic: dict[str, Any],
) -> dict[str, Any]:
    signal_scores = [sentence_length["score"], linking_words["score"], linguistic["ai_score"]]
    spread = statistics.pstdev(signal_scores) if len(signal_scores) > 1 else 0.0
    robustness_score = max(0.0, min(1.0, 1.0 - _normalize(spread, 0.0, 0.28)))
    return {
        "score": round(robustness_score, 3),
        "signal_scores": [round(score, 3) for score in signal_scores],
    }


def calibrate_final_assessment(
    normalized: NormalizedText,
    eligibility: dict[str, Any],
    sentence_length: dict[str, Any],
    linking_words: dict[str, Any],
    linguistic: dict[str, Any],
    robustness: dict[str, Any],
) -> tuple[dict[str, Any], str, int, str]:
    deterministic_score = sentence_length["score"] * 0.58 + linking_words["score"] * 0.42
    base_score = linguistic["ai_score"] * 0.45 + deterministic_score * 0.55
    calibrated_score = 0.5 + (base_score - 0.5) * (0.7 + robustness["score"] * 0.3)

    if not eligibility["strong_verdict_allowed"]:
        calibrated_score = 0.5 + (calibrated_score - 0.5) * 0.6
    if linguistic.get("status") != "ok":
        calibrated_score = 0.5 + (calibrated_score - 0.5) * 0.7

    calibrated_score = max(0.0, min(1.0, calibrated_score))
    ai_percentage = round(calibrated_score * 100)
    human_percentage = round((1.0 - calibrated_score) * 100)
    verdict = "likely AI-generated" if calibrated_score >= 0.5 else "likely human-written"
    percentage = ai_percentage if verdict == "likely AI-generated" else human_percentage
    final_label = f"{percentage}% {verdict}"

    confidence = "high"
    if normalized.word_count < 80 or normalized.sentence_count < 4 or robustness["score"] < 0.55:
        confidence = "medium"
    if (
        normalized.word_count < MIN_WORDS_FOR_STRONG_VERDICT
        or normalized.sentence_count < MIN_SENTENCES_FOR_STRONG_VERDICT
        or not linguistic["available"]
    ):
        confidence = "low"

    document_assessment = {
        "ai_likelihood_score": round(calibrated_score, 2),
        "ai_likelihood_label": _label_likelihood(calibrated_score),
        "confidence": confidence,
    }
    return document_assessment, verdict, percentage, final_label


def build_metrics(
    normalized: NormalizedText,
    sentence_length: dict[str, Any],
    linking_words: dict[str, Any],
    linguistic: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sentence_count": normalized.sentence_count,
        "sentence_lengths": sentence_length["sentence_lengths"],
        "length_variation_score": sentence_length["length_variation_score"],
        "repeated_linking_words": linking_words["repeated_linking_words"],
        "expressive_repetition_score": linking_words["expressive_repetition"]["human_score"],
        "expressive_repeated_phrases": linking_words["expressive_repetition"]["phrases"],
        "linguistic_style_score": linguistic["human_style_score"],
        "sentence_length_range": sentence_length["range"],
        "sentence_length_std_dev": sentence_length["std_dev"],
        "consecutive_diff_over_10": sentence_length["consecutive_diff_over_10"],
        "linking_word_ai_score": linking_words["score"],
        "linguistic_ai_score": linguistic["ai_score"],
    }


def build_explanation(
    verdict: str,
    final_label: str,
    sentence_length: dict[str, Any],
    linking_words: dict[str, Any],
    linguistic: dict[str, Any],
    eligibility: dict[str, Any],
) -> tuple[list[str], list[str], str]:
    bullet_points: list[str] = []
    bullet_points.extend(sentence_length["bullet_points"])
    bullet_points.extend(linking_words["bullet_points"])
    if linguistic.get("status") == "ok":
        bullet_points.extend(linguistic["bullet_points"])
    elif linguistic.get("status") in {"schema_mismatch", "parsing_failed"}:
        recovered_observation = _select_recovered_observation(linguistic.get("observations", []))
        if recovered_observation:
            bullet_points.append(recovered_observation)

    if linguistic["available"]:
        if linguistic.get("status") != "ok":
            pass
        elif linguistic["human_style_score"] >= 0.65:
            bullet_points.append("The wording feels natural and stylistically expressive.")
        elif linguistic["ai_score"] >= 0.6:
            bullet_points.append("The phrasing is coherent but stylistically uniform.")
        else:
            bullet_points.append("The wording signal is mixed rather than strongly one-sided.")

    why = _deduplicate(bullet_points)[:MAX_BULLET_POINTS]

    weakens: list[str] = []
    weakens.extend(_map_eligibility_reasons(eligibility["reasons"]))
    if linguistic.get("status") not in {"schema_mismatch", "parsing_failed", "unavailable"}:
        weakens.extend(linguistic.get("weaknesses", []))
    if linguistic.get("status") == "schema_mismatch":
        weakens.append(
            "The wording detector returned a mismatched response format, so recovered clues were used conservatively."
        )
    elif linguistic.get("status") == "parsing_failed":
        weakens.append("The wording detector response could not be parsed cleanly, so the final score was kept conservative.")
    elif linguistic.get("status") == "unavailable":
        weakens.append("The wording detector was unavailable, so the final score was kept conservative.")
    weakens = _deduplicate(weakens)[:3]

    final_user_message = final_label
    return why, weakens, final_user_message


def build_highlights(
    normalized: NormalizedText,
    linking_words: dict[str, Any],
    linguistic: dict[str, Any],
    verdict: str,
) -> list[dict[str, Any]]:
    highlights: list[dict[str, Any]] = []

    for phrase in linking_words["repeated_linking_words"]:
        highlights.extend(
            _find_phrase_spans(
                normalized.original_text,
                phrase,
                reason="repeated linking word",
                limit=2,
            )
        )

    phrase_reason = "expressive phrasing" if verdict == "likely human-written" else "influential wording"
    for phrase in linguistic.get("important_phrases", []):
        if len(WORD_PATTERN.findall(phrase)) == 0:
            continue
        highlights.extend(
            _find_phrase_spans(
                normalized.original_text,
                phrase,
                reason=phrase_reason,
                limit=1,
            )
        )

    merged = _merge_highlights(highlights)
    return merged[:MAX_HIGHLIGHT_COUNT]


def build_segment_assessment(highlights: list[dict[str, Any]]) -> dict[str, Any]:
    spans = [
        {
            "start": highlight["start"],
            "end": highlight["end"],
            "score": 1.0,
        }
        for highlight in highlights
    ]
    return {
        "available": bool(spans),
        "spans": spans,
    }


def build_response(context: DetectionContext) -> dict[str, Any]:
    return {
        "title": "Statistical Verification Result",
        "verification_title": "Statistical Verification Result",
        "language": context.language,
        "eligibility": context.eligibility,
        "document_assessment": context.document_assessment,
        "signal_breakdown": {
            "semantic_model_score": round(context.linguistic["ai_score"], 2),
            "stylometric_score": round(
                context.sentence_length["score"] * 0.58 + context.linking_words["score"] * 0.42,
                2,
            ),
            "robustness_score": round(context.robustness["score"], 2),
        },
        "why": context.why,
        "what_weakens_the_conclusion": context.weakens,
        "segment_assessment": context.segment_assessment,
        "final_user_message": context.final_user_message,
        "verdict": context.verdict,
        "percentage": context.percentage,
        "final_label": context.final_label,
        "bullet_points": context.bullet_points,
        "summary": context.why,
        "limitations": context.weakens,
        "detector_details": context.detector_details,
        "highlights": context.highlights,
        "metrics": context.metrics,
    }


def run_detection_pipeline(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    language, language_valid = detect_language_and_meta(normalized)
    eligibility = run_eligibility_gate(normalized, language_valid)

    detector = LinguisticDetector()
    sentence_length = run_sentence_length_analysis(normalized)
    linking_words = run_linking_word_analysis(normalized)
    linguistic = run_linguistic_analysis(detector, normalized)
    robustness = run_robustness_checks(sentence_length, linking_words, linguistic)
    document_assessment, verdict, percentage, final_label = calibrate_final_assessment(
        normalized,
        eligibility,
        sentence_length,
        linking_words,
        linguistic,
        robustness,
    )
    metrics = build_metrics(normalized, sentence_length, linking_words, linguistic)
    why, weakens, final_user_message = build_explanation(
        verdict,
        final_label,
        sentence_length,
        linking_words,
        linguistic,
        eligibility,
    )
    highlights = build_highlights(normalized, linking_words, linguistic, verdict)

    context = DetectionContext(
        normalized=normalized,
        language=language,
        language_valid=language_valid,
        eligibility=eligibility,
        linguistic=linguistic,
        sentence_length=sentence_length,
        linking_words=linking_words,
        robustness=robustness,
        document_assessment=document_assessment,
        verdict=verdict,
        percentage=percentage,
        final_label=final_label,
        bullet_points=why,
        highlights=highlights,
        metrics=metrics,
        detector_details=_build_detector_details(linguistic),
        why=why,
        weakens=weakens,
        segment_assessment=build_segment_assessment(highlights),
        final_user_message=final_user_message,
    )
    return build_response(context)


def _fallback_linguistic_assessment(
    message: str,
    details: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    technical_note = message if not details else f"{message} The detector response could not be recovered reliably."
    return {
        "available": False,
        "status": "unavailable",
        "human_style_score": 0.5,
        "ai_score": 0.5,
        "raw_score": None,
        "natural_wording_score": 0.5,
        "expressive_style_score": 0.5,
        "coherence_score": 0.5,
        "awkwardness_score": 0.5,
        "observations": [],
        "bullet_points": ["The wording signal is neutral because the model could not add a stronger judgment."],
        "important_phrases": [],
        "technical_note": technical_note,
        "invoke_diagnostics": diagnostics or _default_invoke_diagnostics(),
        "weaknesses": [message],
    }


def _recover_linguistic_assessment_from_raw(
    raw_text: str,
    status: str = "parsing_failed",
    technical_note: str | None = None,
) -> dict[str, Any]:
    extracted_score = _extract_raw_score(raw_text)
    score_was_extracted = extracted_score is not None
    observations = _extract_labeled_list(raw_text, ["observations", "observation", "details"])
    influential_phrases = _sanitize_important_phrases(
        _extract_labeled_list(raw_text, ["influential phrases", "phrases", "important phrases"])
    )
    if extracted_score is None:
        extracted_score = 0.5

    # Recovered scores are less trustworthy than clean structured scores, so blend toward neutral.
    human_style_score = 0.5 + (extracted_score - 0.5) * 0.5
    ai_score = 1.0 - human_style_score
    if not observations:
        observations = [GENERIC_RECOVERY_OBSERVATION]

    return {
        "available": True,
        "status": status,
        "human_style_score": round(human_style_score, 3),
        "ai_score": round(ai_score, 3),
        "raw_score": round(extracted_score, 3) if score_was_extracted else None,
        "natural_wording_score": round(human_style_score, 3),
        "expressive_style_score": round(human_style_score, 3),
        "coherence_score": round(human_style_score, 3),
        "awkwardness_score": round(1.0 - human_style_score, 3),
        "observations": observations[:3],
        "bullet_points": observations[:3],
        "important_phrases": influential_phrases[:3],
        "technical_note": technical_note or "The detector response could not be parsed in structured format, so details were recovered from raw output.",
        "invoke_diagnostics": _default_invoke_diagnostics(),
        "schema_present_keys": _extract_structured_keys(raw_text),
        "weaknesses": ["The wording detector response could not be parsed cleanly."],
    }


def _parse_linguistic_assessment(raw_text: str) -> LinguisticAssessment:
    cleaned = _extract_json_candidate(raw_text)
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise TypeError("Linguistic detector JSON payload is not an object.")
    LOGGER.debug("Structured linguistic payload keys: %s", sorted(payload.keys()))
    normalized_payload = _normalize_linguistic_payload(payload)
    LOGGER.debug("Normalized linguistic payload keys: %s", sorted(normalized_payload.keys()))
    return LinguisticAssessment(**normalized_payload)


def _build_detector_details(linguistic: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": linguistic.get("status", "ok"),
        "score_semantics": "raw_score is a human-likelihood score from 0 to 1, where 1 means more human-written and 0 means more AI-like.",
        "raw_score": linguistic.get("raw_score"),
        "observations": linguistic.get("observations", []),
        "influential_phrases": linguistic.get("important_phrases", []),
        "technical_note": linguistic.get("technical_note"),
        "schema_present_keys": linguistic.get("schema_present_keys", []),
        "invoke_error_type": linguistic.get("invoke_diagnostics", {}).get("invoke_error_type"),
        "invoke_error_message": linguistic.get("invoke_diagnostics", {}).get("invoke_error_message"),
        "invoke_error_status_code": linguistic.get("invoke_diagnostics", {}).get("invoke_error_status_code"),
        "invoke_error_body": linguistic.get("invoke_diagnostics", {}).get("invoke_error_body"),
        "invoke_error_provider": linguistic.get("invoke_diagnostics", {}).get("invoke_error_provider"),
        "invoke_error_model": linguistic.get("invoke_diagnostics", {}).get("invoke_error_model"),
        "invoke_error_base_url": linguistic.get("invoke_diagnostics", {}).get("invoke_error_base_url"),
        "invoke_error_timeout_seconds": linguistic.get("invoke_diagnostics", {}).get("invoke_error_timeout_seconds"),
        "raw_output_excerpt": linguistic.get("invoke_diagnostics", {}).get("raw_output_excerpt"),
        "diagnostic_timestamp": linguistic.get("invoke_diagnostics", {}).get("diagnostic_timestamp"),
    }


def _default_invoke_diagnostics() -> dict[str, Any]:
    return {
        "invoke_error_type": None,
        "invoke_error_message": None,
        "invoke_error_status_code": None,
        "invoke_error_body": None,
        "invoke_error_provider": "ollama",
        "invoke_error_model": None,
        "invoke_error_base_url": None,
        "invoke_error_timeout_seconds": None,
        "raw_output_excerpt": None,
        "diagnostic_timestamp": None,
    }


def _extract_invoke_error_details(
    exc: Exception,
    provider: str,
    model_name: str,
    base_url: str,
) -> dict[str, Any]:
    diagnostics = _default_invoke_diagnostics()
    diagnostics.update(
        {
            "invoke_error_type": exc.__class__.__name__,
            "invoke_error_message": str(exc)[:500] if str(exc) else None,
            "invoke_error_provider": provider,
            "invoke_error_model": model_name,
            "invoke_error_base_url": base_url,
            "diagnostic_timestamp": datetime.now(UTC).isoformat(),
        }
    )

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    diagnostics["invoke_error_status_code"] = status_code

    body = None
    response = getattr(exc, "response", None)
    if response is not None:
        body = getattr(response, "text", None)
        if body is None:
            content = getattr(response, "content", None)
            if isinstance(content, (bytes, bytearray)):
                body = content.decode("utf-8", errors="replace")
            elif content is not None:
                body = str(content)
    diagnostics["invoke_error_body"] = body[:1000] if isinstance(body, str) and body else None

    request = getattr(exc, "request", None)
    timeout_seconds = getattr(request, "extensions", {}).get("timeout", {}).get("read") if request is not None else None
    diagnostics["invoke_error_timeout_seconds"] = timeout_seconds

    partial_output = getattr(exc, "body", None) or getattr(exc, "message", None)
    if partial_output is not None and not isinstance(partial_output, str):
        partial_output = str(partial_output)
    diagnostics["raw_output_excerpt"] = partial_output[:300] if partial_output else None

    return diagnostics


def _find_phrase_spans(text: str, phrase: str, reason: str, limit: int) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if not phrase.strip():
        return matches

    pattern = re.compile(re.escape(phrase.strip()), re.IGNORECASE)
    for match in pattern.finditer(text):
        matches.append(
            {
                "text": text[match.start():match.end()],
                "start": match.start(),
                "end": match.end(),
                "reason": reason,
            }
        )
        if len(matches) >= limit:
            break
    return matches


def _merge_highlights(highlights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(highlights, key=lambda item: (item["start"], item["end"]))
    merged: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    cursor = -1

    for highlight in ordered:
        key = (highlight["start"], highlight["end"], highlight["reason"])
        if key in seen:
            continue
        if highlight["start"] < cursor:
            continue
        seen.add(key)
        merged.append(highlight)
        cursor = highlight["end"]
    return merged


def _sanitize_important_phrases(phrases: list[str]) -> list[str]:
    sanitized: list[str] = []
    for phrase in _deduplicate(phrases):
        cleaned = WHITESPACE_PATTERN.sub(" ", phrase).strip(" \t\r\n-:;,.!?\"'")
        word_count = len(WORD_PATTERN.findall(cleaned))
        if not cleaned:
            continue
        if word_count == 0 or word_count > MAX_HIGHLIGHT_PHRASE_WORDS:
            continue
        if len(cleaned) > MAX_HIGHLIGHT_PHRASE_CHARS:
            continue
        if cleaned.count(",") > 1 or cleaned.count(".") > 0:
            continue
        sanitized.append(cleaned)
        if len(sanitized) >= 3:
            break
    return sanitized


def _extract_raw_score(raw_text: str) -> float | None:
    patterns = [
        r"human[_\s-]?likelihood[_\s-]?score\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bscore\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ]
    lowered = raw_text.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        value = _normalize_score_value(float(match.group(1)))
        if value is not None:
            return value
    return None


def _extract_json_candidate(raw_text: str) -> str:
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1)

    object_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if object_match:
        return object_match.group(1)

    return raw_text.strip()


def _normalize_linguistic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for target_key, aliases in LINGUISTIC_KEY_ALIASES.items():
        value = None
        for alias in aliases:
            if alias in payload:
                value = payload[alias]
                break
        if value is None:
            if target_key in {"bullet_points", "important_phrases"}:
                normalized[target_key] = []
                continue
            raise KeyError(target_key)
        if target_key in {
            "human_likelihood_score",
            "natural_wording_score",
            "expressive_style_score",
            "coherence_score",
            "awkwardness_score",
        }:
            normalized_score = _normalize_score_value(value)
            if normalized_score is None:
                raise ValueError(f"{target_key} must be a numeric score.")
            normalized[target_key] = normalized_score
            continue
        normalized[target_key] = value
    return normalized


def _normalize_score_value(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= score <= 1.0:
        return score
    if 1.0 < score <= 10.0:
        return score / 10.0
    if 10.0 < score <= 100.0:
        return score / 100.0
    return None


def _extract_structured_keys(raw_text: str) -> list[str]:
    try:
        cleaned = _extract_json_candidate(raw_text)
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return sorted(str(key) for key in payload.keys())
    except Exception:
        return []
    return []


def _extract_labeled_list(raw_text: str, labels: list[str]) -> list[str]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    collected: list[str] = []
    capture = False

    for line in lines:
        normalized = line.strip("* ").lower().rstrip(":")
        if any(normalized == label for label in labels):
            capture = True
            continue
        if capture:
            if re.match(r"^[A-Za-z][A-Za-z\s]+:$", line):
                break
            cleaned = line.lstrip("-*0123456789. )").strip()
            if cleaned:
                collected.append(cleaned)
            if len(collected) >= 5:
                break
    return _deduplicate(collected)


def _select_recovered_observation(observations: list[str]) -> str | None:
    for observation in observations:
        if observation.strip() == GENERIC_RECOVERY_OBSERVATION:
            continue
        safe_observation = _sanitize_detector_text(observation, limit=160)
        if safe_observation:
            return safe_observation
    return None


def _sanitize_detector_text(value: str, limit: int = 180) -> str | None:
    cleaned = WHITESPACE_PATTERN.sub(" ", value).strip(" \t\r\n-*\"'")
    lowered = cleaned.lower()
    blocked_patterns = (
        "invalid json output",
        "output_parsing_failure",
        "for troubleshooting",
        "http://",
        "https://",
        "{",
        "}",
        "```",
    )
    if not cleaned or any(pattern in lowered for pattern in blocked_patterns):
        return None
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 3].rstrip() + "..."
    return cleaned


def _detect_expressive_repetition(normalized: NormalizedText) -> dict[str, Any]:
    lowered_lines = [
        WHITESPACE_PATTERN.sub(" ", line.strip().lower())
        for line in normalized.original_text.splitlines()
        if line.strip()
    ]
    repeated_line_fragments = [
        line for line in lowered_lines
        if len(WORD_PATTERN.findall(line)) >= 2 and lowered_lines.count(line) > 1
    ]

    word_counts: dict[str, int] = {}
    for word in WORD_PATTERN.findall(normalized.analysis_text.lower()):
        if word in EXPRESSIVE_REPEAT_STOPWORDS or len(word) < 4:
            continue
        word_counts[word] = word_counts.get(word, 0) + 1

    expressive_words = [
        word for word, count in sorted(word_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= 3
    ][:3]

    elongated_words = [
        match.group(0) for match in re.finditer(r"\b\w*([a-zA-Z])\1{2,}\w*\b", normalized.original_text)
    ][:3]

    human_score = 0.0
    if expressive_words:
        human_score += min(len(expressive_words) * 0.18, 0.36)
    if repeated_line_fragments:
        human_score += 0.25
    if elongated_words:
        human_score += 0.2

    phrases = _deduplicate([
        *expressive_words,
        *repeated_line_fragments[:2],
        *elongated_words,
    ])[:3]

    return {
        "human_score": round(max(0.0, min(1.0, human_score)), 3),
        "phrases": phrases,
    }


def _normalize(value: float, min_value: float, max_value: float) -> float:
    if min_value == max_value:
        return 0.0
    normalized = (value - min_value) / (max_value - min_value)
    return max(0.0, min(1.0, normalized))


def _label_likelihood(score: float) -> str:
    if score < 0.4:
        return "low"
    if score < 0.6:
        return "moderate"
    return "high"


def _map_eligibility_reasons(reasons: list[str]) -> list[str]:
    mapping = {
        "english_validation_failed": "The text could not be validated strongly enough as English prose.",
        "text_too_short": "The text is short, so the estimate is less stable.",
        "insufficient_sentence_count": "There are too few sentences for a strong sentence-length signal.",
        "mostly_bullet_points": "Bullet-heavy text gives weaker prose signals.",
        "mostly_quotes": "Quoted text gives less original writing to judge.",
        "insufficient_natural_prose": "There is not enough continuous prose for a strong verdict.",
    }
    return [mapping[reason] for reason in reasons if reason in mapping]


def _deduplicate(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered
