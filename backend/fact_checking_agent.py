from __future__ import annotations

import time
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field


ClaimType = Literal[
    "historical_fact",
    "statistical_fact",
    "scientific_fact",
    "technical_fact",
    "biographical_fact",
    "current_fact",
    "other",
]
Verdict = Literal[
    "VERIFIED",
    "LIKELY_TRUE",
    "CONTRADICTED",
    "OUTDATED",
    "UNVERIFIABLE",
]

ALLOWED_CLAIM_TYPES: set[str] = {
    "historical_fact",
    "statistical_fact",
    "scientific_fact",
    "technical_fact",
    "biographical_fact",
    "current_fact",
    "other",
}
ALLOWED_VERDICTS: set[str] = {
    "VERIFIED",
    "LIKELY_TRUE",
    "CONTRADICTED",
    "OUTDATED",
    "UNVERIFIABLE",
}
CLAIM_SCORE_BY_VERDICT: dict[str, int] = {
    "VERIFIED": 100,
    "LIKELY_TRUE": 80,
    "OUTDATED": 45,
    "UNVERIFIABLE": 25,
    "CONTRADICTED": 0,
}

MAX_TEXT_CHARS_FOR_LLM = 12000
MAX_CLAIMS = 12
MAX_SOURCES_PER_CLAIM = 3
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_TIMEOUT_SECONDS = 8
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
MIN_ENTITY_SUPPORT_RATIO = 0.6
MIN_NOUN_SUPPORT_RATIO = 0.5
MIN_NOUNS_FOR_SUPPORT = 2
WEAK_SOURCE_CREDIBILITY_THRESHOLD = 0.4
MEDIUM_SOURCE_CREDIBILITY_THRESHOLD = 0.7
HIGH_SOURCE_CREDIBILITY_THRESHOLD = 0.85
CONTRADICTION_SOURCE_CREDIBILITY_THRESHOLD = 0.7

FACT_CHECK_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "can",
    "claim",
    "could",
    "did",
    "does",
    "during",
    "each",
    "from",
    "had",
    "has",
    "have",
    "having",
    "into",
    "its",
    "itself",
    "joined",
    "made",
    "more",
    "most",
    "not",
    "only",
    "other",
    "over",
    "said",
    "same",
    "should",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "those",
    "through",
    "under",
    "was",
    "were",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}

ENTITY_VARIANTS = {
    "eu": {"eu", "e.u.", "european union"},
    "european union": {"eu", "e.u.", "european union"},
    "uk": {"uk", "u.k.", "united kingdom", "britain", "great britain"},
    "united kingdom": {"uk", "u.k.", "united kingdom", "britain", "great britain"},
    "us": {"us", "u.s.", "usa", "u.s.a.", "united states", "united states of america"},
    "usa": {"us", "u.s.", "usa", "u.s.a.", "united states", "united states of america"},
    "united states": {"us", "u.s.", "usa", "u.s.a.", "united states", "united states of america"},
    "un": {"un", "u.n.", "united nations"},
    "united nations": {"un", "u.n.", "united nations"},
    "france": {"france", "french"},
    "french": {"france", "french"},
    "germany": {"germany", "german"},
    "german": {"germany", "german"},
    "italy": {"italy", "italian"},
    "italian": {"italy", "italian"},
    "romania": {"romania", "romanian"},
    "romanian": {"romania", "romanian"},
    "spain": {"spain", "spanish"},
    "spanish": {"spain", "spanish"},
}


@dataclass(frozen=True)
class ClaimKeyInfo:
    named_entities: list[str]
    important_nouns: list[str]
    numbers: list[str]


@dataclass(frozen=True)
class EvidenceSupportSummary:
    supportive_sources: int
    high_quality_support: int
    medium_quality_support: int
    repeated_low_quality_support: bool
    max_credibility: float


@dataclass(frozen=True)
class EvidenceContradictionSummary:
    contradictory_sources: int
    medium_quality_contradictions: int
    repeated_correction: bool
    related_sources: int
    max_related_credibility: float


@dataclass(frozen=True)
class FactSignature:
    relation: str
    subject: str
    value: str


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()
    module_dir = Path(__file__).resolve().parent
    load_dotenv(module_dir / ".env", override=False)
    load_dotenv(module_dir.parent / ".env", override=False)


_load_dotenv_if_available()


class SourceEvidence(BaseModel):
    title: str
    url: str
    credibility_score: float = Field(ge=0.0, le=1.0)
    snippet: str


class ExtractedClaim(BaseModel):
    claim: str
    type: ClaimType = "other"


class FactCheckedClaim(BaseModel):
    claim: str
    type: ClaimType
    queries: list[str]
    verdict: Verdict
    claim_score: int = Field(ge=0, le=100)
    confidence_score: int = Field(ge=0, le=100)
    sources: list[SourceEvidence]
    explanation: str


class FactCheckingResult(BaseModel):
    agent: Literal["fact_checking"] = "fact_checking"
    overall_trust_score: float
    overall_confidence_score: float
    total_claims: int
    claims: list[FactCheckedClaim]


# def call_llm(prompt: str) -> str:
#     """Call Gemini when configured, otherwise return a safe mock JSON response."""
#     api_key = os.getenv("GEMINI_API_KEY")
#     if api_key:
#         try:
#             return _call_gemini(prompt=prompt, api_key=api_key)
#         except Exception as e: # MODIFICAT AICI
#             print(f"================ EROARE GEMINI: {e} ================") # MODIFICAT AICI
#             pass

#     return _mock_llm_response(prompt)
def call_llm(prompt: str) -> str:
    """Call Gemini when configured, otherwise return a safe mock JSON response."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            time.sleep(1)  # <-- ADAUGĂ ASTA: Pauză de 3 secunde între cereri
            return _call_gemini(prompt=prompt, api_key=api_key)
        except Exception as e:
            print(f"================ call_llm a eșuat și dă fallback: {e} ================")
            pass

    return _mock_llm_response(prompt)

# def _call_gemini(prompt: str, api_key: str) -> str:
#     model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
#     last_error: Exception | None = None

#     try:
#         raw_response = _call_google_genai(prompt, api_key, model_name)
#         return _normalize_llm_json_response(prompt, raw_response)
#     except ImportError as exc:
#         last_error = exc
#     except Exception as exc:
#         last_error = exc

#     try:
#         raw_response = _call_legacy_google_generativeai(prompt, api_key, model_name)
#         return _normalize_llm_json_response(prompt, raw_response)
#     except Exception as exc:
#         raise ValueError("Gemini LLM call failed.") from (last_error or exc)
def _call_gemini(prompt: str, api_key: str) -> str:
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    
    try:
        raw_response = _call_google_genai(prompt, api_key, model_name)
        return _normalize_llm_json_response(prompt, raw_response)
    except Exception as exc:
        print(f"================ EROARE CRITICĂ GENAI: {type(exc).__name__}: {exc} ================")
        raise ValueError("Gemini LLM call failed cu noul pachet google-genai.") from exc

def _call_google_genai(prompt: str, api_key: str, model_name: str) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
        },
    )
    return _extract_gemini_text(response)


def _call_legacy_google_generativeai(prompt: str, api_key: str, model_name: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
        },
    )
    return _extract_gemini_text(response)


def _extract_gemini_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""

    parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                parts.append(part_text)

    return "\n".join(parts)


def _normalize_llm_json_response(prompt: str, raw_response: str) -> str:
    parsed = _parse_json_object(raw_response)

    if "TASK: EXTRACT_FACTUAL_CLAIMS" in prompt:
        claims = _coerce_extracted_claims(parsed.get("claims"))
        return json.dumps(
            {"claims": [{"claim": claim.claim, "type": claim.type} for claim in claims]},
            ensure_ascii=True,
        )

    if "TASK: GENERATE_SEARCH_QUERIES" in prompt:
        queries = _coerce_query_list(parsed.get("queries"))
        return json.dumps({"queries": _dedupe_strings(queries)[:3]}, ensure_ascii=True)

    if "TASK: EVALUATE_FACTUAL_CLAIM" in prompt:
        verdict = _normalize_verdict(parsed.get("verdict"))
        explanation = _clean_text(parsed.get("explanation"))
        if not explanation:
            explanation = _mock_evaluation_payload(prompt)["explanation"]
        return json.dumps(
            {"verdict": verdict, "explanation": explanation},
            ensure_ascii=True,
        )

    return json.dumps(parsed, ensure_ascii=True)


def _mock_llm_response(prompt: str) -> str:
    try:
        if "TASK: EXTRACT_FACTUAL_CLAIMS" in prompt:
            return json.dumps({"claims": _mock_extract_claim_payload(prompt)})
        if "TASK: GENERATE_SEARCH_QUERIES" in prompt:
            return json.dumps({"queries": _mock_query_payload(prompt)})
        if "TASK: EVALUATE_FACTUAL_CLAIM" in prompt:
            return json.dumps(_mock_evaluation_payload(prompt))
    except Exception:
        return "{}"

    return "{}"


def extract_claims(text: str) -> list[ExtractedClaim]:
    prompt = _build_claim_extraction_prompt(text)
    try:
        raw_response = call_llm(prompt)
        parsed = _parse_json_object(raw_response)
        claims = _coerce_extracted_claims(parsed.get("claims"))
        return _dedupe_claims(claims)[:MAX_CLAIMS]
    except Exception:
        return []


# def generate_queries(claim: str) -> list[str]:
#     prompt = _build_query_generation_prompt(claim)
#     try:
#         raw_response = call_llm(prompt)
#         parsed = _parse_json_object(raw_response)
#         queries = _coerce_query_list(parsed.get("queries"))
#     except Exception:
#         queries = []

#     if not queries:
#         queries = _fallback_queries(claim)

#     return _dedupe_strings(queries)[:3]

def generate_queries(claim: str) -> list[str]:
    # Sari peste apelul LLM lent și folosește direct funcția rapidă din Python
    queries = _fallback_queries(claim)
    return _dedupe_strings(queries)[:3]


def search_web_for_evidence(queries: list[str]) -> list[SourceEvidence]:
    """Search Tavily for evidence, falling back to mock data without an API key."""
    try:
        cleaned_queries = _dedupe_strings(queries)[:MAX_SOURCES_PER_CLAIM]
        if not cleaned_queries:
            return []

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return _mock_search_web_for_evidence(cleaned_queries)

        evidence = _search_tavily_for_evidence(cleaned_queries, api_key)
        return evidence or _mock_search_web_for_evidence(cleaned_queries)
    except Exception as e: # MODIFICAT AICI
        print(f"================ EROARE TAVILY: {e} ================") # MODIFICAT AICI
        return _mock_search_web_for_evidence(queries)


def _search_tavily_for_evidence(queries: list[str], api_key: str) -> list[SourceEvidence]:
    evidence: list[SourceEvidence] = []
    seen_urls: set[str] = set()

    for query in queries:
        try:
            payload = _post_tavily_search(query=query, api_key=api_key)
        except Exception:
            continue

        for source in _sources_from_tavily_payload(payload):
            url_key = source.url.lower()
            if url_key in seen_urls:
                continue

            seen_urls.add(url_key)
            evidence.append(source)
            if len(evidence) >= MAX_SOURCES_PER_CLAIM:
                return evidence

    return evidence


def _post_tavily_search(query: str, api_key: str) -> dict[str, Any]:
    request_body = json.dumps(
        {
            "query": query,
            "search_depth": "basic",
            "topic": "general",
            "max_results": MAX_SOURCES_PER_CLAIM,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "include_favicon": False,
            "auto_parameters": False,
            "include_usage": False,
        }
    ).encode("utf-8")
    request = Request(
        TAVILY_SEARCH_URL,
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=TAVILY_TIMEOUT_SECONDS) as response:
            response_text = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"Tavily search request failed: {exc}") from exc

    parsed = json.loads(response_text)
    if not isinstance(parsed, dict):
        raise ValueError("Tavily search response was not a JSON object.")
    return parsed


def _sources_from_tavily_payload(payload: dict[str, Any]) -> list[SourceEvidence]:
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return []

    sources: list[SourceEvidence] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue

        url = _clean_text(item.get("url"))
        if not url:
            continue

        title = _clean_text(item.get("title")) or _extract_host(url) or "Tavily search result"
        snippet = _clean_text(item.get("content") or item.get("raw_content"))
        if not snippet:
            snippet = "Tavily returned this source without a text snippet."

        sources.append(
            SourceEvidence(
                title=title[:200],
                url=url,
                credibility_score=score_source_credibility(url),
                snippet=snippet[:700],
            )
        )

    return sources


def _mock_search_web_for_evidence(queries: list[str]) -> list[SourceEvidence]:
    cleaned_queries = _dedupe_strings(queries)[:MAX_SOURCES_PER_CLAIM]
    evidence: list[SourceEvidence] = []

    for index, query in enumerate(cleaned_queries, start=1):
        url = f"https://example.com/mock-evidence/{index}"
        evidence.append(
            SourceEvidence(
                title=f"Mock evidence placeholder {index}",
                url=url,
                credibility_score=score_source_credibility(url),
                snippet=(
                    "Mock search result for query: "
                    f"{query}. Set TAVILY_API_KEY to enable real Tavily web "
                    "evidence search."
                ),
            )
        )

    return evidence


def score_source_credibility(url: str) -> float:
    host = _extract_host(url)
    normalized_url = (url or "").strip().lower()
    if not host:
        return 0.1

    # 1.0 - Credibilitate Maximă (Știință, Guverne, Fact-Checkers oficiali, Instituții)
    if _has_domain_suffix(host, [".gov", ".edu", ".mil"]) or _matches_domain(
        host,
        {
            # Știință și baze de date academice
            "arxiv.org", "doi.org", "crossref.org", "pubmed.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov", "nature.com", "science.org", "sciencedirect.com",
            "springer.com", "wiley.com", "ieee.org", "acm.org", "jstor.org",
            # Instituții globale și enciclopedii
            "nobelprize.org", "britannica.com", "history.com", "nasa.gov",
            # Site-uri oficiale de Fact-Checking
            "snopes.com", "politifact.com", "factcheck.org", "fullfact.org", "poynter.org",
            # Surse oficiale România
            "agerpres.ro", "insse.ro", "bnr.ro", "guv.ro", "presidency.ro"
        },
    ):
        return 1.0

    if host.startswith(("docs.", "documentation.", "developer.")) or "/docs" in normalized_url:
        return 0.9

    # 0.9 - Instituții majore globale
    if _matches_domain(
        host,
        {
            "who.int", "un.org", "europa.eu", "oecd.org", "worldbank.org",
            "imf.org", "nist.gov", "iso.org", "python.org", "mozilla.org",
            "microsoft.com", "google.com", "apple.com"
        },
    ):
        return 0.9

    if _matches_domain(
        host,
        {
            # Agenții de presă globale și ziare de referință
            "reuters.com", "apnews.com", "afp.com", "bbc.com", "bbc.co.uk",
            "nytimes.com", "washingtonpost.com", "theguardian.com", "wsj.com",
            "economist.com", "nationalgeographic.com", "wikipedia.org", "npr.org",
            # Presă de încredere din România
            "europalibera.org", "digi24.ro", "hotnews.ro", "edupedu.ro", "cursdeguvernare.ro", "biziday.ro"
        },
    ):
        return 0.75

    if _matches_domain(
        host,
        {
            "medium.com",
            "substack.com",
            "wordpress.com",
            "blogspot.com",
            "reddit.com",
            "quora.com",
            "stackoverflow.com",
            "stackexchange.com",
            "hackernews.com",
        },
    ) or "forum" in host or "blog" in host:
        return 0.3

    return 0.6


def evaluate_claim(claim: str | ExtractedClaim, sources: list[SourceEvidence]) -> dict[str, str]:
    claim_text = claim.claim if isinstance(claim, ExtractedClaim) else str(claim)
    scored_sources = _rescore_sources(sources)

    if not scored_sources:
        return {
            "verdict": "UNVERIFIABLE",
            "explanation": "No evidence sources were returned for this claim.",
        }

    prompt = _build_claim_evaluation_prompt(claim_text, scored_sources)
    try:
        raw_response = call_llm(prompt)
        parsed = _parse_json_object(raw_response)
        verdict = _normalize_verdict(parsed.get("verdict"))
        explanation = _clean_text(parsed.get("explanation"))
    except Exception:
        verdict = _fallback_verdict(claim_text, scored_sources)
        explanation = ""

    # Păstrăm verdictul inteligent dat de Gemini
    initial_verdict = verdict
    
    # Comentăm / Ștergem intervenția mecanică a Python-ului
    # support_summary = _analyze_evidence_support(claim_text, scored_sources)
    # contradiction_summary = _analyze_evidence_contradictions(claim_text, scored_sources)
    # verdict = _apply_deterministic_verdict_correction( ... )

    if not explanation:
        explanation = _default_explanation(initial_verdict, scored_sources)

    # Returnăm direct ce a decis Gemini
    return {"verdict": initial_verdict, "explanation": explanation}


def score_claim(verdict: str) -> int:
    return CLAIM_SCORE_BY_VERDICT.get(_normalize_verdict(verdict), 25)


def calculate_confidence_score(sources: list[SourceEvidence]) -> int:
    if not sources:
        return 20

    average_credibility = sum(source.credibility_score for source in sources) / len(sources)
    if average_credibility >= 0.85:
        return 90
    if average_credibility >= 0.70:
        return 75
    if average_credibility >= 0.40:
        return 55
    return 30


def fact_check_text(text: str) -> FactCheckingResult:
    if not text.strip():
        return _empty_result()

    claims = extract_claims(text[:MAX_TEXT_CHARS_FOR_LLM])
    if not claims:
        return _empty_result()

    checked_claims: list[FactCheckedClaim] = []
    for extracted_claim in claims:
        queries = generate_queries(extracted_claim.claim)

        try:
            sources = _rescore_sources(search_web_for_evidence(queries))
        except Exception:
            sources = []

        evaluation = evaluate_claim(extracted_claim, sources)
        verdict = _normalize_verdict(evaluation.get("verdict"))
        claim_score = score_claim(verdict)
        confidence_score = calculate_confidence_score(sources)

        checked_claims.append(
            FactCheckedClaim(
                claim=extracted_claim.claim,
                type=extracted_claim.type,
                queries=queries,
                verdict=verdict,
                claim_score=claim_score,
                confidence_score=confidence_score,
                sources=sources,
                explanation=_clean_text(evaluation.get("explanation"))
                or _default_explanation(verdict, sources),
            )
        )

    total_claims = len(checked_claims)
    if total_claims == 0:
        return _empty_result()

    return FactCheckingResult(
        overall_trust_score=round(
            sum(claim.claim_score for claim in checked_claims) / total_claims,
            2,
        ),
        overall_confidence_score=round(
            sum(claim.confidence_score for claim in checked_claims) / total_claims,
            2,
        ),
        total_claims=total_claims,
        claims=checked_claims,
    )


def _build_claim_extraction_prompt(text: str) -> str:
    return (
        "TASK: EXTRACT_FACTUAL_CLAIMS\n"
        "Extract ALL distinct factual claims from the text. Be exhaustive. Break down complex sentences into individual, standalone claims. Do NOT omit critical words like 'first', 'only', 'best', or specific dates/numbers.\n"
        "Ignore opinions, vague statements, and subjective claims.\n"
        "Allowed claim types: historical_fact, statistical_fact, scientific_fact, technical_fact, biographical_fact, current_fact, other.\n\n"
        "EXAMPLE INPUT:\n"
        "Pământul are două luni. Apa fierbe la 100 de grade. După părerea mea, vara e cel mai frumos anotimp.\n"
        "EXAMPLE OUTPUT:\n"
        '{"claims": [{"claim": "Pământul are două luni.", "type": "scientific_fact"}, {"claim": "Apa fierbe la 100 de grade Celsius.", "type": "scientific_fact"}]}\n\n'
        "Return valid JSON only.\n"
        "INPUT_TEXT_START\n"
        f"{text[:MAX_TEXT_CHARS_FOR_LLM]}\n"
        "INPUT_TEXT_END"
    )


def _build_query_generation_prompt(claim: str) -> str:
    return (
        "TASK: GENERATE_SEARCH_QUERIES\n"
        "Generate 2-3 optimized web search queries in ENGLISH for verifying the factual claim, regardless of the claim's original language.\n"
        "Return valid JSON only in this shape: "
        '{"queries":["english query one","english query two"]}.\n'
        "CLAIM_START\n"
        f"{claim}\n"
        "CLAIM_END"
    )


def _build_claim_evaluation_prompt(claim: str, sources: list[SourceEvidence]) -> str:
    evidence_payload = [
        {
            "title": source.title,
            "url": source.url,
            "credibility_score": source.credibility_score,
            "snippet": source.snippet,
        }
        for source in sources
    ]
    return (
        "TASK: EVALUATE_FACTUAL_CLAIM\n"
        "Assign one verdict based only on the supplied evidence.\n"
        "Allowed verdicts: VERIFIED, LIKELY_TRUE, CONTRADICTED, OUTDATED, "
        "UNVERIFIABLE.\n"
        "Strict rules: prefer UNVERIFIABLE over guessing; prefer LIKELY_TRUE over "
        "VERIFIED unless evidence is strong; never invent source names, URLs, or "
        "facts.\n"
        "IMPORTANT CROSS-LANGUAGE RULE: The claim might be in a different language (e.g., Romanian) than the evidence (which is usually in English). You MUST logically translate and match the concepts, dates, and entities across languages, rather than looking for exact word matches.\n"
        "Use only the Evidence JSON below. You may reference only the provided "
        "snippets and URLs. Do not add outside knowledge, source names, URLs, "
        "statistics, dates, or facts that are not present in Evidence.\n"
        "If the evidence clearly supports the claim, return VERIFIED or "
        "LIKELY_TRUE. If the evidence is weak but supportive, return LIKELY_TRUE. "
        "If the evidence is insufficient, return UNVERIFIABLE.\n"
        "Return valid JSON only in this shape: "
        '{"verdict":"UNVERIFIABLE","explanation":"short explanation"}.\n'
        f"Claim: {claim}\n"
        f"Evidence: {json.dumps(evidence_payload, ensure_ascii=True)}"
    )


def _parse_json_object(raw_response: str) -> dict[str, Any]:
    payload = (raw_response or "").strip()
    if not payload:
        raise ValueError("Empty LLM response.")

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM response did not contain a JSON object.")
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON is not an object.")
    return parsed


def _coerce_extracted_claims(value: Any) -> list[ExtractedClaim]:
    if not isinstance(value, list):
        return []

    claims: list[ExtractedClaim] = []
    for item in value:
        claim_text = ""
        claim_type: ClaimType = "other"

        if isinstance(item, str):
            claim_text = item
        elif isinstance(item, dict):
            claim_text = str(item.get("claim") or "")
            claim_type = _normalize_claim_type(item.get("type"))

        cleaned_claim = _clean_text(claim_text)
        if cleaned_claim and _is_probably_factual(cleaned_claim):
            claims.append(ExtractedClaim(claim=cleaned_claim, type=claim_type))

    return claims


def _coerce_query_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [_clean_text(value)]
    if not isinstance(value, list):
        return []

    queries = []
    for item in value:
        cleaned = _clean_text(item)
        if cleaned:
            queries.append(cleaned[:180])
    return queries


def _mock_extract_claim_payload(prompt: str) -> list[dict[str, str]]:
    text = _extract_between(prompt, "INPUT_TEXT_START", "INPUT_TEXT_END")
    claims = _fallback_extract_claims(text)
    return [{"claim": claim.claim, "type": claim.type} for claim in claims]


def _mock_query_payload(prompt: str) -> list[str]:
    claim = _extract_between(prompt, "CLAIM_START", "CLAIM_END")
    return _fallback_queries(claim)


def _mock_evaluation_payload(prompt: str) -> dict[str, str]:
    claim, sources = _extract_evaluation_inputs(prompt)
    verdict = _fallback_verdict(claim, sources)
    return {
        "verdict": verdict,
        "explanation": _default_explanation(verdict, sources),
    }


def _extract_evaluation_inputs(prompt: str) -> tuple[str, list[SourceEvidence]]:
    claim_match = re.search(r"Claim:\s*(.*?)\nEvidence:", prompt, flags=re.DOTALL)
    claim = _clean_text(claim_match.group(1)) if claim_match else ""

    evidence_match = re.search(r"Evidence:\s*(\[.*\])\s*$", prompt, flags=re.DOTALL)
    if not evidence_match:
        return claim, []

    try:
        raw_sources = json.loads(evidence_match.group(1))
    except json.JSONDecodeError:
        return claim, []

    if not isinstance(raw_sources, list):
        return claim, []

    sources: list[SourceEvidence] = []
    for item in raw_sources:
        if not isinstance(item, dict):
            continue

        url = _clean_text(item.get("url"))
        if not url:
            continue

        try:
            sources.append(
                SourceEvidence(
                    title=_clean_text(item.get("title")) or "Evidence source",
                    url=url,
                    credibility_score=score_source_credibility(url),
                    snippet=_clean_text(item.get("snippet")),
                )
            )
        except Exception:
            continue

    return claim, sources


def _fallback_extract_claims(text: str) -> list[ExtractedClaim]:
    claims: list[ExtractedClaim] = []
    for sentence in _split_sentences(text):
        cleaned = _clean_text(sentence)
        if not cleaned or not _is_probably_factual(cleaned):
            continue

        claims.append(
            ExtractedClaim(
                claim=cleaned,
                type=_classify_claim_type(cleaned),
            )
        )

    return _dedupe_claims(claims)[:MAX_CLAIMS]


def _fallback_queries(claim: str) -> list[str]:
    cleaned_claim = _clean_text(claim)
    if not cleaned_claim:
        return []

    keyword_query = " ".join(_extract_keywords(cleaned_claim, limit=8))
    if not keyword_query:
        keyword_query = cleaned_claim[:120]

    return _dedupe_strings(
        [
            cleaned_claim[:180],
            f"{keyword_query} official source",
            f"{keyword_query} evidence",
        ]
    )[:3]


def _fallback_verdict(claim: str, sources: list[SourceEvidence]) -> str:
    if not sources:
        return "UNVERIFIABLE"

    evidence_text = " ".join(f"{source.title} {source.snippet}" for source in sources)
    overlap = _keyword_overlap_ratio(claim, evidence_text)
    contradiction_terms = {
        "false",
        "incorrect",
        "contradicted",
        "debunked",
        "misleading",
        "not true",
    }
    outdated_terms = {"outdated", "superseded", "no longer", "deprecated"}

    lowered_evidence_text = evidence_text.lower()
    if overlap >= 0.45 and any(term in lowered_evidence_text for term in contradiction_terms):
        return "CONTRADICTED"
    if overlap >= 0.45 and any(term in lowered_evidence_text for term in outdated_terms):
        return "OUTDATED"

    contradiction_summary = _analyze_evidence_contradictions(claim, sources)
    if _has_actionable_contradiction(contradiction_summary):
        return "CONTRADICTED"

    support_summary = _analyze_evidence_support(claim, sources)
    if support_summary.medium_quality_support >= 2:
        return "VERIFIED"
    if support_summary.high_quality_support >= 1:
        return "LIKELY_TRUE"
    if support_summary.medium_quality_support >= 1:
        return "LIKELY_TRUE"
    if support_summary.repeated_low_quality_support:
        return "LIKELY_TRUE"
    return "UNVERIFIABLE"


def _support_summary_allows_likely_true(summary: EvidenceSupportSummary) -> bool:
    return (
        summary.high_quality_support >= 1
        or summary.medium_quality_support >= 1
        or summary.repeated_low_quality_support
    )


def _support_summary_allows_verified(summary: EvidenceSupportSummary) -> bool:
    return summary.medium_quality_support >= 2


def _analyze_evidence_support(claim: str, sources: list[SourceEvidence]) -> EvidenceSupportSummary:
    supportive_sources = 0
    high_quality_support = 0
    medium_quality_support = 0
    max_credibility = max((source.credibility_score for source in sources), default=0.0)
    claim_keys = _extract_claim_key_info(claim)
    claim_signatures = _extract_fact_signatures(claim)
    low_quality_support_counts: dict[str, int] = {}

    for source in sources:
        support_key = _source_support_key(source, claim_keys, claim_signatures)
        if not support_key:
            continue

        supportive_sources += 1
        if source.credibility_score >= HIGH_SOURCE_CREDIBILITY_THRESHOLD:
            high_quality_support += 1
        if source.credibility_score >= MEDIUM_SOURCE_CREDIBILITY_THRESHOLD:
            medium_quality_support += 1
        if source.credibility_score < MEDIUM_SOURCE_CREDIBILITY_THRESHOLD:
            low_quality_support_counts[support_key] = low_quality_support_counts.get(support_key, 0) + 1

    return EvidenceSupportSummary(
        supportive_sources=supportive_sources,
        high_quality_support=high_quality_support,
        medium_quality_support=medium_quality_support,
        repeated_low_quality_support=any(count >= 2 for count in low_quality_support_counts.values()),
        max_credibility=max_credibility,
    )


def _analyze_evidence_contradictions(
    claim: str,
    sources: list[SourceEvidence],
) -> EvidenceContradictionSummary:
    claim_keys = _extract_claim_key_info(claim)
    claim_signatures = _extract_fact_signatures(claim)
    correction_counts: dict[str, int] = {}
    contradictory_sources = 0
    medium_quality_contradictions = 0
    related_sources = 0
    max_related_credibility = 0.0

    for source in sources:
        snippet = _clean_text(source.snippet)
        if not snippet or _is_vague_snippet(snippet):
            continue

        if _source_is_related_to_claim(snippet, claim_keys, claim_signatures):
            related_sources += 1
            max_related_credibility = max(max_related_credibility, source.credibility_score)

        correction_key = _source_contradiction_key(snippet, claim_signatures)
        if not correction_key:
            continue

        contradictory_sources += 1
        correction_counts[correction_key] = correction_counts.get(correction_key, 0) + 1
        if source.credibility_score >= CONTRADICTION_SOURCE_CREDIBILITY_THRESHOLD:
            medium_quality_contradictions += 1

    return EvidenceContradictionSummary(
        contradictory_sources=contradictory_sources,
        medium_quality_contradictions=medium_quality_contradictions,
        repeated_correction=any(count >= 2 for count in correction_counts.values()),
        related_sources=related_sources,
        max_related_credibility=max_related_credibility,
    )


def _has_actionable_contradiction(summary: EvidenceContradictionSummary) -> bool:
    return summary.medium_quality_contradictions >= 1 or summary.repeated_correction


def _apply_deterministic_verdict_correction(
    verdict: str,
    sources: list[SourceEvidence],
    support_summary: EvidenceSupportSummary,
    contradiction_summary: EvidenceContradictionSummary,
) -> Verdict:
    normalized_verdict = _normalize_verdict(verdict)
    if _has_actionable_contradiction(contradiction_summary):
        return "CONTRADICTED"

    if (
        normalized_verdict == "CONTRADICTED"
        and support_summary.supportive_sources == 0
        and contradiction_summary.related_sources > 0
        and (
            contradiction_summary.max_related_credibility >= CONTRADICTION_SOURCE_CREDIBILITY_THRESHOLD
            or contradiction_summary.related_sources >= 2
        )
    ):
        return "CONTRADICTED"

    if not sources:
        return "UNVERIFIABLE"

    if support_summary.supportive_sources == 0:
        return "UNVERIFIABLE"

    if _support_summary_allows_verified(support_summary):
        return "VERIFIED"

    if _support_summary_allows_likely_true(support_summary):
        return "LIKELY_TRUE"

    if support_summary.max_credibility < WEAK_SOURCE_CREDIBILITY_THRESHOLD:
        return "UNVERIFIABLE"

    if normalized_verdict == "VERIFIED":
        return "UNVERIFIABLE"

    if normalized_verdict == "LIKELY_TRUE" and support_summary.medium_quality_support >= 1:
        return "LIKELY_TRUE"

    return "UNVERIFIABLE"


def _deterministic_correction_explanation(
    verdict: str,
    support_summary: EvidenceSupportSummary,
    contradiction_summary: EvidenceContradictionSummary,
) -> str:
    if verdict == "CONTRADICTED":
        return (
            "The evidence discusses the same claim subject but gives a different "
            "factual value, so the claim is treated as contradicted."
        )
    if verdict == "VERIFIED":
        return (
            "At least two medium- or high-credibility evidence snippets contain the "
            "claim's key details, so the claim is treated as verified."
        )
    if verdict == "LIKELY_TRUE":
        if support_summary.repeated_low_quality_support and support_summary.medium_quality_support == 0:
            return (
                "Multiple lower-credibility snippets consistently contain the "
                "claim's key details, so the claim is likely true with low confidence."
            )
        return (
            "At least one credible evidence snippet contains the claim's key "
            "details, but the evidence is limited, so the claim is likely true."
        )
    if support_summary.supportive_sources == 0:
        if contradiction_summary.contradictory_sources > 0:
            return "The contradiction evidence was too weak or isolated to support a contradicted verdict."
        return "The returned snippets do not contain enough key claim details to verify the claim."
    return "The available sources are too weak or limited to verify the claim."


def _extract_claim_key_info(claim: str) -> ClaimKeyInfo:
    named_entities = _extract_named_entities(claim)
    numbers = _extract_key_numbers(claim)
    important_nouns = _extract_important_nouns(claim, named_entities)
    return ClaimKeyInfo(
        named_entities=named_entities,
        important_nouns=important_nouns,
        numbers=numbers,
    )


def _source_supports_claim(source: SourceEvidence, claim_keys: ClaimKeyInfo) -> bool:
    snippet = _clean_text(source.snippet)
    if not snippet or _is_vague_or_contradictory_snippet(snippet):
        return False

    if claim_keys.numbers and not all(_number_present_in_text(number, snippet) for number in claim_keys.numbers):
        return False

    if claim_keys.named_entities:
        return _enough_terms_present(
            terms=claim_keys.named_entities,
            text=snippet,
            minimum_ratio=MIN_ENTITY_SUPPORT_RATIO,
            require_all_when_small=True,
        )

    return _enough_terms_present(
        terms=claim_keys.important_nouns,
        text=snippet,
        minimum_ratio=MIN_NOUN_SUPPORT_RATIO,
        require_all_when_small=False,
    )


def _source_support_key(
    source: SourceEvidence,
    claim_keys: ClaimKeyInfo,
    claim_signatures: list[FactSignature],
) -> str | None:
    snippet = _clean_text(source.snippet)
    if not snippet or _is_vague_or_contradictory_snippet(snippet):
        return None

    if claim_keys.numbers and not all(_number_present_in_text(number, snippet) for number in claim_keys.numbers):
        return None

    if claim_signatures and _source_supports_fact_signatures(snippet, claim_signatures):
        return _claim_signature_support_key(claim_signatures)

    if _source_supports_claim(source, claim_keys):
        return _claim_key_support_key(claim_keys)

    return None


def _source_supports_fact_signatures(
    snippet: str,
    claim_signatures: list[FactSignature],
) -> bool:
    source_signatures = _extract_fact_signatures(snippet)
    if not source_signatures:
        return False

    return all(
        _source_has_matching_support_signature(claim_signature, source_signatures)
        for claim_signature in claim_signatures
    )


def _source_has_matching_support_signature(
    claim_signature: FactSignature,
    source_signatures: list[FactSignature],
) -> bool:
    for source_signature in source_signatures:
        if claim_signature.relation != source_signature.relation:
            continue
        if not _same_fact_subject(claim_signature.subject, source_signature.subject):
            continue
        if _fact_values_support(
            relation=claim_signature.relation,
            claim_value=claim_signature.value,
            source_value=source_signature.value,
        ):
            return True

    return False


def _claim_signature_support_key(claim_signatures: list[FactSignature]) -> str:
    signature_keys = [
        "|".join(
            [
                signature.relation,
                _normalize_for_matching(signature.subject),
                _canonical_fact_value(signature.relation, signature.value),
            ]
        )
        for signature in claim_signatures
    ]
    return "signature:" + ";".join(sorted(signature_keys))


def _claim_key_support_key(claim_keys: ClaimKeyInfo) -> str:
    parts = [
        *(_normalize_for_matching(entity) for entity in claim_keys.named_entities),
        *(_normalize_for_matching(noun) for noun in claim_keys.important_nouns[:4]),
        *(_normalize_for_matching(number) for number in claim_keys.numbers),
    ]
    cleaned_parts = [part for part in parts if part]
    return "keys:" + ";".join(sorted(cleaned_parts))


def _source_is_related_to_claim(
    snippet: str,
    claim_keys: ClaimKeyInfo,
    claim_signatures: list[FactSignature],
) -> bool:
    if claim_signatures:
        for signature in claim_signatures:
            if _term_present_in_text(signature.subject, snippet) and _relation_marker_present(
                signature.relation,
                snippet,
            ):
                return True

    if claim_keys.named_entities and _enough_terms_present(
        terms=claim_keys.named_entities,
        text=snippet,
        minimum_ratio=MIN_ENTITY_SUPPORT_RATIO,
        require_all_when_small=False,
    ):
        return True

    return _enough_terms_present(
        terms=claim_keys.important_nouns,
        text=snippet,
        minimum_ratio=MIN_NOUN_SUPPORT_RATIO,
        require_all_when_small=False,
    )


def _source_contradiction_key(
    snippet: str,
    claim_signatures: list[FactSignature],
) -> str | None:
    if not claim_signatures:
        return None

    source_signatures = _extract_fact_signatures(snippet)
    for claim_signature in claim_signatures:
        for source_signature in source_signatures:
            if claim_signature.relation != source_signature.relation:
                continue
            if not _same_fact_subject(claim_signature.subject, source_signature.subject):
                continue
            if not _fact_values_conflict(
                relation=claim_signature.relation,
                claim_value=claim_signature.value,
                source_value=source_signature.value,
            ):
                continue

            return "|".join(
                [
                    claim_signature.relation,
                    _normalize_for_matching(claim_signature.subject),
                    _canonical_fact_value(claim_signature.relation, source_signature.value),
                ]
            )

    return None


def _extract_fact_signatures(text: str) -> list[FactSignature]:
    signatures: list[FactSignature] = []

    relation_patterns = [
        (
            "capital",
            r"\bcapital\s+of\s+(?P<subject>[A-Za-z][A-Za-z .'-]{1,70}?)\s+(?:is|was|=|:)\s+(?P<value>[A-Za-z][A-Za-z .'-]{1,70})",
        ),
        (
            "capital",
            r"\b(?P<value>[A-Za-z][A-Za-z .'-]{1,70}?)\s+(?:is|was)\s+(?:the\s+)?capital\s+of\s+(?P<subject>[A-Za-z][A-Za-z .'-]{1,70})",
        ),
        (
            "capital",
            r"\b(?P<subject>[A-Za-z][A-Za-z .'-]{1,70})'s\s+capital\s+(?:is|was|=|:)\s+(?P<value>[A-Za-z][A-Za-z .'-]{1,70})",
        ),
        (
            "capital",
            r"\b(?P<value>[A-Za-z][A-Za-z .'-]{1,70}?)\s+(?:is|was)\s+(?P<subject>[A-Za-z][A-Za-z .'-]{1,70})'s\s+capital\b",
        ),
        (
            "capital",
            r"\b(?P<subject>[A-Za-z][A-Za-z .'-]{2,40})\s+capital\s+(?P<value>[A-Za-z][A-Za-z .'-]{1,70})",
        ),
        (
            "located_in",
            r"\b(?P<subject>(?:the\s+)?[A-Za-z][A-Za-z0-9 .'-]{1,90}?)\s+(?:is|was)\s+(?:located\s+)?in\s+(?P<value>[A-Za-z][A-Za-z .'-]{1,70})",
        ),
        (
            "founded_by",
            r"\b(?P<subject>[A-Za-z][A-Za-z0-9 .&'-]{1,90}?)\s+(?:is|was)?\s*founded\s+by\s+(?P<value>[A-Za-z][A-Za-z .&'-]{1,120})",
        ),
        (
            "founded_by",
            r"\b(?P<value>[A-Za-z][A-Za-z .&'-]{1,120}?)\s+founded\s+(?P<subject>[A-Z][A-Za-z0-9 .&'-]{1,90})(?:\s+with\s+(?P<extra>[A-Za-z][A-Za-z .&'-]{1,80}))?",
        ),
        (
            "created_by",
            r"\b(?P<subject>[A-Za-z][A-Za-z0-9 .&'-]{1,90}?)\s+(?:is|was)?\s*created\s+by\s+(?P<value>[A-Za-z][A-Za-z .&'-]{1,120})",
        ),
        (
            "created_by",
            r"\b(?P<value>[A-Za-z][A-Za-z .&'-]{1,120}?)\s+created\s+(?P<subject>[A-Z][A-Za-z0-9 .&'-]{1,90})",
        ),
        (
            "released_year",
            r"\b(?P<subject>[A-Za-z][A-Za-z0-9 .&'-]{1,90}?)\s+(?:was\s+)?(?:first\s+)?released\s+in\s+(?P<value>\d{4})",
        ),
        (
            "released_year",
            r"\b(?P<subject>[A-Za-z][A-Za-z0-9 .&'-]{1,90}?)\s+(?:initial|first)\s+release\s+(?:was\s+)?(?:in\s+)?(?P<value>\d{4})",
        ),
        (
            "made_of",
            r"\b(?P<subject>(?:the\s+)?[A-Za-z][A-Za-z .'-]{1,70}?)\s+(?:is|was)\s+(?:made|composed)\s+of\s+(?P<value>[^.;]+)",
        ),
        (
            "made_of",
            r"\b(?P<subject>(?:the\s+)?[A-Za-z][A-Za-z .'-]{1,70}?)\s+consists\s+of\s+(?P<value>[^.;]+)",
        ),
        (
            "shape",
            r"\b(?P<subject>(?:the\s+)?(?:earth|moon|planet\s+earth))\s+(?:is|was)\s+(?P<value>flat|round|spherical|an?\s+oblate\s+spheroid)\b",
        ),
        (
            "boils_at",
            r"\b(?P<subject>water)\s+boils\s+at\s+(?P<value>\d+(?:\.\d+)?\s*(?:degrees\s*)?(?:c|f|celsius|fahrenheit)?)",
        ),
        (
            "boils_at",
            r"\bboiling\s+point\s+of\s+(?P<subject>water)\s+(?:is|=)\s+(?P<value>\d+(?:\.\d+)?\s*(?:degrees\s*)?(?:c|f|celsius|fahrenheit)?)",
        ),
        (
            "boils_at",
            r"\b(?P<subject>water)\s+boils\s+at\s+(?P<value>\d+(?:\.\d+)?\s*(?:°\s*)?(?:c|f|celsius|fahrenheit)?)",
        ),
        (
            "boils_at",
            r"\bboiling\s+point\s+of\s+(?P<subject>water)\s+(?:is|=)\s+(?P<value>\d+(?:\.\d+)?\s*(?:°\s*)?(?:c|f|celsius|fahrenheit)?)",
        ),
        (
            "joined_year",
            r"\b(?P<member>[A-Za-z][A-Za-z .'-]{1,60}?)\s+joined\s+(?:the\s+)?(?P<org>EU|E\.U\.|European\s+Union|NATO|United\s+Nations|UN)\s+in\s+(?P<value>\d{4})",
        ),
        (
            "joined_year",
            r"\b(?P<member>[A-Za-z][A-Za-z .'-]{1,60}?)\s+became\s+a\s+member\s+of\s+(?:the\s+)?(?P<org>EU|E\.U\.|European\s+Union|NATO|United\s+Nations|UN)\s+in\s+(?P<value>\d{4})",
        ),
    ]

    for relation, pattern in relation_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if relation == "joined_year":
                subject = f"{match.group('member')} {match.group('org')}"
            else:
                subject = match.group("subject")
            if relation == "released_year":
                subject = re.split(r"\s+(?:is|was)?\s*created\s+by\s+", subject, maxsplit=1, flags=re.IGNORECASE)[0]
            value = match.group("value")
            extra_value = match.groupdict().get("extra")
            if extra_value:
                value = f"{value} and {extra_value}"

            signature = _make_fact_signature(
                relation=relation,
                subject=subject,
                value=value,
            )
            if signature:
                signatures.append(signature)

    return _dedupe_fact_signatures(signatures)


def _make_fact_signature(relation: str, subject: str, value: str) -> FactSignature | None:
    cleaned_subject = _clean_entity(subject)
    cleaned_value = _clean_fact_value(value)
    if not cleaned_subject or not cleaned_value:
        return None
    if _is_malformed_fact_signature(relation, cleaned_subject, cleaned_value):
        return None
    return FactSignature(relation=relation, subject=cleaned_subject, value=cleaned_value)


def _is_malformed_fact_signature(relation: str, subject: str, value: str) -> bool:
    normalized_subject = _normalize_for_matching(subject)
    normalized_value = _normalize_for_matching(value)
    invalid_single_terms = {"a", "an", "the", "by", "of", "is", "was", "are", "were"}

    if normalized_subject in invalid_single_terms or normalized_value in invalid_single_terms:
        return True
    if normalized_subject.startswith(("by ", "of ")) or normalized_value.startswith(("by ", "of ")):
        return True
    if normalized_subject.endswith((" is", " was", " are", " were")):
        return True
    if normalized_value.endswith((" is", " was", " are", " were")):
        return True
    if relation in {"founded_by", "created_by"} and " by " in f" {normalized_subject} ":
        return True
    return False


def _dedupe_fact_signatures(signatures: list[FactSignature]) -> list[FactSignature]:
    seen: set[str] = set()
    deduped: list[FactSignature] = []
    for signature in signatures:
        key = "|".join(
            [
                signature.relation,
                _normalize_for_matching(signature.subject),
                _canonical_fact_value(signature.relation, signature.value),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(signature)
    return deduped[:8]


def _same_fact_subject(left: str, right: str) -> bool:
    return _term_present_in_text(left, right) or _term_present_in_text(right, left)


def _fact_values_conflict(relation: str, claim_value: str, source_value: str) -> bool:
    if _fact_values_support(relation, claim_value, source_value):
        return False

    if relation in {"founded_by", "created_by"}:
        claim_terms = _extract_fact_value_terms(claim_value)
        source_terms = _extract_fact_value_terms(source_value)
        if claim_terms and source_terms:
            return not any(_term_present_in_text(term, source_value) for term in claim_terms)
        return False

    claim_group = _canonical_fact_value(relation, claim_value)
    source_group = _canonical_fact_value(relation, source_value)
    if claim_group and source_group and claim_group != source_group:
        return True

    if relation in {"boils_at", "joined_year", "released_year"}:
        claim_numbers = _extract_key_numbers(claim_value)
        source_numbers = _extract_key_numbers(source_value)
        return bool(claim_numbers and source_numbers and not set(claim_numbers) & set(source_numbers))

    return relation in {"capital", "located_in", "made_of", "shape"}


def _fact_values_support(relation: str, claim_value: str, source_value: str) -> bool:
    if _fact_values_equivalent(claim_value, source_value):
        return True

    if relation in {"boils_at", "joined_year", "released_year"}:
        claim_numbers = {_normalize_for_matching(number) for number in _extract_key_numbers(claim_value)}
        source_numbers = {_normalize_for_matching(number) for number in _extract_key_numbers(source_value)}
        return bool(claim_numbers and claim_numbers <= source_numbers)

    if relation in {"founded_by", "created_by"}:
        claim_terms = _extract_fact_value_terms(claim_value)
        if not claim_terms:
            return False
        return all(_term_present_in_text(term, source_value) for term in claim_terms)

    claim_group = _canonical_fact_value(relation, claim_value)
    source_group = _canonical_fact_value(relation, source_value)
    return bool(claim_group and source_group and claim_group == source_group)


def _extract_fact_value_terms(value: str) -> list[str]:
    terms: list[str] = []
    for entity in _extract_named_entities(value):
        parts = [
            _clean_text(part)
            for part in re.split(r"\s+(?:and|with)\s+", entity, flags=re.IGNORECASE)
        ]
        terms.extend(part for part in parts if part)

    if not terms:
        terms = _extract_keywords(value, limit=6)

    return _dedupe_strings(terms)[:6]


def _fact_values_equivalent(left: str, right: str) -> bool:
    normalized_left = _normalize_for_matching(left)
    normalized_right = _normalize_for_matching(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True
    return _contains_normalized_phrase(normalized_left, normalized_right) or _contains_normalized_phrase(
        normalized_right,
        normalized_left,
    )


def _canonical_fact_value(relation: str, value: str) -> str:
    normalized = _normalize_for_matching(value)
    if relation == "shape":
        if _contains_any(normalized, {"flat"}):
            return "flat"
        if _contains_any(normalized, {"round", "spherical", "sphere", "oblate spheroid"}):
            return "round"

    if relation == "made_of":
        if _contains_any(normalized, {"cheese"}):
            return "cheese"
        if _contains_any(normalized, {"rock", "rocks", "dust", "regolith", "basalt", "mineral"}):
            return "rock_dust_regolith"

    if relation in {"boils_at", "joined_year", "released_year"}:
        numbers = _extract_key_numbers(value)
        return _normalize_for_matching(numbers[0]) if numbers else normalized

    return normalized


def _relation_marker_present(relation: str, text: str) -> bool:
    normalized = _normalize_for_matching(text)
    markers_by_relation = {
        "capital": {"capital"},
        "located_in": {"located in", " in "},
        "founded_by": {"founded by", "founded"},
        "created_by": {"created by", "created"},
        "released_year": {"released in", "first released", "initial release"},
        "made_of": {"made of", "composed of", "consists of"},
        "shape": {"flat", "round", "spherical", "sphere", "oblate spheroid"},
        "boils_at": {"boils at", "boiling point"},
        "joined_year": {"joined", "became a member"},
    }
    markers = markers_by_relation.get(relation, set())
    return any(_contains_normalized_phrase(normalized, _normalize_for_matching(marker)) for marker in markers)


def _clean_fact_value(value: str) -> str:
    cleaned = _clean_text(value)
    cleaned = re.split(
        r"\s+and\s+(?:first\s+)?released\b|\b(?:according\s+to|although|because|but|while|which|where|when)\b|[.;]",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.strip(" ,:!?").split()[:8])


def _contains_any(normalized_text: str, terms: set[str]) -> bool:
    return any(
        _contains_normalized_phrase(normalized_text, _normalize_for_matching(term))
        for term in terms
    )


def _extract_named_entities(text: str) -> list[str]:
    entity_pattern = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9&.-]*(?:\s+(?:of|and|the|de|la|[A-Z][A-Za-z0-9&.-]*))*|[A-Z]{2,})\b"
    )
    entities: list[str] = []
    for match in entity_pattern.finditer(text):
        entity = _clean_entity(match.group(0))
        if not entity:
            continue

        lowered = entity.lower()
        if lowered in FACT_CHECK_STOPWORDS or len(lowered) < 2:
            continue
        if entity.isupper() and len(entity) == 1:
            continue
        entities.append(entity)

    return _dedupe_strings(entities)[:8]


def _extract_key_numbers(text: str) -> list[str]:
    number_pattern = re.compile(
        r"(?<![A-Za-z0-9])\d{1,4}(?:[,.]\d{3})*(?:\.\d+)?(?:\s*(?:%|percent|million|billion|trillion|km|kg|usd|eur|dollars|euros))?(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    return _dedupe_strings([match.group(0) for match in number_pattern.finditer(text)])[:6]


def _extract_important_nouns(text: str, named_entities: list[str]) -> list[str]:
    entity_words = {
        word.lower()
        for entity in named_entities
        for word in re.findall(r"[A-Za-z0-9]+", entity)
    }
    nouns: list[str] = []

    for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text):
        lowered = word.lower()
        if lowered in FACT_CHECK_STOPWORDS or lowered in entity_words:
            continue
        if any(char.isdigit() for char in lowered):
            continue
        nouns.append(lowered)

    return _dedupe_strings(nouns)[:8]


def _enough_terms_present(
    terms: list[str],
    text: str,
    minimum_ratio: float,
    require_all_when_small: bool,
) -> bool:
    cleaned_terms = _dedupe_strings(terms)
    if not cleaned_terms:
        return False

    matched_terms = sum(1 for term in cleaned_terms if _term_present_in_text(term, text))
    if require_all_when_small and len(cleaned_terms) <= 2:
        return matched_terms == len(cleaned_terms)

    required_matches = max(MIN_NOUNS_FOR_SUPPORT, round(len(cleaned_terms) * minimum_ratio))
    required_matches = min(required_matches, len(cleaned_terms))
    return matched_terms >= required_matches


def _term_present_in_text(term: str, text: str) -> bool:
    normalized_text = _normalize_for_matching(text)
    for variant in _term_variants(term):
        normalized_variant = _normalize_for_matching(variant)
        if not normalized_variant:
            continue
        if _contains_normalized_phrase(normalized_text, normalized_variant):
            return True
    return False


def _term_variants(term: str) -> set[str]:
    cleaned = _clean_text(term)
    normalized = _normalize_for_matching(cleaned)
    variants = {cleaned, normalized}
    variants.update(ENTITY_VARIANTS.get(normalized, set()))

    words = normalized.split()
    if len(words) > 1:
        acronym = "".join(word[0] for word in words if word and word not in {"of", "and", "the"})
        if len(acronym) >= 2:
            variants.add(acronym)

    if normalized.endswith("s") and len(normalized) > 4:
        variants.add(normalized[:-1])
    else:
        variants.add(f"{normalized}s")

    return {variant for variant in variants if variant}


def _number_present_in_text(number: str, text: str) -> bool:
    normalized_text = _normalize_for_matching(text)
    return any(
        _contains_normalized_phrase(normalized_text, _normalize_for_matching(variant))
        for variant in _number_variants(number)
    )


def _number_variants(number: str) -> set[str]:
    cleaned = _clean_text(number).lower()
    compact = cleaned.replace(",", "")
    variants = {cleaned, compact}

    percent_match = re.match(r"^(\d+(?:\.\d+)?)\s*(?:%|percent)$", compact)
    if percent_match:
        value = percent_match.group(1)
        variants.add(f"{value}%")
        variants.add(f"{value} percent")

    return {variant for variant in variants if variant}


def _contains_normalized_phrase(normalized_text: str, normalized_phrase: str) -> bool:
    if not normalized_phrase:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])", normalized_text) is not None


def _normalize_for_matching(value: str) -> str:
    normalized = value.lower().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9%]+", " ", normalized)
    return " ".join(normalized.split())


def _clean_entity(entity: str) -> str:
    cleaned = _clean_text(entity)
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:is|was|were|are|has|have|had)$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" .,:;!?")


def _is_vague_or_contradictory_snippet(snippet: str) -> bool:
    lowered = snippet.lower()
    vague_terms = {
        "may refer to",
        "disambiguation",
        "possibly",
        "rumor",
        "unconfirmed",
    }
    contradiction_terms = {
        "false",
        "incorrect",
        "contradicted",
        "debunked",
        "misleading",
        "not true",
        "did not",
        "never",
    }
    return any(term in lowered for term in vague_terms | contradiction_terms)


def _is_vague_snippet(snippet: str) -> bool:
    lowered = snippet.lower()
    vague_terms = {
        "may refer to",
        "disambiguation",
        "possibly",
        "rumor",
        "unconfirmed",
    }
    return any(term in lowered for term in vague_terms)


def _guard_verdict_with_evidence(verdict: str, sources: list[SourceEvidence]) -> str:
    normalized = _normalize_verdict(verdict)
    if not sources:
        return "UNVERIFIABLE"

    average_credibility = sum(source.credibility_score for source in sources) / len(sources)
    if normalized == "VERIFIED" and (len(sources) < 2 or average_credibility < 0.85):
        return "LIKELY_TRUE" if average_credibility >= 0.70 else "UNVERIFIABLE"

    return normalized


def _default_explanation(verdict: str, sources: list[SourceEvidence]) -> str:
    if not sources:
        return "No evidence was available, so the claim is treated as unverifiable."
    if verdict == "UNVERIFIABLE":
        return "The returned evidence is not strong enough to verify or contradict the claim."
    if verdict == "VERIFIED":
        return "Multiple medium- or high-credibility sources strongly support the claim."
    if verdict == "LIKELY_TRUE":
        return "The available evidence appears supportive but is not strong enough for a verified verdict."
    if verdict == "OUTDATED":
        return "The available evidence suggests the claim may have been true but is no longer current."
    return "The available evidence contradicts the claim."


def _rescore_sources(sources: list[SourceEvidence]) -> list[SourceEvidence]:
    rescored: list[SourceEvidence] = []
    for source in sources[:MAX_SOURCES_PER_CLAIM]:
        try:
            rescored.append(
                SourceEvidence(
                    title=_clean_text(source.title) or "Untitled source",
                    url=_clean_text(source.url),
                    credibility_score=score_source_credibility(source.url),
                    snippet=_clean_text(source.snippet),
                )
            )
        except Exception:
            continue
    return rescored


def _normalize_claim_type(value: Any) -> ClaimType:
    normalized = str(value or "").strip().lower()
    if normalized in ALLOWED_CLAIM_TYPES:
        return normalized  # type: ignore[return-value]
    return "other"


def _normalize_verdict(value: Any) -> Verdict:
    normalized = str(value or "").strip().upper()
    if normalized in ALLOWED_VERDICTS:
        return normalized  # type: ignore[return-value]
    return "UNVERIFIABLE"


def _classify_claim_type(claim: str) -> ClaimType:
    lowered = claim.lower()
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:%|percent|million|billion|trillion)\b", lowered):
        return "statistical_fact"
    if any(term in lowered for term in ["study", "species", "molecule", "planet", "cell", "climate"]):
        return "scientific_fact"
    if any(term in lowered for term in ["software", "api", "algorithm", "database", "python", "server"]):
        return "technical_fact"
    if any(term in lowered for term in ["born", "died", "biography", "president", "ceo"]):
        return "biographical_fact"
    if any(term in lowered for term in ["currently", "today", "recently", "as of", "now"]):
        return "current_fact"
    if re.search(r"\b(?:1[5-9]\d{2}|20\d{2})\b", lowered):
        return "historical_fact"
    return "other"

def _is_probably_factual(sentence: str) -> bool:
    # Lăsăm LLM-ul să decidă ce e factual, nu blocăm din Python!
    return True

# def _is_probably_factual(sentence: str) -> bool:
#     lowered = sentence.lower().strip()
#     if len(lowered.split()) < 4:
#         return False

#     subjective_markers = {
#         "i think",
#         "i believe",
#         "in my opinion",
#         "best",
#         "worst",
#         "beautiful",
#         "amazing",
#         "boring",
#         "should",
#         "could",
#         "might",
#         "may",
#         "will probably",
#     }
#     if any(marker in lowered for marker in subjective_markers):
#         return False

#     factual_patterns = [
#         r"\b\d+(?:\.\d+)?\b",
#         r"\b(?:is|are|was|were|has|have|had|contains|uses|supports|released|founded)\b",
#         r"\b(?:born|died|located|invented|discovered|published|created|developed)\b",
#         r"\b(?:according to|as of|currently)\b",
#     ]
#     return any(re.search(pattern, lowered) for pattern in factual_patterns)


def _extract_keywords(text: str, limit: int) -> list[str]:
    stopwords = {
        "the",
        "and",
        "that",
        "this",
        "with",
        "from",
        "into",
        "than",
        "then",
        "there",
        "their",
        "have",
        "has",
        "had",
        "was",
        "were",
        "are",
        "for",
        "about",
        "claim",
    }
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", text)
    keywords = []
    for word in words:
        lowered = word.lower()
        if lowered in stopwords or len(lowered) < 3:
            continue
        keywords.append(word)
    return keywords[:limit]


def _keyword_overlap_ratio(claim: str, evidence_text: str) -> float:
    claim_keywords = {word.lower() for word in _extract_keywords(claim, limit=20)}
    if not claim_keywords:
        return 0.0

    evidence_words = {word.lower() for word in _extract_keywords(evidence_text, limit=200)}
    if not evidence_words:
        return 0.0

    return len(claim_keywords & evidence_words) / len(claim_keywords)


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    start_index = text.find(start_marker)
    end_index = text.find(end_marker)
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        return ""
    return text[start_index + len(start_marker) : end_index].strip()


def _extract_host(url: str) -> str:
    cleaned = (url or "").strip()
    if not cleaned:
        return ""

    parsed = urlparse(cleaned if "://" in cleaned else f"https://{cleaned}")
    host = parsed.netloc.lower()
    if "@" in host:
        host = host.rsplit("@", maxsplit=1)[-1]
    if ":" in host:
        host = host.split(":", maxsplit=1)[0]
    return host.removeprefix("www.")


def _has_domain_suffix(host: str, suffixes: list[str]) -> bool:
    return any(host.endswith(suffix) for suffix in suffixes)


def _matches_domain(host: str, domains: set[str]) -> bool:
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _dedupe_claims(claims: list[ExtractedClaim]) -> list[ExtractedClaim]:
    seen: set[str] = set()
    deduped: list[ExtractedClaim] = []
    for claim in claims:
        key = claim.claim.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(claim)
    return deduped


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        cleaned = _clean_text(value)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        value = "" if value is None else str(value)
    return " ".join(value.strip().split())


def _empty_result() -> FactCheckingResult:
    return FactCheckingResult(
        overall_trust_score=0,
        overall_confidence_score=0,
        total_claims=0,
        claims=[],
    )
