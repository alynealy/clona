from __future__ import annotations

from .fact_checking_agent import FactCheckingResult, fact_check_text #nu stiu daca e bine

import logging
from typing import Any, Literal
import re
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .db import fetch_history_for_user, init_db
from .services import build_text_verification_result
import trafilatura

logger = logging.getLogger(__name__)


class TextVerificationRequest(BaseModel):
    user_email: str = Field(min_length=3)
    text: str = Field(min_length=10)
    input_type: str = Field(default="text")


class EligibilityResponse(BaseModel):
    strong_verdict_allowed: bool
    reasons: list[str]


class DocumentAssessmentResponse(BaseModel):
    ai_likelihood_score: float
    ai_likelihood_label: str
    confidence: str


class SignalBreakdownResponse(BaseModel):
    semantic_model_score: float
    stylometric_score: float
    robustness_score: float


class SegmentSpanResponse(BaseModel):
    start: int
    end: int
    score: float


class SegmentAssessmentResponse(BaseModel):
    available: bool
    spans: list[SegmentSpanResponse]


class HighlightResponse(BaseModel):
    text: str
    start: int
    end: int
    reason: str


class TextAnalysisMetricsResponse(BaseModel):
    sentence_count: int
    sentence_lengths: list[int]
    length_variation_score: float
    repeated_linking_words: dict[str, int]
    expressive_repetition_score: float
    expressive_repeated_phrases: list[str]
    linguistic_style_score: float
    sentence_length_range: int
    sentence_length_std_dev: float
    consecutive_diff_over_10: bool
    linking_word_ai_score: float
    linguistic_ai_score: float


class DetectorDetailsResponse(BaseModel):
    status: str
    score_semantics: str
    raw_score: float | None
    observations: list[str]
    influential_phrases: list[str]
    technical_note: str | None
    schema_present_keys: list[str] | None = None
    invoke_error_type: str | None = None
    invoke_error_message: str | None = None
    invoke_error_status_code: int | None = None
    invoke_error_body: str | None = None
    invoke_error_provider: str | None = None
    invoke_error_model: str | None = None
    invoke_error_base_url: str | None = None
    invoke_error_timeout_seconds: float | None = None
    raw_output_excerpt: str | None = None
    diagnostic_timestamp: str | None = None


class GrammaticalResultResponse(BaseModel):
    score: int = Field(ge=0, le=100)
    confidence: Literal["low", "medium", "high"]
    reasons_for_rating: list[str]
    lowered_confidence_reasons: list[str]


class TextVerificationResponse(BaseModel):
    title: str
    verification_title: str
    language: str
    eligibility: EligibilityResponse
    document_assessment: DocumentAssessmentResponse
    signal_breakdown: SignalBreakdownResponse
    why: list[str]
    what_weakens_the_conclusion: list[str]
    segment_assessment: SegmentAssessmentResponse
    final_user_message: str
    verdict: str
    percentage: int
    final_label: str
    bullet_points: list[str]
    summary: list[str]
    limitations: list[str]
    detector_details: DetectorDetailsResponse
    highlights: list[HighlightResponse]
    metrics: TextAnalysisMetricsResponse
    grammatical_result: GrammaticalResultResponse
    fact_checking_result: dict[str, Any] | None    #adaugat
    master_result: dict[str, Any]


class HistoryEntry(BaseModel):
    id: int
    user_email: str
    input_type: str
    submitted_text: str
    text_preview: str
    verification_rating: int | None
    statistical_percentage: int
    confidence: str
    structured_result: dict[str, Any]
    created_at: str


app = FastAPI(title="CheckWise Backend", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    init_db()


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/api/fact-check", response_model=FactCheckingResult)   #nu stiu daca e bine
def fact_check(payload: TextVerificationRequest) -> FactCheckingResult:
    try:
        return fact_check_text(payload.text)
    except Exception as exc:
        logger.exception("Fact-checking failed for text_length=%s", len(payload.text))
        raise HTTPException(status_code=500, detail=f"Fact-checking failed: {exc}") from exc


# @app.post("/api/text/verify", response_model=TextVerificationResponse)
# def verify_text(payload: TextVerificationRequest) -> TextVerificationResponse:
#     if payload.input_type != "text":
#         raise HTTPException(status_code=400, detail="Only text verification is supported by this endpoint.")

#     try:
#         result = build_text_verification_result(user_email=payload.user_email, text=payload.text)
#         return TextVerificationResponse(**result)
#     except ValueError as exc:
#         raise HTTPException(status_code=400, detail=str(exc)) from exc
#     except Exception as exc:
#         logger.exception(
#             "Text verification failed for user_email=%s text_length=%s",
#             payload.user_email,
#             len(payload.text),
#         )
#         raise HTTPException(status_code=500, detail=f"Text verification failed: {exc}") from exc
def sanitize_extracted_text(text: str) -> str:
    """Curăță textul extras pentru a preveni erorile de parsare JSON în agenți."""
    if not text:
        return ""
    # Eliminăm caracterele de control (ASCII 0-31)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # Înlocuim ghilimelele duble cu simple pentru a nu sparge JSON-ul generat de LLM
    text = text.replace('"', "'").replace('“', "'").replace('”', "'")
    # Consolidăm spațiile albe
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.post("/api/text/verify", response_model=TextVerificationResponse)
def verify_text(payload: TextVerificationRequest) -> TextVerificationResponse:
    is_url = payload.input_type == "url" or payload.text.strip().startswith(("http://", "https://"))
    final_text = payload.text
    
    if is_url:
        try:
            downloaded = trafilatura.fetch_url(payload.text.strip())
            extracted = trafilatura.extract(downloaded)
            
            if not extracted:
                raise HTTPException(status_code=400, detail="Nu am putut extrage text util de la acest URL.")
            
            # Sanitizăm textul extras
            sanitized = sanitize_extracted_text(extracted)
            
            # Limităm lungimea la ~3500 caractere pentru a evita tăierea răspunsului JSON de către AI
            final_text = sanitized[:3500]
            
            print(f"DEBUG: Text extras și sanitizat (lungime {len(final_text)})")
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Eroare la procesarea URL-ului: {str(e)}")

    try:
        result = build_text_verification_result(user_email=payload.user_email, text=final_text)
        return TextVerificationResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "Text verification failed for user_email=%s text_length=%s",
            payload.user_email,
            len(final_text),
        )
        raise HTTPException(status_code=500, detail=f"Text verification failed: {exc}") from exc
    
@app.get("/api/history", response_model=list[HistoryEntry])
def get_history(user_email: str = Query(min_length=3)) -> list[HistoryEntry]:
    entries = fetch_history_for_user(user_email)
    return [HistoryEntry(**entry) for entry in entries]
