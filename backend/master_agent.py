from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


MASTER_AGENT_MODEL = "llama3.2:1b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class AgentScoreDetail(TypedDict):
    agent: str
    label: str
    score: float | None
    original_score: float | None
    used: bool
    source_field: str
    explanation: str


class MasterAgentResult(TypedDict):
    agent: Literal["master"]
    title: str
    model: str
    score: int | None
    raw_score: float | None
    label: str
    available: bool
    used_agents: list[str]
    missing_agents: list[str]
    formula: str
    details: dict[str, Any]


class MasterAgentState(TypedDict, total=False):
    statistical_result: dict[str, Any] | None
    grammatical_result: dict[str, Any] | None
    fact_checking_result: dict[str, Any] | None
    score_details: list[AgentScoreDetail]
    missing_agents: list[str]
    raw_score: float | None
    final_score: int | None
    result: MasterAgentResult


@dataclass
class MasterAgent:
    model_name: str = MASTER_AGENT_MODEL
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
        graph = StateGraph(MasterAgentState)
        graph.add_node("collect_scores", self.collect_scores)
        graph.add_node("calculate_average", self.calculate_average)
        graph.add_node("finalize_result", self.finalize_result)

        graph.add_edge(START, "collect_scores")
        graph.add_edge("collect_scores", "calculate_average")
        graph.add_edge("calculate_average", "finalize_result")
        graph.add_edge("finalize_result", END)
        return graph.compile()

    def run(
        self,
        statistical_result: dict[str, Any] | None,
        grammatical_result: dict[str, Any] | None,
        fact_checking_result: dict[str, Any] | None,
    ) -> MasterAgentResult:
        final_state = self.graph.invoke(
            {
                "statistical_result": statistical_result,
                "grammatical_result": grammatical_result,
                "fact_checking_result": fact_checking_result,
            }
        )
        return final_state["result"]

    def collect_scores(self, state: MasterAgentState) -> MasterAgentState:
        details: list[AgentScoreDetail] = []
        missing_agents: list[str] = []

        statistical_score = _extract_statistical_ai_score(state.get("statistical_result"))
        _append_score_detail(
            details,
            missing_agents,
            agent="statistical",
            label="Statistical Agent",
            score=statistical_score,
            source_field="document_assessment.ai_likelihood_score",
            explanation="Statistical score is the AI likelihood percentage from the statistical agent.",
        )

        grammatical_score = _extract_grammatical_ai_score(state.get("grammatical_result"))
        _append_score_detail(
            details,
            missing_agents,
            agent="grammatical",
            label="Grammatical Agent",
            score=grammatical_score,
            source_field="grammatical_result.score",
            explanation="Grammatical score is the AI likelihood percentage from the grammatical agent.",
        )

        factual_trust = _extract_fact_checking_trust_score(state.get("fact_checking_result"))
        fact_ai_score = None if factual_trust is None else 100.0 - factual_trust
        _append_score_detail(
            details,
            missing_agents,
            agent="fact_checking",
            label="Fact-Checking Agent",
            score=fact_ai_score,
            original_score=factual_trust,
            source_field="100 - fact_checking_result.overall_trust_score",
            explanation=(
                "Fact-checking returns factual trust, so the Master Agent converts it into AI suspicion "
                "with 100 - factual trust."
            ),
        )

        return {
            "score_details": details,
            "missing_agents": missing_agents,
        }

    def calculate_average(self, state: MasterAgentState) -> MasterAgentState:
        scores = [
            detail["score"]
            for detail in state.get("score_details", [])
            if detail.get("used") and detail.get("score") is not None
        ]
        if not scores:
            return {"raw_score": None, "final_score": None}

        raw_score = sum(scores) / len(scores)
        return {"raw_score": raw_score, "final_score": round(raw_score)}

    def finalize_result(self, state: MasterAgentState) -> MasterAgentState:
        raw_score = state.get("raw_score")
        final_score = state.get("final_score")
        score_details = state.get("score_details", [])
        missing_agents = state.get("missing_agents", [])
        used_agents = [detail["agent"] for detail in score_details if detail["used"]]

        result: MasterAgentResult = {
            "agent": "master",
            "title": "Master Verification Result",
            "model": self.model_name,
            "score": final_score,
            "raw_score": round(raw_score, 2) if raw_score is not None else None,
            "label": (
                f"{final_score}% overall likely AI-written"
                if final_score is not None
                else "Not enough agent results to calculate an overall AI-written score"
            ),
            "available": final_score is not None,
            "used_agents": used_agents,
            "missing_agents": missing_agents,
            "formula": "average(statistical_score, grammatical_score, 100 - factual_trust)",
            "details": {
                "scores": score_details,
                "fact_check_conversion": "fact_ai_score = 100 - factual_trust",
                "average_formula": _build_average_formula(score_details),
                "final_rounded_result": final_score,
                "missing_agents": missing_agents,
                "note": (
                    "The Master Agent does not analyze the text. It only combines the existing agent results."
                ),
            },
        }
        return {"result": result}


def run_master_agent(
    statistical_result: dict[str, Any] | None,
    grammatical_result: dict[str, Any] | None,
    fact_checking_result: dict[str, Any] | None,
) -> MasterAgentResult:
    return MasterAgent().run(
        statistical_result=statistical_result,
        grammatical_result=grammatical_result,
        fact_checking_result=fact_checking_result,
    )


def _append_score_detail(
    details: list[AgentScoreDetail],
    missing_agents: list[str],
    *,
    agent: str,
    label: str,
    score: float | None,
    source_field: str,
    explanation: str,
    original_score: float | None = None,
) -> None:
    used = score is not None
    if not used:
        missing_agents.append(agent)
    details.append(
        {
            "agent": agent,
            "label": label,
            "score": round(score, 2) if score is not None else None,
            "original_score": round(original_score, 2) if original_score is not None else None,
            "used": used,
            "source_field": source_field,
            "explanation": explanation if used else f"{label} result was missing or did not contain a usable score.",
        }
    )


def _extract_statistical_ai_score(result: dict[str, Any] | None) -> float | None:
    if not isinstance(result, dict):
        return None
    document_assessment = result.get("document_assessment")
    if isinstance(document_assessment, dict):
        score = _coerce_percentage(document_assessment.get("ai_likelihood_score"), scale_fraction=True)
        if score is not None:
            return score
    return _coerce_percentage(result.get("percentage"))


def _extract_grammatical_ai_score(result: dict[str, Any] | None) -> float | None:
    if not isinstance(result, dict):
        return None
    return _coerce_percentage(result.get("score"))


def _extract_fact_checking_trust_score(result: dict[str, Any] | None) -> float | None:
    if not isinstance(result, dict):
        return None
    if result.get("total_claims") == 0:
        return None
    return _coerce_percentage(result.get("overall_trust_score"))


def _coerce_percentage(value: Any, *, scale_fraction: bool = False) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if scale_fraction and 0.0 <= score <= 1.0:
        score *= 100.0
    if 0.0 <= score <= 100.0:
        return score
    return None


def _build_average_formula(score_details: list[AgentScoreDetail]) -> str:
    used_scores = [detail["score"] for detail in score_details if detail["used"] and detail["score"] is not None]
    if not used_scores:
        return "No available scores."
    formatted_scores = " + ".join(_format_score(score) for score in used_scores)
    return f"({formatted_scores}) / {len(used_scores)}"


def _format_score(score: float) -> str:
    if float(score).is_integer():
        return str(int(score))
    return f"{score:.2f}"
