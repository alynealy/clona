from __future__ import annotations

from typing import Any, Literal, TypedDict

import pandas as pd
from pydantic import BaseModel, Field


class ParsedRequest(BaseModel):
    intent: Literal["descriptive", "t_test", "chi_square", "unsupported"] = Field(
        description="The statistical intent inferred from the user question."
    )
    target_columns: list[str] = Field(
        default_factory=list,
        description="Columns directly referenced by the user for analysis.",
    )
    group_column: str | None = Field(
        default=None,
        description="Grouping column for comparative tests when required.",
    )
    explanation_goal: str = Field(
        default="Give a concise statistical answer in plain English.",
        description="How the final answer should be phrased for the user.",
    )


class StatisticalAgentState(TypedDict, total=False):
    question: str
    data: pd.DataFrame | list[dict[str, Any]] | dict[str, list[Any]]
    dataframe: pd.DataFrame
    parsed_request: ParsedRequest
    data_summary: dict[str, Any]
    selected_method: str
    method_reason: str
    analysis_result: dict[str, Any]
    answer: str
    llm_error: str | None
    execution_error: str | None
