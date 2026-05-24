from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.stats.weightstats import DescrStatsW

from .state import ParsedRequest, StatisticalAgentState

DEFAULT_STATISTICAL_MODEL = os.getenv("CHECKWISE_STATS_MODEL", "llama3.2:1b")
DEFAULT_STATISTICAL_OLLAMA_BASE_URL = os.getenv(
    "CHECKWISE_STATS_OLLAMA_BASE_URL",
    "http://localhost:11434",
)
DEFAULT_STATISTICAL_OLLAMA_API_KEY = (
    os.getenv("CHECKWISE_STATS_OLLAMA_API_KEY") or os.getenv("OLLAMA_API_KEY")
)
SCHEMA_PREVIEW_COLUMN_LIMIT = 12


PARSE_REQUEST_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify the request from the question and schema only. "
            "Choose exactly one intent: descriptive, t_test, chi_square, or unsupported. "
            "Do not calculate anything.",
        ),
        (
            "human",
            "Question: {question}\nSchema:\n{schema}",
        ),
    ]
)


EXPLAIN_RESULTS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's statistical question in plain English. "
            "Use only the provided result. Be concise. "
            "Do not add extra analysis or invented details.",
        ),
        (
            "human",
            "Question: {question}\nMethod: {method}\nResult JSON: {analysis_json}",
        ),
    ]
)


@dataclass
class StatisticalAgent:
    model_name: str = DEFAULT_STATISTICAL_MODEL
    base_url: str = DEFAULT_STATISTICAL_OLLAMA_BASE_URL
    temperature: float = 0.0

    def __post_init__(self) -> None:
        self.llm = ChatOllama(**self._build_llm_config())
        self.graph = self._build_graph()

    def _build_llm_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "model": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
        }
        if DEFAULT_STATISTICAL_OLLAMA_API_KEY and "ollama.com" in self.base_url.lower():
            config["client_kwargs"] = {
                "headers": {
                    "Authorization": f"Bearer {DEFAULT_STATISTICAL_OLLAMA_API_KEY}",
                }
            }
        return config

    def run(
        self,
        question: str,
        data: pd.DataFrame | list[dict[str, Any]] | dict[str, list[Any]],
    ) -> StatisticalAgentState:
        initial_state: StatisticalAgentState = {
            "question": question,
            "data": data,
            "llm_error": None,
            "execution_error": None,
        }
        return self.graph.invoke(initial_state)

    def _build_graph(self):
        graph = StateGraph(StatisticalAgentState)
        graph.add_node("parse_request", self.parse_request)
        graph.add_node("inspect_data", self.inspect_data)
        graph.add_node("select_method", self.select_method)
        graph.add_node("run_analysis", self.run_analysis)
        graph.add_node("explain_results", self.explain_results)

        graph.add_edge(START, "parse_request")
        graph.add_edge("parse_request", "inspect_data")
        graph.add_edge("inspect_data", "select_method")
        graph.add_edge("select_method", "run_analysis")
        graph.add_edge("run_analysis", "explain_results")
        graph.add_edge("explain_results", END)
        return graph.compile()

    def parse_request(self, state: StatisticalAgentState) -> StatisticalAgentState:
        dataframe = self._coerce_dataframe(state["data"])
        schema = self._build_schema_preview(dataframe)
        structured_llm = self.llm.with_structured_output(ParsedRequest)

        try:
            parsed_request = structured_llm.invoke(
                PARSE_REQUEST_PROMPT.format_messages(
                    question=state["question"],
                    schema=schema,
                )
            )
            llm_error = None
        except Exception as exc:  # pragma: no cover - network/model availability dependent
            parsed_request = self._fallback_parse_request(state["question"], dataframe)
            llm_error = str(exc)

        return {
            "dataframe": dataframe,
            "parsed_request": parsed_request,
            "llm_error": llm_error,
        }

    def inspect_data(self, state: StatisticalAgentState) -> StatisticalAgentState:
        dataframe = state["dataframe"]
        numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
        categorical_columns = dataframe.select_dtypes(exclude="number").columns.tolist()

        data_summary = {
            "row_count": int(len(dataframe)),
            "column_count": int(len(dataframe.columns)),
            "columns": dataframe.columns.tolist(),
            "dtypes": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "missing_values": {
                column: int(count) for column, count in dataframe.isna().sum().to_dict().items()
            },
            "unique_values": {
                column: int(dataframe[column].nunique(dropna=True))
                for column in dataframe.columns
            },
        }
        return {"data_summary": data_summary}

    def select_method(self, state: StatisticalAgentState) -> StatisticalAgentState:
        parsed = state["parsed_request"]
        summary = state["data_summary"]
        dataframe = state["dataframe"]
        target_columns = [column for column in parsed.target_columns if column in dataframe.columns]
        group_column = parsed.group_column if parsed.group_column in dataframe.columns else None

        if parsed.intent == "descriptive":
            if target_columns:
                method_reason = "The user asked for a summary of specific columns."
            elif summary["numeric_columns"]:
                target_columns = summary["numeric_columns"][:3]
                method_reason = "No specific columns were detected, so the agent selected the first numeric columns."
            elif summary["categorical_columns"]:
                target_columns = summary["categorical_columns"][:1]
                method_reason = "No numeric columns were available, so the agent will summarize the first categorical column."
            else:
                return {
                    "selected_method": "unsupported",
                    "method_reason": "The dataset does not contain analyzable columns.",
                }
            return {
                "selected_method": "descriptive",
                "method_reason": method_reason,
                "parsed_request": parsed.model_copy(update={"target_columns": target_columns}),
            }

        if parsed.intent == "t_test":
            numeric_target = self._resolve_numeric_target(target_columns, summary["numeric_columns"])
            if not numeric_target:
                return {
                    "selected_method": "unsupported",
                    "method_reason": "A t-test requires one numeric target column.",
                }

            inferred_group_column = group_column or self._infer_group_column(
                dataframe,
                exclude_columns=[numeric_target],
            )
            if not inferred_group_column:
                return {
                    "selected_method": "unsupported",
                    "method_reason": "A t-test requires a categorical grouping column with exactly two groups.",
                }

            group_count = int(dataframe[inferred_group_column].dropna().nunique())
            if group_count != 2:
                return {
                    "selected_method": "unsupported",
                    "method_reason": "The selected grouping column does not have exactly two observed groups.",
                }

            return {
                "selected_method": "t_test",
                "method_reason": "The question asks for a difference between two groups on a numeric measure.",
                "parsed_request": parsed.model_copy(
                    update={"target_columns": [numeric_target], "group_column": inferred_group_column}
                ),
            }

        if parsed.intent == "chi_square":
            categorical_targets = [
                column for column in target_columns if column in summary["categorical_columns"]
            ]
            resolved_target = categorical_targets[0] if categorical_targets else self._infer_categorical_column(dataframe)
            resolved_group = group_column or self._infer_group_column(dataframe, exclude_columns=[resolved_target] if resolved_target else [])

            if not resolved_target or not resolved_group:
                return {
                    "selected_method": "unsupported",
                    "method_reason": "A chi-square test requires two categorical columns.",
                }

            return {
                "selected_method": "chi_square",
                "method_reason": "The question asks about association between categorical variables.",
                "parsed_request": parsed.model_copy(
                    update={"target_columns": [resolved_target], "group_column": resolved_group}
                ),
            }

        return {
            "selected_method": "unsupported",
            "method_reason": "The request does not map cleanly to the supported statistical methods.",
        }

    def run_analysis(self, state: StatisticalAgentState) -> StatisticalAgentState:
        method = state["selected_method"]
        dataframe = state["dataframe"]
        parsed = state["parsed_request"]

        try:
            if method == "descriptive":
                analysis_result = self._run_descriptive(dataframe, parsed.target_columns)
            elif method == "t_test":
                analysis_result = self._run_t_test(
                    dataframe,
                    target_column=parsed.target_columns[0],
                    group_column=parsed.group_column or "",
                )
            elif method == "chi_square":
                analysis_result = self._run_chi_square(
                    dataframe,
                    target_column=parsed.target_columns[0],
                    group_column=parsed.group_column or "",
                )
            else:
                analysis_result = {
                    "method": "unsupported",
                    "message": state["method_reason"],
                }
            execution_error = None
        except Exception as exc:
            analysis_result = {
                "method": "error",
                "message": "The statistical analysis could not be completed safely.",
            }
            execution_error = str(exc)

        return {
            "analysis_result": analysis_result,
            "execution_error": execution_error,
        }

    def explain_results(self, state: StatisticalAgentState) -> StatisticalAgentState:
        analysis_result = state["analysis_result"]
        if analysis_result.get("method") in {"unsupported", "error"}:
            answer = analysis_result["message"]
            if state.get("execution_error"):
                answer = f"{answer} Details: {state['execution_error']}"
            return {"answer": answer}

        try:
            response = self.llm.invoke(
                EXPLAIN_RESULTS_PROMPT.format_messages(
                    question=state["question"],
                    method=state["selected_method"],
                    analysis_json=json.dumps(
                        self._build_explanation_payload(analysis_result),
                        separators=(",", ":"),
                        default=str,
                    ),
                )
            )
            answer = response.content if isinstance(response.content, str) else str(response.content)
        except Exception:  # pragma: no cover - network/model availability dependent
            answer = self._fallback_explanation(
                question=state["question"],
                method=state["selected_method"],
                analysis_result=analysis_result,
            )

        return {"answer": answer}

    @staticmethod
    def _coerce_dataframe(
        data: pd.DataFrame | list[dict[str, Any]] | dict[str, list[Any]]
    ) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        return pd.DataFrame(data)

    @staticmethod
    def _build_schema_preview(dataframe: pd.DataFrame) -> str:
        lines = [f"rows={len(dataframe)}, columns={len(dataframe.columns)}"]
        preview_columns = list(dataframe.columns[:SCHEMA_PREVIEW_COLUMN_LIMIT])
        for column in preview_columns:
            non_null = int(dataframe[column].notna().sum())
            unique = int(dataframe[column].nunique(dropna=True))
            lines.append(
                f"- {column}: dtype={dataframe[column].dtype}, non_null={non_null}, unique={unique}"
            )
        remaining_columns = len(dataframe.columns) - len(preview_columns)
        if remaining_columns > 0:
            lines.append(f"- ... {remaining_columns} more columns omitted")
        return "\n".join(lines)

    @staticmethod
    def _build_explanation_payload(analysis_result: dict[str, Any]) -> dict[str, Any]:
        method = analysis_result.get("method")
        if method == "descriptive":
            compact_summaries: list[dict[str, Any]] = []
            for summary in analysis_result.get("summaries", []):
                compact_summary = {
                    "column": summary.get("column"),
                    "type": summary.get("type"),
                }
                if summary.get("type") == "numeric":
                    compact_summary.update(
                        {
                            "count": summary.get("count"),
                            "mean": summary.get("mean"),
                            "median": summary.get("median"),
                            "std": summary.get("std"),
                            "min": summary.get("min"),
                            "max": summary.get("max"),
                        }
                    )
                elif summary.get("type") == "categorical":
                    compact_summary.update(
                        {
                            "count": summary.get("count"),
                            "unique": summary.get("unique"),
                            "top_category": summary.get("top_category"),
                            "top_category_count": summary.get("top_category_count"),
                        }
                    )
                else:
                    compact_summary["message"] = summary.get("message")
                compact_summaries.append(compact_summary)
            return {"method": method, "summaries": compact_summaries}

        if method == "t_test":
            return {
                "method": method,
                "target_column": analysis_result.get("target_column"),
                "group_column": analysis_result.get("group_column"),
                "group_a": {
                    "name": analysis_result.get("group_a", {}).get("name"),
                    "count": analysis_result.get("group_a", {}).get("count"),
                    "mean": analysis_result.get("group_a", {}).get("mean"),
                },
                "group_b": {
                    "name": analysis_result.get("group_b", {}).get("name"),
                    "count": analysis_result.get("group_b", {}).get("count"),
                    "mean": analysis_result.get("group_b", {}).get("mean"),
                },
                "difference_in_means": analysis_result.get("difference_in_means"),
                "t_statistic": analysis_result.get("t_statistic"),
                "p_value": analysis_result.get("p_value"),
                "significant": analysis_result.get("significant"),
            }

        if method == "chi_square":
            return {
                "method": method,
                "target_column": analysis_result.get("target_column"),
                "group_column": analysis_result.get("group_column"),
                "chi_square_statistic": analysis_result.get("chi_square_statistic"),
                "degrees_of_freedom": analysis_result.get("degrees_of_freedom"),
                "p_value": analysis_result.get("p_value"),
                "significant": analysis_result.get("significant"),
            }

        return analysis_result

    @staticmethod
    def _fallback_parse_request(question: str, dataframe: pd.DataFrame) -> ParsedRequest:
        lowered = question.lower()
        columns = dataframe.columns.tolist()
        mentioned_columns = [
            column for column in columns if column.lower() in lowered
        ]

        if any(keyword in lowered for keyword in ["chi-square", "chi square", "association", "relationship"]):
            return ParsedRequest(intent="chi_square", target_columns=mentioned_columns[:2])
        if any(keyword in lowered for keyword in ["t-test", "t test", "difference", "compare", "significant"]):
            return ParsedRequest(intent="t_test", target_columns=mentioned_columns[:1])
        if any(keyword in lowered for keyword in ["average", "mean", "median", "describe", "summary", "distribution"]):
            return ParsedRequest(intent="descriptive", target_columns=mentioned_columns)
        return ParsedRequest(intent="descriptive", target_columns=mentioned_columns)

    @staticmethod
    def _resolve_numeric_target(target_columns: list[str], numeric_columns: list[str]) -> str | None:
        for column in target_columns:
            if column in numeric_columns:
                return column
        return numeric_columns[0] if numeric_columns else None

    @staticmethod
    def _infer_group_column(dataframe: pd.DataFrame, exclude_columns: list[str]) -> str | None:
        for column in dataframe.columns:
            if column in exclude_columns:
                continue
            unique_count = int(dataframe[column].dropna().nunique())
            if unique_count == 2 and not pd.api.types.is_numeric_dtype(dataframe[column]):
                return column
        for column in dataframe.columns:
            if column in exclude_columns:
                continue
            unique_count = int(dataframe[column].dropna().nunique())
            if unique_count == 2:
                return column
        return None

    @staticmethod
    def _infer_categorical_column(dataframe: pd.DataFrame) -> str | None:
        for column in dataframe.columns:
            if not pd.api.types.is_numeric_dtype(dataframe[column]):
                return column
        return None

    @staticmethod
    def _run_descriptive(dataframe: pd.DataFrame, target_columns: list[str]) -> dict[str, Any]:
        summaries: list[dict[str, Any]] = []
        for column in target_columns:
            series = dataframe[column].dropna()
            if series.empty:
                summaries.append(
                    {
                        "column": column,
                        "type": "empty",
                        "message": "The column has no non-missing values.",
                    }
                )
                continue

            if pd.api.types.is_numeric_dtype(series):
                numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                stats = DescrStatsW(numeric_series)
                ci_low, ci_high = stats.tconfint_mean()
                summaries.append(
                    {
                        "column": column,
                        "type": "numeric",
                        "count": int(numeric_series.count()),
                        "mean": float(numeric_series.mean()),
                        "median": float(numeric_series.median()),
                        "std": float(numeric_series.std(ddof=1)) if len(numeric_series) > 1 else 0.0,
                        "min": float(numeric_series.min()),
                        "max": float(numeric_series.max()),
                        "mean_95_ci": [float(ci_low), float(ci_high)],
                    }
                )
            else:
                value_counts = series.astype(str).value_counts()
                top_category = value_counts.index[0]
                summaries.append(
                    {
                        "column": column,
                        "type": "categorical",
                        "count": int(series.count()),
                        "unique": int(series.nunique()),
                        "top_category": str(top_category),
                        "top_category_count": int(value_counts.iloc[0]),
                        "distribution": {
                            str(index): int(value) for index, value in value_counts.head(10).items()
                        },
                    }
                )

        return {
            "method": "descriptive",
            "columns": target_columns,
            "summaries": summaries,
        }

    @staticmethod
    def _run_t_test(dataframe: pd.DataFrame, target_column: str, group_column: str) -> dict[str, Any]:
        clean_data = dataframe[[target_column, group_column]].dropna()
        observed_groups = clean_data[group_column].unique().tolist()
        if len(observed_groups) != 2:
            raise ValueError("The t-test requires exactly two observed groups after removing missing values.")

        group_a_name, group_b_name = observed_groups[0], observed_groups[1]
        group_a = pd.to_numeric(
            clean_data.loc[clean_data[group_column] == group_a_name, target_column],
            errors="coerce",
        ).dropna()
        group_b = pd.to_numeric(
            clean_data.loc[clean_data[group_column] == group_b_name, target_column],
            errors="coerce",
        ).dropna()

        if len(group_a) < 2 or len(group_b) < 2:
            raise ValueError("Each group needs at least two numeric observations for a stable t-test.")

        statistic, p_value = ttest_ind(group_a, group_b, equal_var=False)
        group_a_stats = DescrStatsW(group_a)
        group_b_stats = DescrStatsW(group_b)
        a_ci_low, a_ci_high = group_a_stats.tconfint_mean()
        b_ci_low, b_ci_high = group_b_stats.tconfint_mean()

        return {
            "method": "t_test",
            "target_column": target_column,
            "group_column": group_column,
            "group_a": {
                "name": str(group_a_name),
                "count": int(group_a.count()),
                "mean": float(group_a.mean()),
                "std": float(group_a.std(ddof=1)),
                "mean_95_ci": [float(a_ci_low), float(a_ci_high)],
            },
            "group_b": {
                "name": str(group_b_name),
                "count": int(group_b.count()),
                "mean": float(group_b.mean()),
                "std": float(group_b.std(ddof=1)),
                "mean_95_ci": [float(b_ci_low), float(b_ci_high)],
            },
            "difference_in_means": float(group_a.mean() - group_b.mean()),
            "t_statistic": float(statistic),
            "p_value": float(p_value),
            "alpha": 0.05,
            "significant": bool(p_value < 0.05),
        }

    @staticmethod
    def _run_chi_square(dataframe: pd.DataFrame, target_column: str, group_column: str) -> dict[str, Any]:
        clean_data = dataframe[[target_column, group_column]].dropna()
        contingency = pd.crosstab(clean_data[group_column], clean_data[target_column])

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            raise ValueError("The chi-square test requires at least a 2x2 contingency table.")

        chi2, p_value, degrees_of_freedom, expected = chi2_contingency(contingency)
        return {
            "method": "chi_square",
            "target_column": target_column,
            "group_column": group_column,
            "contingency_table": {
                str(row): {str(column): int(value) for column, value in values.items()}
                for row, values in contingency.to_dict(orient="index").items()
            },
            "expected_frequencies": expected.round(4).tolist(),
            "chi_square_statistic": float(chi2),
            "degrees_of_freedom": int(degrees_of_freedom),
            "p_value": float(p_value),
            "alpha": 0.05,
            "significant": bool(p_value < 0.05),
        }

    @staticmethod
    def _fallback_explanation(
        question: str,
        method: str,
        analysis_result: dict[str, Any],
    ) -> str:
        if method == "descriptive":
            fragments = []
            for summary in analysis_result["summaries"]:
                if summary["type"] == "numeric":
                    fragments.append(
                        f"{summary['column']} has mean {summary['mean']:.3f}, median {summary['median']:.3f}, "
                        f"standard deviation {summary['std']:.3f}, and ranges from {summary['min']:.3f} to {summary['max']:.3f}."
                    )
                elif summary["type"] == "categorical":
                    fragments.append(
                        f"{summary['column']} is most often '{summary['top_category']}' "
                        f"({summary['top_category_count']} of {summary['count']} non-missing rows)."
                    )
                else:
                    fragments.append(f"{summary['column']}: {summary['message']}")
            return " ".join(fragments)

        if method == "t_test":
            return (
                f"For '{question}', the Welch t-test compared {analysis_result['group_a']['name']} and "
                f"{analysis_result['group_b']['name']} on {analysis_result['target_column']}. "
                f"The mean difference was {analysis_result['difference_in_means']:.3f} with "
                f"t = {analysis_result['t_statistic']:.3f} and p = {analysis_result['p_value']:.4f}. "
                f"This result is {'statistically significant' if analysis_result['significant'] else 'not statistically significant'} at alpha = 0.05."
            )

        if method == "chi_square":
            return (
                f"For '{question}', the chi-square test between {analysis_result['group_column']} and "
                f"{analysis_result['target_column']} returned chi-square = {analysis_result['chi_square_statistic']:.3f}, "
                f"degrees of freedom = {analysis_result['degrees_of_freedom']}, and p = {analysis_result['p_value']:.4f}. "
                f"This result is {'statistically significant' if analysis_result['significant'] else 'not statistically significant'} at alpha = 0.05."
            )

        return analysis_result.get("message", "No explanation could be generated.")


def run_statistical_agent(
    question: str,
    data: pd.DataFrame | list[dict[str, Any]] | dict[str, list[Any]],
) -> StatisticalAgentState:
    agent = StatisticalAgent()
    return agent.run(question=question, data=data)
