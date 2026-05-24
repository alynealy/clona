from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


DB_PATH = Path(__file__).resolve().parent / "checkwise.db"


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _column_exists(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in rows)


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS verification_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                input_type TEXT NOT NULL,
                submitted_text TEXT NOT NULL,
                text_preview TEXT NOT NULL,
                verification_rating TEXT NOT NULL,
                statistical_percentage INTEGER NOT NULL,
                confidence TEXT NOT NULL,
                explanation TEXT,
                structured_result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        if not _column_exists(connection, "verification_history", "explanation"):
            connection.execute("ALTER TABLE verification_history ADD COLUMN explanation TEXT")
            connection.execute(
                "UPDATE verification_history SET explanation = '' WHERE explanation IS NULL"
            )
        if not _column_exists(connection, "verification_history", "structured_result_json"):
            connection.execute("ALTER TABLE verification_history ADD COLUMN structured_result_json TEXT")
            connection.execute(
                "UPDATE verification_history SET structured_result_json = '{}' WHERE structured_result_json IS NULL"
            )

        if not _column_exists(connection, "verification_history", "confidence"):
            connection.execute("ALTER TABLE verification_history ADD COLUMN confidence TEXT")
            connection.execute(
                "UPDATE verification_history SET confidence = 'low' WHERE confidence IS NULL"
            )


def insert_history_entry(entry: dict[str, Any]) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO verification_history (
                user_email,
                input_type,
                submitted_text,
                text_preview,
                verification_rating,
                statistical_percentage,
                confidence,
                explanation,
                structured_result_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["user_email"],
                entry["input_type"],
                entry["submitted_text"],
                entry["text_preview"],
                entry["verification_rating"],
                entry["statistical_percentage"],
                entry["confidence"],
                entry["explanation"],
                json.dumps(entry["structured_result"], ensure_ascii=False),
                entry["created_at"],
            ),
        )


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return round(value)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return round(float(stripped))
        except ValueError:
            return None

    return None


def fetch_history_for_user(user_email: str) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                user_email,
                input_type,
                submitted_text,
                text_preview,
                verification_rating,
                statistical_percentage,
                confidence,
                structured_result_json,
                created_at
            FROM verification_history
            WHERE user_email = ?
            ORDER BY datetime(created_at) DESC, id DESC
            """,
            (user_email,),
        ).fetchall()

    entries: list[dict[str, Any]] = []
    for row in rows:
        entry = dict(row)
        entry["verification_rating"] = _coerce_optional_int(entry.get("verification_rating"))
        raw_structured_result = entry.get("structured_result_json") or "{}"
        entry["structured_result"] = _sanitize_legacy_structured_result(json.loads(raw_structured_result))
        entry.pop("structured_result_json", None)
        entries.append(entry)
    return entries


def _sanitize_legacy_structured_result(structured_result: Any) -> Any:
    if not isinstance(structured_result, dict):
        return structured_result

    cleaned = dict(structured_result)
    safe_detector_message = "The wording detector could not return a reliable structured result."

    for field_name in ("what_weakens_the_conclusion", "limitations"):
        field_value = cleaned.get(field_name)
        if isinstance(field_value, list):
            cleaned[field_name] = [
                safe_detector_message if _contains_legacy_parser_dump(item) else item
                for item in field_value
                if isinstance(item, str)
            ]

    detector_details = cleaned.get("detector_details")
    if isinstance(detector_details, dict):
        normalized_details = dict(detector_details)
        technical_note = normalized_details.get("technical_note")
        if isinstance(technical_note, str) and _contains_legacy_parser_dump(technical_note):
            normalized_details["technical_note"] = safe_detector_message

        observations = normalized_details.get("observations")
        if isinstance(observations, list):
            normalized_details["observations"] = [
                item for item in observations
                if isinstance(item, str) and not _contains_legacy_parser_dump(item)
            ]

        influential_phrases = normalized_details.get("influential_phrases")
        if isinstance(influential_phrases, list):
            normalized_details["influential_phrases"] = [
                item for item in influential_phrases
                if isinstance(item, str) and not _contains_legacy_parser_dump(item)
            ]

        cleaned["detector_details"] = normalized_details

    return cleaned


def _contains_legacy_parser_dump(value: str) -> bool:
    lowered = value.lower()
    return "invalid json output" in lowered or "output_parsing_failure" in lowered or "for troubleshooting" in lowered
