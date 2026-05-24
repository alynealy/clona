from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .agent import StatisticalAgent


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        return pd.DataFrame(raw)
    raise ValueError("Only .csv and .json data sources are supported by the CLI.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CheckWise statistical LangGraph agent.")
    parser.add_argument("--question", required=True, help="User question to answer statistically.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a CSV or JSON file that can be loaded into a pandas DataFrame.",
    )
    parser.add_argument(
        "--show-state",
        action="store_true",
        help="Print the structured intermediate graph state as JSON after the final answer.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataframe = load_data(Path(args.data))
    result = StatisticalAgent().run(question=args.question, data=dataframe)

    print(result["answer"])
    if args.show_state:
        printable_state = {
            key: (value.to_dict(orient="records") if isinstance(value, pd.DataFrame) else value)
            for key, value in result.items()
        }
        print(json.dumps(printable_state, indent=2, default=str))


if __name__ == "__main__":
    main()
