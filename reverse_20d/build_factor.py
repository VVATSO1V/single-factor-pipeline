"""Build factor.csv for the 20-day reversal example.

The factor is:

    reversal_20d(T) = -(post_close(T) / post_close(T-20) - 1)

It reads reverse_20d/data/market_panel.csv and writes
reverse_20d/data/factor.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
KEYS = ["date", "stock_code"]


def build_reversal_factor(market_panel: pd.DataFrame, lookback: int) -> pd.DataFrame:
    required = {*KEYS, "post_close"}
    missing = sorted(required - set(market_panel.columns))
    if missing:
        raise ValueError(f"market_panel is missing columns: {', '.join(missing)}")

    panel = market_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["post_close"] = pd.to_numeric(panel["post_close"], errors="coerce")

    close = panel.pivot(index="date", columns="stock_code", values="post_close")
    close = close.sort_index()
    factor = -(close / close.shift(lookback) - 1.0)
    result = (
        factor.stack(future_stack=True)
        .rename("factor_value")
        .reset_index()
        .dropna(subset=["factor_value"])
    )
    return result[[*KEYS, "factor_value"]].sort_values(KEYS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 20-day reversal factor CSV.")
    parser.add_argument(
        "--market-panel-path",
        type=Path,
        default=SCRIPT_DIR / "data" / "market_panel.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=SCRIPT_DIR / "data" / "factor.csv",
    )
    parser.add_argument("--lookback", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    market_panel = pd.read_csv(args.market_panel_path)
    factor = build_reversal_factor(market_panel, args.lookback)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    factor.to_csv(args.output_path, index=False, encoding="utf-8-sig")
    print(f"factor written: {args.output_path.resolve()} shape={factor.shape}")


if __name__ == "__main__":
    main()
