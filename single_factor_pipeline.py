"""Standard cross-sectional single-factor testing pipeline.

The pipeline is intentionally data-source agnostic. Users prepare standardized
CSV files, then this module performs alignment, preprocessing, evaluation, and
reporting. The current public interface uses two inputs:

    factor.csv       date, stock_code, factor_value
    market_panel.csv all non-factor fields keyed by date, stock_code

See README.md for schemas and Ricequant download examples.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KEYS = ["date", "stock_code"]


@dataclass(frozen=True)
class PipelineConfig:
    factor_name: str
    factor_path: Path
    market_panel_path: Path
    output_dir: Path
    return_windows: tuple[int, ...] = (1, 5, 10)
    quantiles: int = 10
    min_listing_days: int = 120
    mad_width: float = 3.0
    direction: str = "positive"

    def __post_init__(self) -> None:
        path_fields = [
            "factor_path",
            "market_panel_path",
            "output_dir",
        ]
        for name in path_fields:
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))

        if not self.factor_name.strip():
            raise ValueError("factor_name cannot be empty")
        if self.direction not in {"positive", "negative"}:
            raise ValueError("direction must be 'positive' or 'negative'")
        if self.quantiles < 2:
            raise ValueError("quantiles must be at least 2")
        if self.min_listing_days < 0:
            raise ValueError("min_listing_days cannot be negative")
        if self.mad_width <= 0:
            raise ValueError("mad_width must be positive")
        if not self.return_windows or any(window <= 0 for window in self.return_windows):
            raise ValueError("return_windows must contain positive integers")


def _normalize_stock_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit() and len(text) <= 6:
        return text.zfill(6)
    return text


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    if "stock_code" in result.columns:
        result["stock_code"] = result["stock_code"].map(_normalize_stock_code)
    return result


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def _require_unique_keys(frame: pd.DataFrame, name: str) -> None:
    duplicates = frame.duplicated(KEYS, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, KEYS].head(5).to_dict("records")
        raise ValueError(f"{name} has duplicate date/stock_code rows: {sample}")


def _to_bool(series: pd.Series, name: str) -> pd.Series:
    if series.dtype == bool:
        return series
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }
    converted = series.map(lambda value: mapping.get(str(value).strip().lower()))
    if converted.isna().any():
        bad = series[converted.isna()].astype(str).unique()[:5]
        raise ValueError(f"{name} contains invalid boolean values: {bad}")
    return converted.astype(bool)


def load_factor(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if {"date", "stock_code", "factor_value"}.issubset(frame.columns):
        result = frame[["date", "stock_code", "factor_value"]].copy()
    else:
        if "date" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "date"})
        value_columns = [column for column in frame.columns if column != "date"]
        if not value_columns:
            raise ValueError("wide factor file must contain stock columns")
        result = frame.melt(
            id_vars="date",
            value_vars=value_columns,
            var_name="stock_code",
            value_name="factor_value",
        )
    result = _normalize_keys(result)
    result["factor_value"] = pd.to_numeric(result["factor_value"], errors="coerce")
    _require_unique_keys(result, "factor file")
    return result.dropna(subset=["factor_value"]).sort_values(KEYS)


def load_standard_table(
    path: Path,
    required: Iterable[str],
    name: str,
) -> pd.DataFrame:
    frame = _normalize_keys(pd.read_csv(path))
    _require_columns(frame, required, name)
    _require_unique_keys(frame, name)
    return frame


def load_market_panel(path: Path) -> pd.DataFrame:
    required = [
        *KEYS,
        "in_universe",
        "post_close",
        "raw_open",
        "raw_close",
        "limit_up",
        "limit_down",
        "is_st",
        "is_suspended",
        "listing_days",
        "industry",
        "market_cap",
    ]
    panel = load_standard_table(path, required, "market panel")
    panel["in_universe"] = _to_bool(panel["in_universe"], "in_universe")
    panel["is_st"] = _to_bool(panel["is_st"], "is_st")
    panel["is_suspended"] = _to_bool(panel["is_suspended"], "is_suspended")
    panel["listing_days"] = pd.to_numeric(panel["listing_days"], errors="coerce")
    panel["market_cap"] = pd.to_numeric(panel["market_cap"], errors="coerce")
    for column in ["post_close", "raw_open", "raw_close", "limit_up", "limit_down"]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    return panel.sort_values(KEYS)


def build_forward_returns(
    prices: pd.DataFrame,
    windows: tuple[int, ...],
) -> pd.DataFrame:
    _require_columns(prices, [*KEYS, "post_close"], "price file")
    calendar = pd.Index(sorted(prices["date"].unique()), name="date")
    stocks = pd.Index(sorted(prices["stock_code"].unique()), name="stock_code")
    full_index = pd.MultiIndex.from_product(
        [stocks, calendar],
        names=["stock_code", "date"],
    )
    panel = (
        prices.set_index(["stock_code", "date"])[["post_close"]]
        .reindex(full_index)
        .sort_index()
    )
    grouped = panel.groupby(level="stock_code", group_keys=False)
    for window in windows:
        future_close = grouped["post_close"].shift(-window)
        panel[f"return_{window}d"] = future_close / panel["post_close"] - 1.0
    return panel.reset_index()


def _build_execution_state(
    prices: pd.DataFrame,
    status: pd.DataFrame,
) -> pd.DataFrame:
    calendar = pd.Index(sorted(prices["date"].unique()))
    execution = pd.DataFrame({"date": calendar[:-1], "execution_date": calendar[1:]})

    state = status.merge(
        prices[
            [
                "date",
                "stock_code",
                "raw_open",
                "limit_up",
                "limit_down",
            ]
        ],
        on=KEYS,
        how="outer",
        validate="one_to_one",
    ).rename(columns={"date": "execution_date"})
    state = execution.merge(state, on="execution_date", how="left")
    state["has_next_trading_day"] = state["execution_date"].notna()
    return state


def build_analysis_sample(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor = load_factor(config.factor_path)
    market_panel = load_market_panel(config.market_panel_path)
    prices = market_panel[
        [
            *KEYS,
            "post_close",
            "raw_open",
            "raw_close",
            "limit_up",
            "limit_down",
        ]
    ].copy()
    status = market_panel[[*KEYS, "is_st", "is_suspended", "listing_days"]].copy()

    returns = build_forward_returns(prices, config.return_windows)
    execution_state = _build_execution_state(prices, status)

    panel_columns = [
        *KEYS,
        "in_universe",
        "industry",
        "market_cap",
    ]
    sample = factor.merge(
        market_panel[panel_columns],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    sample = sample.merge(
        returns[
            [*KEYS, *[f"return_{window}d" for window in config.return_windows]]
        ],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    sample = sample.merge(
        execution_state,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    tolerance = 1e-10
    sample["open_limit_up"] = (
        sample["raw_open"] >= sample["limit_up"] - tolerance
    )
    sample["open_limit_down"] = (
        sample["raw_open"] <= sample["limit_down"] + tolerance
    )
    in_universe = sample["in_universe"].astype("boolean").fillna(False)
    is_st = sample["is_st"].astype("boolean").fillna(True)
    is_suspended = sample["is_suspended"].astype("boolean").fillna(True)
    sample["eligible"] = (
        in_universe
        & ~is_st
        & ~is_suspended
        & (sample["listing_days"] >= config.min_listing_days)
        & sample["raw_open"].notna()
        & ~sample["open_limit_up"]
        & ~sample["open_limit_down"]
    )

    quality = pd.DataFrame(
        [
            {
                "factor_rows": len(factor),
                "factor_dates": factor["date"].nunique(),
                "factor_stocks": factor["stock_code"].nunique(),
                "market_panel_rows": len(market_panel),
                "market_panel_dates": market_panel["date"].nunique(),
                "market_panel_stocks": market_panel["stock_code"].nunique(),
                "eligible_rows": int(sample["eligible"].sum()),
                "excluded_not_in_universe": int(
                    (~in_universe).sum()
                ),
                "excluded_st": int(is_st.sum()),
                "excluded_suspended": int(is_suspended.sum()),
                "excluded_listing_days": int(
                    (sample["listing_days"] < config.min_listing_days).fillna(True).sum()
                ),
                "excluded_open_limit": int(
                    (sample["open_limit_up"] | sample["open_limit_down"]).sum()
                ),
            }
        ]
    )
    return sample[sample["eligible"]].copy(), quality


def _mad_winsorize(values: pd.Series, width: float) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    median = values.median()
    mad = (values - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return values
    scale = 1.4826 * mad
    return values.clip(median - width * scale, median + width * scale)


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=1)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (values - values.mean()) / std


def _spearman_correlation(left: pd.Series, right: pd.Series) -> float:
    clean = pd.concat([left, right], axis=1).dropna()
    if len(clean) < 3:
        return np.nan
    return clean.iloc[:, 0].rank(method="average").corr(
        clean.iloc[:, 1].rank(method="average"),
        method="pearson",
    )


def preprocess_factor(sample: pd.DataFrame, mad_width: float) -> pd.DataFrame:
    result_parts = []
    for _, group in sample.groupby("date", sort=True):
        current = group.copy()
        winsorized = _mad_winsorize(current["factor_value"], mad_width)
        current["factor_raw"] = _zscore(winsorized)

        neutralized = pd.Series(np.nan, index=current.index, dtype=float)
        valid = current[
            ["factor_raw", "industry", "market_cap"]
        ].dropna()
        valid = valid[valid["market_cap"] > 0]
        if len(valid) >= 5:
            industry_dummies = pd.get_dummies(
                current.loc[valid.index, "industry"].astype(str),
                prefix="industry",
                drop_first=True,
                dtype=float,
            )
            design = pd.concat(
                [
                    pd.Series(1.0, index=valid.index, name="intercept"),
                    np.log(current.loc[valid.index, "market_cap"]).rename(
                        "log_market_cap"
                    ),
                    industry_dummies,
                ],
                axis=1,
            )
            x = design.to_numpy(dtype=float)
            y = current.loc[valid.index, "factor_raw"].to_numpy(dtype=float)
            if len(valid) > x.shape[1] and np.linalg.matrix_rank(x) == x.shape[1]:
                beta, *_ = np.linalg.lstsq(x, y, rcond=None)
                neutralized.loc[valid.index] = y - x @ beta
        current["factor_neutralized"] = _zscore(neutralized)
        result_parts.append(current)
    if not result_parts:
        raise ValueError("no eligible factor observations remain after filtering")
    return pd.concat(result_parts).sort_values(KEYS).reset_index(drop=True)


def _cross_section_correlation(
    group: pd.DataFrame,
    factor_column: str,
    return_column: str,
) -> tuple[float, float, int]:
    clean = group[[factor_column, return_column]].dropna()
    if len(clean) < 3:
        return np.nan, np.nan, len(clean)
    pearson = clean[factor_column].corr(clean[return_column], method="pearson")
    rank_ic = _spearman_correlation(
        clean[factor_column],
        clean[return_column],
    )
    return pearson, rank_ic, len(clean)


def compute_ic(
    sample: pd.DataFrame,
    variants: dict[str, str],
    windows: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summaries = []
    for variant, factor_column in variants.items():
        for window in windows:
            return_column = f"return_{window}d"
            for date, group in sample.groupby("date", sort=True):
                pearson, rank_ic, count = _cross_section_correlation(
                    group,
                    factor_column,
                    return_column,
                )
                rows.append(
                    {
                        "date": date,
                        "variant": variant,
                        "return_window": window,
                        "pearson_ic": pearson,
                        "rank_ic": rank_ic,
                        "n_stocks": count,
                    }
                )

            current = pd.DataFrame(rows)
            current = current[
                (current["variant"] == variant)
                & (current["return_window"] == window)
            ]
            rank_values = current["rank_ic"].dropna()
            pearson_values = current["pearson_ic"].dropna()
            rank_std = rank_values.std(ddof=1)
            pearson_std = pearson_values.std(ddof=1)
            summaries.append(
                {
                    "variant": variant,
                    "return_window": window,
                    "rank_ic_mean": rank_values.mean(),
                    "rank_ic_std": rank_std,
                    "rank_icir": (
                        rank_values.mean() / rank_std
                        if pd.notna(rank_std) and rank_std > 0
                        else np.nan
                    ),
                    "rank_icir_annualized": (
                        rank_values.mean() / rank_std * np.sqrt(252)
                        if pd.notna(rank_std) and rank_std > 0
                        else np.nan
                    ),
                    "pearson_ic_mean": pearson_values.mean(),
                    "pearson_icir": (
                        pearson_values.mean() / pearson_std
                        if pd.notna(pearson_std) and pearson_std > 0
                        else np.nan
                    ),
                    "ic_positive_ratio": (
                        (rank_values > 0).mean() if len(rank_values) else np.nan
                    ),
                    "n_dates": len(rank_values),
                }
            )
    series = pd.DataFrame(rows)
    series["cumulative_rank_ic"] = series.groupby(
        ["variant", "return_window"]
    )["rank_ic"].cumsum()
    return series, pd.DataFrame(summaries)


def _assign_quantiles(
    group: pd.DataFrame,
    factor_column: str,
    quantiles: int,
) -> pd.Series:
    valid = group[factor_column].dropna()
    result = pd.Series(np.nan, index=group.index)
    if len(valid) < quantiles:
        return result
    ranks = valid.rank(method="first")
    result.loc[valid.index] = (
        pd.qcut(ranks, q=quantiles, labels=False, duplicates="drop") + 1
    )
    return result


def compute_quantile_returns(
    sample: pd.DataFrame,
    variants: dict[str, str],
    windows: tuple[int, ...],
    quantiles: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    daily_parts = []
    membership: dict[str, pd.DataFrame] = {}
    for variant, factor_column in variants.items():
        assigned_parts = []
        for _, group in sample.groupby("date", sort=True):
            current = group[["date", "stock_code", factor_column]].copy()
            current["quantile"] = _assign_quantiles(
                group,
                factor_column,
                quantiles,
            )
            assigned_parts.append(current)
        assigned = pd.concat(assigned_parts, ignore_index=True)
        membership[variant] = assigned.dropna(subset=["quantile"]).copy()

        merged = sample.merge(
            assigned[["date", "stock_code", "quantile"]],
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
        for window in windows:
            return_column = f"return_{window}d"
            current = (
                merged.dropna(subset=["quantile", return_column])
                .groupby(["date", "quantile"], as_index=False)
                .agg(
                    mean_return=(return_column, "mean"),
                    n_stocks=("stock_code", "count"),
                )
            )
            current["variant"] = variant
            current["return_window"] = window
            daily_parts.append(current)

    daily = pd.concat(daily_parts, ignore_index=True)
    summary_rows = []
    for (variant, window), group in daily.groupby(
        ["variant", "return_window"],
        sort=True,
    ):
        group_means = group.groupby("quantile")["mean_return"].mean().sort_index()
        monotonicity = _spearman_correlation(
            pd.Series(group_means.index, index=group_means.index, dtype=float),
            group_means,
        )
        summary_rows.append(
            {
                "variant": variant,
                "return_window": window,
                "monotonicity": monotonicity,
                "top_group_return": group_means.iloc[-1],
                "bottom_group_return": group_means.iloc[0],
                "top_bottom_spread": group_means.iloc[-1] - group_means.iloc[0],
            }
        )
    return daily, pd.DataFrame(summary_rows), membership


def compute_long_short_nav(
    quantile_daily: pd.DataFrame,
    direction: str,
    quantiles: int,
) -> pd.DataFrame:
    rows = []
    daily = quantile_daily[quantile_daily["return_window"] == 1]
    for variant, group in daily.groupby("variant", sort=True):
        wide = group.pivot(
            index="date",
            columns="quantile",
            values="mean_return",
        ).sort_index()
        if 1 not in wide.columns or quantiles not in wide.columns:
            continue
        high = wide[quantiles]
        low = wide[1]
        long_return = high if direction == "positive" else low
        short_underlying_return = low if direction == "positive" else high
        current = pd.DataFrame(index=wide.index)
        current["variant"] = variant
        current["long_return"] = long_return
        current["short_return"] = -short_underlying_return
        current["long_short_return"] = long_return - short_underlying_return
        current["long_nav"] = (1.0 + current["long_return"].fillna(0.0)).cumprod()
        current["short_nav"] = (1.0 + current["short_return"].fillna(0.0)).cumprod()
        current["long_short_nav"] = (
            1.0 + current["long_short_return"].fillna(0.0)
        ).cumprod()
        rows.append(current.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_turnover(
    membership: dict[str, pd.DataFrame],
    quantiles: int,
) -> pd.DataFrame:
    rows = []
    for variant, frame in membership.items():
        factor_columns = [
            column
            for column in frame.columns
            if column not in {"date", "stock_code", "quantile"}
        ]
        factor_column = factor_columns[0]
        previous_ranks = None
        previous_long = None
        previous_short = None
        for date, group in frame.groupby("date", sort=True):
            ranks = (
                group.set_index("stock_code")[factor_column]
                .rank(method="average", pct=True)
            )
            long_members = set(group.loc[group["quantile"] == quantiles, "stock_code"])
            short_members = set(group.loc[group["quantile"] == 1, "stock_code"])

            common = (
                ranks.index.intersection(previous_ranks.index)
                if previous_ranks is not None
                else []
            )
            rank_correlation = (
                _spearman_correlation(
                    ranks.loc[common],
                    previous_ranks.loc[common],
                )
                if len(common) >= 3
                else np.nan
            )
            factor_rank_turnover = (
                1.0 - rank_correlation if pd.notna(rank_correlation) else np.nan
            )
            long_turnover = (
                1.0 - len(long_members & previous_long) / len(previous_long)
                if previous_long
                else np.nan
            )
            short_turnover = (
                1.0 - len(short_members & previous_short) / len(previous_short)
                if previous_short
                else np.nan
            )
            rows.append(
                {
                    "date": date,
                    "variant": variant,
                    "factor_rank_turnover": factor_rank_turnover,
                    "long_turnover": long_turnover,
                    "short_turnover": short_turnover,
                    "long_count": len(long_members),
                    "short_count": len(short_members),
                }
            )
            previous_ranks = ranks
            previous_long = long_members
            previous_short = short_members
    return pd.DataFrame(rows)


def _max_drawdown(returns: pd.Series) -> float:
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    return drawdown.min()


def compute_yearly_performance(
    ic_series: pd.DataFrame,
    long_short_nav: pd.DataFrame,
    turnover: pd.DataFrame,
) -> pd.DataFrame:
    ic = ic_series.copy()
    ic["year"] = pd.to_datetime(ic["date"]).dt.year
    yearly_ic = (
        ic.groupby(["year", "variant", "return_window"], as_index=False)
        .agg(
            rank_ic_mean=("rank_ic", "mean"),
            rank_ic_std=("rank_ic", "std"),
            ic_positive_ratio=("rank_ic", lambda values: (values > 0).mean()),
            ic_days=("rank_ic", "count"),
            mean_stock_count=("n_stocks", "mean"),
        )
    )
    yearly_ic["rank_icir"] = (
        yearly_ic["rank_ic_mean"] / yearly_ic["rank_ic_std"]
    )

    if long_short_nav.empty:
        return yearly_ic

    nav = long_short_nav.copy()
    nav["year"] = pd.to_datetime(nav["date"]).dt.year
    yearly_nav_rows = []
    for (year, variant), group in nav.groupby(["year", "variant"], sort=True):
        returns = group["long_short_return"]
        yearly_nav_rows.append(
            {
                "year": year,
                "variant": variant,
                "long_short_return": (1.0 + returns.fillna(0.0)).prod() - 1.0,
                "long_short_volatility": returns.std(ddof=1) * np.sqrt(252),
                "long_short_sharpe": (
                    returns.mean() / returns.std(ddof=1) * np.sqrt(252)
                    if returns.std(ddof=1) > 0
                    else np.nan
                ),
                "long_short_max_drawdown": _max_drawdown(returns),
                "long_short_win_rate": (returns > 0).mean(),
            }
        )
    yearly_nav = pd.DataFrame(yearly_nav_rows)

    turnover_copy = turnover.copy()
    turnover_copy["year"] = pd.to_datetime(turnover_copy["date"]).dt.year
    yearly_turnover = (
        turnover_copy.groupby(["year", "variant"], as_index=False)
        .agg(
            factor_rank_turnover=("factor_rank_turnover", "mean"),
            long_turnover=("long_turnover", "mean"),
            short_turnover=("short_turnover", "mean"),
        )
    )

    result = yearly_ic.merge(
        yearly_nav,
        on=["year", "variant"],
        how="left",
    )
    result = result.merge(
        yearly_turnover,
        on=["year", "variant"],
        how="left",
    )
    nav_columns = [
        "long_short_return",
        "long_short_volatility",
        "long_short_sharpe",
        "long_short_max_drawdown",
        "long_short_win_rate",
    ]
    result.loc[result["return_window"] != 1, nav_columns] = np.nan
    return result


def _json_safe_config(config: PipelineConfig) -> dict:
    values = asdict(config)
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    return values


def write_outputs(
    config: PipelineConfig,
    outputs: dict[str, pd.DataFrame],
) -> Path:
    target = config.output_dir
    target.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(target / f"{name}.csv", index=False)
    with (target / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(_json_safe_config(config), file, ensure_ascii=False, indent=2)
    return target


def run_pipeline(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    sample, data_quality = build_analysis_sample(config)
    sample = preprocess_factor(sample, config.mad_width)
    variants = {"raw": "factor_raw"}
    if sample["factor_neutralized"].notna().any():
        variants["neutralized"] = "factor_neutralized"

    ic_series, summary = compute_ic(sample, variants, config.return_windows)
    quantile_daily, quantile_summary, membership = compute_quantile_returns(
        sample,
        variants,
        config.return_windows,
        config.quantiles,
    )
    long_short_nav = compute_long_short_nav(
        quantile_daily,
        config.direction,
        config.quantiles,
    )
    turnover = compute_turnover(membership, config.quantiles)
    yearly_performance = compute_yearly_performance(
        ic_series,
        long_short_nav,
        turnover,
    )

    outputs = {
        "summary": summary,
        "ic_series": ic_series,
        "quantile_daily_returns": quantile_daily,
        "quantile_summary": quantile_summary,
        "long_short_nav": long_short_nav,
        "turnover": turnover,
        "yearly_performance": yearly_performance,
        "data_quality": data_quality,
    }
    write_outputs(config, outputs)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the standard cross-sectional single-factor pipeline."
    )
    parser.add_argument("--factor-name", required=True)
    parser.add_argument("--factor-path", required=True, type=Path)
    parser.add_argument("--market-panel-path", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path)
    parser.add_argument("--return-windows", default="1,5,10")
    parser.add_argument("--quantiles", default=10, type=int)
    parser.add_argument("--min-listing-days", default=120, type=int)
    parser.add_argument("--mad-width", default=3.0, type=float)
    parser.add_argument(
        "--direction",
        choices=["positive", "negative"],
        default="positive",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = tuple(
        int(value.strip())
        for value in args.return_windows.split(",")
        if value.strip()
    )
    config = PipelineConfig(
        factor_name=args.factor_name,
        factor_path=args.factor_path,
        market_panel_path=args.market_panel_path,
        output_dir=args.output_dir,
        return_windows=windows,
        quantiles=args.quantiles,
        min_listing_days=args.min_listing_days,
        mad_width=args.mad_width,
        direction=args.direction,
    )
    run_pipeline(config)
    print(f"Pipeline completed: {config.output_dir}")


if __name__ == "__main__":
    main()
