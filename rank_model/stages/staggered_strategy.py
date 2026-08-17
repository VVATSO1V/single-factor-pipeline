"""Schedule and simulate staggered ten-day strategy paths."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from rank_model.stages.dataset import file_sha256
from rank_model.stages.strategy import (
    LOCKED_MODEL_NAMES,
    StrategyBundle,
    StrategySettings,
    PortfolioState,
    _DAILY_NAV_COLUMNS,
    _DIAGNOSTIC_COLUMNS,
    _ENDING_POSITION_COLUMNS,
    _POSITION_COLUMNS,
    _STRATEGY_OBSERVATION_CONVENTIONS,
    _STRATEGY_OUTPUT_NAMES,
    _STRATEGY_SOURCE_NAMES,
    _SUMMARY_FORMULAS,
    _TRADE_COLUMNS,
    _acquire_strategy_lock,
    _calendar_dates,
    _canonical_json_sha256,
    _ending_position_frame,
    _is_sha256,
    _normalize_market_panel,
    _position_rows,
    _release_strategy_lock,
    _settings_manifest,
    _write_json_file,
    load_locked_strategy_inputs,
    select_daily_top,
    summarize_strategy,
    transition_at_open,
)


STAGGERED_HORIZON = 10
STAGGERED_OFFSET_COUNT = 10
STAGGERED_COMPLETE_SHIFT = 11
_FIRST_ELIGIBLE_SIGNAL = pd.Timestamp("2024-01-02")
_LAST_ELIGIBLE_SIGNAL = pd.Timestamp("2025-12-16")
_ELIGIBLE_SIGNAL_COUNT = 474
OFFSET_STATISTICS = ("mean", "median", "std", "min", "max")
AVERAGE_METRICS = (
    "elapsed_trading_observations",
    "cumulative_return",
    "cagr",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "average_cash_ratio",
    "gross_turnover",
    "one_way_turnover",
    "average_gross_turnover",
    "average_one_way_turnover",
    "average_turnover",
    "annualized_gross_turnover",
    "annualized_one_way_turnover",
    "annualized_turnover",
    "total_cost",
    "ending_nav",
)
_OFFSET_SCHEDULE_COLUMNS = (
    "offset",
    "first_signal_date",
    "first_execution_date",
    "final_signal_date",
    "final_horizon_date",
)
_OFFSET_METRIC_COLUMNS = tuple(_SUMMARY_FORMULAS)
_AVERAGE_ECONOMIC_COLUMNS = (
    "pre_trade_nav",
    "nav",
    "cash",
    "buy_notional",
    "sell_notional",
    "total_cost",
)
_AVERAGE_NAV_COLUMNS = (
    "pre_trade_nav",
    "nav",
    "gross_return",
    "net_return",
    "cash",
    "cash_ratio",
    "buy_notional",
    "sell_notional",
    "total_cost",
    "gross_turnover",
    "one_way_turnover",
)
_AVERAGE_FORMULAS = {name: _SUMMARY_FORMULAS[name] for name in AVERAGE_METRICS}
_STAGGERED_MODEL_OUTPUT_NAMES = (
    "offset_metrics.csv",
    "offset_summary.csv",
    "average_nav.csv",
    "average_metrics.json",
    "manifest.json",
)
_STAGGERED_MODEL_PAYLOAD_NAMES = _STAGGERED_MODEL_OUTPUT_NAMES[:-1]
_STAGGERED_MODEL_MANIFEST_FIELDS = {
    "schema_version",
    "status",
    "purpose",
    "model_name",
    "settings",
    "rows",
    "formulas",
    "observation_conventions",
    "input_sha256",
    "schedule",
    "offsets",
    "contract_sha256",
    "output_sha256",
    "manifest_hash_convention",
}
_STAGGERED_OFFSET_MANIFEST_FIELDS = {
    "schema_version",
    "status",
    "purpose",
    "model_name",
    "offset",
    "settings",
    "schedule",
    "rows",
    "formulas",
    "observation_conventions",
    "input_sha256",
    "contract_sha256",
    "output_sha256",
}
_MANIFEST_HASH_CONVENTION = (
    "sha256 of canonical JSON after removing output_sha256['manifest.json']"
)


@dataclass(frozen=True)
class OffsetSchedule:
    offset: int
    signal_dates: Sequence[pd.Timestamp]
    execution_dates: Sequence[pd.Timestamp]
    final_horizon_date: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "signal_dates",
            tuple(pd.Timestamp(date).normalize() for date in self.signal_dates),
        )
        object.__setattr__(
            self,
            "execution_dates",
            tuple(pd.Timestamp(date).normalize() for date in self.execution_dates),
        )
        object.__setattr__(
            self,
            "final_horizon_date",
            pd.Timestamp(self.final_horizon_date).normalize(),
        )

    @property
    def first_execution_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.execution_dates[0])


@dataclass
class StaggeredPathBundle:
    offset: int
    schedule: OffsetSchedule
    strategy: StrategyBundle


def _validated_paths(
    paths: Sequence[StaggeredPathBundle],
) -> tuple[StaggeredPathBundle, ...]:
    result = tuple(paths)
    if len(result) != STAGGERED_OFFSET_COUNT:
        raise ValueError("staggered aggregation requires exactly 10 paths")
    if tuple(path.offset for path in result) != tuple(
        range(1, STAGGERED_OFFSET_COUNT + 1)
    ):
        raise ValueError("staggered paths must be ordered from offset 1 through 10")
    for path in result:
        if not isinstance(path, StaggeredPathBundle):
            raise ValueError("staggered aggregation requires path bundles")
        if path.schedule.offset != path.offset:
            raise ValueError("staggered path offset does not match its schedule")
        daily_nav = path.strategy.daily_nav
        if daily_nav.empty or not isinstance(daily_nav.index, pd.DatetimeIndex):
            raise ValueError("staggered path requires a dated NAV frame")
        if not daily_nav.index.is_unique or not daily_nav.index.is_monotonic_increasing:
            raise ValueError("staggered path NAV dates must be unique and ordered")
        if pd.Timestamp(daily_nav.index[-1]) != path.schedule.final_horizon_date:
            raise ValueError("staggered path NAV must end at its final horizon")
        if set(path.strategy.metrics_summary) != set(_OFFSET_METRIC_COLUMNS):
            raise ValueError("staggered path metric formula contract is incomplete")
    return result


def build_offset_metrics(paths: Sequence[StaggeredPathBundle]) -> pd.DataFrame:
    """Build one schedule and metric detail row for each timing offset."""
    rows: list[dict[str, Any]] = []
    for path in _validated_paths(paths):
        rows.append(
            {
                "offset": path.offset,
                "first_signal_date": path.schedule.signal_dates[0],
                "first_execution_date": path.schedule.execution_dates[0],
                "final_signal_date": path.schedule.signal_dates[-1],
                "final_horizon_date": path.schedule.final_horizon_date,
                **path.strategy.metrics_summary,
            }
        )
    return pd.DataFrame(
        rows,
        columns=(*_OFFSET_SCHEDULE_COLUMNS, *_OFFSET_METRIC_COLUMNS),
    )


def build_offset_summary(offset_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize timing dispersion with fixed sample statistics."""
    expected_columns = (*_OFFSET_SCHEDULE_COLUMNS, *_OFFSET_METRIC_COLUMNS)
    if tuple(offset_metrics.columns) != expected_columns or len(offset_metrics) != 10:
        raise ValueError("offset metric detail schema is invalid")
    if offset_metrics["offset"].tolist() != list(range(1, 11)):
        raise ValueError("offset metric detail must contain offsets 1 through 10")
    summary = offset_metrics.loc[:, _OFFSET_METRIC_COLUMNS].agg(
        list(OFFSET_STATISTICS)
    )
    summary.index = pd.Index(summary.index, name="statistic")
    return summary.loc[list(OFFSET_STATISTICS), list(_OFFSET_METRIC_COLUMNS)]


def build_average_nav(
    paths: Sequence[StaggeredPathBundle], settings: StrategySettings
) -> pd.DataFrame:
    """Average economic columns over the genuine ten-path date intersection."""
    validated = _validated_paths(paths)
    common_dates = validated[0].strategy.daily_nav.index
    for path in validated[1:]:
        common_dates = common_dates.intersection(path.strategy.daily_nav.index)
    common_dates = pd.DatetimeIndex(common_dates, name="date").sort_values()
    if common_dates.empty:
        raise ValueError("staggered paths have no common NAV dates")
    earliest_horizon = min(path.schedule.final_horizon_date for path in validated)
    if common_dates[-1] != earliest_horizon:
        raise ValueError("average NAV must end at the earliest genuine final horizon")

    economic_frames: list[pd.DataFrame] = []
    for path in validated:
        daily_nav = path.strategy.daily_nav
        missing = set(_AVERAGE_ECONOMIC_COLUMNS).difference(daily_nav.columns)
        if missing:
            raise ValueError("staggered path NAV economic columns are incomplete")
        economic_frames.append(
            daily_nav.loc[common_dates, _AVERAGE_ECONOMIC_COLUMNS].astype("float64")
        )
    economic = sum(economic_frames[1:], economic_frames[0].copy()) / len(
        economic_frames
    )
    if not np.isfinite(economic.to_numpy()).all():
        raise ValueError("average NAV economic columns must be finite")
    if not np.isclose(float(economic["nav"].iloc[0]), settings.initial_nav):
        raise ValueError("average NAV must begin at the configured initial NAV")

    prior_nav = economic["nav"].shift(1)
    average = economic.copy()
    average["gross_return"] = average["pre_trade_nav"] / prior_nav - 1.0
    average["net_return"] = average["nav"] / prior_nav - 1.0
    average.iloc[0, average.columns.get_loc("gross_return")] = 0.0
    average.iloc[0, average.columns.get_loc("net_return")] = 0.0
    average["cash_ratio"] = np.where(
        average["nav"] > 0, average["cash"] / average["nav"], 0.0
    )
    average["gross_turnover"] = np.where(
        average["pre_trade_nav"] > 0,
        (average["buy_notional"] + average["sell_notional"])
        / average["pre_trade_nav"],
        0.0,
    )
    average["one_way_turnover"] = average["gross_turnover"] / 2.0
    average.index = common_dates
    return average.loc[:, _AVERAGE_NAV_COLUMNS]


def summarize_average_nav(
    average_nav: pd.DataFrame, settings: StrategySettings
) -> dict[str, float | int]:
    """Recompute the timing-average return, risk, cash, cost, and turnover."""
    if (
        average_nav.empty
        or tuple(average_nav.columns) != _AVERAGE_NAV_COLUMNS
        or not isinstance(average_nav.index, pd.DatetimeIndex)
    ):
        raise ValueError("average NAV schema is invalid")
    observations = average_nav.iloc[1:]
    returns = observations["net_return"].astype("float64")
    elapsed = len(returns)
    ending_nav = float(average_nav["nav"].iloc[-1])
    cumulative_return = ending_nav / settings.initial_nav - 1.0
    if elapsed == 0:
        cagr = 0.0
    elif ending_nav <= 0:
        cagr = -1.0
    else:
        cagr = (ending_nav / settings.initial_nav) ** (
            settings.annualization_days / elapsed
        ) - 1.0
    daily_volatility = float(returns.std(ddof=1)) if elapsed > 1 else 0.0
    if not np.isfinite(daily_volatility):
        daily_volatility = 0.0
    annualized_volatility = daily_volatility * np.sqrt(settings.annualization_days)
    sharpe_ratio = (
        float(returns.mean()) / daily_volatility * np.sqrt(settings.annualization_days)
        if daily_volatility > 0
        else 0.0
    )
    nav = average_nav["nav"].astype("float64")
    average_gross_turnover = (
        float(observations["gross_turnover"].mean()) if elapsed else 0.0
    )
    average_one_way_turnover = (
        float(observations["one_way_turnover"].mean()) if elapsed else 0.0
    )
    result: dict[str, float | int] = {
        "elapsed_trading_observations": elapsed,
        "cumulative_return": cumulative_return,
        "cagr": cagr,
        "annualized_return": cagr,
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(-(nav / nav.cummax() - 1.0).min()),
        "win_rate": float(returns.gt(0).mean()) if elapsed else 0.0,
        "average_cash_ratio": float(average_nav["cash_ratio"].mean()),
        "gross_turnover": float(observations["gross_turnover"].sum()),
        "one_way_turnover": float(observations["one_way_turnover"].sum()),
        "average_gross_turnover": average_gross_turnover,
        "average_one_way_turnover": average_one_way_turnover,
        "average_turnover": average_one_way_turnover,
        "annualized_gross_turnover": average_gross_turnover
        * settings.annualization_days,
        "annualized_one_way_turnover": average_one_way_turnover
        * settings.annualization_days,
        "annualized_turnover": average_one_way_turnover
        * settings.annualization_days,
        "total_cost": float(observations["total_cost"].sum()),
        "ending_nav": ending_nav,
    }
    return {name: result[name] for name in AVERAGE_METRICS}


def _schedule_manifest(schedule: OffsetSchedule) -> dict[str, Any]:
    return {
        "offset": schedule.offset,
        "signal_dates": [date.date().isoformat() for date in schedule.signal_dates],
        "execution_dates": [
            date.date().isoformat() for date in schedule.execution_dates
        ],
        "final_horizon_date": schedule.final_horizon_date.date().isoformat(),
    }


def _path_row_counts(bundle: StrategyBundle) -> dict[str, int]:
    return {
        "daily_nav": len(bundle.daily_nav),
        "execution_dates": len(bundle.execution_diagnostics),
        "trades": len(bundle.trades),
        "positions": len(bundle.positions),
        "ending_positions": len(bundle.ending_positions),
    }


def _offset_contract_hashes(
    settings: Mapping[str, Any], schedule: Mapping[str, Any]
) -> dict[str, str]:
    return {
        "settings": _canonical_json_sha256(settings),
        "schedule": _canonical_json_sha256(schedule),
        "formulas": _canonical_json_sha256(_SUMMARY_FORMULAS),
        "observation_conventions": _canonical_json_sha256(
            _STRATEGY_OBSERVATION_CONVENTIONS
        ),
    }


def _average_observation_conventions() -> dict[str, Any]:
    return {
        **_STRATEGY_OBSERVATION_CONVENTIONS,
        "date_intersection": (
            "exact intersection of all ten genuine path NAV dates; no forward fill"
        ),
        "economic_aggregation": (
            "arithmetic mean of pre_trade_nav, nav, cash, buy_notional, "
            "sell_notional, and total_cost before ratio and metric recomputation"
        ),
    }


def _model_contract_hashes(
    settings: Mapping[str, Any], schedules: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    return {
        "settings": _canonical_json_sha256(settings),
        "schedule": _canonical_json_sha256(list(schedules)),
        "offset_formulas": _canonical_json_sha256(_SUMMARY_FORMULAS),
        "average_formulas": _canonical_json_sha256(_AVERAGE_FORMULAS),
        "observation_conventions": _canonical_json_sha256(
            _average_observation_conventions()
        ),
    }


def _write_staggered_offset(
    path: StaggeredPathBundle,
    directory: Path,
    settings: StrategySettings,
    model_name: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    directory.mkdir()
    bundle = path.strategy
    bundle.daily_nav.reset_index().to_csv(directory / "daily_nav.csv", index=False)
    bundle.trades.to_parquet(directory / "trades.parquet", index=False)
    bundle.positions.to_parquet(directory / "positions.parquet", index=False)
    bundle.execution_diagnostics.to_csv(
        directory / "execution_diagnostics.csv", index=False
    )
    bundle.ending_positions.to_csv(directory / "ending_positions.csv", index=False)
    _write_json_file(directory / "metrics_summary.json", bundle.metrics_summary)
    output_hashes = {
        name: file_sha256(directory / name) for name in _STRATEGY_OUTPUT_NAMES
    }
    settings_payload = _settings_manifest(settings)
    schedule_payload = _schedule_manifest(path.schedule)
    rows = _path_row_counts(bundle)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "purpose": "frozen_rank_model_staggered_10d_offset_2024_2025",
        "model_name": model_name,
        "offset": path.offset,
        "settings": settings_payload,
        "schedule": schedule_payload,
        "rows": rows,
        "formulas": _SUMMARY_FORMULAS,
        "observation_conventions": _STRATEGY_OBSERVATION_CONVENTIONS,
        "input_sha256": dict(source_hashes),
        "contract_sha256": _offset_contract_hashes(
            settings_payload, schedule_payload
        ),
        "output_sha256": output_hashes,
    }
    _write_json_file(directory / "manifest.json", manifest)
    return {
        "offset": path.offset,
        "path": directory.name,
        "schedule": schedule_payload,
        "rows": rows,
        "output_sha256": output_hashes,
        "manifest_sha256": file_sha256(directory / "manifest.json"),
    }


def _validate_source_provenance(
    source_paths: Mapping[str, Path], expected_hashes: Mapping[str, str]
) -> None:
    if (
        not isinstance(source_paths, Mapping)
        or set(source_paths) != set(_STRATEGY_SOURCE_NAMES)
        or not isinstance(expected_hashes, Mapping)
        or set(expected_hashes) != set(_STRATEGY_SOURCE_NAMES)
    ):
        raise ValueError("staggered source provenance set is invalid")
    for name in _STRATEGY_SOURCE_NAMES:
        path = Path(source_paths[name])
        expected = expected_hashes[name]
        if _is_reparse_escape(path):
            raise ValueError(
                "staggered source artifact may not be a symlink, junction, or "
                f"Windows reparse point: {name}"
            )
        if not path.is_file() or not _is_sha256(expected):
            raise ValueError(f"staggered source provenance is invalid: {name}")
        if file_sha256(path) != expected:
            raise ValueError(f"staggered source hash changed: {name}")


def _is_windows_platform() -> bool:
    return os.name == "nt"


def _is_reparse_escape(path: Path) -> bool:
    if path.is_symlink():
        return True
    if not _is_windows_platform():
        return False
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ValueError(
            f"cannot inspect Windows reparse attributes: {path}"
        ) from error
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _validate_no_reparse_ancestry(path: Path, label: str) -> None:
    for component in (path, *path.parents):
        if _is_reparse_escape(component):
            raise ValueError(
                f"{label} may not contain a symlink, junction, or reparse point: "
                f"{component}"
            )


def _validate_no_reparse_subtree(root: Path, label: str) -> None:
    _validate_no_reparse_ancestry(root, label)
    pending = [root]
    while pending:
        directory = pending.pop()
        if _is_reparse_escape(directory):
            raise ValueError(
                f"{label} may not contain a symlink, junction, or reparse point: "
                f"{directory}"
            )
        try:
            children = list(directory.iterdir())
        except OSError as error:
            raise ValueError(f"{label} is unreadable: {directory}") from error
        for child in children:
            if _is_reparse_escape(child):
                raise ValueError(
                    f"{label} may not contain a symlink, junction, or reparse point: "
                    f"{child}"
                )
            if child.is_dir():
                pending.append(child)


def _remove_staging_best_effort(staging: Path) -> None:
    for _ in range(2):
        try:
            if not staging.exists():
                return
            shutil.rmtree(staging)
            return
        except Exception:
            # Preserve the publication failure and retry transient cleanup once.
            pass


def _require_exact_children(
    directory: Path, expected_names: set[str], label: str
) -> None:
    if _is_reparse_escape(directory) or not directory.is_dir():
        raise ValueError(f"{label} is missing or escaped: {directory}")
    children = list(directory.iterdir())
    if any(_is_reparse_escape(child) for child in children):
        raise ValueError(
            f"{label} contains a symlink, junction, or reparse escape: {directory}"
        )
    if {child.name for child in children} != expected_names:
        raise ValueError(f"{label} artifact set is incomplete: {directory}")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if _is_reparse_escape(path) or not path.is_file():
        raise ValueError(f"{label} is missing or escaped: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object: {path}")
    return value


def _stored_offset_row_counts(directory: Path) -> dict[str, int]:
    try:
        return {
            "daily_nav": len(pd.read_csv(directory / "daily_nav.csv")),
            "execution_dates": len(
                pd.read_csv(directory / "execution_diagnostics.csv")
            ),
            "trades": len(pd.read_parquet(directory / "trades.parquet")),
            "positions": len(pd.read_parquet(directory / "positions.parquet")),
            "ending_positions": len(
                pd.read_csv(directory / "ending_positions.csv")
            ),
        }
    except Exception as error:
        raise ValueError(
            f"staggered offset row counts are unreadable: {directory}"
        ) from error


def _validate_offset_publication(
    run_directory: Path,
    model_name: str,
    descriptor: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> None:
    expected_files = {*_STRATEGY_OUTPUT_NAMES, "manifest.json"}
    _require_exact_children(run_directory, expected_files, "staggered offset")
    manifest_path = run_directory / "manifest.json"
    manifest = _read_json_object(manifest_path, "staggered offset manifest")
    if (
        set(manifest) != _STAGGERED_OFFSET_MANIFEST_FIELDS
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "completed"
        or manifest.get("purpose")
        != "frozen_rank_model_staggered_10d_offset_2024_2025"
        or manifest.get("model_name") != model_name
        or manifest.get("offset") != descriptor.get("offset")
    ):
        raise ValueError(
            f"staggered offset manifest contract is invalid: {run_directory}"
        )
    if descriptor.get("path") != run_directory.name:
        raise ValueError("staggered offset path escaped its model directory")
    if manifest.get("input_sha256") != dict(source_hashes):
        raise ValueError("staggered offset source hashes do not match")
    if manifest.get("formulas") != _SUMMARY_FORMULAS:
        raise ValueError("staggered offset formula contract does not match")
    if manifest.get("observation_conventions") != _STRATEGY_OBSERVATION_CONVENTIONS:
        raise ValueError("staggered offset observation contract does not match")
    schedule = manifest.get("schedule")
    settings_payload = manifest.get("settings")
    if not isinstance(schedule, Mapping) or not isinstance(settings_payload, Mapping):
        raise ValueError("staggered offset schedule or settings contract is invalid")
    if manifest.get("contract_sha256") != _offset_contract_hashes(
        settings_payload, schedule
    ):
        raise ValueError("staggered offset schedule or formula hash does not match")
    if descriptor.get("schedule") != schedule:
        raise ValueError("staggered offset schedule does not match model manifest")

    output_hashes = manifest.get("output_sha256")
    if (
        not isinstance(output_hashes, Mapping)
        or set(output_hashes) != set(_STRATEGY_OUTPUT_NAMES)
        or any(not _is_sha256(output_hashes[name]) for name in _STRATEGY_OUTPUT_NAMES)
    ):
        raise ValueError("staggered offset output hash contract is invalid")
    for name in _STRATEGY_OUTPUT_NAMES:
        if file_sha256(run_directory / name) != output_hashes[name]:
            raise ValueError(f"staggered offset output hash does not match: {name}")
    if descriptor.get("output_sha256") != output_hashes:
        raise ValueError("staggered offset output hashes do not match model manifest")
    if descriptor.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("staggered offset manifest hash does not match")

    rows = _stored_offset_row_counts(run_directory)
    if manifest.get("rows") != rows or descriptor.get("rows") != rows:
        raise ValueError("staggered offset row counts do not match")
    metrics = _read_json_object(
        run_directory / "metrics_summary.json", "staggered offset metrics"
    )
    if set(metrics) != set(_SUMMARY_FORMULAS):
        raise ValueError("staggered offset metric formula contract is incomplete")
    daily = pd.read_csv(run_directory / "daily_nav.csv", usecols=["date"])
    if not schedule.get("signal_dates") or not schedule.get("execution_dates"):
        raise ValueError("staggered offset schedule dates are incomplete")
    if len(schedule["signal_dates"]) != len(schedule["execution_dates"]):
        raise ValueError("staggered offset schedule date counts do not match")
    if str(daily["date"].iloc[-1])[:10] != schedule.get("final_horizon_date"):
        raise ValueError("staggered offset final horizon does not match its NAV")


def _model_manifest_logical_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    output_hashes = dict(payload.get("output_sha256", {}))
    output_hashes.pop("manifest.json", None)
    payload["output_sha256"] = output_hashes
    return _canonical_json_sha256(payload)


def _validate_staggered_publication(
    run_directory: Path,
    model_name: str,
    source_paths: Mapping[str, Path],
    source_hashes: Mapping[str, str],
) -> None:
    """Validate a complete model publication and its current provenance."""
    run_directory = Path(run_directory)
    _validate_source_provenance(source_paths, source_hashes)
    expected_root = {
        *(f"offset_{offset:02d}" for offset in range(1, 11)),
        *_STAGGERED_MODEL_OUTPUT_NAMES,
    }
    _require_exact_children(run_directory, expected_root, "staggered model run")
    manifest = _read_json_object(
        run_directory / "manifest.json", "staggered model manifest"
    )
    if (
        set(manifest) != _STAGGERED_MODEL_MANIFEST_FIELDS
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "completed"
        or manifest.get("purpose")
        != "frozen_rank_model_staggered_10d_strategy_2024_2025"
        or manifest.get("model_name") != model_name
        or manifest.get("manifest_hash_convention") != _MANIFEST_HASH_CONVENTION
    ):
        raise ValueError("staggered model manifest contract is invalid")
    if manifest.get("input_sha256") != dict(source_hashes):
        raise ValueError("staggered model source hashes do not match")
    expected_formulas = {
        "offset_metrics": _SUMMARY_FORMULAS,
        "average_metrics": _AVERAGE_FORMULAS,
    }
    if manifest.get("formulas") != expected_formulas:
        raise ValueError("staggered model formula contract does not match")
    if manifest.get("observation_conventions") != _average_observation_conventions():
        raise ValueError("staggered model observation contract does not match")

    output_hashes = manifest.get("output_sha256")
    if (
        not isinstance(output_hashes, Mapping)
        or set(output_hashes) != set(_STAGGERED_MODEL_OUTPUT_NAMES)
        or any(
            not _is_sha256(output_hashes[name])
            for name in _STAGGERED_MODEL_OUTPUT_NAMES
        )
    ):
        raise ValueError("staggered model output hash contract is invalid")
    for name in _STAGGERED_MODEL_PAYLOAD_NAMES:
        if file_sha256(run_directory / name) != output_hashes[name]:
            raise ValueError(f"staggered model output hash does not match: {name}")
    if output_hashes["manifest.json"] != _model_manifest_logical_hash(manifest):
        raise ValueError("staggered model manifest logical hash does not match")

    offsets = manifest.get("offsets")
    if not isinstance(offsets, list) or len(offsets) != STAGGERED_OFFSET_COUNT:
        raise ValueError("staggered model must bind exactly ten offsets")
    schedules: list[Mapping[str, Any]] = []
    for expected_offset, descriptor in enumerate(offsets, start=1):
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "offset",
            "path",
            "schedule",
            "rows",
            "output_sha256",
            "manifest_sha256",
        }:
            raise ValueError("staggered model offset descriptor is invalid")
        if descriptor.get("offset") != expected_offset:
            raise ValueError("staggered model offsets are not ordered")
        expected_path = f"offset_{expected_offset:02d}"
        if descriptor.get("path") != expected_path:
            raise ValueError("staggered model offset path is escaped")
        schedule = descriptor.get("schedule")
        if not isinstance(schedule, Mapping):
            raise ValueError("staggered model offset schedule is invalid")
        schedules.append(schedule)
        _validate_offset_publication(
            run_directory / expected_path,
            model_name,
            descriptor,
            source_hashes,
        )

    settings_payload = manifest.get("settings")
    if not isinstance(settings_payload, Mapping):
        raise ValueError("staggered model settings contract is invalid")
    if manifest.get("contract_sha256") != _model_contract_hashes(
        settings_payload, schedules
    ):
        raise ValueError("staggered model schedule or formula hash does not match")

    try:
        offset_metrics = pd.read_csv(run_directory / "offset_metrics.csv")
        offset_summary = pd.read_csv(run_directory / "offset_summary.csv")
        average_nav = pd.read_csv(run_directory / "average_nav.csv")
    except Exception as error:
        raise ValueError("staggered model aggregate rows are unreadable") from error
    average_metrics = _read_json_object(
        run_directory / "average_metrics.json", "staggered average metrics"
    )
    rows = {
        "offsets": len(offsets),
        "offset_metrics": len(offset_metrics),
        "offset_summary": len(offset_summary),
        "average_nav": len(average_nav),
        "average_metrics": len(average_metrics),
    }
    if manifest.get("rows") != rows:
        raise ValueError("staggered model aggregate row counts do not match")
    if tuple(offset_metrics.columns) != (
        *_OFFSET_SCHEDULE_COLUMNS,
        *_OFFSET_METRIC_COLUMNS,
    ):
        raise ValueError("staggered offset metric detail schema is invalid")
    if tuple(offset_summary.columns) != ("statistic", *_OFFSET_METRIC_COLUMNS):
        raise ValueError("staggered offset summary schema is invalid")
    if tuple(average_nav.columns) != ("date", *_AVERAGE_NAV_COLUMNS):
        raise ValueError("staggered average NAV schema is invalid")
    if set(average_metrics) != set(AVERAGE_METRICS):
        raise ValueError("staggered average metric formula contract is incomplete")
    schedule_summary = manifest.get("schedule")
    expected_schedule_summary = {
        "horizon_trading_days": STAGGERED_HORIZON,
        "offset_count": STAGGERED_OFFSET_COUNT,
        "common_start_date": str(average_nav["date"].iloc[0])[:10],
        "common_end_date": str(average_nav["date"].iloc[-1])[:10],
        "earliest_final_horizon_date": min(
            schedule["final_horizon_date"] for schedule in schedules
        ),
    }
    if schedule_summary != expected_schedule_summary:
        raise ValueError("staggered model common schedule does not match")


def _validate_publication_destination(model_name: str, destination: Path) -> Path:
    destination = Path(destination)
    if model_name not in LOCKED_MODEL_NAMES:
        raise ValueError(f"strategy model is not frozen: {model_name}")
    if (
        destination.name != model_name
        or destination.parent.name != "strategy_10d_runs"
        or ".." in destination.parts
    ):
        raise ValueError(
            "staggered strategy destination must be strategy_10d_runs/<model>"
        )
    _validate_no_reparse_ancestry(destination, "staggered strategy destination")
    return destination


def backtest_staggered_strategy(
    model_name: str,
    *,
    destination: Path,
    settings: StrategySettings,
    source_paths: Mapping[str, Path],
) -> Path:
    """Publish all ten offset paths and aggregate reports atomically."""
    destination = _validate_publication_destination(model_name, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _validate_no_reparse_ancestry(destination, "staggered strategy destination")
    lock_path = destination.parent / ".strategy-10d.lock"
    if _is_reparse_escape(lock_path):
        raise ValueError(
            "staggered strategy publication lock may not be a symlink, junction, "
            "or reparse point"
        )
    lock = _acquire_strategy_lock(lock_path)
    staging: Path | None = None
    try:
        if destination.exists():
            raise FileExistsError(
                f"staggered strategy run already exists: {destination}"
            )
        stale_staging = list(destination.parent.glob(f".{destination.name}.*"))
        if stale_staging:
            raise FileExistsError(
                f"partial staggered strategy staging already exists: {stale_staging[0]}"
            )

        inputs = load_locked_strategy_inputs(model_name, settings, source_paths)
        loaded_paths = {name: Path(path) for name, path in inputs.source_paths.items()}
        requested_paths = {name: Path(path) for name, path in source_paths.items()}
        if loaded_paths != requested_paths:
            raise ValueError("staggered source provenance paths changed during loading")
        _validate_source_provenance(loaded_paths, inputs.source_hashes)
        schedules = build_offset_schedules(inputs.trading_calendar, settings)
        paths = tuple(
            simulate_staggered_offset(
                inputs.predictions,
                inputs.market_panel,
                inputs.trading_calendar,
                settings,
                schedule,
            )
            for schedule in schedules
        )
        _validated_paths(paths)
        offset_metrics = build_offset_metrics(paths)
        offset_summary = build_offset_summary(offset_metrics)
        average_nav = build_average_nav(paths, settings)
        average_metrics = summarize_average_nav(average_nav, settings)

        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        offset_descriptors = [
            _write_staggered_offset(
                path,
                staging / f"offset_{path.offset:02d}",
                settings,
                model_name,
                inputs.source_hashes,
            )
            for path in paths
        ]
        offset_metrics.to_csv(staging / "offset_metrics.csv", index=False)
        offset_summary.reset_index().to_csv(
            staging / "offset_summary.csv", index=False
        )
        average_nav.reset_index().to_csv(staging / "average_nav.csv", index=False)
        _write_json_file(staging / "average_metrics.json", average_metrics)
        payload_hashes = {
            name: file_sha256(staging / name)
            for name in _STAGGERED_MODEL_PAYLOAD_NAMES
        }
        settings_payload = _settings_manifest(settings)
        schedule_payloads = [
            descriptor["schedule"] for descriptor in offset_descriptors
        ]
        model_manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "completed",
            "purpose": "frozen_rank_model_staggered_10d_strategy_2024_2025",
            "model_name": model_name,
            "settings": settings_payload,
            "rows": {
                "offsets": len(offset_descriptors),
                "offset_metrics": len(offset_metrics),
                "offset_summary": len(offset_summary),
                "average_nav": len(average_nav),
                "average_metrics": len(average_metrics),
            },
            "formulas": {
                "offset_metrics": _SUMMARY_FORMULAS,
                "average_metrics": _AVERAGE_FORMULAS,
            },
            "observation_conventions": _average_observation_conventions(),
            "input_sha256": dict(inputs.source_hashes),
            "schedule": {
                "horizon_trading_days": STAGGERED_HORIZON,
                "offset_count": STAGGERED_OFFSET_COUNT,
                "common_start_date": average_nav.index[0].date().isoformat(),
                "common_end_date": average_nav.index[-1].date().isoformat(),
                "earliest_final_horizon_date": min(
                    path.schedule.final_horizon_date for path in paths
                )
                .date()
                .isoformat(),
            },
            "offsets": offset_descriptors,
            "contract_sha256": _model_contract_hashes(
                settings_payload, schedule_payloads
            ),
            "output_sha256": payload_hashes,
            "manifest_hash_convention": _MANIFEST_HASH_CONVENTION,
        }
        model_manifest["output_sha256"]["manifest.json"] = (
            _model_manifest_logical_hash(model_manifest)
        )
        _write_json_file(staging / "manifest.json", model_manifest)

        _validate_staggered_publication(
            staging,
            model_name,
            loaded_paths,
            inputs.source_hashes,
        )
        _validate_source_provenance(loaded_paths, inputs.source_hashes)
        _validate_no_reparse_ancestry(
            destination, "staggered strategy destination"
        )
        _validate_no_reparse_subtree(staging, "staggered strategy staging")
        os.replace(staging, destination)
        staging = None
    finally:
        try:
            if staging is not None:
                _remove_staging_best_effort(staging)
        finally:
            _release_strategy_lock(lock)
    return destination


def _validate_offset_schedules(
    schedules: Sequence[OffsetSchedule], period: pd.DatetimeIndex
) -> None:
    if len(schedules) != STAGGERED_OFFSET_COUNT:
        raise ValueError("staggered schedules must contain exactly 10 offsets")

    eligible = tuple(pd.Timestamp(date) for date in period[:-STAGGERED_COMPLETE_SHIFT])
    assigned = tuple(
        pd.Timestamp(date) for schedule in schedules for date in schedule.signal_dates
    )
    if len(assigned) != len(set(assigned)) or set(assigned) != set(eligible):
        raise ValueError("staggered schedules must partition every eligible signal date")

    expected_offsets = tuple(range(1, STAGGERED_OFFSET_COUNT + 1))
    if tuple(schedule.offset for schedule in schedules) != expected_offsets:
        raise ValueError("staggered schedule offsets must be ordered from 1 through 10")

    date_positions = {pd.Timestamp(date): index for index, date in enumerate(period)}
    for schedule in schedules:
        expected_signals = tuple(
            eligible[index]
            for index in range(schedule.offset - 1, len(eligible), STAGGERED_OFFSET_COUNT)
        )
        signals = tuple(pd.Timestamp(date) for date in schedule.signal_dates)
        executions = tuple(pd.Timestamp(date) for date in schedule.execution_dates)
        if signals != expected_signals:
            raise ValueError("staggered schedule signal dates do not match their offset")
        if len(executions) != len(signals):
            raise ValueError("staggered schedule execution dates do not match signal dates")
        expected_executions = tuple(period[date_positions[date] + 1] for date in signals)
        if executions != expected_executions:
            raise ValueError("staggered schedule execution dates must be T+1")
        expected_horizon = period[date_positions[signals[-1]] + STAGGERED_COMPLETE_SHIFT]
        if pd.Timestamp(schedule.final_horizon_date) != expected_horizon:
            raise ValueError("staggered schedule final horizon must be T+11")


def build_offset_schedules(
    calendar: Any,
    settings: StrategySettings,
) -> Sequence[OffsetSchedule]:
    """Partition the sealed complete signal period into ten daily offsets."""
    official = _calendar_dates(calendar)
    period = official[(official >= settings.start) & (official <= settings.end)]
    if len(period) < STAGGERED_COMPLETE_SHIFT + 1:
        raise ValueError("strategy period requires at least 12 official trading dates")

    eligible = period[:-STAGGERED_COMPLETE_SHIFT]
    if len(eligible) != _ELIGIBLE_SIGNAL_COUNT:
        raise ValueError("staggered strategy requires exactly 474 eligible signal dates")
    if eligible[0] != _FIRST_ELIGIBLE_SIGNAL or eligible[-1] != _LAST_ELIGIBLE_SIGNAL:
        raise ValueError("staggered strategy eligible dates must span 2024-01-02 through 2025-12-16")

    schedules: list[OffsetSchedule] = []
    for offset in range(1, STAGGERED_OFFSET_COUNT + 1):
        indices = list(range(offset - 1, len(eligible), STAGGERED_OFFSET_COUNT))
        schedules.append(
            OffsetSchedule(
                offset=offset,
                signal_dates=tuple(eligible[index] for index in indices),
                execution_dates=tuple(period[index + 1] for index in indices),
                final_horizon_date=period[indices[-1] + STAGGERED_COMPLETE_SHIFT],
            )
        )
    result = tuple(schedules)
    _validate_offset_schedules(result, period)
    return result


_STAGGERED_DAILY_COLUMNS = (
    "offset",
    "is_rebalance",
    "active_signal_date",
    *_DAILY_NAV_COLUMNS,
)
_STAGGERED_DIAGNOSTIC_COLUMNS = (
    "offset",
    "is_rebalance",
    "active_signal_date",
    *_DIAGNOSTIC_COLUMNS,
)


def _validate_path_schedule(
    schedule: OffsetSchedule, period: pd.DatetimeIndex
) -> None:
    if not isinstance(schedule, OffsetSchedule):
        raise ValueError("staggered path requires an offset schedule")
    if (
        isinstance(schedule.offset, bool)
        or not isinstance(schedule.offset, int)
        or not 1 <= schedule.offset <= STAGGERED_OFFSET_COUNT
    ):
        raise ValueError("staggered path offset must be from 1 through 10")

    signals = tuple(pd.Timestamp(date) for date in schedule.signal_dates)
    executions = tuple(pd.Timestamp(date) for date in schedule.execution_dates)
    if not signals or len(signals) != len(executions):
        raise ValueError("staggered path requires matching non-empty schedule dates")
    if len(signals) != len(set(signals)) or len(executions) != len(set(executions)):
        raise ValueError("staggered path schedule dates must be unique")

    positions = {pd.Timestamp(date): index for index, date in enumerate(period)}
    if any(date not in positions for date in (*signals, *executions)):
        raise ValueError("staggered path schedule dates must be official period dates")
    signal_positions = [positions[date] for date in signals]
    execution_positions = [positions[date] for date in executions]
    if signal_positions != sorted(signal_positions):
        raise ValueError("staggered path signal dates must be ordered")
    if signal_positions[0] != schedule.offset - 1 or any(
        current - previous != STAGGERED_OFFSET_COUNT
        for previous, current in zip(signal_positions, signal_positions[1:])
    ):
        raise ValueError("staggered path signal dates must match the offset cadence")
    if any(execution != signal + 1 for signal, execution in zip(signal_positions, execution_positions)):
        raise ValueError("staggered path execution dates must be T+1")

    final_horizon_date = pd.Timestamp(schedule.final_horizon_date)
    expected_horizon_position = signal_positions[-1] + STAGGERED_COMPLETE_SHIFT
    if (
        final_horizon_date not in positions
        or positions[final_horizon_date] != expected_horizon_position
    ):
        raise ValueError("staggered path final horizon date must be T+11")


def _notional_totals(trades: pd.DataFrame) -> tuple[float, float]:
    if trades.empty:
        return 0.0, 0.0
    executed = trades["status"].eq("executed")
    buy_notional = float(
        trades.loc[executed & trades["side"].eq("buy"), "gross_notional"].sum()
    )
    sell_notional = float(
        trades.loc[executed & trades["side"].eq("sell"), "gross_notional"].sum()
    )
    return buy_notional, sell_notional


def simulate_staggered_offset(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    calendar: Any,
    settings: StrategySettings,
    schedule: OffsetSchedule,
) -> StaggeredPathBundle:
    """Value one offset daily and update its desired set only on schedule."""
    official = _calendar_dates(calendar)
    period = official[(official >= settings.start) & (official <= settings.end)]
    if period.empty:
        raise ValueError("strategy period has no official trading dates")
    _validate_path_schedule(schedule, period)

    desired_by_date = select_daily_top(predictions, settings, official)
    if set(desired_by_date) != set(period):
        raise ValueError("prediction dates must exactly match the strategy trading calendar")
    market_panel = _normalize_market_panel(market)
    market_by_date = {
        pd.Timestamp(date): group.drop(columns="date").reset_index(drop=True)
        for date, group in market_panel.groupby("date", sort=False)
    }

    final_horizon_date = pd.Timestamp(schedule.final_horizon_date)
    path_dates = period[period <= final_horizon_date]
    signal_by_execution = {
        pd.Timestamp(execution): pd.Timestamp(signal)
        for signal, execution in zip(schedule.signal_dates, schedule.execution_dates)
    }
    first_date = pd.Timestamp(path_dates[0])
    state = PortfolioState(
        cash=settings.initial_nav,
        positions={},
        previous_nav=settings.initial_nav,
    )
    active_signal_date = pd.NaT
    current_desired: tuple[str, ...] = ()
    daily_rows: list[dict[str, Any]] = [
        {
            "date": first_date,
            "offset": schedule.offset,
            "is_rebalance": False,
            "active_signal_date": active_signal_date,
            "signal_date": active_signal_date,
            "pre_trade_nav": settings.initial_nav,
            "nav": settings.initial_nav,
            "gross_return": 0.0,
            "net_return": 0.0,
            "cash": settings.initial_nav,
            "cash_ratio": 1.0,
            "holding_count": 0,
            "desired_count": 0,
            "buy_notional": 0.0,
            "sell_notional": 0.0,
            "total_cost": 0.0,
            "gross_turnover": 0.0,
            "one_way_turnover": 0.0,
            "attempted_order_count": 0,
            "executed_order_count": 0,
            "blocked_order_count": 0,
        }
    ]
    trade_frames: list[pd.DataFrame] = []
    position_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for date in path_dates[1:]:
        execution_date = pd.Timestamp(date)
        is_rebalance = execution_date in signal_by_execution
        if is_rebalance:
            active_signal_date = signal_by_execution[execution_date]
            current_desired = desired_by_date[active_signal_date]
        prior_nav = state.previous_nav
        transition_signal_date = (
            active_signal_date if not pd.isna(active_signal_date) else first_date
        )
        transition = transition_at_open(
            state,
            current_desired,
            market_by_date.get(execution_date, pd.DataFrame(columns=["stock_code"])),
            transition_signal_date,
            execution_date,
            settings,
            allow_buys=is_rebalance,
        )
        state = transition.state
        trades = transition.trades.reset_index()
        if not trades.empty:
            trade_frames.append(trades)
        buy_notional, sell_notional = _notional_totals(trades)
        gross_turnover = (
            (buy_notional + sell_notional) / transition.pre_trade_nav
            if transition.pre_trade_nav > 0
            else 0.0
        )
        daily_rows.append(
            {
                "date": execution_date,
                "offset": schedule.offset,
                "is_rebalance": is_rebalance,
                "active_signal_date": active_signal_date,
                "signal_date": active_signal_date,
                "pre_trade_nav": transition.pre_trade_nav,
                "nav": transition.end_nav,
                "gross_return": (
                    transition.pre_trade_nav / prior_nav - 1.0 if prior_nav > 0 else 0.0
                ),
                "net_return": transition.end_nav / prior_nav - 1.0 if prior_nav > 0 else 0.0,
                "cash": state.cash,
                "cash_ratio": state.cash / transition.end_nav if transition.end_nav > 0 else 0.0,
                "holding_count": len(state.positions),
                "desired_count": len(current_desired),
                "buy_notional": buy_notional,
                "sell_notional": sell_notional,
                "total_cost": transition.total_cost,
                "gross_turnover": gross_turnover,
                "one_way_turnover": gross_turnover / 2.0,
                "attempted_order_count": transition.diagnostics["attempted_order_count"],
                "executed_order_count": transition.diagnostics["executed_order_count"],
                "blocked_order_count": transition.diagnostics["blocked_order_count"],
            }
        )
        position_rows.extend(_position_rows(state, execution_date))
        diagnostic_rows.append(
            {
                "offset": schedule.offset,
                "is_rebalance": is_rebalance,
                "active_signal_date": active_signal_date,
                "signal_date": active_signal_date,
                "execution_date": execution_date,
                **transition.diagnostics,
            }
        )

    daily_nav = pd.DataFrame(daily_rows).set_index("date")
    daily_nav.index = pd.DatetimeIndex(daily_nav.index, name="date")
    daily_nav = daily_nav.loc[:, _STAGGERED_DAILY_COLUMNS]
    strategy = StrategyBundle(
        daily_nav=daily_nav,
        trades=(
            pd.concat(trade_frames, ignore_index=True)
            if trade_frames
            else pd.DataFrame(columns=_TRADE_COLUMNS)
        ),
        positions=pd.DataFrame(position_rows, columns=_POSITION_COLUMNS),
        execution_diagnostics=pd.DataFrame(
            diagnostic_rows, columns=_STAGGERED_DIAGNOSTIC_COLUMNS
        ),
        ending_positions=pd.DataFrame(
            _ending_position_frame(state), columns=_ENDING_POSITION_COLUMNS
        ),
        metrics_summary={},
    )
    strategy.metrics_summary = summarize_strategy(strategy, settings)
    return StaggeredPathBundle(
        offset=schedule.offset,
        schedule=schedule,
        strategy=strategy,
    )
