"""Publish dynamic forward strategy runs without touching locked artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd

from rank_model.stages.dataset import file_sha256
from rank_model.stages.staggered_strategy import (
    AVERAGE_METRICS,
    OffsetSchedule,
    StaggeredPathBundle,
    build_average_nav,
    build_dynamic_offset_schedules,
    build_offset_metrics,
    build_offset_summary,
    simulate_staggered_offset,
    summarize_average_nav,
)
from rank_model.stages.strategy import StrategyBundle, StrategySettings


MODEL_NAMES = (
    "ridge_rank_regression",
    "xgboost_rank_regression",
    "lightgbm_rank_regression",
    "lightgbm_lambdarank",
    "mlp_top100_hybrid_rank",
)
_MODEL_OUTPUTS = (
    "daily_nav.csv",
    "trades.parquet",
    "positions.parquet",
    "execution_diagnostics.csv",
    "ending_positions.csv",
    "metrics_summary.json",
)


def strategy_settings(
    config: Mapping[str, Any],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> StrategySettings:
    """Build ordinary strategy settings with a dynamic date window."""
    values = config.get("strategy")
    if not isinstance(values, Mapping):
        raise ValueError("forward config is missing [strategy]")

    def integer(name: str) -> int:
        value = values.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"strategy {name} must be a positive integer")
        return value

    def number(name: str) -> float:
        value = values.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"strategy {name} must be numeric")
        value = float(value)
        if not pd.notna(value) or value < 0:
            raise ValueError(f"strategy {name} must be finite and nonnegative")
        return value

    settings = StrategySettings(
        start=pd.Timestamp(start).normalize(),
        end=pd.Timestamp(end).normalize(),
        top_k=integer("top_k"),
        expected_cross_section_size=integer("expected_cross_section_size"),
        min_listing_days=integer("min_listing_days"),
        initial_nav=number("initial_nav"),
        commission_rate=number("commission_bps") / 10_000,
        slippage_rate=number("slippage_bps") / 10_000,
        sell_stamp_duty_rate=number("sell_stamp_duty_bps") / 10_000,
        limit_tolerance=number("limit_tolerance"),
        annualization_days=integer("annualization_days"),
    )
    if settings.top_k != 100 or settings.expected_cross_section_size != 1000:
        raise ValueError("forward strategy must use Top100 and 1000-stock cross sections")
    if settings.start > settings.end:
        raise ValueError("forward strategy start must not be after end")
    return settings


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_model_path(path: StaggeredPathBundle, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    strategy: StrategyBundle = path.strategy
    strategy.daily_nav.reset_index().to_csv(destination / "daily_nav.csv", index=False)
    strategy.trades.to_parquet(destination / "trades.parquet", index=False)
    strategy.positions.to_parquet(destination / "positions.parquet", index=False)
    strategy.execution_diagnostics.to_csv(
        destination / "execution_diagnostics.csv", index=False
    )
    strategy.ending_positions.to_csv(destination / "ending_positions.csv", index=False)
    _write_json(destination / "metrics_summary.json", strategy.metrics_summary)
    return {
        "offset": path.offset,
        "first_signal_date": path.schedule.signal_dates[0].date().isoformat(),
        "first_execution_date": path.schedule.execution_dates[0].date().isoformat(),
        "final_signal_date": path.schedule.signal_dates[-1].date().isoformat(),
        "final_horizon_date": path.schedule.final_horizon_date.date().isoformat(),
        "rows": {
            "daily_nav": int(len(strategy.daily_nav)),
            "trades": int(len(strategy.trades)),
            "positions": int(len(strategy.positions)),
            "execution_diagnostics": int(len(strategy.execution_diagnostics)),
        },
        "output_sha256": {
            name: file_sha256(destination / name) for name in _MODEL_OUTPUTS
        },
    }


def _write_model_run(
    model_name: str,
    paths: Sequence[StaggeredPathBundle],
    destination: Path,
    *,
    source_hashes: Mapping[str, str],
    settings: StrategySettings,
) -> dict[str, Any]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unsupported forward strategy model: {model_name}")
    if destination.exists():
        raise FileExistsError(f"forward strategy model output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    path_descriptors = []
    for path in paths:
        path_descriptors.append(
            _write_model_path(path, destination / f"offset_{path.offset:02d}")
        )
    offset_metrics = build_offset_metrics(paths)
    offset_summary = build_offset_summary(offset_metrics)
    average_nav = build_average_nav(paths, settings)
    average_metrics = summarize_average_nav(average_nav, settings)
    offset_metrics.to_csv(destination / "offset_metrics.csv", index=False)
    offset_summary.reset_index().to_csv(destination / "offset_summary.csv", index=False)
    average_nav.reset_index().to_csv(destination / "average_nav.csv", index=False)
    _write_json(destination / "average_metrics.json", average_metrics)
    output_names = (
        "offset_metrics.csv",
        "offset_summary.csv",
        "average_nav.csv",
        "average_metrics.json",
    )
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "purpose": "forward_rank_model_staggered_10d_strategy",
        "model_name": model_name,
        "settings": {
            "start": settings.start.date().isoformat(),
            "end": settings.end.date().isoformat(),
            "top_k": settings.top_k,
            "horizon_trading_days": 10,
            "offset_count": 10,
            "min_listing_days": settings.min_listing_days,
            "initial_nav": settings.initial_nav,
            "commission_rate": settings.commission_rate,
            "slippage_rate": settings.slippage_rate,
            "sell_stamp_duty_rate": settings.sell_stamp_duty_rate,
            "annualization_days": settings.annualization_days,
        },
        "source_sha256": dict(source_hashes),
        "schedule": path_descriptors,
        "average_metrics": average_metrics,
        "output_sha256": {
            name: file_sha256(destination / name) for name in output_names
        },
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def _comparison_frame(
    model_manifests: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        metrics = model_manifests[model_name].get("average_metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"forward model manifest has no average metrics: {model_name}")
        rows.append(
            {
                "model_name": model_name,
                **{metric: metrics[metric] for metric in AVERAGE_METRICS},
            }
        )
    return pd.DataFrame(rows, columns=("model_name", *AVERAGE_METRICS))


def publish_forward_run(
    run_id: str,
    *,
    feature_frame: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    market: pd.DataFrame,
    calendar: pd.DataFrame,
    settings: StrategySettings,
    output_root: Path,
    metadata: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> Path:
    """Run and atomically publish all five dynamic strategy results."""
    if not isinstance(run_id, str) or not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in run_id
    ):
        raise ValueError("run_id may contain only letters, numbers, underscore, and hyphen")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / run_id
    if destination.exists():
        raise FileExistsError(f"forward run already exists: {destination}")
    if set(predictions) != set(MODEL_NAMES):
        raise ValueError("forward run requires predictions for exactly five models")
    feature_keys = feature_frame[["date", "stock_code"]].reset_index(drop=True)
    for model_name in MODEL_NAMES:
        frame = predictions[model_name]
        if not frame[["date", "stock_code"]].reset_index(drop=True).equals(feature_keys):
            raise ValueError(f"prediction keys differ from feature keys: {model_name}")

    stale = list(output_root.glob(f".{run_id}.*"))
    if stale:
        raise FileExistsError(f"partial forward run exists: {stale[0]}")
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=output_root))
    model_manifests: dict[str, dict[str, Any]] = {}
    try:
        feature_frame.to_parquet(staging / "feature_snapshot.parquet", index=False)
        prediction_root = staging / "predictions"
        prediction_root.mkdir()
        for model_name in MODEL_NAMES:
            predictions[model_name].to_parquet(
                prediction_root / f"{model_name}.parquet", index=False
            )

        strategy_root = staging / "strategy_10d_runs"
        strategy_root.mkdir()
        schedules = build_dynamic_offset_schedules(calendar, settings)
        for model_name in MODEL_NAMES:
            paths = tuple(
                simulate_staggered_offset(
                    predictions[model_name],
                    market,
                    calendar,
                    settings,
                    schedule,
                    prediction_split="forward",
                )
                for schedule in schedules
            )
            model_manifests[model_name] = _write_model_run(
                model_name,
                paths,
                strategy_root / model_name,
                source_hashes=source_hashes,
                settings=settings,
            )

        comparison = _comparison_frame(model_manifests)
        comparison.to_csv(staging / "strategy_10d_comparison.csv", index=False)
        factor_update_manifest = metadata.get("factor_update_manifest")
        if factor_update_manifest is not None:
            if not isinstance(factor_update_manifest, Mapping):
                raise ValueError("factor update manifest must be a mapping")
            _write_json(staging / "factor_update_manifest.json", factor_update_manifest)
        as_of = metadata.get("as_of_date")
        last_signal = metadata.get("last_complete_signal")
        _write_json(
            staging / "asof.json",
            {
                **dict(metadata),
                "as_of_date": as_of,
                "last_complete_signal": last_signal,
                "model_names": list(MODEL_NAMES),
            },
        )
        top_manifest = {
            "schema_version": 1,
            "status": "completed",
            "purpose": "forward_rank_model_staggered_10d_strategy",
            "run_id": run_id,
            "metadata": dict(metadata),
            "source_sha256": dict(source_hashes),
            "model_names": list(MODEL_NAMES),
            "feature_rows": int(len(feature_frame)),
            "prediction_rows": {
                model_name: int(len(predictions[model_name])) for model_name in MODEL_NAMES
            },
            "model_manifests": model_manifests,
            "output_sha256": {
                "asof.json": file_sha256(staging / "asof.json"),
                "feature_snapshot.parquet": file_sha256(
                    staging / "feature_snapshot.parquet"
                ),
                "strategy_10d_comparison.csv": file_sha256(
                    staging / "strategy_10d_comparison.csv"
                ),
                **(
                    {
                        "factor_update_manifest.json": file_sha256(
                            staging / "factor_update_manifest.json"
                        )
                    }
                    if (staging / "factor_update_manifest.json").exists()
                    else {}
                ),
            },
        }
        _write_json(staging / "manifest.json", top_manifest)
        os.replace(staging, destination)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return destination
