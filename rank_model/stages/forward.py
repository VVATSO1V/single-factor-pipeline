"""Build point-in-time forward features and score frozen rank models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from rank_model.stages.dataset import file_sha256
from rank_model.stages.locked_test import (
    _build_daily_context,
    _normalize_industry,
    _transform_daily_cross_sections,
    _validate_daily_size,
    _validate_keys,
    _verify_artifact_hashes,
)
from rank_model.stages.preprocessing import RankPreprocessor, predicted_percentiles
from rank_model.stages.training import _load_persisted_model, _predict_model


KEY_COLUMNS = ("date", "stock_code")
FIVE_FROZEN_MODELS = (
    "ridge_rank_regression",
    "xgboost_rank_regression",
    "lightgbm_rank_regression",
    "lightgbm_lambdarank",
    "mlp_top100_hybrid_rank",
)
UNKNOWN_INDUSTRY = "UNKNOWN"


def _calendar_index(calendar: Any) -> pd.DatetimeIndex:
    if isinstance(calendar, pd.DataFrame):
        if "date" not in calendar.columns:
            raise ValueError("trading calendar is missing date")
        values = calendar["date"]
    else:
        values = calendar
    dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise")).normalize()
    dates = dates.sort_values().unique()
    if len(dates) == 0 or dates.hasnans:
        raise ValueError("trading calendar must be non-empty and valid")
    return pd.DatetimeIndex(dates, name="date")


def resolve_as_of_date(
    calendar: Any,
    requested_end: pd.Timestamp | str | None = None,
) -> pd.Timestamp:
    """Return the latest official date not later than the requested endpoint."""
    dates = _calendar_index(calendar)
    if requested_end is None:
        return pd.Timestamp(dates[-1]).normalize()
    endpoint = pd.Timestamp(requested_end).normalize()
    eligible = dates[dates <= endpoint]
    if len(eligible) == 0:
        raise ValueError("calendar has no date at or before requested endpoint")
    return pd.Timestamp(eligible[-1]).normalize()


def last_complete_signal(
    calendar: Any,
    as_of_date: pd.Timestamp | str,
    horizon: int = 10,
) -> pd.Timestamp:
    """Return the last T whose T+1 entry and T+11 exit are available."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    dates = _calendar_index(calendar)
    as_of = resolve_as_of_date(dates, as_of_date)
    as_of_position = int(dates.searchsorted(as_of, side="right") - 1)
    signal_position = as_of_position - horizon - 1
    if signal_position < 0:
        raise ValueError("calendar does not contain a complete forward horizon")
    return pd.Timestamp(dates[signal_position]).normalize()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object: {path}")
    return value


def _numeric_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    return result


def build_forward_frame(
    factor_wide: pd.DataFrame,
    market: pd.DataFrame,
    calendar: Any,
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    feature_schema: Mapping[str, Any],
    mad_width: float = 3.0,
    expected_cross_section_size: int = 1000,
    date_chunk_size: int = 60,
    windows: Sequence[int] = (1, 5, 20),
    cap_coverage_threshold: float = 0.95,
) -> pd.DataFrame:
    """Build the frozen rank feature contract for a dynamic forward window."""
    official = _calendar_index(calendar)
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if start_date > end_date:
        raise ValueError("forward start must not be after forward end")
    period = official[(official >= start_date) & (official <= end_date)]
    if len(period) < 12:
        raise ValueError("forward period requires at least 12 official trading dates")

    factors = _validate_keys(factor_wide, "forward factor-wide table")
    factor_columns = [column for column in factors.columns if column not in KEY_COLUMNS]
    if not factor_columns:
        raise ValueError("forward factor-wide table has no factor columns")
    factors = _numeric_columns(factors, factor_columns)
    factors = factors.loc[factors["date"].isin(period)].copy()
    if factors.empty:
        raise ValueError("forward factor-wide table has no rows in the requested period")
    _validate_daily_size(
        factors,
        expected_cross_section_size,
        "forward factor-wide table",
    )

    panel = _validate_keys(market, "forward market panel")
    required_market = {
        "in_universe",
        "post_close",
        "raw_close",
        "limit_up",
        "limit_down",
        "industry",
        "market_cap",
    }
    missing_market = sorted(required_market.difference(panel.columns))
    if missing_market:
        raise ValueError(f"forward market panel is missing columns: {missing_market}")
    panel = panel.copy()
    panel["in_universe"] = (
        panel["in_universe"].astype("boolean").fillna(False).astype(bool)
    )
    panel["industry"] = _normalize_industry(panel["industry"])
    panel = _numeric_columns(
        panel,
        ["post_close", "raw_close", "limit_up", "limit_down", "market_cap"],
    )

    metadata = panel.loc[
        panel["in_universe"], [*KEY_COLUMNS, "industry", "market_cap"]
    ].drop_duplicates(KEY_COLUMNS)
    data = factors.merge(
        metadata,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator="_metadata_match",
    )
    missing_metadata = data["_metadata_match"].ne("both")
    if missing_metadata.any():
        sample = data.loc[missing_metadata, list(KEY_COLUMNS)].head(5).to_dict("records")
        raise ValueError(f"forward factor keys lack T-day market metadata: {sample}")
    data = data.drop(columns="_metadata_match")

    transformed, factor_features = _transform_daily_cross_sections(
        data,
        factor_columns,
        mad_width=float(mad_width),
        expected_cross_section_size=expected_cross_section_size,
        date_chunk_size=int(date_chunk_size),
    )
    factor_frame = pd.DataFrame(transformed, columns=factor_features, index=data.index)
    data = pd.concat([data.drop(columns=factor_columns), factor_frame], axis=1)
    data["industry"] = _normalize_industry(data["industry"])
    data["industry_missing"] = data["industry"].eq(UNKNOWN_INDUSTRY).astype(np.float32)
    raw_cap = pd.to_numeric(data["market_cap"], errors="coerce").where(
        lambda values: values.gt(0)
    )
    log_cap = np.log1p(raw_cap)
    cap_median = log_cap.groupby(data["date"], sort=False).transform("median")
    data["market_cap_missing"] = log_cap.isna().astype(np.float32)
    data["log_market_cap"] = log_cap.fillna(cap_median).fillna(0.0).astype(np.float64)
    data = data.drop(columns="market_cap")

    market_context, industry_context = _build_daily_context(
        panel,
        official,
        windows=tuple(int(window) for window in windows),
        cap_coverage_threshold=float(cap_coverage_threshold),
    )
    market_context = market_context.loc[
        market_context["date"].between(start_date, end_date)
    ]
    industry_context = industry_context.loc[
        industry_context["date"].between(start_date, end_date)
    ]
    market_features = [column for column in market_context if column != "date"]
    industry_features = [
        column for column in industry_context if column not in {"date", "industry"}
    ]
    data = data.merge(
        market_context,
        on="date",
        how="left",
        validate="many_to_one",
    )
    data = data.merge(
        industry_context,
        on=["date", "industry"],
        how="left",
        validate="many_to_one",
    )
    unknown = data["industry"].eq(UNKNOWN_INDUSTRY)
    for column in industry_features:
        data.loc[unknown, column] = 1.0 if column.endswith("__missing") else 0.0

    expected_continuous = feature_schema.get("continuous_feature_columns")
    expected_industry = feature_schema.get("industry_column")
    continuous = [
        *factor_features,
        "log_market_cap",
        "market_cap_missing",
        "industry_missing",
        *market_features,
        *industry_features,
    ]
    if not isinstance(expected_continuous, list) or continuous != expected_continuous:
        raise ValueError("forward continuous feature order differs from frozen schema")
    if expected_industry != "industry":
        raise ValueError("forward industry feature contract differs from frozen schema")
    values = data[continuous].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("forward continuous features must be finite")
    result = data[[*KEY_COLUMNS, *continuous, "industry"]].sort_values(
        list(KEY_COLUMNS), kind="mergesort"
    )
    result = result.reset_index(drop=True)
    _validate_daily_size(result, expected_cross_section_size, "forward feature frame")
    if not result.duplicated(list(KEY_COLUMNS)).any():
        return result
    raise ValueError("forward feature frame contains duplicate keys")


def _prediction_frame(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise ValueError("forward scores must be finite and cover every key")
    result = frame[list(KEY_COLUMNS)].copy()
    result["split"] = "forward"
    result["horizon"] = 10
    result["score_raw"] = values
    result["pred_rank_pct"] = predicted_percentiles(values, result["date"])
    result["pred_rank_position"] = result.groupby("date", sort=False)[
        "score_raw"
    ].rank(method="average", ascending=True)
    return result


def _load_model_bundle(
    model_name: str,
    final_runs_dir: Path,
    frozen_spec_path: Path,
) -> tuple[RankPreprocessor, Any, dict[str, Any]]:
    final_directory = Path(final_runs_dir) / model_name
    manifest_path = final_directory / "manifest.json"
    manifest = _read_json_object(manifest_path, f"{model_name} final-model manifest")
    if (
        manifest.get("status") != "completed"
        or manifest.get("purpose") != "final_refit_2019_2023"
        or manifest.get("model_name") != model_name
    ):
        raise ValueError(f"{model_name} is not a completed frozen final model")
    _verify_artifact_hashes(final_directory, manifest)
    feature_schema = _read_json_object(
        final_directory / "feature_schema.json",
        f"{model_name} feature schema",
    )
    frozen_spec = _read_json_object(frozen_spec_path, "frozen model specification")
    candidates = [
        candidate
        for candidate in frozen_spec.get("candidates", [])
        if isinstance(candidate, Mapping) and candidate.get("model_name") == model_name
    ]
    if len(candidates) != 1:
        raise ValueError(f"frozen specification has no unique candidate: {model_name}")
    candidate = candidates[0]
    preprocessor = joblib.load(final_directory / "preprocessor.joblib")
    if not isinstance(preprocessor, RankPreprocessor):
        raise ValueError(f"{model_name} preprocessor has an unexpected type")
    if preprocessor.feature_names != candidate.get("transformed_feature_columns"):
        raise ValueError(f"{model_name} preprocessor feature order differs from frozen spec")
    if preprocessor.feature_names != feature_schema.get("transformed_feature_columns"):
        raise ValueError(f"{model_name} preprocessor feature order differs from schema")
    model = _load_persisted_model(final_directory)
    return preprocessor, model, {
        "manifest_sha256": file_sha256(manifest_path),
        "artifact_sha256": manifest.get("artifact_sha256"),
        "feature_schema_sha256": file_sha256(final_directory / "feature_schema.json"),
        "feature_schema": feature_schema,
    }


def predict_frozen_models(
    frame: pd.DataFrame,
    *,
    final_runs_dir: Path,
    frozen_spec_path: Path,
    model_names: Sequence[str] = FIVE_FROZEN_MODELS,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Score one forward feature frame with the five frozen models."""
    names = tuple(model_names)
    if names != FIVE_FROZEN_MODELS:
        raise ValueError("forward inference requires exactly the five frozen models")
    predictions: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, Any]] = {}
    expected_keys = frame[list(KEY_COLUMNS)].reset_index(drop=True)
    for position, model_name in enumerate(names, start=1):
        print(
            f"forward prediction progress: {position}/{len(names)} {model_name}",
            flush=True,
        )
        preprocessor, model, bundle_metadata = _load_model_bundle(
            model_name,
            Path(final_runs_dir),
            Path(frozen_spec_path),
        )
        features = preprocessor.transform(
            frame,
            scale_continuous=not isinstance(model, (xgb.Booster, lgb.Booster)),
        )
        scores = _predict_model(model, features)
        output = _prediction_frame(frame, scores)
        _validate_daily_size(output, 1000, f"{model_name} forward predictions")
        if not output[list(KEY_COLUMNS)].reset_index(drop=True).equals(expected_keys):
            raise ValueError(f"{model_name} prediction keys differ from feature keys")
        predictions[model_name] = output
        metadata[model_name] = bundle_metadata
        print(
            f"forward prediction complete: {position}/{len(names)} {model_name}",
            flush=True,
        )
    return predictions, metadata
