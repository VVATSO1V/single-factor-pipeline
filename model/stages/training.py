"""Train deterministic development-only model baselines."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from model.stages.preprocessing import transform_daily_cross_sections


KEY_COLUMNS = ["date", "stock_code"]
SUPPORTED_HORIZONS = (1, 5, 10)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def validate_run_id(run_id: str) -> str:
    """Reject unsafe or ambiguous output directory names."""
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}"
        )
    return run_id


def mean_daily_rank_ic(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, int]:
    """Return equal-date-weighted cross-sectional Spearman Rank IC."""
    if len(dates) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("dates, y_true, and y_pred must have equal lengths")
    metric_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="raise").dt.normalize(),
            "y_true": np.asarray(y_true, dtype=np.float64),
            "y_pred": np.asarray(y_pred, dtype=np.float64),
        }
    )
    daily_values: list[float] = []
    for _, group in metric_frame.groupby("date", sort=False):
        finite = np.isfinite(group["y_true"]) & np.isfinite(group["y_pred"])
        valid = group.loc[finite]
        if len(valid) < 2:
            continue
        true_ranks = valid["y_true"].rank(method="average").to_numpy()
        predicted_ranks = valid["y_pred"].rank(method="average").to_numpy()
        if np.ptp(true_ranks) == 0 or np.ptp(predicted_ranks) == 0:
            continue
        correlation = float(np.corrcoef(true_ranks, predicted_ranks)[0, 1])
        if np.isfinite(correlation):
            daily_values.append(correlation)
    if not daily_values:
        return float("nan"), 0
    return float(np.mean(daily_values)), len(daily_values)


def build_true_deciles(
    dates: pd.Series,
    targets: np.ndarray,
) -> np.ndarray:
    """Assign descending same-date target deciles while preserving ties."""
    normalized_dates = pd.DatetimeIndex(
        pd.to_datetime(dates, errors="raise")
    ).normalize()
    values = np.asarray(targets, dtype=np.float64)
    if len(normalized_dates) != len(values):
        raise ValueError("dates and targets must have equal lengths")
    if len(values) == 0:
        raise ValueError("decile input must not be empty")
    if normalized_dates.isna().any() or not np.isfinite(values).all():
        raise ValueError("decile dates and targets must be finite")

    frame = pd.DataFrame(
        {
            "date": normalized_dates,
            "target": values,
            "position": np.arange(len(values), dtype=np.int64),
        }
    )
    result = np.empty(len(frame), dtype=np.int8)
    for _, group in frame.groupby("date", sort=False):
        if len(group) < 10:
            raise ValueError("each decile date must contain at least 10 rows")
        ranks = group["target"].rank(
            method="average",
            ascending=False,
        ).to_numpy(dtype=np.float64)
        deciles = np.ceil(10.0 * ranks / len(group))
        deciles = np.clip(deciles, 1, 10).astype(np.int8)
        result[group["position"].to_numpy(dtype=np.int64)] = deciles
    return result


def build_purged_oof_ridge(
    feature_matrix: np.ndarray,
    dates: pd.Series,
    exit_dates: pd.Series,
    targets: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    *,
    prediction_years: tuple[int, ...] = (2020, 2021, 2022),
    ridge_lambda: float = 1.0,
) -> dict[str, Any]:
    """Build expanding OOF Ridge predictions with exit-date purging."""
    features = np.asarray(feature_matrix)
    normalized_dates = pd.DatetimeIndex(
        pd.to_datetime(dates, errors="raise")
    ).normalize()
    normalized_exits = pd.DatetimeIndex(
        pd.to_datetime(exit_dates, errors="coerce")
    ).normalize()
    values = np.asarray(targets, dtype=np.float64)
    train = np.asarray(train_mask, dtype=bool)
    validation = np.asarray(validation_mask, dtype=bool)
    row_count = len(values)
    if any(
        len(value) != row_count
        for value in (
            features,
            normalized_dates,
            normalized_exits,
            train,
            validation,
        )
    ):
        raise ValueError("OOF Ridge inputs must have equal row counts")
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("OOF Ridge features must be a non-empty matrix")
    if row_count == 0 or not train.any() or not validation.any():
        raise ValueError("OOF Ridge requires train and validation rows")
    if train[validation].any():
        raise ValueError("train and validation masks must not overlap")
    selected = train | validation
    if (
        normalized_dates[selected].isna().any()
        or normalized_exits[selected].isna().any()
        or not np.isfinite(values[selected]).all()
        or not np.isfinite(features[selected]).all()
    ):
        raise ValueError("OOF Ridge selected inputs must be finite")
    if (
        not isinstance(ridge_lambda, (int, float))
        or isinstance(ridge_lambda, bool)
        or ridge_lambda <= 0
    ):
        raise ValueError("ridge_lambda must be positive")
    if (
        not prediction_years
        or len(prediction_years) != len(set(prediction_years))
        or tuple(sorted(prediction_years)) != prediction_years
    ):
        raise ValueError("prediction_years must be unique and increasing")

    oof_predictions = np.full(row_count, np.nan, dtype=np.float64)
    oof_mask = np.zeros(row_count, dtype=bool)
    fold_metadata: list[dict[str, Any]] = []
    fold_models: list[Ridge] = []
    fold_scalers: list[StandardScaler] = []
    date_years = normalized_dates.year.to_numpy()
    for prediction_year in prediction_years:
        prediction_start = pd.Timestamp(prediction_year, 1, 1)
        fold_train = (
            train
            & (normalized_dates < prediction_start)
            & (normalized_exits < prediction_start)
        )
        fold_predict = train & (date_years == prediction_year)
        if not fold_train.any() or not fold_predict.any():
            raise ValueError(
                f"OOF Ridge year {prediction_year} has empty train or predict"
            )
        if oof_mask[fold_predict].any():
            raise RuntimeError("OOF Ridge prediction rows overlap")
        fold_scaler = StandardScaler(copy=True)
        x_train = fold_scaler.fit_transform(features[fold_train])
        x_predict = fold_scaler.transform(features[fold_predict])
        fold_model = Ridge(
            alpha=float(ridge_lambda) * int(fold_train.sum()),
            fit_intercept=True,
            solver="cholesky",
        )
        fold_model.fit(x_train, values[fold_train])
        fold_prediction = np.asarray(
            fold_model.predict(x_predict),
            dtype=np.float64,
        )
        if not np.isfinite(fold_prediction).all():
            raise ValueError("OOF Ridge generated nonfinite predictions")
        oof_predictions[fold_predict] = fold_prediction
        oof_mask[fold_predict] = True
        fold_models.append(fold_model)
        fold_scalers.append(fold_scaler)
        fold_metadata.append(
            {
                "prediction_year": int(prediction_year),
                "train_rows": int(fold_train.sum()),
                "train_dates": int(
                    normalized_dates[fold_train].nunique()
                ),
                "train_start_date": normalized_dates[fold_train]
                .min()
                .strftime("%Y-%m-%d"),
                "train_end_date": normalized_dates[fold_train]
                .max()
                .strftime("%Y-%m-%d"),
                "max_train_exit_date": normalized_exits[fold_train]
                .max()
                .strftime("%Y-%m-%d"),
                "prediction_rows": int(fold_predict.sum()),
                "prediction_dates": int(
                    normalized_dates[fold_predict].nunique()
                ),
                "prediction_start_date": normalized_dates[fold_predict]
                .min()
                .strftime("%Y-%m-%d"),
                "prediction_end_date": normalized_dates[fold_predict]
                .max()
                .strftime("%Y-%m-%d"),
            }
        )

    expected_oof = train & np.isin(date_years, prediction_years)
    if not np.array_equal(oof_mask, expected_oof):
        missing_rows = int(np.count_nonzero(expected_oof & ~oof_mask))
        extra_rows = int(np.count_nonzero(oof_mask & ~expected_oof))
        raise RuntimeError(
            "OOF Ridge coverage mismatch: "
            f"missing={missing_rows} extra={extra_rows}"
        )

    development_scaler = StandardScaler(copy=True)
    x_train = development_scaler.fit_transform(features[train])
    x_validation = development_scaler.transform(features[validation])
    development_model = Ridge(
        alpha=float(ridge_lambda) * int(train.sum()),
        fit_intercept=True,
        solver="cholesky",
    )
    development_model.fit(x_train, values[train])
    validation_predictions = np.asarray(
        development_model.predict(x_validation),
        dtype=np.float64,
    )
    if not np.isfinite(validation_predictions).all():
        raise ValueError("development Ridge generated nonfinite predictions")

    return {
        "oof_predictions": oof_predictions,
        "oof_mask": oof_mask,
        "validation_predictions": validation_predictions,
        "fold_metadata": fold_metadata,
        "fold_models": fold_models,
        "fold_scalers": fold_scalers,
        "development_model": development_model,
        "development_scaler": development_scaler,
    }


def residual_rank_metrics(
    dates: pd.Series,
    stock_codes: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    """Compute the shared cross-sectional and regression diagnostics."""
    normalized_dates = pd.DatetimeIndex(
        pd.to_datetime(dates, errors="raise")
    ).normalize()
    normalized_codes = pd.Series(stock_codes, copy=False).astype("string")
    frame = pd.DataFrame(
        {
            "date": normalized_dates,
            "stock_code": normalized_codes.to_numpy(),
            "y_true": np.asarray(y_true, dtype=np.float64),
            "y_pred": np.asarray(y_pred, dtype=np.float64),
            "position": np.arange(len(y_true), dtype=np.int64),
        }
    )
    if not (
        len(frame) == len(stock_codes) == len(y_true) == len(y_pred)
    ):
        raise ValueError("metric inputs must have equal row counts")
    if (
        frame.empty
        or frame["date"].isna().any()
        or frame["stock_code"].isna().any()
        or frame["stock_code"].eq("").any()
        or not np.isfinite(frame[["y_true", "y_pred"]].to_numpy()).all()
    ):
        raise ValueError("metric inputs must be non-empty and finite")

    true_groups = build_true_deciles(frame["date"], frame["y_true"].to_numpy())
    predicted_groups = build_true_deciles(
        frame["date"],
        frame["y_pred"].to_numpy(),
    )
    daily_rank_ic: list[float] = []
    top100_recalls: list[float] = []
    top100_mean_targets: list[float] = []
    for _, group in frame.groupby("date", sort=False):
        true_ranks = group["y_true"].rank(method="average").to_numpy()
        predicted_ranks = group["y_pred"].rank(method="average").to_numpy()
        if np.ptp(true_ranks) > 0 and np.ptp(predicted_ranks) > 0:
            correlation = float(
                np.corrcoef(true_ranks, predicted_ranks)[0, 1]
            )
            if np.isfinite(correlation):
                daily_rank_ic.append(correlation)

        selection_size = min(100, len(group))
        predicted_top = group.sort_values(
            ["y_pred", "stock_code"],
            ascending=[False, True],
            kind="mergesort",
        ).head(selection_size)
        true_threshold = float(
            group["y_true"].nlargest(selection_size).iloc[-1]
        )
        true_top_positions = set(
            group.loc[
                group["y_true"].ge(true_threshold),
                "position",
            ].astype(int)
        )
        predicted_top_positions = set(
            predicted_top["position"].astype(int)
        )
        top100_recalls.append(
            len(true_top_positions & predicted_top_positions)
            / len(true_top_positions)
        )
        top100_mean_targets.append(float(predicted_top["y_true"].mean()))

    if not daily_rank_ic:
        raise ValueError("metrics contain no valid daily Rank IC")
    daily_ic = np.asarray(daily_rank_ic, dtype=np.float64)
    ic_std = float(np.std(daily_ic, ddof=1)) if len(daily_ic) > 1 else 0.0
    ic_mean = float(np.mean(daily_ic))
    errors = frame["y_pred"].to_numpy() - frame["y_true"].to_numpy()
    mse = float(np.mean(np.square(errors)))
    target_centered = frame["y_true"].to_numpy() - float(
        frame["y_true"].mean()
    )
    target_ss = float(np.sum(np.square(target_centered)))
    r_squared = (
        float(1.0 - np.sum(np.square(errors)) / target_ss)
        if target_ss > 0
        else float("nan")
    )
    return {
        "rows": int(len(frame)),
        "dates": int(frame["date"].nunique()),
        "rank_ic_valid_dates": int(len(daily_ic)),
        "mean_daily_rank_ic": ic_mean,
        "daily_rank_ic_std": ic_std,
        "rank_icir": float(ic_mean / ic_std) if ic_std > 0 else float("nan"),
        "rank_icir_annualized": (
            float(ic_mean / ic_std * np.sqrt(252.0))
            if ic_std > 0
            else float("nan")
        ),
        "positive_rank_ic_rate": float(np.mean(daily_ic > 0)),
        "group_mae": float(
            np.mean(
                np.abs(
                    true_groups.astype(np.float64)
                    - predicted_groups.astype(np.float64)
                )
            )
        ),
        "top100_recall": float(np.mean(top100_recalls)),
        "top100_mean_target": float(np.mean(top100_mean_targets)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(errors))),
        "r_squared": r_squared,
        "prediction_std": float(
            np.std(frame["y_pred"].to_numpy(), ddof=1)
        ),
    }


def build_date_slices(dates: pd.Series) -> list[np.ndarray]:
    """Return contiguous complete-date row positions."""
    normalized_dates = pd.DatetimeIndex(
        pd.to_datetime(dates, errors="raise")
    ).normalize()
    if len(normalized_dates) == 0 or normalized_dates.isna().any():
        raise ValueError("date slices require non-empty valid dates")
    date_values = normalized_dates.asi8
    starts = np.r_[0, np.flatnonzero(date_values[1:] != date_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(date_values)]
    block_dates = date_values[starts]
    if len(np.unique(block_dates)) != len(block_dates):
        raise ValueError("each date must occupy exactly one contiguous block")
    return [
        np.arange(start, end, dtype=np.int64)
        for start, end in zip(starts, ends, strict=True)
    ]


def sample_group_weighted_pairs(
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    epoch: int,
    date_value: Any,
    pairs_per_stock: int = 8,
    adjacent_pair_fraction: float = 0.5,
    top_group_weight: float = 2.0,
    second_group_weight: float = 1.5,
) -> dict[str, np.ndarray]:
    """Sample deterministic within-date pairs stratified by true decile."""
    values = np.asarray(targets, dtype=np.float64)
    deciles = np.asarray(groups, dtype=np.int16)
    if (
        len(values) != len(deciles)
        or len(values) < 2
        or not np.isfinite(values).all()
        or np.any((deciles < 1) | (deciles > 10))
    ):
        raise ValueError("pair inputs must contain finite aligned deciles")
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or epoch < 0
        or not isinstance(pairs_per_stock, int)
        or isinstance(pairs_per_stock, bool)
        or pairs_per_stock <= 0
        or not 0 <= adjacent_pair_fraction <= 1
        or top_group_weight <= 0
        or second_group_weight <= 0
    ):
        raise ValueError("pair sampling configuration is invalid")

    timestamp = pd.Timestamp(date_value).normalize()
    if pd.isna(timestamp):
        raise ValueError("pair date must be valid")
    date_number = int(timestamp.value // (24 * 60 * 60 * 1_000_000_000))
    date_bits = int(np.uint64(date_number))
    random_state = np.random.default_rng(
        np.random.SeedSequence(
            [
                seed & 0xFFFFFFFF,
                epoch & 0xFFFFFFFF,
                date_bits & 0xFFFFFFFF,
                (date_bits >> 32) & 0xFFFFFFFF,
            ]
        )
    )
    group_members = {
        group: np.flatnonzero(deciles == group)
        for group in range(1, 11)
    }
    all_indices = np.arange(len(values), dtype=np.int64)
    adjacent_count = int(round(pairs_per_stock * adjacent_pair_fraction))
    far_count = pairs_per_stock - adjacent_count
    source_parts: list[np.ndarray] = []
    opponent_parts: list[np.ndarray] = []

    for group in range(1, 11):
        anchors = group_members[group]
        if len(anchors) == 0:
            continue
        different_group = all_indices[deciles != group]
        if len(different_group) == 0:
            continue
        adjacent_pool = all_indices[np.abs(deciles - group) == 1]
        far_pool = all_indices[np.abs(deciles - group) >= 3]
        if len(adjacent_pool) == 0:
            adjacent_pool = different_group
        if len(far_pool) == 0:
            far_pool = different_group
        if adjacent_count:
            source_parts.append(np.repeat(anchors, adjacent_count))
            opponent_parts.append(
                random_state.choice(
                    adjacent_pool,
                    size=len(anchors) * adjacent_count,
                    replace=True,
                )
            )
        if far_count:
            source_parts.append(np.repeat(anchors, far_count))
            opponent_parts.append(
                random_state.choice(
                    far_pool,
                    size=len(anchors) * far_count,
                    replace=True,
                )
            )

    if not source_parts:
        raise ValueError("date contains no cross-group pair candidates")
    sources = np.concatenate(source_parts).astype(np.int64, copy=False)
    opponents = np.concatenate(opponent_parts).astype(np.int64, copy=False)
    lower = np.minimum(sources, opponents)
    upper = np.maximum(sources, opponents)
    valid = (lower != upper) & (values[lower] != values[upper])
    canonical = np.column_stack([lower[valid], upper[valid]])
    if len(canonical) == 0:
        raise ValueError("date contains no unequal-return pairs")
    canonical = np.unique(canonical, axis=0)

    covered = np.unique(canonical)
    missing_anchors = np.setdiff1d(all_indices, covered, assume_unique=True)
    additions: list[tuple[int, int]] = []
    for anchor in missing_anchors:
        candidates = all_indices[
            (deciles != deciles[anchor]) & (values != values[anchor])
        ]
        if len(candidates):
            opponent = int(random_state.choice(candidates))
            additions.append((min(int(anchor), opponent), max(int(anchor), opponent)))
    if additions:
        canonical = np.unique(
            np.vstack(
                [
                    canonical,
                    np.asarray(additions, dtype=np.int64),
                ]
            ),
            axis=0,
        )

    sources = canonical[:, 0]
    opponents = canonical[:, 1]
    target_difference = values[sources] - values[opponents]
    signs = np.sign(target_difference).astype(np.float32)
    group_distance = np.abs(
        deciles[sources] - deciles[opponents]
    ).astype(np.float32)
    if np.any(signs == 0) or np.any(group_distance == 0):
        raise RuntimeError("invalid equal pair survived filtering")
    stock_importance = np.ones(len(values), dtype=np.float32)
    stock_importance[deciles == 2] = float(second_group_weight)
    stock_importance[deciles == 1] = float(top_group_weight)
    pair_importance = (
        stock_importance[sources] + stock_importance[opponents]
    ) / 2.0
    weights = group_distance / 9.0 * pair_importance
    weights /= float(weights.mean())
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError("pair weights must be finite and positive")
    return {
        "source_index": sources,
        "opponent_index": opponents,
        "sign": signs,
        "group_distance": group_distance,
        "weight": weights.astype(np.float32, copy=False),
    }


def group_weighted_pairwise_loss(
    predictions: Any,
    pairs: dict[str, np.ndarray],
    *,
    temperature: float,
) -> Any:
    """Return normalized weighted logistic loss for one complete date."""
    import torch

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("pair temperature must be finite and positive")
    if not pairs["source_index"].size:
        raise ValueError("pair loss requires at least one pair")
    device = predictions.device
    source = torch.as_tensor(
        pairs["source_index"],
        dtype=torch.long,
        device=device,
    )
    opponent = torch.as_tensor(
        pairs["opponent_index"],
        dtype=torch.long,
        device=device,
    )
    signs = torch.as_tensor(
        pairs["sign"],
        dtype=predictions.dtype,
        device=device,
    )
    weights = torch.as_tensor(
        pairs["weight"],
        dtype=predictions.dtype,
        device=device,
    )
    prediction_difference = (
        predictions.reshape(-1)[source]
        - predictions.reshape(-1)[opponent]
    ) / float(temperature)
    losses = torch.nn.functional.softplus(-signs * prediction_difference)
    return torch.mean(weights * losses) / np.log(2.0)


def choose_best_candidate(
    candidates: list[dict[str, float]],
    *,
    tolerance: float,
) -> dict[str, float]:
    """Select by Rank IC, then RMSE, then stronger regularization."""
    if not candidates:
        raise ValueError("at least one Ridge candidate is required")
    best = candidates[0]
    for candidate in candidates[1:]:
        rank_difference = (
            candidate["mean_daily_rank_ic"]
            - best["mean_daily_rank_ic"]
        )
        if rank_difference > tolerance:
            best = candidate
            continue
        if abs(rank_difference) > tolerance:
            continue
        if candidate["validation_rmse"] < best["validation_rmse"]:
            best = candidate
            continue
        if (
            candidate["validation_rmse"] == best["validation_rmse"]
            and candidate["lambda"] > best["lambda"]
        ):
            best = candidate
    return best


def choose_best_xgboost_candidate(
    candidates: list[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Select near-best Rank IC by RMSE, then prefer the simpler tree."""
    if not candidates:
        raise ValueError("at least one XGBoost candidate is required")
    best_rank_ic = max(
        float(candidate["mean_daily_rank_ic"])
        for candidate in candidates
    )
    eligible = [
        candidate
        for candidate in candidates
        if best_rank_ic - float(candidate["mean_daily_rank_ic"]) <= tolerance
    ]
    return min(
        eligible,
        key=lambda candidate: (
            float(candidate["validation_rmse"]),
            int(candidate["parameters"]["max_depth"]),
            -float(candidate["parameters"]["min_child_weight"]),
            int(candidate["boosting_rounds"]),
            str(candidate["candidate_name"]),
        ),
    )


def choose_best_lightgbm_candidate(
    candidates: list[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Select near-best Rank IC by RMSE, then prefer fewer leaves."""
    if not candidates:
        raise ValueError("at least one LightGBM candidate is required")
    best_rank_ic = max(
        float(candidate["mean_daily_rank_ic"])
        for candidate in candidates
    )
    eligible = [
        candidate
        for candidate in candidates
        if best_rank_ic - float(candidate["mean_daily_rank_ic"]) <= tolerance
    ]
    return min(
        eligible,
        key=lambda candidate: (
            float(candidate["validation_rmse"]),
            int(candidate["parameters"]["num_leaves"]),
            int(candidate["parameters"]["max_depth"]),
            -int(candidate["parameters"]["min_data_in_leaf"]),
            int(candidate["boosting_rounds"]),
            str(candidate["candidate_name"]),
        ),
    )


def choose_best_mlp_candidate(
    candidates: list[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Select near-best Rank IC by RMSE, then prefer fewer parameters."""
    if not candidates:
        raise ValueError("at least one MLP candidate is required")
    best_rank_ic = max(
        float(candidate["mean_daily_rank_ic"])
        for candidate in candidates
    )
    eligible = [
        candidate
        for candidate in candidates
        if best_rank_ic - float(candidate["mean_daily_rank_ic"]) <= tolerance
    ]
    return min(
        eligible,
        key=lambda candidate: (
            float(candidate["validation_rmse"]),
            int(candidate["parameter_count"]),
            -float(candidate["parameters"]["dropout"]),
            -float(candidate["parameters"]["weight_decay"]),
            int(candidate["best_epoch"]),
            str(candidate["candidate_name"]),
        ),
    )


def _near_best_metric_key(
    candidate: dict[str, Any],
    *,
    regularization_key: str,
) -> tuple[float, float, float, int, str]:
    return (
        float(candidate["metrics"]["group_mae"]),
        float(candidate["metrics"]["rmse"]),
        float(candidate[regularization_key]),
        int(candidate["best_epoch"]),
        str(candidate.get("candidate_name", "")),
    )


def choose_residual_candidate(
    candidates: list[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Choose Stage C by near-best Rank IC and conservative tie breaks."""
    if not candidates or tolerance < 0:
        raise ValueError("residual candidates and tolerance are required")
    best_rank_ic = max(
        float(candidate["metrics"]["mean_daily_rank_ic"])
        for candidate in candidates
    )
    eligible = [
        candidate
        for candidate in candidates
        if best_rank_ic
        - float(candidate["metrics"]["mean_daily_rank_ic"])
        <= tolerance
    ]
    return min(
        eligible,
        key=lambda candidate: _near_best_metric_key(
            candidate,
            regularization_key="gamma",
        ),
    )


def choose_rank_candidate(
    candidates: list[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Choose Stage D by near-best Rank IC, then smaller rank lambda."""
    if not candidates or tolerance < 0:
        raise ValueError("rank candidates and tolerance are required")
    best_rank_ic = max(
        float(candidate["metrics"]["mean_daily_rank_ic"])
        for candidate in candidates
    )
    eligible = [
        candidate
        for candidate in candidates
        if best_rank_ic
        - float(candidate["metrics"]["mean_daily_rank_ic"])
        <= tolerance
    ]
    return min(
        eligible,
        key=lambda candidate: _near_best_metric_key(
            candidate,
            regularization_key="lambda_rank",
        ),
    )


def build_mlp_model(
    input_dim: int,
    hidden_layers: list[int],
    dropout: float,
    *,
    output_bias: float,
) -> Any:
    """Build the shared ReLU MLP without importing PyTorch at module import."""
    import torch

    layers: list[torch.nn.Module] = []
    previous_width = input_dim
    for width in hidden_layers:
        layers.append(torch.nn.Linear(previous_width, width))
        layers.append(torch.nn.ReLU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        previous_width = width
    output_layer = torch.nn.Linear(previous_width, 1)
    torch.nn.init.zeros_(output_layer.weight)
    torch.nn.init.constant_(output_layer.bias, output_bias)
    layers.append(output_layer)
    return torch.nn.Sequential(*layers)


def _resolve_path(config_path: Path, raw_path: str) -> Path:
    return (config_path.resolve().parent / raw_path).resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _published_lineage(
    schema_path: Path,
    summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    schema = _read_json(schema_path)
    summary = _read_json(summary_path)
    dataset_hash = schema.get("parquet_sha256")
    sample_hash = summary.get("sample_index_sha256")
    for label, value in (
        ("model dataset", dataset_hash),
        ("sample index", sample_hash),
    ):
        if (
            not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", value)
        ):
            raise ValueError(f"published {label} SHA256 is invalid")
    if summary.get("source_model_dataset_sha256") != dataset_hash:
        raise ValueError("sample index references a different model dataset")
    if summary.get("uses_entry_tradeable") is not False:
        raise ValueError("sample index must not use entry_tradeable")
    return schema, summary, dataset_hash.lower(), sample_hash.lower()


def _read_development_parquet(
    path: Path,
    columns: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("date", ">=", start_date.to_pydatetime()),
            ("date", "<=", end_date.to_pydatetime()),
        ],
    )
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing requested columns: {missing}")
    return frame


def _normalize_keys(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(
        result["date"],
        errors="raise",
    ).dt.normalize()
    result["stock_code"] = result["stock_code"].astype("string").str.strip()
    if (
        result["date"].isna().any()
        or result["stock_code"].isna().any()
        or result["stock_code"].eq("").any()
    ):
        raise ValueError(f"{path} contains missing model keys")
    duplicate_mask = result.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        sample = result.loc[duplicate_mask, KEY_COLUMNS]
        raise ValueError(
            f"{path} contains duplicate model keys: "
            f"{sample.head(5).to_dict('records')}"
        )
    return result.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _load_development_data(
    config: dict[str, Any],
    config_path: Path,
    horizons: list[int],
) -> tuple[
    pd.DataFrame,
    list[str],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, list[str]],
]:
    paths = {
        name: _resolve_path(config_path, config["paths"][name])
        for name in (
            "model_dataset",
            "model_schema",
            "sample_index",
            "split_summary",
        )
    }
    schema, summary, dataset_hash, sample_hash = _published_lineage(
        paths["model_schema"],
        paths["split_summary"],
    )
    factor_columns = schema.get("feature_columns")
    if (
        not isinstance(factor_columns, list)
        or not factor_columns
        or not all(isinstance(column, str) and column for column in factor_columns)
        or len(factor_columns) != len(set(factor_columns))
    ):
        raise ValueError("model schema feature_columns are invalid")
    forbidden = {
        "stock_code",
        "industry",
        "market_cap",
        *(f"target_{horizon}d" for horizon in SUPPORTED_HORIZONS),
    }
    conflicts = sorted(
        column
        for column in factor_columns
        if column in forbidden or column.startswith("entry_")
    )
    if conflicts:
        raise ValueError(f"forbidden model features: {conflicts}")

    train_start = pd.Timestamp(config["split"]["train"]["start"]).normalize()
    validation_end = pd.Timestamp(
        config["split"]["validation"]["end"]
    ).normalize()
    target_columns = [f"target_{horizon}d" for horizon in horizons]
    split_columns = [f"split_{horizon}d" for horizon in horizons]
    dataset_columns = [*KEY_COLUMNS, *factor_columns, *target_columns]
    sample_columns = [*KEY_COLUMNS, *split_columns]
    data = _read_development_parquet(
        paths["model_dataset"],
        dataset_columns,
        train_start,
        validation_end,
    )
    sample = _read_development_parquet(
        paths["sample_index"],
        sample_columns,
        train_start,
        validation_end,
    )
    data = _normalize_keys(data, paths["model_dataset"])
    sample = _normalize_keys(sample, paths["sample_index"])
    if len(data) != len(sample) or not data[KEY_COLUMNS].equals(
        sample[KEY_COLUMNS]
    ):
        raise ValueError("development dataset and sample-index keys differ")
    if data.empty:
        raise ValueError("development data is empty")
    if data["date"].max() > validation_end:
        raise RuntimeError("locked test dates entered development data")
    for column in split_columns:
        data[column] = sample[column].astype("string")

    return (
        data,
        list(factor_columns),
        schema,
        summary,
        {
            "model_dataset_sha256": dataset_hash,
            "sample_index_sha256": sample_hash,
        },
        {
            "model_dataset": dataset_columns,
            "sample_index": sample_columns,
        },
    )


def _load_residual_rank_data(
    config: dict[str, Any],
    config_path: Path,
    horizons: list[int],
) -> tuple[
    pd.DataFrame,
    list[str],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, list[str]],
]:
    """Load development data plus horizon exit dates for OOF purging."""
    (
        data,
        factor_columns,
        schema,
        summary,
        source_hashes,
        read_columns,
    ) = _load_development_data(config, config_path, horizons)
    sample_path = _resolve_path(
        config_path,
        config["paths"]["sample_index"],
    )
    train_start = pd.Timestamp(config["split"]["train"]["start"]).normalize()
    validation_end = pd.Timestamp(
        config["split"]["validation"]["end"]
    ).normalize()
    if validation_end > pd.Timestamp("2023-12-31"):
        raise RuntimeError(
            "residual_rank_mlp development may not read beyond 2023"
        )
    exit_columns = [f"exit_date_{horizon}d" for horizon in horizons]
    sample_columns = [*KEY_COLUMNS, *exit_columns]
    exits = _read_development_parquet(
        sample_path,
        sample_columns,
        train_start,
        validation_end,
    )
    exits = _normalize_keys(exits, sample_path)
    if len(data) != len(exits) or not data[KEY_COLUMNS].equals(
        exits[KEY_COLUMNS]
    ):
        raise ValueError("development data and exit-date keys differ")

    selected_mask = np.zeros(len(data), dtype=bool)
    for horizon in horizons:
        split_column = f"split_{horizon}d"
        target_column = f"target_{horizon}d"
        exit_column = f"exit_date_{horizon}d"
        data[exit_column] = pd.to_datetime(
            exits[exit_column],
            errors="coerce",
        ).dt.normalize()
        horizon_selected = data[split_column].isin(
            ["train", "validation"]
        ).to_numpy()
        selected_mask |= horizon_selected
        if (
            data.loc[horizon_selected, exit_column].isna().any()
            or not np.isfinite(
                pd.to_numeric(
                    data.loc[horizon_selected, target_column],
                    errors="coerce",
                ).to_numpy(dtype=np.float64)
            ).all()
        ):
            raise ValueError(
                f"{horizon}d selected targets or exit dates are invalid"
            )
    if not selected_mask.any():
        raise ValueError("residual rank development has no selected rows")
    if data.loc[selected_mask, "date"].max() > pd.Timestamp("2023-12-31"):
        raise RuntimeError("locked test dates entered residual development")
    read_columns["sample_index"] = [
        *read_columns["sample_index"],
        *exit_columns,
    ]
    return (
        data,
        factor_columns,
        schema,
        summary,
        source_hashes,
        read_columns,
    )


def _sample_stats(dates: pd.Series, mask: np.ndarray) -> dict[str, Any]:
    selected = dates.loc[mask]
    return {
        "rows": int(mask.sum()),
        "dates": int(selected.nunique()),
        "start_date": selected.min().strftime("%Y-%m-%d"),
        "end_date": selected.max().strftime("%Y-%m-%d"),
    }


def _git_state(config_path: Path) -> dict[str, Any]:
    repository = config_path.resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _environment_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "scipy": importlib.metadata.version("scipy"),
        "joblib": importlib.metadata.version("joblib"),
        "pyarrow": importlib.metadata.version("pyarrow"),
    }
    try:
        versions["xgboost"] = importlib.metadata.version("xgboost")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        versions["lightgbm"] = importlib.metadata.version("lightgbm")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        versions["torch"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        pass
    return versions


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def train_ridge_run(
    config: dict[str, Any],
    config_path: Path,
    *,
    horizons: list[int],
    run_id: str,
) -> Path:
    """Tune Ridge on 2019-2022, validate on 2023, and publish one run."""
    validate_run_id(run_id)
    selected_horizons = list(horizons)
    if (
        not selected_horizons
        or len(selected_horizons) != len(set(selected_horizons))
        or any(
            horizon not in SUPPORTED_HORIZONS
            for horizon in selected_horizons
        )
    ):
        raise ValueError("horizons must be unique values from 1, 5, 10")

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    runs_directory = _resolve_path(
        config_path,
        config["paths"]["runs_dir"],
    )
    runs_directory.mkdir(parents=True, exist_ok=True)
    final_run_path = runs_directory / run_id
    if final_run_path.exists():
        raise FileExistsError(f"run already exists: {final_run_path}")

    (
        development,
        factor_columns,
        _schema,
        _summary,
        lineage,
        read_columns,
    ) = _load_development_data(
        config,
        config_path,
        selected_horizons,
    )
    preprocessing_config = config["preprocessing"]
    feature_matrix, derived_columns = transform_daily_cross_sections(
        development,
        factor_columns,
        mad_width=float(preprocessing_config["mad_width"]),
        expected_cross_section_size=int(
            preprocessing_config["expected_cross_section_size"]
        ),
        date_chunk_size=int(preprocessing_config["date_chunk_size"]),
    )
    if feature_matrix.dtype != np.float32:
        raise RuntimeError("cross-sectional features must use float32")

    temporary_path = Path(
        tempfile.mkdtemp(
            dir=runs_directory,
            prefix=f".{run_id}.",
            suffix=".tmp",
        )
    )
    try:
        shutil.copy2(config_path, temporary_path / "config_snapshot.toml")
        ridge_config = config["models"]["ridge"]
        lambda_grid = [float(value) for value in ridge_config["lambda_grid"]]
        solver = str(ridge_config["solver"])
        tolerance = float(ridge_config["rank_ic_tolerance"])
        predictions: list[pd.DataFrame] = []
        horizon_results: dict[str, Any] = {}

        for horizon in selected_horizons:
            suffix = f"{horizon}d"
            target_column = f"target_{suffix}"
            split_column = f"split_{suffix}"
            train_mask = (
                development[split_column]
                .eq("train")
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            validation_mask = (
                development[split_column]
                .eq("validation")
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            if not train_mask.any() or not validation_mask.any():
                raise ValueError(f"{suffix} has no train or validation rows")

            targets = pd.to_numeric(
                development[target_column],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            if not np.isfinite(targets[train_mask]).all():
                raise ValueError(f"{suffix} training targets are not finite")
            if not np.isfinite(targets[validation_mask]).all():
                raise ValueError(f"{suffix} validation targets are not finite")

            x_train = feature_matrix[train_mask]
            x_validation = feature_matrix[validation_mask]
            y_train = targets[train_mask]
            y_validation = targets[validation_mask]
            validation_dates = development.loc[validation_mask, "date"]

            scaler = StandardScaler(copy=True)
            x_train_scaled = scaler.fit_transform(x_train)
            x_validation_scaled = scaler.transform(x_validation)
            candidate_results: list[dict[str, float | int]] = []
            for candidate_lambda in lambda_grid:
                alpha = candidate_lambda * len(y_train)
                candidate_model = Ridge(
                    alpha=alpha,
                    fit_intercept=True,
                    solver=solver,
                    copy_X=True,
                )
                candidate_model.fit(x_train_scaled, y_train)
                candidate_predictions = candidate_model.predict(
                    x_validation_scaled
                )
                rank_ic, valid_date_count = mean_daily_rank_ic(
                    validation_dates,
                    y_validation,
                    candidate_predictions,
                )
                if not np.isfinite(rank_ic):
                    raise ValueError(
                        f"{suffix} lambda={candidate_lambda} has no valid Rank IC"
                    )
                rmse = float(
                    np.sqrt(
                        np.mean(
                            np.square(candidate_predictions - y_validation)
                        )
                    )
                )
                candidate_results.append(
                    {
                        "lambda": candidate_lambda,
                        "alpha": alpha,
                        "mean_daily_rank_ic": rank_ic,
                        "rank_ic_valid_dates": valid_date_count,
                        "validation_rmse": rmse,
                    }
                )

            best = choose_best_candidate(
                candidate_results,
                tolerance=tolerance,
            )
            selected_model = Ridge(
                alpha=float(best["alpha"]),
                fit_intercept=True,
                solver=solver,
                copy_X=True,
            )
            selected_model.fit(x_train_scaled, y_train)
            selected_predictions = selected_model.predict(
                x_validation_scaled
            )
            joblib.dump(
                scaler,
                temporary_path / f"preprocessor_{suffix}.joblib",
                compress=3,
            )
            joblib.dump(
                selected_model,
                temporary_path / f"model_{suffix}.joblib",
                compress=3,
            )

            horizon_prediction = development.loc[
                validation_mask,
                KEY_COLUMNS,
            ].copy()
            horizon_prediction["split"] = "validation"
            horizon_prediction["horizon"] = horizon
            horizon_prediction["y_true"] = y_validation
            horizon_prediction["y_pred"] = selected_predictions
            predictions.append(horizon_prediction)
            horizon_results[suffix] = {
                "target_column": target_column,
                "split_column": split_column,
                "train": _sample_stats(
                    development["date"],
                    train_mask,
                ),
                "validation": _sample_stats(
                    development["date"],
                    validation_mask,
                ),
                "lambda_candidates": candidate_results,
                "selected_lambda": float(best["lambda"]),
                "selected_alpha": float(best["alpha"]),
                "selected_mean_daily_rank_ic": float(
                    best["mean_daily_rank_ic"]
                ),
                "selected_validation_rmse": float(
                    best["validation_rmse"]
                ),
                "rank_ic_valid_dates": int(
                    best["rank_ic_valid_dates"]
                ),
            }
            print(
                f"{suffix}: selected lambda={best['lambda']:.6g} "
                f"rank_ic={best['mean_daily_rank_ic']:.6f} "
                f"rmse={best['validation_rmse']:.6f}",
                flush=True,
            )
            del (
                x_train,
                x_validation,
                x_train_scaled,
                x_validation_scaled,
                y_train,
                y_validation,
                selected_predictions,
            )
            gc.collect()

        prediction_table = pd.concat(predictions, ignore_index=True)
        prediction_table = prediction_table[
            [
                "date",
                "stock_code",
                "split",
                "horizon",
                "y_true",
                "y_pred",
            ]
        ].sort_values(
            ["date", "stock_code", "horizon"],
            ignore_index=True,
        )
        prediction_table.to_parquet(
            temporary_path / "predictions.parquet",
            index=False,
            compression="zstd",
        )

        manifest = {
            "schema_version": 1,
            "status": "completed",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": "ridge",
            "horizons": selected_horizons,
            "development_data": {
                "configured_start_date": config["split"]["train"]["start"],
                "configured_validation_end": config["split"]["validation"][
                    "end"
                ],
                "min_date": development["date"].min().strftime("%Y-%m-%d"),
                "max_date": development["date"].max().strftime("%Y-%m-%d"),
                "rows": len(development),
                "dates": int(development["date"].nunique()),
                "expected_cross_section_size": int(
                    preprocessing_config["expected_cross_section_size"]
                ),
            },
            "features": {
                "raw": factor_columns,
                "derived": derived_columns,
                "raw_count": len(factor_columns),
                "derived_count": len(derived_columns),
            },
            "preprocessing": {
                "daily_cross_section_first": True,
                "mad_width": float(preprocessing_config["mad_width"]),
                "mad_normalization": 1.4826,
                "missing_fill_value": 0.0,
                "dtype": preprocessing_config["dtype"],
                "standard_scaler_fit_scope": "horizon_train_only",
            },
            "selection": {
                "primary_metric": "mean_daily_spearman_rank_ic",
                "first_tie_breaker": "lower_validation_rmse",
                "second_tie_breaker": "larger_lambda",
                "rank_ic_tolerance": tolerance,
                "lambda_grid": lambda_grid,
                "sklearn_alpha_formula": "lambda * n_train",
                "solver": solver,
            },
            "horizon_results": horizon_results,
            "read_columns": read_columns,
            "uses_entry_tradeable": False,
            "prediction_rows": len(prediction_table),
            "sources": {
                "config_sha256": _file_sha256(config_path),
                **lineage,
                "lineage_validation": "published_schema_and_summary",
            },
            "git": _git_state(config_path),
            "environment": _environment_versions(),
        }
        _write_json(temporary_path / "manifest.json", manifest)
        os.rename(temporary_path, final_run_path)
    except BaseException:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise

    print(f"run published: {final_run_path}", flush=True)
    return final_run_path


def train_xgboost_run(
    config: dict[str, Any],
    config_path: Path,
    *,
    horizons: list[int],
    run_id: str,
) -> Path:
    """Tune XGBoost on 2019-2022 and validate on 2023."""
    import xgboost as xgb

    validate_run_id(run_id)
    selected_horizons = list(horizons)
    if (
        not selected_horizons
        or len(selected_horizons) != len(set(selected_horizons))
        or any(
            horizon not in SUPPORTED_HORIZONS
            for horizon in selected_horizons
        )
    ):
        raise ValueError("horizons must be unique values from 1, 5, 10")

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    runs_directory = _resolve_path(
        config_path,
        config["paths"]["runs_dir"],
    )
    runs_directory.mkdir(parents=True, exist_ok=True)
    final_run_path = runs_directory / run_id
    if final_run_path.exists():
        raise FileExistsError(f"run already exists: {final_run_path}")

    (
        development,
        factor_columns,
        _schema,
        _summary,
        lineage,
        read_columns,
    ) = _load_development_data(
        config,
        config_path,
        selected_horizons,
    )
    preprocessing_config = config["preprocessing"]
    feature_matrix, derived_columns = transform_daily_cross_sections(
        development,
        factor_columns,
        mad_width=float(preprocessing_config["mad_width"]),
        expected_cross_section_size=int(
            preprocessing_config["expected_cross_section_size"]
        ),
        date_chunk_size=int(preprocessing_config["date_chunk_size"]),
    )
    if feature_matrix.dtype != np.float32:
        raise RuntimeError("cross-sectional features must use float32")

    temporary_path = Path(
        tempfile.mkdtemp(
            dir=runs_directory,
            prefix=f".{run_id}.",
            suffix=".tmp",
        )
    )
    try:
        shutil.copy2(config_path, temporary_path / "config_snapshot.toml")
        xgboost_config = config["models"]["xgboost"]
        candidates = list(xgboost_config["candidates"])
        tolerance = float(xgboost_config["rank_ic_tolerance"])
        max_rounds = int(xgboost_config["max_boost_rounds"])
        early_stopping_rounds = int(
            xgboost_config["early_stopping_rounds"]
        )
        fixed_parameters = {
            "objective": str(xgboost_config["objective"]),
            "eval_metric": str(xgboost_config["eval_metric"]),
            "tree_method": str(xgboost_config["tree_method"]),
            "eta": float(xgboost_config["learning_rate"]),
            "subsample": float(xgboost_config["subsample"]),
            "colsample_bytree": float(
                xgboost_config["colsample_bytree"]
            ),
            "lambda": float(xgboost_config["reg_lambda"]),
            "alpha": float(xgboost_config["reg_alpha"]),
            "gamma": float(xgboost_config["gamma"]),
            "max_bin": int(xgboost_config["max_bin"]),
            "seed": int(xgboost_config["seed"]),
            "nthread": int(xgboost_config["nthread"]),
            "verbosity": 0,
        }
        predictions: list[pd.DataFrame] = []
        horizon_results: dict[str, Any] = {}

        for horizon in selected_horizons:
            suffix = f"{horizon}d"
            target_column = f"target_{suffix}"
            split_column = f"split_{suffix}"
            train_mask = (
                development[split_column]
                .eq("train")
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            validation_mask = (
                development[split_column]
                .eq("validation")
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            if not train_mask.any() or not validation_mask.any():
                raise ValueError(f"{suffix} has no train or validation rows")

            targets = pd.to_numeric(
                development[target_column],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            if not np.isfinite(targets[train_mask]).all():
                raise ValueError(f"{suffix} training targets are not finite")
            if not np.isfinite(targets[validation_mask]).all():
                raise ValueError(f"{suffix} validation targets are not finite")

            x_train = feature_matrix[train_mask]
            x_validation = feature_matrix[validation_mask]
            y_train = targets[train_mask]
            y_validation = targets[validation_mask]
            validation_dates = development.loc[validation_mask, "date"]
            dtrain = xgb.DMatrix(
                x_train,
                label=y_train,
                feature_names=derived_columns,
                nthread=fixed_parameters["nthread"],
            )
            dvalidation = xgb.DMatrix(
                x_validation,
                label=y_validation,
                feature_names=derived_columns,
                nthread=fixed_parameters["nthread"],
            )

            candidate_results: list[dict[str, Any]] = []
            trained_models: dict[str, xgb.Booster] = {}
            for index, candidate in enumerate(candidates, start=1):
                candidate_name = str(candidate["name"])
                candidate_parameters = {
                    "max_depth": int(candidate["max_depth"]),
                    "min_child_weight": float(
                        candidate["min_child_weight"]
                    ),
                }
                parameters = {
                    **fixed_parameters,
                    **candidate_parameters,
                }
                booster = xgb.train(
                    parameters,
                    dtrain,
                    num_boost_round=max_rounds,
                    evals=[(dvalidation, "validation")],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False,
                )
                best_iteration = int(booster.best_iteration)
                selected_booster = booster[: best_iteration + 1]
                candidate_predictions = selected_booster.predict(dvalidation)
                rank_ic, valid_date_count = mean_daily_rank_ic(
                    validation_dates,
                    y_validation,
                    candidate_predictions,
                )
                if not np.isfinite(rank_ic):
                    raise ValueError(
                        f"{suffix} candidate={candidate_name} "
                        "has no valid Rank IC"
                    )
                rmse = float(
                    np.sqrt(
                        np.mean(
                            np.square(candidate_predictions - y_validation)
                        )
                    )
                )
                candidate_results.append(
                    {
                        "candidate_name": candidate_name,
                        "parameters": candidate_parameters,
                        "best_iteration": best_iteration,
                        "boosting_rounds": best_iteration + 1,
                        "mean_daily_rank_ic": rank_ic,
                        "rank_ic_valid_dates": valid_date_count,
                        "validation_rmse": rmse,
                    }
                )
                trained_models[candidate_name] = selected_booster
                print(
                    f"{suffix} candidate {index}/{len(candidates)} "
                    f"{candidate_name}: rounds={best_iteration + 1} "
                    f"rank_ic={rank_ic:.6f} rmse={rmse:.6f}",
                    flush=True,
                )
                del booster, candidate_predictions
                gc.collect()

            best = choose_best_xgboost_candidate(
                candidate_results,
                tolerance=tolerance,
            )
            selected_name = str(best["candidate_name"])
            selected_model = trained_models[selected_name]
            selected_predictions = selected_model.predict(dvalidation)
            model_filename = f"model_{suffix}.json"
            selected_model.save_model(temporary_path / model_filename)

            horizon_prediction = development.loc[
                validation_mask,
                KEY_COLUMNS,
            ].copy()
            horizon_prediction["split"] = "validation"
            horizon_prediction["horizon"] = horizon
            horizon_prediction["y_true"] = y_validation
            horizon_prediction["y_pred"] = selected_predictions
            predictions.append(horizon_prediction)
            horizon_results[suffix] = {
                "target_column": target_column,
                "split_column": split_column,
                "train": _sample_stats(
                    development["date"],
                    train_mask,
                ),
                "validation": _sample_stats(
                    development["date"],
                    validation_mask,
                ),
                "candidate_results": candidate_results,
                "selected_candidate": selected_name,
                "selected_parameters": dict(best["parameters"]),
                "selected_best_iteration": int(best["best_iteration"]),
                "selected_boosting_rounds": int(best["boosting_rounds"]),
                "selected_mean_daily_rank_ic": float(
                    best["mean_daily_rank_ic"]
                ),
                "selected_validation_rmse": float(
                    best["validation_rmse"]
                ),
                "rank_ic_valid_dates": int(
                    best["rank_ic_valid_dates"]
                ),
                "model_artifact": model_filename,
            }
            print(
                f"{suffix}: selected {selected_name} "
                f"rounds={best['boosting_rounds']} "
                f"rank_ic={best['mean_daily_rank_ic']:.6f} "
                f"rmse={best['validation_rmse']:.6f}",
                flush=True,
            )
            del (
                x_train,
                x_validation,
                y_train,
                y_validation,
                dtrain,
                dvalidation,
                selected_predictions,
                selected_model,
                trained_models,
            )
            gc.collect()

        prediction_table = pd.concat(predictions, ignore_index=True)
        prediction_table = prediction_table[
            [
                "date",
                "stock_code",
                "split",
                "horizon",
                "y_true",
                "y_pred",
            ]
        ].sort_values(
            ["date", "stock_code", "horizon"],
            ignore_index=True,
        )
        prediction_table.to_parquet(
            temporary_path / "predictions.parquet",
            index=False,
            compression="zstd",
        )

        manifest = {
            "schema_version": 1,
            "status": "completed",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": "xgboost",
            "horizons": selected_horizons,
            "development_data": {
                "configured_start_date": config["split"]["train"]["start"],
                "configured_validation_end": config["split"]["validation"][
                    "end"
                ],
                "min_date": development["date"].min().strftime("%Y-%m-%d"),
                "max_date": development["date"].max().strftime("%Y-%m-%d"),
                "rows": len(development),
                "dates": int(development["date"].nunique()),
                "expected_cross_section_size": int(
                    preprocessing_config["expected_cross_section_size"]
                ),
            },
            "features": {
                "raw": factor_columns,
                "derived": derived_columns,
                "raw_count": len(factor_columns),
                "derived_count": len(derived_columns),
            },
            "preprocessing": {
                "daily_cross_section_first": True,
                "mad_width": float(preprocessing_config["mad_width"]),
                "mad_normalization": 1.4826,
                "missing_fill_value": 0.0,
                "dtype": preprocessing_config["dtype"],
                "second_stage_scaler": False,
            },
            "selection": {
                "primary_metric": "mean_daily_spearman_rank_ic",
                "near_best_rank_ic_tolerance": tolerance,
                "first_tie_breaker": "lower_validation_rmse",
                "second_tie_breaker": "simpler_tree",
                "early_stopping_metric": xgboost_config["eval_metric"],
                "max_boost_rounds": max_rounds,
                "early_stopping_rounds": early_stopping_rounds,
                "fixed_parameters": fixed_parameters,
                "candidate_count": len(candidates),
            },
            "horizon_results": horizon_results,
            "read_columns": read_columns,
            "uses_entry_tradeable": False,
            "prediction_rows": len(prediction_table),
            "sources": {
                "config_sha256": _file_sha256(config_path),
                **lineage,
                "lineage_validation": "published_schema_and_summary",
            },
            "git": _git_state(config_path),
            "environment": _environment_versions(),
        }
        _write_json(temporary_path / "manifest.json", manifest)
        os.rename(temporary_path, final_run_path)
    except BaseException:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise

    print(f"run published: {final_run_path}", flush=True)
    return final_run_path


def train_lightgbm_run(
    config: dict[str, Any],
    config_path: Path,
    *,
    horizons: list[int],
    run_id: str,
) -> Path:
    """Tune LightGBM on 2019-2022 and validate on 2023."""
    import lightgbm as lgb

    validate_run_id(run_id)
    selected_horizons = list(horizons)
    if (
        not selected_horizons
        or len(selected_horizons) != len(set(selected_horizons))
        or any(
            horizon not in SUPPORTED_HORIZONS
            for horizon in selected_horizons
        )
    ):
        raise ValueError("horizons must be unique values from 1, 5, 10")

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    runs_directory = _resolve_path(
        config_path,
        config["paths"]["runs_dir"],
    )
    runs_directory.mkdir(parents=True, exist_ok=True)
    final_run_path = runs_directory / run_id
    if final_run_path.exists():
        raise FileExistsError(f"run already exists: {final_run_path}")

    (
        development,
        factor_columns,
        _schema,
        _summary,
        lineage,
        read_columns,
    ) = _load_development_data(
        config,
        config_path,
        selected_horizons,
    )
    preprocessing_config = config["preprocessing"]
    feature_matrix, derived_columns = transform_daily_cross_sections(
        development,
        factor_columns,
        mad_width=float(preprocessing_config["mad_width"]),
        expected_cross_section_size=int(
            preprocessing_config["expected_cross_section_size"]
        ),
        date_chunk_size=int(preprocessing_config["date_chunk_size"]),
    )
    if feature_matrix.dtype != np.float32:
        raise RuntimeError("cross-sectional features must use float32")

    temporary_path = Path(
        tempfile.mkdtemp(
            dir=runs_directory,
            prefix=f".{run_id}.",
            suffix=".tmp",
        )
    )
    try:
        shutil.copy2(config_path, temporary_path / "config_snapshot.toml")
        lightgbm_config = config["models"]["lightgbm"]
        candidates = list(lightgbm_config["candidates"])
        tolerance = float(lightgbm_config["rank_ic_tolerance"])
        max_rounds = int(lightgbm_config["max_boost_rounds"])
        early_stopping_rounds = int(
            lightgbm_config["early_stopping_rounds"]
        )
        fixed_parameters = {
            "objective": str(lightgbm_config["objective"]),
            "metric": str(lightgbm_config["metric"]),
            "boosting_type": str(lightgbm_config["boosting_type"]),
            "learning_rate": float(lightgbm_config["learning_rate"]),
            "bagging_fraction": float(
                lightgbm_config["bagging_fraction"]
            ),
            "bagging_freq": int(lightgbm_config["bagging_freq"]),
            "feature_fraction": float(
                lightgbm_config["feature_fraction"]
            ),
            "lambda_l2": float(lightgbm_config["lambda_l2"]),
            "lambda_l1": float(lightgbm_config["lambda_l1"]),
            "min_gain_to_split": float(
                lightgbm_config["min_gain_to_split"]
            ),
            "max_bin": int(lightgbm_config["max_bin"]),
            "seed": int(lightgbm_config["seed"]),
            "data_random_seed": int(lightgbm_config["seed"]),
            "feature_fraction_seed": int(lightgbm_config["seed"]),
            "bagging_seed": int(lightgbm_config["seed"]),
            "num_threads": int(lightgbm_config["num_threads"]),
            "deterministic": bool(lightgbm_config["deterministic"]),
            "force_col_wise": bool(lightgbm_config["force_col_wise"]),
            "feature_pre_filter": False,
            "verbosity": -1,
        }
        predictions: list[pd.DataFrame] = []
        horizon_results: dict[str, Any] = {}

        for horizon in selected_horizons:
            suffix = f"{horizon}d"
            target_column = f"target_{suffix}"
            split_column = f"split_{suffix}"
            train_mask = (
                development[split_column]
                .eq("train")
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            validation_mask = (
                development[split_column]
                .eq("validation")
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            if not train_mask.any() or not validation_mask.any():
                raise ValueError(f"{suffix} has no train or validation rows")

            targets = pd.to_numeric(
                development[target_column],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            if not np.isfinite(targets[train_mask]).all():
                raise ValueError(f"{suffix} training targets are not finite")
            if not np.isfinite(targets[validation_mask]).all():
                raise ValueError(f"{suffix} validation targets are not finite")

            x_train = feature_matrix[train_mask]
            x_validation = feature_matrix[validation_mask]
            y_train = targets[train_mask]
            y_validation = targets[validation_mask]
            validation_dates = development.loc[validation_mask, "date"]
            train_set = lgb.Dataset(
                x_train,
                label=y_train,
                feature_name=derived_columns,
                free_raw_data=False,
                params={
                    "max_bin": fixed_parameters["max_bin"],
                    "feature_pre_filter": False,
                },
            )
            validation_set = lgb.Dataset(
                x_validation,
                label=y_validation,
                reference=train_set,
                feature_name=derived_columns,
                free_raw_data=False,
                params={
                    "max_bin": fixed_parameters["max_bin"],
                    "feature_pre_filter": False,
                },
            )

            candidate_results: list[dict[str, Any]] = []
            trained_models: dict[str, lgb.Booster] = {}
            for index, candidate in enumerate(candidates, start=1):
                candidate_name = str(candidate["name"])
                candidate_parameters = {
                    "num_leaves": int(candidate["num_leaves"]),
                    "max_depth": int(candidate["max_depth"]),
                    "min_data_in_leaf": int(
                        candidate["min_data_in_leaf"]
                    ),
                }
                parameters = {
                    **fixed_parameters,
                    **candidate_parameters,
                }
                booster = lgb.train(
                    parameters,
                    train_set,
                    num_boost_round=max_rounds,
                    valid_sets=[validation_set],
                    valid_names=["validation"],
                    callbacks=[
                        lgb.early_stopping(
                            early_stopping_rounds,
                            first_metric_only=True,
                            verbose=False,
                        ),
                        lgb.log_evaluation(period=0),
                    ],
                )
                best_iteration = int(booster.best_iteration)
                candidate_predictions = booster.predict(
                    x_validation,
                    num_iteration=best_iteration,
                )
                rank_ic, valid_date_count = mean_daily_rank_ic(
                    validation_dates,
                    y_validation,
                    candidate_predictions,
                )
                if not np.isfinite(rank_ic):
                    raise ValueError(
                        f"{suffix} candidate={candidate_name} "
                        "has no valid Rank IC"
                    )
                rmse = float(
                    np.sqrt(
                        np.mean(
                            np.square(candidate_predictions - y_validation)
                        )
                    )
                )
                candidate_results.append(
                    {
                        "candidate_name": candidate_name,
                        "parameters": candidate_parameters,
                        "best_iteration": best_iteration,
                        "boosting_rounds": best_iteration,
                        "mean_daily_rank_ic": rank_ic,
                        "rank_ic_valid_dates": valid_date_count,
                        "validation_rmse": rmse,
                    }
                )
                trained_models[candidate_name] = booster
                print(
                    f"{suffix} candidate {index}/{len(candidates)} "
                    f"{candidate_name}: rounds={best_iteration} "
                    f"rank_ic={rank_ic:.6f} rmse={rmse:.6f}",
                    flush=True,
                )
                del candidate_predictions
                gc.collect()

            best = choose_best_lightgbm_candidate(
                candidate_results,
                tolerance=tolerance,
            )
            selected_name = str(best["candidate_name"])
            selected_model = trained_models[selected_name]
            selected_iterations = int(best["boosting_rounds"])
            selected_predictions = selected_model.predict(
                x_validation,
                num_iteration=selected_iterations,
            )
            model_filename = f"model_{suffix}.txt"
            (temporary_path / model_filename).write_text(
                selected_model.model_to_string(
                    num_iteration=selected_iterations,
                ),
                encoding="utf-8",
            )

            horizon_prediction = development.loc[
                validation_mask,
                KEY_COLUMNS,
            ].copy()
            horizon_prediction["split"] = "validation"
            horizon_prediction["horizon"] = horizon
            horizon_prediction["y_true"] = y_validation
            horizon_prediction["y_pred"] = selected_predictions
            predictions.append(horizon_prediction)
            horizon_results[suffix] = {
                "target_column": target_column,
                "split_column": split_column,
                "train": _sample_stats(
                    development["date"],
                    train_mask,
                ),
                "validation": _sample_stats(
                    development["date"],
                    validation_mask,
                ),
                "candidate_results": candidate_results,
                "selected_candidate": selected_name,
                "selected_parameters": dict(best["parameters"]),
                "selected_best_iteration": int(best["best_iteration"]),
                "selected_boosting_rounds": selected_iterations,
                "selected_mean_daily_rank_ic": float(
                    best["mean_daily_rank_ic"]
                ),
                "selected_validation_rmse": float(
                    best["validation_rmse"]
                ),
                "rank_ic_valid_dates": int(
                    best["rank_ic_valid_dates"]
                ),
                "model_artifact": model_filename,
            }
            print(
                f"{suffix}: selected {selected_name} "
                f"rounds={selected_iterations} "
                f"rank_ic={best['mean_daily_rank_ic']:.6f} "
                f"rmse={best['validation_rmse']:.6f}",
                flush=True,
            )
            del (
                x_train,
                x_validation,
                y_train,
                y_validation,
                train_set,
                validation_set,
                selected_predictions,
                selected_model,
                trained_models,
            )
            gc.collect()

        prediction_table = pd.concat(predictions, ignore_index=True)
        prediction_table = prediction_table[
            [
                "date",
                "stock_code",
                "split",
                "horizon",
                "y_true",
                "y_pred",
            ]
        ].sort_values(
            ["date", "stock_code", "horizon"],
            ignore_index=True,
        )
        prediction_table.to_parquet(
            temporary_path / "predictions.parquet",
            index=False,
            compression="zstd",
        )

        manifest = {
            "schema_version": 1,
            "status": "completed",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": "lightgbm",
            "horizons": selected_horizons,
            "development_data": {
                "configured_start_date": config["split"]["train"]["start"],
                "configured_validation_end": config["split"]["validation"][
                    "end"
                ],
                "min_date": development["date"].min().strftime("%Y-%m-%d"),
                "max_date": development["date"].max().strftime("%Y-%m-%d"),
                "rows": len(development),
                "dates": int(development["date"].nunique()),
                "expected_cross_section_size": int(
                    preprocessing_config["expected_cross_section_size"]
                ),
            },
            "features": {
                "raw": factor_columns,
                "derived": derived_columns,
                "raw_count": len(factor_columns),
                "derived_count": len(derived_columns),
            },
            "preprocessing": {
                "daily_cross_section_first": True,
                "mad_width": float(preprocessing_config["mad_width"]),
                "mad_normalization": 1.4826,
                "missing_fill_value": 0.0,
                "dtype": preprocessing_config["dtype"],
                "second_stage_scaler": False,
            },
            "selection": {
                "primary_metric": "mean_daily_spearman_rank_ic",
                "near_best_rank_ic_tolerance": tolerance,
                "first_tie_breaker": "lower_validation_rmse",
                "second_tie_breaker": "simpler_tree",
                "early_stopping_metric": lightgbm_config["metric"],
                "max_boost_rounds": max_rounds,
                "early_stopping_rounds": early_stopping_rounds,
                "fixed_parameters": fixed_parameters,
                "candidate_count": len(candidates),
            },
            "horizon_results": horizon_results,
            "read_columns": read_columns,
            "uses_entry_tradeable": False,
            "prediction_rows": len(prediction_table),
            "sources": {
                "config_sha256": _file_sha256(config_path),
                **lineage,
                "lineage_validation": "published_schema_and_summary",
            },
            "git": _git_state(config_path),
            "environment": _environment_versions(),
        }
        _write_json(temporary_path / "manifest.json", manifest)
        os.rename(temporary_path, final_run_path)
    except BaseException:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise

    print(f"run published: {final_run_path}", flush=True)
    return final_run_path


def _predict_mlp(
    model: Any,
    features: Any,
    *,
    batch_size: int,
    device: Any,
) -> np.ndarray:
    import torch

    predictions = np.empty(len(features), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            batch = features[start:end].to(device)
            output = model(batch).squeeze(-1)
            predictions[start:end] = output.cpu().numpy()
    return predictions


def _train_mlp_candidate(
    *,
    train_features: Any,
    train_targets: Any,
    validation_features: Any,
    validation_targets: np.ndarray,
    input_dim: int,
    hidden_layers: list[int],
    dropout: float,
    weight_decay: float,
    training_config: dict[str, Any],
    candidate_name: str,
    suffix: str,
) -> dict[str, Any]:
    import torch

    seed = int(training_config["seed"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    configured_threads = int(training_config["num_threads"])
    if configured_threads > 0:
        torch.set_num_threads(configured_threads)
    device = torch.device(str(training_config["device"]))
    output_bias = float(train_targets.double().mean().item())
    model = build_mlp_model(
        input_dim,
        hidden_layers,
        dropout,
        output_bias=output_bias,
    ).to(device)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config["lr_scheduler_factor"]),
        patience=int(training_config["lr_scheduler_patience"]),
        min_lr=float(training_config["min_learning_rate"]),
    )
    loss_function = torch.nn.MSELoss()
    batch_size = int(training_config["batch_size"])
    max_epochs = int(training_config["max_epochs"])
    early_stopping_patience = int(
        training_config["early_stopping_patience"]
    )
    min_delta = float(training_config["early_stopping_min_delta"])
    gradient_clip_norm = float(training_config["gradient_clip_norm"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    best_validation_mse = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    epochs_trained = 0
    last_training_mse = float("nan")
    print(
        f"{suffix} candidate {candidate_name}: training started "
        f"parameters={parameter_count}",
        flush=True,
    )

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(
            len(train_features),
            generator=generator,
        )
        squared_error_sum = 0.0
        observation_count = 0
        for start in range(0, len(train_features), batch_size):
            indices = permutation[start : start + batch_size]
            feature_batch = train_features[indices].to(device)
            target_batch = train_targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(feature_batch).squeeze(-1)
            loss = loss_function(output, target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip_norm,
            )
            optimizer.step()
            batch_observations = len(indices)
            squared_error_sum += float(loss.item()) * batch_observations
            observation_count += batch_observations

        last_training_mse = squared_error_sum / observation_count
        validation_predictions = _predict_mlp(
            model,
            validation_features,
            batch_size=batch_size,
            device=device,
        )
        validation_errors = (
            validation_predictions.astype(np.float64)
            - validation_targets
        )
        validation_mse = float(np.mean(np.square(validation_errors)))
        scheduler.step(validation_mse)
        epochs_trained = epoch

        if best_validation_mse - validation_mse > min_delta:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{suffix} candidate {candidate_name}: "
                f"epoch={epoch} train_rmse={np.sqrt(last_training_mse):.6f} "
                f"validation_rmse={np.sqrt(validation_mse):.6f}",
                flush=True,
            )
        if epochs_without_improvement >= early_stopping_patience:
            break

    if best_state is None or best_epoch <= 0:
        raise RuntimeError(
            f"{suffix} candidate {candidate_name} did not produce a model"
        )
    model.load_state_dict(best_state)
    best_predictions = _predict_mlp(
        model,
        validation_features,
        batch_size=batch_size,
        device=device,
    )
    final_learning_rate = float(optimizer.param_groups[0]["lr"])
    del model, optimizer, scheduler
    gc.collect()
    return {
        "state_dict": best_state,
        "predictions": best_predictions,
        "parameter_count": int(parameter_count),
        "output_bias_initialization": output_bias,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "best_validation_mse": best_validation_mse,
        "last_training_mse": last_training_mse,
        "final_learning_rate": final_learning_rate,
    }


def train_mlp_run(
    config: dict[str, Any],
    config_path: Path,
    *,
    horizons: list[int],
    run_id: str,
) -> Path:
    """Tune a raw-target PyTorch MLP on 2019-2022 and validate on 2023."""
    import torch

    validate_run_id(run_id)
    selected_horizons = list(horizons)
    if (
        not selected_horizons
        or len(selected_horizons) != len(set(selected_horizons))
        or any(
            horizon not in SUPPORTED_HORIZONS
            for horizon in selected_horizons
        )
    ):
        raise ValueError("horizons must be unique values from 1, 5, 10")

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    runs_directory = _resolve_path(
        config_path,
        config["paths"]["runs_dir"],
    )
    runs_directory.mkdir(parents=True, exist_ok=True)
    final_run_path = runs_directory / run_id
    if final_run_path.exists():
        raise FileExistsError(f"run already exists: {final_run_path}")

    (
        development,
        factor_columns,
        _schema,
        _summary,
        lineage,
        read_columns,
    ) = _load_development_data(
        config,
        config_path,
        selected_horizons,
    )
    preprocessing_config = config["preprocessing"]
    feature_matrix, derived_columns = transform_daily_cross_sections(
        development,
        factor_columns,
        mad_width=float(preprocessing_config["mad_width"]),
        expected_cross_section_size=int(
            preprocessing_config["expected_cross_section_size"]
        ),
        date_chunk_size=int(preprocessing_config["date_chunk_size"]),
    )
    if feature_matrix.dtype != np.float32:
        raise RuntimeError("cross-sectional features must use float32")

    temporary_path = Path(
        tempfile.mkdtemp(
            dir=runs_directory,
            prefix=f".{run_id}.",
            suffix=".tmp",
        )
    )
    try:
        shutil.copy2(config_path, temporary_path / "config_snapshot.toml")
        mlp_config = config["models"]["mlp"]
        candidates = list(mlp_config["candidates"])
        tolerance = float(mlp_config["rank_ic_tolerance"])
        predictions: list[pd.DataFrame] = []
        horizon_results: dict[str, Any] = {}

        for horizon in selected_horizons:
            suffix = f"{horizon}d"
            target_column = f"target_{suffix}"
            split_column = f"split_{suffix}"
            train_mask = (
                development[split_column]
                .eq("train")
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            validation_mask = (
                development[split_column]
                .eq("validation")
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            if not train_mask.any() or not validation_mask.any():
                raise ValueError(f"{suffix} has no train or validation rows")

            targets = pd.to_numeric(
                development[target_column],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            if not np.isfinite(targets[train_mask]).all():
                raise ValueError(f"{suffix} training targets are not finite")
            if not np.isfinite(targets[validation_mask]).all():
                raise ValueError(f"{suffix} validation targets are not finite")

            x_train = feature_matrix[train_mask]
            x_validation = feature_matrix[validation_mask]
            y_train = targets[train_mask]
            y_validation = targets[validation_mask]
            validation_dates = development.loc[validation_mask, "date"]
            scaler = StandardScaler(copy=True)
            x_train_scaled = np.ascontiguousarray(
                scaler.fit_transform(x_train),
                dtype=np.float32,
            )
            x_validation_scaled = np.ascontiguousarray(
                scaler.transform(x_validation),
                dtype=np.float32,
            )
            train_feature_tensor = torch.from_numpy(x_train_scaled)
            validation_feature_tensor = torch.from_numpy(
                x_validation_scaled
            )
            train_target_tensor = torch.from_numpy(
                np.ascontiguousarray(y_train, dtype=np.float32)
            )

            candidate_results: list[dict[str, Any]] = []
            candidate_states: dict[str, dict[str, Any]] = {}
            candidate_predictions: dict[str, np.ndarray] = {}
            candidate_biases: dict[str, float] = {}
            for candidate in candidates:
                candidate_name = str(candidate["name"])
                hidden_layers = [
                    int(width)
                    for width in candidate["hidden_layers"]
                ]
                dropout = float(candidate["dropout"])
                weight_decay = float(candidate["weight_decay"])
                training_result = _train_mlp_candidate(
                    train_features=train_feature_tensor,
                    train_targets=train_target_tensor,
                    validation_features=validation_feature_tensor,
                    validation_targets=y_validation,
                    input_dim=len(derived_columns),
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                    weight_decay=weight_decay,
                    training_config=mlp_config,
                    candidate_name=candidate_name,
                    suffix=suffix,
                )
                validation_prediction = training_result["predictions"]
                rank_ic, valid_date_count = mean_daily_rank_ic(
                    validation_dates,
                    y_validation,
                    validation_prediction,
                )
                if not np.isfinite(rank_ic):
                    raise ValueError(
                        f"{suffix} candidate={candidate_name} "
                        "has no valid Rank IC"
                    )
                rmse = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                validation_prediction.astype(np.float64)
                                - y_validation
                            )
                        )
                    )
                )
                parameters = {
                    "hidden_layers": hidden_layers,
                    "dropout": dropout,
                    "weight_decay": weight_decay,
                }
                candidate_results.append(
                    {
                        "candidate_name": candidate_name,
                        "parameters": parameters,
                        "parameter_count": int(
                            training_result["parameter_count"]
                        ),
                        "best_epoch": int(training_result["best_epoch"]),
                        "epochs_trained": int(
                            training_result["epochs_trained"]
                        ),
                        "mean_daily_rank_ic": rank_ic,
                        "rank_ic_valid_dates": valid_date_count,
                        "validation_rmse": rmse,
                        "last_training_rmse": float(
                            np.sqrt(training_result["last_training_mse"])
                        ),
                        "final_learning_rate": float(
                            training_result["final_learning_rate"]
                        ),
                    }
                )
                candidate_states[candidate_name] = training_result[
                    "state_dict"
                ]
                candidate_predictions[candidate_name] = (
                    validation_prediction
                )
                candidate_biases[candidate_name] = float(
                    training_result["output_bias_initialization"]
                )
                print(
                    f"{suffix} candidate {candidate_name}: "
                    f"best_epoch={training_result['best_epoch']} "
                    f"rank_ic={rank_ic:.6f} rmse={rmse:.6f}",
                    flush=True,
                )

            best = choose_best_mlp_candidate(
                candidate_results,
                tolerance=tolerance,
            )
            selected_name = str(best["candidate_name"])
            selected_predictions = candidate_predictions[selected_name]
            selected_parameters = dict(best["parameters"])
            model_filename = f"model_{suffix}.pt"
            model_payload = {
                "schema_version": 1,
                "model_name": "mlp",
                "input_dim": len(derived_columns),
                "feature_names": derived_columns,
                "hidden_layers": selected_parameters["hidden_layers"],
                "dropout": selected_parameters["dropout"],
                "activation": mlp_config["activation"],
                "output_bias_initialization": candidate_biases[
                    selected_name
                ],
                "target_standardization": False,
                "state_dict": candidate_states[selected_name],
            }
            buffer = io.BytesIO()
            torch.save(model_payload, buffer)
            (temporary_path / model_filename).write_bytes(buffer.getvalue())
            joblib.dump(
                scaler,
                temporary_path / f"preprocessor_{suffix}.joblib",
                compress=3,
            )

            horizon_prediction = development.loc[
                validation_mask,
                KEY_COLUMNS,
            ].copy()
            horizon_prediction["split"] = "validation"
            horizon_prediction["horizon"] = horizon
            horizon_prediction["y_true"] = y_validation
            horizon_prediction["y_pred"] = selected_predictions
            predictions.append(horizon_prediction)
            horizon_results[suffix] = {
                "target_column": target_column,
                "split_column": split_column,
                "train": _sample_stats(
                    development["date"],
                    train_mask,
                ),
                "validation": _sample_stats(
                    development["date"],
                    validation_mask,
                ),
                "candidate_results": candidate_results,
                "selected_candidate": selected_name,
                "selected_parameters": selected_parameters,
                "selected_parameter_count": int(best["parameter_count"]),
                "selected_best_epoch": int(best["best_epoch"]),
                "selected_epochs_trained": int(best["epochs_trained"]),
                "selected_mean_daily_rank_ic": float(
                    best["mean_daily_rank_ic"]
                ),
                "selected_validation_rmse": float(
                    best["validation_rmse"]
                ),
                "rank_ic_valid_dates": int(
                    best["rank_ic_valid_dates"]
                ),
                "model_artifact": model_filename,
                "preprocessor_artifact": f"preprocessor_{suffix}.joblib",
            }
            print(
                f"{suffix}: selected {selected_name} "
                f"epoch={best['best_epoch']} "
                f"rank_ic={best['mean_daily_rank_ic']:.6f} "
                f"rmse={best['validation_rmse']:.6f}",
                flush=True,
            )
            del (
                x_train,
                x_validation,
                x_train_scaled,
                x_validation_scaled,
                y_train,
                y_validation,
                train_feature_tensor,
                validation_feature_tensor,
                train_target_tensor,
                selected_predictions,
                candidate_states,
                candidate_predictions,
                candidate_biases,
                scaler,
            )
            gc.collect()

        prediction_table = pd.concat(predictions, ignore_index=True)
        prediction_table = prediction_table[
            [
                "date",
                "stock_code",
                "split",
                "horizon",
                "y_true",
                "y_pred",
            ]
        ].sort_values(
            ["date", "stock_code", "horizon"],
            ignore_index=True,
        )
        prediction_table.to_parquet(
            temporary_path / "predictions.parquet",
            index=False,
            compression="zstd",
        )

        manifest = {
            "schema_version": 1,
            "status": "completed",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": "mlp",
            "horizons": selected_horizons,
            "development_data": {
                "configured_start_date": config["split"]["train"]["start"],
                "configured_validation_end": config["split"]["validation"][
                    "end"
                ],
                "min_date": development["date"].min().strftime("%Y-%m-%d"),
                "max_date": development["date"].max().strftime("%Y-%m-%d"),
                "rows": len(development),
                "dates": int(development["date"].nunique()),
                "expected_cross_section_size": int(
                    preprocessing_config["expected_cross_section_size"]
                ),
            },
            "features": {
                "raw": factor_columns,
                "derived": derived_columns,
                "raw_count": len(factor_columns),
                "derived_count": len(derived_columns),
            },
            "preprocessing": {
                "daily_cross_section_first": True,
                "mad_width": float(preprocessing_config["mad_width"]),
                "mad_normalization": 1.4826,
                "missing_fill_value": 0.0,
                "dtype": preprocessing_config["dtype"],
                "standard_scaler_fit_scope": "horizon_train_only",
                "target_standardization": False,
            },
            "selection": {
                "primary_metric": "mean_daily_spearman_rank_ic",
                "near_best_rank_ic_tolerance": tolerance,
                "first_tie_breaker": "lower_validation_rmse",
                "second_tie_breaker": "smaller_network",
                "early_stopping_metric": "raw_target_mse",
                "fixed_training_parameters": {
                    key: mlp_config[key]
                    for key in (
                        "framework",
                        "loss",
                        "optimizer",
                        "activation",
                        "device",
                        "learning_rate",
                        "batch_size",
                        "max_epochs",
                        "early_stopping_patience",
                        "early_stopping_min_delta",
                        "lr_scheduler_patience",
                        "lr_scheduler_factor",
                        "min_learning_rate",
                        "gradient_clip_norm",
                        "seed",
                        "num_threads",
                        "target_standardization",
                    )
                },
                "candidate_count": len(candidates),
            },
            "horizon_results": horizon_results,
            "read_columns": read_columns,
            "uses_entry_tradeable": False,
            "prediction_rows": len(prediction_table),
            "sources": {
                "config_sha256": _file_sha256(config_path),
                **lineage,
                "lineage_validation": "published_schema_and_summary",
            },
            "git": _git_state(config_path),
            "environment": _environment_versions(),
        }
        _write_json(temporary_path / "manifest.json", manifest)
        os.rename(temporary_path, final_run_path)
    except BaseException:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise

    print(f"run published: {final_run_path}", flush=True)
    return final_run_path


def _huber_values(errors: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(np.asarray(errors, dtype=np.float64))
    return np.where(
        absolute <= delta,
        0.5 * np.square(absolute),
        delta * (absolute - 0.5 * delta),
    )


def _residual_loss_constants(
    dates: pd.Series,
    targets: np.ndarray,
    ridge_predictions: np.ndarray,
    *,
    mad_multiplier: float,
) -> dict[str, float]:
    values = np.asarray(targets, dtype=np.float64)
    ridge = np.asarray(ridge_predictions, dtype=np.float64)
    residuals = values - ridge
    residual_median = float(np.median(residuals))
    residual_mad = float(np.median(np.abs(residuals - residual_median)))
    huber_delta = float(mad_multiplier * 1.4826 * residual_mad)
    date_frame = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(
                pd.to_datetime(dates, errors="raise")
            ).normalize(),
            "target": values,
        }
    )
    temperature = float(
        date_frame.groupby("date", sort=False)["target"]
        .std(ddof=1)
        .median()
    )
    ridge_huber_baseline = float(
        np.mean(_huber_values(ridge - values, huber_delta))
    )
    for name, value in (
        ("huber_delta", huber_delta),
        ("temperature", temperature),
        ("ridge_huber_baseline", ridge_huber_baseline),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return {
        "residual_median": residual_median,
        "residual_mad": residual_mad,
        "huber_delta": huber_delta,
        "temperature": temperature,
        "ridge_huber_baseline": ridge_huber_baseline,
    }


def _monthly_rank_ic_diagnostics(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(
                pd.to_datetime(dates, errors="raise")
            ).normalize(),
            "y_true": np.asarray(y_true, dtype=np.float64),
            "y_pred": np.asarray(y_pred, dtype=np.float64),
        }
    )
    daily_rows: list[dict[str, Any]] = []
    for date, group in frame.groupby("date", sort=False):
        true_rank = group["y_true"].rank(method="average").to_numpy()
        pred_rank = group["y_pred"].rank(method="average").to_numpy()
        if np.ptp(true_rank) == 0 or np.ptp(pred_rank) == 0:
            continue
        rank_ic = float(np.corrcoef(true_rank, pred_rank)[0, 1])
        if np.isfinite(rank_ic):
            daily_rows.append({"date": date, "rank_ic": rank_ic})
    if not daily_rows:
        raise ValueError("monthly diagnostics contain no valid Rank IC")
    daily = pd.DataFrame(daily_rows)
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    monthly = (
        daily.groupby("month", sort=True)["rank_ic"].mean().astype(float)
    )
    return {
        "valid_months": int(len(monthly)),
        "positive_months": int(monthly.gt(0).sum()),
        "positive_month_rate": float(monthly.gt(0).mean()),
        "monthly_rank_ic": {
            month: float(value)
            for month, value in monthly.items()
        },
    }


def _numpy_pair_loss(
    predictions: np.ndarray,
    dates: pd.Series,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    temperature: float,
    pair_config: dict[str, Any],
    seed: int,
) -> float:
    date_slices = build_date_slices(dates)
    values = np.asarray(targets, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    losses: list[float] = []
    normalized_dates = pd.DatetimeIndex(
        pd.to_datetime(dates, errors="raise")
    ).normalize()
    for date_slice in date_slices:
        pairs = sample_group_weighted_pairs(
            values[date_slice],
            groups[date_slice],
            seed=seed,
            epoch=0,
            date_value=normalized_dates[date_slice[0]],
            pairs_per_stock=int(pair_config["pairs_per_stock"]),
            adjacent_pair_fraction=float(
                pair_config["adjacent_pair_fraction"]
            ),
            top_group_weight=float(pair_config["top_group_weight"]),
            second_group_weight=float(
                pair_config["second_group_weight"]
            ),
        )
        difference = (
            predicted[date_slice][pairs["source_index"]]
            - predicted[date_slice][pairs["opponent_index"]]
        ) / temperature
        pair_loss = np.logaddexp(
            0.0,
            -pairs["sign"].astype(np.float64) * difference,
        )
        losses.append(
            float(np.mean(pairs["weight"] * pair_loss) / np.log(2.0))
        )
    return float(np.mean(losses))


def _train_residual_candidate(
    *,
    train_features: np.ndarray,
    train_dates: pd.Series,
    train_targets: np.ndarray,
    train_ridge_predictions: np.ndarray,
    validation_features: np.ndarray,
    validation_dates: pd.Series,
    validation_stock_codes: pd.Series,
    validation_targets: np.ndarray,
    validation_ridge_predictions: np.ndarray,
    input_dim: int,
    hidden_layers: list[int],
    dropout: float,
    weight_decay: float,
    gamma_grid: list[float],
    lambda_rank: float,
    rank_stage: bool,
    loss_constants: dict[str, float],
    training_config: dict[str, Any],
    candidate_name: str,
    suffix: str,
    seed: int,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(training_config["num_threads"]))
    device = torch.device("cpu")
    train_x = torch.from_numpy(
        np.ascontiguousarray(train_features, dtype=np.float32)
    )
    validation_x = torch.from_numpy(
        np.ascontiguousarray(validation_features, dtype=np.float32)
    )
    train_y = np.asarray(train_targets, dtype=np.float64)
    train_ridge = np.asarray(
        train_ridge_predictions,
        dtype=np.float64,
    )
    validation_y = np.asarray(validation_targets, dtype=np.float64)
    validation_ridge = np.asarray(
        validation_ridge_predictions,
        dtype=np.float64,
    )
    residual_targets = train_y - train_ridge
    train_y_tensor = torch.from_numpy(
        np.ascontiguousarray(train_y, dtype=np.float32)
    )
    train_ridge_tensor = torch.from_numpy(
        np.ascontiguousarray(train_ridge, dtype=np.float32)
    )
    residual_target_tensor = torch.from_numpy(
        np.ascontiguousarray(residual_targets, dtype=np.float32)
    )
    output_bias = float(np.mean(residual_targets))
    model = build_mlp_model(
        input_dim,
        hidden_layers,
        dropout,
        output_bias=output_bias,
    ).to(device)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config["lr_scheduler_factor"]),
        patience=int(training_config["lr_scheduler_patience"]),
        min_lr=float(training_config["min_learning_rate"]),
    )
    delta = float(loss_constants["huber_delta"])
    baseline_loss = float(loss_constants["ridge_huber_baseline"])
    temperature = float(loss_constants["temperature"])
    max_epochs = int(training_config["max_epochs"])
    patience = int(training_config["early_stopping_patience"])
    min_delta = float(training_config["early_stopping_min_delta"])
    gradient_clip = float(training_config["gradient_clip_norm"])
    tolerance = float(training_config["rank_ic_tolerance"])
    dates_per_batch = int(training_config["dates_per_batch"])
    date_slices = build_date_slices(train_dates)
    normalized_train_dates = pd.DatetimeIndex(
        pd.to_datetime(train_dates, errors="raise")
    ).normalize()
    train_groups = (
        build_true_deciles(train_dates, train_y)
        if rank_stage and lambda_rank > 0
        else None
    )
    validation_groups = (
        build_true_deciles(validation_dates, validation_y)
        if rank_stage and lambda_rank > 0
        else None
    )
    records: list[dict[str, Any]] = []
    epoch_states: dict[int, dict[str, Any]] = {}
    best_identity: tuple[Any, ...] | None = None
    epochs_without_improvement = 0
    epochs_trained = 0
    stage_name = "D" if rank_stage else "C"
    print(
        f"{suffix} {stage_name} {candidate_name}: training started "
        f"seed={seed} parameters={parameter_count}",
        flush=True,
    )

    for epoch in range(1, max_epochs + 1):
        model.train()
        date_rng = np.random.default_rng(
            np.random.SeedSequence([seed, epoch])
        )
        shuffled_dates = date_rng.permutation(len(date_slices))
        training_loss_sum = 0.0
        batch_count = 0
        for start in range(0, len(shuffled_dates), dates_per_batch):
            selected_date_ids = shuffled_dates[
                start : start + dates_per_batch
            ]
            selected_slices = [
                date_slices[int(date_id)]
                for date_id in selected_date_ids
            ]
            batch_indices = np.concatenate(selected_slices)
            index_tensor = torch.from_numpy(batch_indices)
            feature_batch = train_x[index_tensor].to(device)
            target_batch = train_y_tensor[index_tensor].to(device)
            ridge_batch = train_ridge_tensor[index_tensor].to(device)
            residual_target_batch = residual_target_tensor[
                index_tensor
            ].to(device)
            optimizer.zero_grad(set_to_none=True)
            residual_output = model(feature_batch).squeeze(-1)
            if not rank_stage:
                return_loss = torch.nn.functional.huber_loss(
                    residual_output,
                    residual_target_batch,
                    reduction="mean",
                    delta=delta,
                )
                total_loss = return_loss
            else:
                final_output = (
                    ridge_batch
                    + float(gamma_grid[0]) * residual_output
                )
                return_loss = torch.nn.functional.huber_loss(
                    final_output,
                    target_batch,
                    reduction="mean",
                    delta=delta,
                ) / baseline_loss
                if lambda_rank > 0:
                    pair_losses: list[Any] = []
                    offset = 0
                    for date_slice in selected_slices:
                        date_length = len(date_slice)
                        local_prediction = final_output[
                            offset : offset + date_length
                        ]
                        pairs = sample_group_weighted_pairs(
                            train_y[date_slice],
                            train_groups[date_slice],
                            seed=seed,
                            epoch=epoch,
                            date_value=normalized_train_dates[
                                date_slice[0]
                            ],
                            pairs_per_stock=int(
                                training_config["pairs_per_stock"]
                            ),
                            adjacent_pair_fraction=float(
                                training_config[
                                    "adjacent_pair_fraction"
                                ]
                            ),
                            top_group_weight=float(
                                training_config["top_group_weight"]
                            ),
                            second_group_weight=float(
                                training_config["second_group_weight"]
                            ),
                        )
                        pair_losses.append(
                            group_weighted_pairwise_loss(
                                local_prediction,
                                pairs,
                                temperature=temperature,
                            )
                        )
                        offset += date_length
                    pair_loss = torch.stack(pair_losses).mean()
                    total_loss = return_loss + lambda_rank * pair_loss
                else:
                    total_loss = return_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip,
            )
            optimizer.step()
            training_loss_sum += float(total_loss.item())
            batch_count += 1

        residual_validation = _predict_mlp(
            model,
            validation_x,
            batch_size=dates_per_batch * 1000,
            device=device,
        ).astype(np.float64)
        if not rank_stage:
            validation_total_loss = float(
                np.mean(
                    _huber_values(
                        residual_validation
                        - (validation_y - validation_ridge),
                        delta,
                    )
                )
            )
        else:
            final_validation_for_loss = (
                validation_ridge
                + float(gamma_grid[0]) * residual_validation
            )
            validation_return_loss = float(
                np.mean(
                    _huber_values(
                        final_validation_for_loss - validation_y,
                        delta,
                    )
                )
                / baseline_loss
            )
            if lambda_rank > 0:
                validation_pair_loss = _numpy_pair_loss(
                    final_validation_for_loss,
                    validation_dates,
                    validation_y,
                    validation_groups,
                    temperature=temperature,
                    pair_config=training_config,
                    seed=seed,
                )
                validation_total_loss = (
                    validation_return_loss
                    + lambda_rank * validation_pair_loss
                )
            else:
                validation_total_loss = validation_return_loss
        scheduler.step(validation_total_loss)
        epochs_trained = epoch
        epoch_states[epoch] = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        for gamma in gamma_grid:
            final_validation = (
                validation_ridge + gamma * residual_validation
            )
            metrics = residual_rank_metrics(
                validation_dates,
                validation_stock_codes,
                validation_y,
                final_validation,
            )
            records.append(
                {
                    "candidate_name": candidate_name,
                    "gamma": float(gamma),
                    "lambda_rank": float(lambda_rank),
                    "best_epoch": int(epoch),
                    "metrics": metrics,
                    "validation_total_loss": validation_total_loss,
                }
            )
        chooser = (
            choose_rank_candidate
            if rank_stage
            else choose_residual_candidate
        )
        selected_so_far = chooser(records, tolerance=tolerance)
        selected_identity = (
            selected_so_far["candidate_name"],
            selected_so_far["gamma"],
            selected_so_far["lambda_rank"],
            selected_so_far["best_epoch"],
        )
        if selected_identity != best_identity:
            best_identity = selected_identity
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0:
            selected_metrics = selected_so_far["metrics"]
            print(
                f"{suffix} {stage_name} {candidate_name}: "
                f"epoch={epoch} train_loss="
                f"{training_loss_sum / batch_count:.6f} "
                f"validation_loss={validation_total_loss:.6f} "
                f"best_rank_ic="
                f"{selected_metrics['mean_daily_rank_ic']:.6f} "
                f"best_group_mae={selected_metrics['group_mae']:.6f}",
                flush=True,
            )
        if epochs_without_improvement >= patience:
            break

    chooser = (
        choose_rank_candidate
        if rank_stage
        else choose_residual_candidate
    )
    selected = chooser(records, tolerance=tolerance)
    selected_epoch = int(selected["best_epoch"])
    selected_state = epoch_states[selected_epoch]
    model.load_state_dict(selected_state)
    selected_residual_prediction = _predict_mlp(
        model,
        validation_x,
        batch_size=dates_per_batch * 1000,
        device=device,
    ).astype(np.float64)
    selected_final_prediction = (
        validation_ridge
        + float(selected["gamma"]) * selected_residual_prediction
    )
    if not np.allclose(
        selected_final_prediction,
        validation_ridge
        + float(selected["gamma"]) * selected_residual_prediction,
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("component prediction identity failed")
    selected["metrics"] = residual_rank_metrics(
        validation_dates,
        validation_stock_codes,
        validation_y,
        selected_final_prediction,
    )
    final_learning_rate = float(optimizer.param_groups[0]["lr"])
    del model, optimizer, scheduler, train_x, validation_x
    gc.collect()
    return {
        "selected_record": selected,
        "state_dict": selected_state,
        "residual_predictions": selected_residual_prediction,
        "final_predictions": selected_final_prediction,
        "parameter_count": int(parameter_count),
        "output_bias_initialization": output_bias,
        "epochs_trained": int(epochs_trained),
        "final_learning_rate": final_learning_rate,
        "epoch_records": records,
    }


def _selected_training_run(
    runs: list[dict[str, Any]],
    selected: dict[str, Any],
) -> dict[str, Any]:
    for run in runs:
        record = run["selected_record"]
        if (
            record["candidate_name"] == selected["candidate_name"]
            and float(record["gamma"]) == float(selected["gamma"])
            and float(record["lambda_rank"])
            == float(selected["lambda_rank"])
            and int(record["best_epoch"]) == int(selected["best_epoch"])
        ):
            return run
    raise RuntimeError("selected training run is missing")


def _stability_summary(
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_names = (
        "mean_daily_rank_ic",
        "group_mae",
        "top100_recall",
    )
    summary: dict[str, Any] = {
        "seeds": [
            {
                "seed": int(result["seed"]),
                "metrics": result["metrics"],
            }
            for result in seed_results
        ]
    }
    for metric_name in metric_names:
        values = np.asarray(
            [
                result["metrics"][metric_name]
                for result in seed_results
            ],
            dtype=np.float64,
        )
        summary[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    summary["all_rank_ic_positive"] = bool(
        all(
            result["metrics"]["mean_daily_rank_ic"] > 0
            for result in seed_results
        )
    )
    return summary


def _restore_residual_rank_components(
    run_path: Path,
    *,
    suffix: str,
    feature_matrix: np.ndarray,
    validation_mask: np.ndarray,
    expected_ridge: np.ndarray,
    expected_residual: np.ndarray,
    expected_final: np.ndarray,
    batch_size: int,
) -> None:
    import torch

    ridge_model = joblib.load(run_path / f"ridge_model_{suffix}.joblib")
    ridge_scaler = joblib.load(
        run_path / f"ridge_preprocessor_{suffix}.joblib"
    )
    mlp_scaler = joblib.load(
        run_path / f"mlp_preprocessor_{suffix}.joblib"
    )
    payload = torch.load(
        run_path / f"residual_mlp_{suffix}.pt",
        map_location="cpu",
        weights_only=False,
    )
    x_validation = feature_matrix[validation_mask]
    restored_ridge = np.asarray(
        ridge_model.predict(ridge_scaler.transform(x_validation)),
        dtype=np.float64,
    )
    restored_mlp_features = np.ascontiguousarray(
        mlp_scaler.transform(x_validation),
        dtype=np.float32,
    )
    model = build_mlp_model(
        int(payload["input_dim"]),
        [int(width) for width in payload["hidden_layers"]],
        float(payload["dropout"]),
        output_bias=float(payload["output_bias_initialization"]),
    )
    model.load_state_dict(payload["state_dict"])
    restored_residual = _predict_mlp(
        model,
        torch.from_numpy(restored_mlp_features),
        batch_size=batch_size,
        device=torch.device("cpu"),
    ).astype(np.float64)
    restored_final = (
        restored_ridge + float(payload["gamma"]) * restored_residual
    )
    for name, restored, expected in (
        ("ridge", restored_ridge, expected_ridge),
        ("residual", restored_residual, expected_residual),
        ("final", restored_final, expected_final),
    ):
        if not np.allclose(
            restored,
            expected,
            rtol=1e-6,
            atol=1e-8,
        ):
            maximum_error = float(np.max(np.abs(restored - expected)))
            raise RuntimeError(
                f"{suffix} restored {name} predictions differ: "
                f"max_error={maximum_error}"
            )


def train_residual_rank_mlp_run(
    config: dict[str, Any],
    config_path: Path,
    *,
    horizons: list[int],
    run_id: str,
) -> Path:
    """Run the development-only Ridge plus residual rank-aware MLP."""
    import torch

    validate_run_id(run_id)
    selected_horizons = list(horizons)
    if (
        not selected_horizons
        or len(selected_horizons) != len(set(selected_horizons))
        or any(
            horizon not in SUPPORTED_HORIZONS
            for horizon in selected_horizons
        )
    ):
        raise ValueError("horizons must be unique values from 1, 5, 10")

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    runs_directory = _resolve_path(
        config_path,
        config["paths"]["runs_dir"],
    )
    runs_directory.mkdir(parents=True, exist_ok=True)
    final_run_path = runs_directory / run_id
    if final_run_path.exists():
        raise FileExistsError(f"run already exists: {final_run_path}")

    (
        development,
        factor_columns,
        _schema,
        _summary,
        lineage,
        read_columns,
    ) = _load_residual_rank_data(
        config,
        config_path,
        selected_horizons,
    )
    preprocessing_config = config["preprocessing"]
    feature_matrix, derived_columns = transform_daily_cross_sections(
        development,
        factor_columns,
        mad_width=float(preprocessing_config["mad_width"]),
        expected_cross_section_size=int(
            preprocessing_config["expected_cross_section_size"]
        ),
        date_chunk_size=int(preprocessing_config["date_chunk_size"]),
    )
    if feature_matrix.dtype != np.float32:
        raise RuntimeError("cross-sectional features must use float32")

    training_config = config["models"]["residual_rank_mlp"]
    tolerance = float(training_config["rank_ic_tolerance"])
    candidates = list(training_config["candidates"])
    gamma_grid = [
        float(value) for value in training_config["gamma_grid"]
    ]
    rank_lambda_grid = [
        float(value) for value in training_config["rank_lambda_grid"]
    ]
    base_seed = int(training_config["seed"])
    stability_seeds = [
        int(value) for value in training_config["stability_seeds"]
    ]
    batch_size = int(training_config["dates_per_batch"]) * int(
        preprocessing_config["expected_cross_section_size"]
    )
    temporary_path = Path(
        tempfile.mkdtemp(
            dir=runs_directory,
            prefix=f".{run_id}.",
            suffix=".tmp",
        )
    )
    predictions: list[pd.DataFrame] = []
    component_predictions: list[pd.DataFrame] = []
    horizon_results: dict[str, Any] = {}
    restore_inputs: list[dict[str, Any]] = []

    try:
        shutil.copy2(config_path, temporary_path / "config_snapshot.toml")
        for horizon in selected_horizons:
            suffix = f"{horizon}d"
            target_column = f"target_{suffix}"
            split_column = f"split_{suffix}"
            exit_column = f"exit_date_{suffix}"
            train_mask = (
                development[split_column]
                .eq("train")
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            validation_mask = (
                development[split_column]
                .eq("validation")
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            targets = pd.to_numeric(
                development[target_column],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            ridge_result = build_purged_oof_ridge(
                feature_matrix,
                development["date"],
                development[exit_column],
                targets,
                train_mask,
                validation_mask,
                ridge_lambda=float(training_config["ridge_lambda"]),
            )
            oof_mask = ridge_result["oof_mask"]
            expected_oof_counts = {
                1: (726, 726000),
                5: (722, 722000),
                10: (717, 717000),
            }
            actual_oof = (
                int(development.loc[oof_mask, "date"].nunique()),
                int(oof_mask.sum()),
            )
            if actual_oof != expected_oof_counts[horizon]:
                raise RuntimeError(
                    f"{suffix} OOF count mismatch: {actual_oof}"
                )

            oof_targets = targets[oof_mask]
            validation_targets = targets[validation_mask]
            oof_ridge = ridge_result["oof_predictions"][oof_mask]
            validation_ridge = ridge_result["validation_predictions"]
            oof_dates = development.loc[oof_mask, "date"].reset_index(
                drop=True
            )
            validation_dates = development.loc[
                validation_mask,
                "date",
            ].reset_index(drop=True)
            validation_codes = development.loc[
                validation_mask,
                "stock_code",
            ].reset_index(drop=True)
            loss_constants = _residual_loss_constants(
                oof_dates,
                oof_targets,
                oof_ridge,
                mad_multiplier=float(
                    training_config["huber_mad_multiplier"]
                ),
            )
            mlp_scaler = StandardScaler(copy=True)
            oof_features = np.ascontiguousarray(
                mlp_scaler.fit_transform(feature_matrix[oof_mask]),
                dtype=np.float32,
            )
            validation_features = np.ascontiguousarray(
                mlp_scaler.transform(feature_matrix[validation_mask]),
                dtype=np.float32,
            )
            ridge_metrics = residual_rank_metrics(
                validation_dates,
                validation_codes,
                validation_targets,
                validation_ridge,
            )
            ridge_monthly = _monthly_rank_ic_diagnostics(
                validation_dates,
                validation_targets,
                validation_ridge,
            )

            stage_c_runs: list[dict[str, Any]] = []
            for candidate in candidates:
                parameters = {
                    "hidden_layers": [
                        int(width)
                        for width in candidate["hidden_layers"]
                    ],
                    "dropout": float(candidate["dropout"]),
                    "weight_decay": float(candidate["weight_decay"]),
                }
                result = _train_residual_candidate(
                    train_features=oof_features,
                    train_dates=oof_dates,
                    train_targets=oof_targets,
                    train_ridge_predictions=oof_ridge,
                    validation_features=validation_features,
                    validation_dates=validation_dates,
                    validation_stock_codes=validation_codes,
                    validation_targets=validation_targets,
                    validation_ridge_predictions=validation_ridge,
                    input_dim=len(derived_columns),
                    **parameters,
                    gamma_grid=gamma_grid,
                    lambda_rank=0.0,
                    rank_stage=False,
                    loss_constants=loss_constants,
                    training_config=training_config,
                    candidate_name=str(candidate["name"]),
                    suffix=suffix,
                    seed=base_seed,
                )
                result["parameters"] = parameters
                result["selected_record"]["parameters"] = parameters
                stage_c_runs.append(result)
            stage_c_selected = choose_residual_candidate(
                [
                    result["selected_record"]
                    for result in stage_c_runs
                ],
                tolerance=tolerance,
            )
            stage_c_run = _selected_training_run(
                stage_c_runs,
                stage_c_selected,
            )
            stage_c_monthly = _monthly_rank_ic_diagnostics(
                validation_dates,
                validation_targets,
                stage_c_run["final_predictions"],
            )
            residual_gate_reasons: list[str] = []
            if float(stage_c_selected["gamma"]) == 0:
                residual_gate_reasons.append("selected_gamma_is_zero")
            if (
                stage_c_selected["metrics"]["mean_daily_rank_ic"]
                < ridge_metrics["mean_daily_rank_ic"] - tolerance
            ):
                residual_gate_reasons.append("rank_ic_below_ridge_tolerance")
            if (
                stage_c_selected["metrics"]["group_mae"]
                >= ridge_metrics["group_mae"]
            ):
                residual_gate_reasons.append(
                    "group_mae_not_below_ridge"
                )
            if (
                stage_c_monthly["valid_months"] < 10
                or stage_c_monthly["positive_month_rate"] < 0.5
            ):
                residual_gate_reasons.append(
                    "monthly_rank_ic_is_concentrated"
                )

            rank_stage_executed = not residual_gate_reasons
            stage_d_runs: list[dict[str, Any]] = []
            rank_gate_reasons: list[str] = []
            stability: dict[str, Any] | None = None
            selected_run = stage_c_run
            selected_record = stage_c_selected
            development_decision = "reject_residual"
            if residual_gate_reasons:
                selected_run = dict(stage_c_run)
                selected_run["final_predictions"] = validation_ridge
                selected_record = dict(stage_c_selected)
                selected_record["gamma"] = 0.0
                selected_record["lambda_rank"] = 0.0
                selected_record["metrics"] = ridge_metrics
            if rank_stage_executed:
                selected_parameters = stage_c_run["parameters"]
                selected_gamma = float(stage_c_selected["gamma"])
                for lambda_rank in rank_lambda_grid:
                    result = _train_residual_candidate(
                        train_features=oof_features,
                        train_dates=oof_dates,
                        train_targets=oof_targets,
                        train_ridge_predictions=oof_ridge,
                        validation_features=validation_features,
                        validation_dates=validation_dates,
                        validation_stock_codes=validation_codes,
                        validation_targets=validation_targets,
                        validation_ridge_predictions=validation_ridge,
                        input_dim=len(derived_columns),
                        **selected_parameters,
                        gamma_grid=[selected_gamma],
                        lambda_rank=lambda_rank,
                        rank_stage=True,
                        loss_constants=loss_constants,
                        training_config=training_config,
                        candidate_name=(
                            f"{stage_c_selected['candidate_name']}"
                            f"_lambda_{lambda_rank:g}"
                        ),
                        suffix=suffix,
                        seed=base_seed,
                    )
                    result["parameters"] = selected_parameters
                    result["selected_record"][
                        "parameters"
                    ] = selected_parameters
                    stage_d_runs.append(result)
                stage_d_selected = choose_rank_candidate(
                    [
                        result["selected_record"]
                        for result in stage_d_runs
                    ],
                    tolerance=tolerance,
                )
                stage_d_run = _selected_training_run(
                    stage_d_runs,
                    stage_d_selected,
                )
                control_run = next(
                    result
                    for result in stage_d_runs
                    if float(
                        result["selected_record"]["lambda_rank"]
                    )
                    == 0
                )
                control_metrics = control_run["selected_record"]["metrics"]
                selected_metrics = stage_d_selected["metrics"]
                if float(stage_d_selected["lambda_rank"]) <= 0:
                    rank_gate_reasons.append(
                        "selected_rank_lambda_is_zero"
                    )
                if (
                    selected_metrics["mean_daily_rank_ic"]
                    <= control_metrics["mean_daily_rank_ic"]
                ):
                    rank_gate_reasons.append(
                        "rank_ic_not_above_zero_control"
                    )
                if (
                    selected_metrics["group_mae"]
                    >= control_metrics["group_mae"]
                ):
                    rank_gate_reasons.append(
                        "group_mae_not_below_zero_control"
                    )
                if (
                    selected_metrics["top100_recall"]
                    < control_metrics["top100_recall"]
                ):
                    rank_gate_reasons.append(
                        "top100_recall_below_zero_control"
                    )
                if (
                    selected_metrics["rmse"]
                    > control_metrics["rmse"] * 1.01
                ):
                    rank_gate_reasons.append(
                        "rmse_worse_than_one_percent"
                    )
                if (
                    not np.isfinite(selected_metrics["prediction_std"])
                    or selected_metrics["prediction_std"]
                    > max(
                        2.0 * control_metrics["prediction_std"],
                        1e-12,
                    )
                ):
                    rank_gate_reasons.append(
                        "prediction_std_is_explosive"
                    )
                if rank_gate_reasons:
                    selected_run = control_run
                    selected_record = control_run["selected_record"]
                    development_decision = (
                        "accept_residual_reject_rank_loss"
                    )
                else:
                    selected_run = stage_d_run
                    selected_record = stage_d_selected
                    development_decision = "accept_residual_and_rank_loss"

                seed_results = [
                    {
                        "seed": base_seed,
                        "metrics": selected_record["metrics"],
                    }
                ]
                for stability_seed in stability_seeds:
                    if stability_seed == base_seed:
                        continue
                    stability_run = _train_residual_candidate(
                        train_features=oof_features,
                        train_dates=oof_dates,
                        train_targets=oof_targets,
                        train_ridge_predictions=oof_ridge,
                        validation_features=validation_features,
                        validation_dates=validation_dates,
                        validation_stock_codes=validation_codes,
                        validation_targets=validation_targets,
                        validation_ridge_predictions=validation_ridge,
                        input_dim=len(derived_columns),
                        **selected_run["parameters"],
                        gamma_grid=[float(selected_record["gamma"])],
                        lambda_rank=float(
                            selected_record["lambda_rank"]
                        ),
                        rank_stage=True,
                        loss_constants=loss_constants,
                        training_config=training_config,
                        candidate_name=(
                            f"stability_seed_{stability_seed}"
                        ),
                        suffix=suffix,
                        seed=stability_seed,
                    )
                    seed_results.append(
                        {
                            "seed": stability_seed,
                            "metrics": stability_run[
                                "selected_record"
                            ]["metrics"],
                        }
                    )
                    del stability_run
                    gc.collect()
                stability = _stability_summary(seed_results)
                final_monthly = _monthly_rank_ic_diagnostics(
                    validation_dates,
                    validation_targets,
                    selected_run["final_predictions"],
                )
                final_metrics = selected_record["metrics"]
                final_gate_reasons: list[str] = []
                if not stability["all_rank_ic_positive"]:
                    final_gate_reasons.append(
                        "rank_ic_direction_not_stable"
                    )
                if (
                    final_metrics["mean_daily_rank_ic"]
                    < ridge_metrics["mean_daily_rank_ic"] - tolerance
                ):
                    final_gate_reasons.append(
                        "final_rank_ic_below_ridge_tolerance"
                    )
                if final_metrics["group_mae"] >= ridge_metrics["group_mae"]:
                    final_gate_reasons.append(
                        "final_group_mae_not_below_ridge"
                    )
                if (
                    final_monthly["valid_months"] < 10
                    or final_monthly["positive_month_rate"] < 0.5
                ):
                    final_gate_reasons.append(
                        "final_monthly_rank_ic_is_concentrated"
                    )
                if final_gate_reasons:
                    residual_gate_reasons.extend(final_gate_reasons)
                    development_decision = "reject_unstable_residual"
                    selected_run = stage_c_run
                    selected_record = dict(stage_c_selected)
                    selected_record["gamma"] = 0.0
                    selected_record["lambda_rank"] = 0.0
                    selected_record["metrics"] = ridge_metrics
                    selected_run = dict(selected_run)
                    selected_run["final_predictions"] = validation_ridge

            final_gamma = float(selected_record["gamma"])
            final_lambda = float(selected_record["lambda_rank"])
            final_ridge_predictions = validation_ridge
            final_residual_predictions = selected_run[
                "residual_predictions"
            ]
            final_predictions = (
                final_ridge_predictions
                + final_gamma * final_residual_predictions
            )
            if not np.allclose(
                final_predictions,
                selected_run["final_predictions"],
                rtol=1e-6,
                atol=1e-8,
            ):
                raise RuntimeError(
                    f"{suffix} selected component predictions differ"
                )
            final_metrics = residual_rank_metrics(
                validation_dates,
                validation_codes,
                validation_targets,
                final_predictions,
            )
            final_monthly = _monthly_rank_ic_diagnostics(
                validation_dates,
                validation_targets,
                final_predictions,
            )

            ridge_model_name = f"ridge_model_{suffix}.joblib"
            ridge_scaler_name = f"ridge_preprocessor_{suffix}.joblib"
            residual_model_name = f"residual_mlp_{suffix}.pt"
            mlp_scaler_name = f"mlp_preprocessor_{suffix}.joblib"
            joblib.dump(
                ridge_result["development_model"],
                temporary_path / ridge_model_name,
                compress=3,
            )
            joblib.dump(
                ridge_result["development_scaler"],
                temporary_path / ridge_scaler_name,
                compress=3,
            )
            joblib.dump(
                mlp_scaler,
                temporary_path / mlp_scaler_name,
                compress=3,
            )
            selected_parameters = selected_run["parameters"]
            model_payload = {
                "schema_version": 1,
                "model_name": "residual_rank_mlp",
                "input_dim": len(derived_columns),
                "feature_names": derived_columns,
                "hidden_layers": selected_parameters["hidden_layers"],
                "dropout": selected_parameters["dropout"],
                "activation": "relu",
                "output_bias_initialization": float(
                    selected_run["output_bias_initialization"]
                ),
                "gamma": final_gamma,
                "lambda_rank": final_lambda,
                "loss_constants": loss_constants,
                "target_standardization": False,
                "state_dict": selected_run["state_dict"],
            }
            buffer = io.BytesIO()
            torch.save(model_payload, buffer)
            (temporary_path / residual_model_name).write_bytes(
                buffer.getvalue()
            )

            horizon_prediction = development.loc[
                validation_mask,
                KEY_COLUMNS,
            ].copy()
            horizon_prediction["split"] = "validation"
            horizon_prediction["horizon"] = horizon
            horizon_prediction["y_true"] = validation_targets
            horizon_prediction["y_pred"] = final_predictions
            predictions.append(horizon_prediction)
            component = horizon_prediction[
                ["date", "stock_code", "horizon"]
            ].copy()
            component["ridge_pred"] = final_ridge_predictions
            component["residual_pred"] = final_residual_predictions
            component["gamma"] = final_gamma
            component["final_pred"] = final_predictions
            component_predictions.append(component)
            artifact_names = (
                ridge_model_name,
                ridge_scaler_name,
                residual_model_name,
                mlp_scaler_name,
            )
            horizon_results[suffix] = {
                "target_column": target_column,
                "split_column": split_column,
                "exit_date_column": exit_column,
                "train": _sample_stats(development["date"], train_mask),
                "validation": _sample_stats(
                    development["date"],
                    validation_mask,
                ),
                "oof": {
                    "rows": actual_oof[1],
                    "dates": actual_oof[0],
                    "folds": ridge_result["fold_metadata"],
                },
                "loss_constants": loss_constants,
                "ridge_metrics": ridge_metrics,
                "ridge_monthly": ridge_monthly,
                "stage_c": {
                    "candidate_results": [
                        {
                            "candidate_name": result[
                                "selected_record"
                            ]["candidate_name"],
                            "parameters": result["parameters"],
                            "selected": result["selected_record"],
                            "epoch_records": result["epoch_records"],
                        }
                        for result in stage_c_runs
                    ],
                    "selected": stage_c_selected,
                    "monthly": stage_c_monthly,
                    "gate_reasons": residual_gate_reasons,
                },
                "stage_d": {
                    "executed": rank_stage_executed,
                    "candidate_results": [
                        {
                            "candidate_name": result[
                                "selected_record"
                            ]["candidate_name"],
                            "parameters": result["parameters"],
                            "selected": result["selected_record"],
                            "epoch_records": result["epoch_records"],
                        }
                        for result in stage_d_runs
                    ],
                    "gate_reasons": rank_gate_reasons,
                },
                "stability": stability,
                "development_decision": development_decision,
                "rank_stage_executed": rank_stage_executed,
                "selected": {
                    "candidate_name": selected_record[
                        "candidate_name"
                    ],
                    "parameters": selected_parameters,
                    "gamma": final_gamma,
                    "lambda_rank": final_lambda,
                    "best_epoch": int(selected_record["best_epoch"]),
                    "metrics": final_metrics,
                    "monthly": final_monthly,
                },
                "artifacts": {
                    name: {
                        "sha256": _file_sha256(temporary_path / name)
                    }
                    for name in artifact_names
                },
            }
            restore_inputs.append(
                {
                    "suffix": suffix,
                    "validation_mask": validation_mask,
                    "expected_ridge": final_ridge_predictions,
                    "expected_residual": final_residual_predictions,
                    "expected_final": final_predictions,
                }
            )
            print(
                f"{suffix}: decision={development_decision} "
                f"gamma={final_gamma:g} lambda_rank={final_lambda:g} "
                f"rank_ic={final_metrics['mean_daily_rank_ic']:.6f} "
                f"group_mae={final_metrics['group_mae']:.6f}",
                flush=True,
            )
            del (
                oof_features,
                validation_features,
                stage_c_runs,
                stage_d_runs,
                ridge_result["fold_models"],
                ridge_result["fold_scalers"],
            )
            gc.collect()

        prediction_table = pd.concat(predictions, ignore_index=True)[
            [
                "date",
                "stock_code",
                "split",
                "horizon",
                "y_true",
                "y_pred",
            ]
        ].sort_values(
            ["date", "stock_code", "horizon"],
            ignore_index=True,
        )
        component_table = pd.concat(
            component_predictions,
            ignore_index=True,
        )[
            [
                "date",
                "stock_code",
                "horizon",
                "ridge_pred",
                "residual_pred",
                "gamma",
                "final_pred",
            ]
        ].sort_values(
            ["date", "stock_code", "horizon"],
            ignore_index=True,
        )
        if not np.allclose(
            component_table["final_pred"],
            component_table["ridge_pred"]
            + component_table["gamma"]
            * component_table["residual_pred"],
            rtol=1e-6,
            atol=1e-8,
        ):
            raise RuntimeError("published component identity failed")
        prediction_table.to_parquet(
            temporary_path / "predictions.parquet",
            index=False,
            compression="zstd",
        )
        component_table.to_parquet(
            temporary_path / "component_predictions.parquet",
            index=False,
            compression="zstd",
        )

        for restore_input in restore_inputs:
            _restore_residual_rank_components(
                temporary_path,
                feature_matrix=feature_matrix,
                batch_size=batch_size,
                **restore_input,
            )
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": "residual_rank_mlp",
            "horizons": selected_horizons,
            "development_data": {
                "configured_start_date": config["split"]["train"]["start"],
                "configured_validation_end": config["split"]["validation"][
                    "end"
                ],
                "min_date": development["date"].min().strftime("%Y-%m-%d"),
                "max_date": development["date"].max().strftime("%Y-%m-%d"),
                "rows": int(len(development)),
                "dates": int(development["date"].nunique()),
            },
            "features": {
                "raw": factor_columns,
                "derived": derived_columns,
                "raw_count": len(factor_columns),
                "derived_count": len(derived_columns),
            },
            "preprocessing": {
                "daily_cross_section_first": True,
                "mad_width": float(preprocessing_config["mad_width"]),
                "mad_normalization": 1.4826,
                "missing_fill_value": 0.0,
                "dtype": preprocessing_config["dtype"],
                "ridge_scaler_scope": "fold_train_or_2019_2022_train",
                "mlp_scaler_scope": "2020_2022_purged_oof_rows",
                "target_standardization": False,
            },
            "selection": {
                "primary_metric": "mean_daily_spearman_rank_ic",
                "near_best_rank_ic_tolerance": tolerance,
                "tie_breakers": [
                    "lower_group_mae",
                    "lower_rmse",
                    "smaller_gamma_or_lambda",
                    "earlier_epoch",
                ],
                "monthly_gate": {
                    "minimum_valid_months": 10,
                    "minimum_positive_month_rate": 0.5,
                },
                "rank_gate": {
                    "rmse_max_relative_to_control": 1.01,
                    "prediction_std_max_relative_to_control": 2.0,
                },
            },
            "training": {
                key: training_config[key]
                for key in training_config
                if key != "candidates"
            },
            "candidate_architectures": candidates,
            "horizon_results": horizon_results,
            "read_columns": read_columns,
            "test_data_read": False,
            "uses_entry_tradeable": False,
            "target_standardization": False,
            "prediction_rows": int(len(prediction_table)),
            "sources": {
                "config_sha256": _file_sha256(config_path),
                **lineage,
                "lineage_validation": "published_schema_and_summary",
            },
            "git": _git_state(config_path),
            "environment": _environment_versions(),
        }
        _write_json(temporary_path / "manifest.json", manifest)
        os.rename(temporary_path, final_run_path)
    except BaseException:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise

    print(f"run published: {final_run_path}", flush=True)
    return final_run_path
