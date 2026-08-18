"""OOF selection, calibration, models, and metrics for 10-day context research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


UNKNOWN_INDUSTRY = "UNKNOWN"
OOF_TIE_TOLERANCE = 1e-8


def _normalized_industries(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    missing = values.isna() | values.eq("") | values.str.lower().eq("unknown")
    return values.mask(missing, UNKNOWN_INDUSTRY).astype("string")


@dataclass
class ContextPreprocessor:
    continuous_columns: list[str]
    industry_column: str | None
    industry_categories: list[str]
    scaler: StandardScaler

    @property
    def feature_names(self) -> list[str]:
        if self.industry_column is None:
            return list(self.continuous_columns)
        return [
            *self.continuous_columns,
            *(
                f"{self.industry_column}__{category}"
                for category in self.industry_categories
            ),
        ]

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        scale_continuous: bool,
    ) -> np.ndarray:
        required = set(self.continuous_columns)
        if self.industry_column is not None:
            required.add(self.industry_column)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"context transform is missing columns: {missing}")
        continuous = frame[self.continuous_columns].to_numpy(
            dtype=np.float64, copy=True
        )
        if not np.isfinite(continuous).all():
            raise ValueError("context continuous features must be finite")
        if scale_continuous:
            continuous = self.scaler.transform(continuous)
        if self.industry_column is None:
            return continuous.astype(np.float32, copy=False)
        industries = _normalized_industries(frame[self.industry_column])
        known = set(self.industry_categories)
        industries = industries.where(industries.isin(known), UNKNOWN_INDUSTRY)
        category_positions = {
            category: position
            for position, category in enumerate(self.industry_categories)
        }
        one_hot = np.zeros(
            (len(frame), len(self.industry_categories)), dtype=np.float32
        )
        positions = industries.map(category_positions).to_numpy(dtype=np.int64)
        one_hot[np.arange(len(frame)), positions] = 1.0
        return np.concatenate(
            [continuous.astype(np.float32, copy=False), one_hot], axis=1
        )


def fit_context_preprocessor(
    frame: pd.DataFrame,
    continuous_columns: list[str],
    *,
    industry_column: str | None = "industry",
) -> ContextPreprocessor:
    """Fit continuous scaling and point-in-time industry categories."""
    columns = list(continuous_columns)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("continuous_columns must be non-empty and unique")
    required = set(columns)
    if industry_column is not None:
        required.add(industry_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"context fit is missing columns: {missing}")
    if frame.empty:
        raise ValueError("context preprocessor training frame is empty")
    continuous = frame[columns].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(continuous).all():
        raise ValueError("context continuous training features must be finite")
    scaler = StandardScaler(copy=True).fit(continuous)
    if industry_column is None:
        categories: list[str] = []
    else:
        observed = sorted(
            set(_normalized_industries(frame[industry_column]).astype(str))
            - {UNKNOWN_INDUSTRY}
        )
        categories = [*observed, UNKNOWN_INDUSTRY]
    return ContextPreprocessor(
        continuous_columns=columns,
        industry_column=industry_column,
        industry_categories=categories,
        scaler=scaler,
    )


@dataclass(frozen=True)
class ExpandingFold:
    prediction_year: int
    train_mask: np.ndarray
    predict_mask: np.ndarray
    metadata: dict[str, Any]


def iter_expanding_folds(
    dates: pd.Series,
    exit_dates: pd.Series,
    eligible_mask: np.ndarray,
    *,
    prediction_years: tuple[int, ...] = (2020, 2021, 2022),
) -> list[ExpandingFold]:
    """Create expanding folds whose training targets exit before prediction."""
    normalized_dates = pd.DatetimeIndex(
        pd.to_datetime(dates, errors="raise")
    ).normalize()
    normalized_exits = pd.DatetimeIndex(
        pd.to_datetime(exit_dates, errors="coerce")
    ).normalize()
    eligible = np.asarray(eligible_mask, dtype=bool)
    if len(normalized_dates) != len(normalized_exits) or len(eligible) != len(
        normalized_dates
    ):
        raise ValueError("fold inputs must have equal row counts")
    if (
        not prediction_years
        or tuple(sorted(prediction_years)) != prediction_years
        or len(prediction_years) != len(set(prediction_years))
    ):
        raise ValueError("prediction_years must be unique and increasing")
    if eligible.any() and normalized_dates[eligible].max() > pd.Timestamp("2023-12-31"):
        raise RuntimeError("locked test dates entered OOF development")

    years = normalized_dates.year.to_numpy()
    folds: list[ExpandingFold] = []
    for prediction_year in prediction_years:
        boundary = pd.Timestamp(prediction_year, 1, 1)
        train_mask = (
            eligible
            & (normalized_dates < boundary)
            & (normalized_exits < boundary)
        )
        predict_mask = eligible & (years == prediction_year)
        if not train_mask.any() or not predict_mask.any():
            raise ValueError(
                f"OOF year {prediction_year} has empty train or prediction rows"
            )
        if np.any(train_mask & predict_mask):
            raise RuntimeError("OOF train and prediction rows overlap")
        max_exit = normalized_exits[train_mask].max()
        if pd.isna(max_exit) or max_exit >= boundary:
            raise RuntimeError("OOF exit-date purge failed")
        metadata = {
            "prediction_year": int(prediction_year),
            "train_rows": int(train_mask.sum()),
            "prediction_rows": int(predict_mask.sum()),
            "train_start_date": normalized_dates[train_mask]
            .min()
            .strftime("%Y-%m-%d"),
            "train_end_date": normalized_dates[train_mask]
            .max()
            .strftime("%Y-%m-%d"),
            "max_train_exit_date": max_exit.strftime("%Y-%m-%d"),
            "prediction_start_date": normalized_dates[predict_mask]
            .min()
            .strftime("%Y-%m-%d"),
            "prediction_end_date": normalized_dates[predict_mask]
            .max()
            .strftime("%Y-%m-%d"),
        }
        folds.append(
            ExpandingFold(
                prediction_year=int(prediction_year),
                train_mask=train_mask,
                predict_mask=predict_mask,
                metadata=metadata,
            )
        )
    return folds


def select_oof_candidate(
    records: list[dict[str, Any]],
    *,
    complexity_key: str,
    prefer_larger_on_tie: bool,
) -> dict[str, Any]:
    """Select minimum OOF RMSE with the documented deterministic tie rule."""
    if not records:
        raise ValueError("candidate records must not be empty")
    for record in records:
        if complexity_key not in record or "oof_rmse" not in record:
            raise ValueError("candidate record is missing selection fields")
        if not np.isfinite(float(record["oof_rmse"])):
            raise ValueError("candidate OOF RMSE must be finite")
    best_rmse = min(float(record["oof_rmse"]) for record in records)
    tied = [
        record
        for record in records
        if float(record["oof_rmse"]) <= best_rmse + OOF_TIE_TOLERANCE
    ]
    return dict(
        sorted(
            tied,
            key=lambda record: float(record[complexity_key]),
            reverse=prefer_larger_on_tie,
        )[0]
    )


@dataclass(frozen=True)
class CalibrationResult:
    intercept: float
    slope: float
    use_calibrated: bool

    def calibrated(self, predictions: np.ndarray) -> np.ndarray:
        values = np.asarray(predictions, dtype=np.float64)
        return self.intercept + self.slope * values

    def official(self, predictions: np.ndarray) -> np.ndarray:
        values = np.asarray(predictions, dtype=np.float64)
        return self.calibrated(values) if self.use_calibrated else values.copy()

    def to_dict(self) -> dict[str, float | bool | str]:
        return {
            "intercept": float(self.intercept),
            "slope": float(self.slope),
            "use_calibrated": bool(self.use_calibrated),
            "official_prediction": "calibrated"
            if self.use_calibrated
            else "raw",
        }


def fit_calibrator(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> CalibrationResult:
    """Fit an equal-row-weight OOF affine calibrator."""
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if len(truth) == 0 or len(truth) != len(prediction):
        raise ValueError("calibration inputs must be non-empty and equal length")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("calibration inputs must be finite")
    centered = prediction - float(prediction.mean())
    denominator = float(np.dot(centered, centered))
    if denominator <= np.finfo(np.float64).eps:
        slope = 0.0
    else:
        slope = float(
            np.dot(centered, truth - float(truth.mean())) / denominator
        )
    intercept = float(truth.mean() - slope * prediction.mean())
    return CalibrationResult(
        intercept=intercept,
        slope=slope,
        use_calibrated=bool(np.isfinite(slope) and slope > 0),
    )


@dataclass
class CandidateResult:
    name: str
    candidate_records: list[dict[str, Any]]
    selected_parameters: dict[str, Any]
    oof_predictions_raw: np.ndarray
    oof_mask: np.ndarray
    validation_predictions_raw: np.ndarray
    validation_predictions_calibrated: np.ndarray
    validation_predictions_official: np.ndarray
    calibration: CalibrationResult
    fold_metadata: list[dict[str, Any]]
    final_models: dict[str, Any]
    preprocessors: dict[str, ContextPreprocessor]
    validation_components: pd.DataFrame | None = None
    oof_components: pd.DataFrame | None = None


def _validate_model_masks(
    frame: pd.DataFrame,
    target_column: str,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    *,
    require_validation: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target_column not in frame.columns:
        raise ValueError(f"model frame is missing {target_column}")
    train = np.asarray(train_mask, dtype=bool)
    validation = np.asarray(validation_mask, dtype=bool)
    if len(train) != len(frame) or len(validation) != len(frame):
        raise ValueError("model masks must match frame length")
    if (
        np.any(train & validation)
        or not train.any()
        or (require_validation and not validation.any())
    ):
        raise ValueError("model masks must be nonempty as required and disjoint")
    targets = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(targets[train | validation]).all():
        raise ValueError("selected model targets must be finite")
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if dates.loc[train | validation].max() > pd.Timestamp("2023-12-31"):
        raise RuntimeError("locked test dates entered model development")
    return train, validation, targets


def _weighted_rmse(
    truth: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray | None,
) -> float:
    squared = np.square(
        np.asarray(prediction, dtype=np.float64)
        - np.asarray(truth, dtype=np.float64)
    )
    if weights is None:
        return float(np.sqrt(np.mean(squared)))
    selected_weights = np.asarray(weights, dtype=np.float64)
    if (
        len(selected_weights) != len(squared)
        or not np.isfinite(selected_weights).all()
        or np.any(selected_weights <= 0)
    ):
        raise ValueError("sample weights must be finite and positive")
    return float(np.sqrt(np.average(squared, weights=selected_weights)))


def _run_ridge_table(
    frame: pd.DataFrame,
    *,
    name: str,
    continuous_columns: list[str],
    target_column: str,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    lambda_grid: tuple[float, ...],
    prediction_years: tuple[int, ...],
    sample_weight_column: str | None = None,
    fit_validation: bool = True,
) -> CandidateResult:
    train, validation, targets = _validate_model_masks(
        frame,
        target_column,
        train_mask,
        validation_mask,
        require_validation=fit_validation,
    )
    lambdas = tuple(float(value) for value in lambda_grid)
    if (
        not lambdas
        or len(lambdas) != len(set(lambdas))
        or any(value <= 0 for value in lambdas)
    ):
        raise ValueError("lambda_grid must contain unique positive values")
    weights = None
    if sample_weight_column is not None:
        weights = pd.to_numeric(
            frame[sample_weight_column], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(weights[train | validation]).all() or np.any(
            weights[train | validation] <= 0
        ):
            raise ValueError("selected sample weights must be finite and positive")
    folds = iter_expanding_folds(
        frame["date"],
        frame["exit_date_10d"],
        train,
        prediction_years=prediction_years,
    )
    candidate_predictions = {
        value: np.full(len(frame), np.nan, dtype=np.float64) for value in lambdas
    }
    oof_mask = np.zeros(len(frame), dtype=bool)
    for fold in folds:
        preprocessor = fit_context_preprocessor(
            frame.loc[fold.train_mask],
            continuous_columns,
            industry_column="industry",
        )
        x_train = preprocessor.transform(
            frame.loc[fold.train_mask], scale_continuous=True
        )
        x_predict = preprocessor.transform(
            frame.loc[fold.predict_mask], scale_continuous=True
        )
        fold_weights = weights[fold.train_mask] if weights is not None else None
        for value in lambdas:
            model = Ridge(
                alpha=float(value) * int(fold.train_mask.sum()),
                fit_intercept=True,
                solver="cholesky",
            )
            model.fit(
                x_train,
                targets[fold.train_mask],
                sample_weight=fold_weights,
            )
            candidate_predictions[value][fold.predict_mask] = model.predict(
                x_predict
            )
        oof_mask |= fold.predict_mask
    if not oof_mask.any():
        raise RuntimeError("Ridge OOF produced no predictions")
    candidate_records: list[dict[str, Any]] = []
    for value in lambdas:
        prediction = candidate_predictions[value]
        if not np.isfinite(prediction[oof_mask]).all():
            raise RuntimeError("Ridge OOF prediction coverage failed")
        candidate_records.append(
            {
                "lambda": float(value),
                "oof_rmse": _weighted_rmse(
                    targets[oof_mask],
                    prediction[oof_mask],
                    weights[oof_mask] if weights is not None else None,
                ),
            }
        )
    selected = select_oof_candidate(
        candidate_records,
        complexity_key="lambda",
        prefer_larger_on_tie=True,
    )
    selected_lambda = float(selected["lambda"])
    selected_oof = candidate_predictions[selected_lambda]
    calibration = fit_calibrator(targets[oof_mask], selected_oof[oof_mask])
    if not fit_validation:
        empty = np.empty(0, dtype=np.float64)
        return CandidateResult(
            name=name,
            candidate_records=candidate_records,
            selected_parameters={"lambda": selected_lambda},
            oof_predictions_raw=selected_oof,
            oof_mask=oof_mask,
            validation_predictions_raw=empty,
            validation_predictions_calibrated=empty,
            validation_predictions_official=empty,
            calibration=calibration,
            fold_metadata=[fold.metadata for fold in folds],
            final_models={},
            preprocessors={},
        )
    final_preprocessor = fit_context_preprocessor(
        frame.loc[train], continuous_columns, industry_column="industry"
    )
    x_train = final_preprocessor.transform(
        frame.loc[train], scale_continuous=True
    )
    x_validation = final_preprocessor.transform(
        frame.loc[validation], scale_continuous=True
    )
    final_model = Ridge(
        alpha=selected_lambda * int(train.sum()),
        fit_intercept=True,
        solver="cholesky",
    )
    final_model.fit(
        x_train,
        targets[train],
        sample_weight=weights[train] if weights is not None else None,
    )
    validation_raw = np.asarray(
        final_model.predict(x_validation), dtype=np.float64
    )
    validation_calibrated = calibration.calibrated(validation_raw)
    return CandidateResult(
        name=name,
        candidate_records=candidate_records,
        selected_parameters={"lambda": selected_lambda},
        oof_predictions_raw=selected_oof,
        oof_mask=oof_mask,
        validation_predictions_raw=validation_raw,
        validation_predictions_calibrated=validation_calibrated,
        validation_predictions_official=calibration.official(validation_raw),
        calibration=calibration,
        fold_metadata=[fold.metadata for fold in folds],
        final_models={"model": final_model},
        preprocessors={"preprocessor": final_preprocessor},
    )


def run_context_ridge(
    frame: pd.DataFrame,
    *,
    continuous_columns: list[str],
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    lambda_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
    prediction_years: tuple[int, ...] = (2020, 2021, 2022),
    fit_validation: bool = True,
) -> CandidateResult:
    """Select and fit direct Context Ridge using raw pooled OOF RMSE."""
    return _run_ridge_table(
        frame,
        name="context_ridge_10d",
        continuous_columns=continuous_columns,
        target_column="target_10d",
        train_mask=train_mask,
        validation_mask=validation_mask,
        lambda_grid=lambda_grid,
        prediction_years=prediction_years,
        fit_validation=fit_validation,
    )


def run_context_lightgbm(
    frame: pd.DataFrame,
    *,
    continuous_columns: list[str],
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    rounds_grid: tuple[int, ...] = (10, 20, 40, 80, 120, 200),
    prediction_years: tuple[int, ...] = (2020, 2021, 2022),
    fixed_parameters: dict[str, Any],
    fit_validation: bool = True,
) -> CandidateResult:
    """Select LightGBM rounds from expanding OOF without 2023 early stopping."""
    import lightgbm as lgb

    train, validation, targets = _validate_model_masks(
        frame,
        "target_10d",
        train_mask,
        validation_mask,
        require_validation=fit_validation,
    )
    rounds = tuple(int(value) for value in rounds_grid)
    if (
        not rounds
        or tuple(sorted(rounds)) != rounds
        or len(rounds) != len(set(rounds))
        or any(value <= 0 for value in rounds)
    ):
        raise ValueError("rounds_grid must contain unique increasing values")
    folds = iter_expanding_folds(
        frame["date"],
        frame["exit_date_10d"],
        train,
        prediction_years=prediction_years,
    )
    candidate_predictions = {
        value: np.full(len(frame), np.nan, dtype=np.float64) for value in rounds
    }
    oof_mask = np.zeros(len(frame), dtype=bool)
    parameters = {**fixed_parameters, "feature_pre_filter": False}
    for fold in folds:
        preprocessor = fit_context_preprocessor(
            frame.loc[fold.train_mask],
            continuous_columns,
            industry_column="industry",
        )
        x_train = preprocessor.transform(
            frame.loc[fold.train_mask], scale_continuous=False
        )
        x_predict = preprocessor.transform(
            frame.loc[fold.predict_mask], scale_continuous=False
        )
        train_set = lgb.Dataset(
            x_train,
            label=targets[fold.train_mask],
            feature_name=preprocessor.feature_names,
            free_raw_data=True,
            params={"feature_pre_filter": False},
        )
        booster = lgb.train(
            parameters,
            train_set,
            num_boost_round=max(rounds),
            callbacks=[lgb.log_evaluation(period=0)],
        )
        for value in rounds:
            candidate_predictions[value][fold.predict_mask] = booster.predict(
                x_predict, num_iteration=value
            )
        oof_mask |= fold.predict_mask
    candidate_records: list[dict[str, Any]] = []
    for value in rounds:
        prediction = candidate_predictions[value]
        if not np.isfinite(prediction[oof_mask]).all():
            raise RuntimeError("LightGBM OOF prediction coverage failed")
        candidate_records.append(
            {
                "rounds": int(value),
                "oof_rmse": _weighted_rmse(
                    targets[oof_mask], prediction[oof_mask], None
                ),
            }
        )
    selected = select_oof_candidate(
        candidate_records,
        complexity_key="rounds",
        prefer_larger_on_tie=False,
    )
    selected_rounds = int(selected["rounds"])
    selected_oof = candidate_predictions[selected_rounds]
    calibration = fit_calibrator(targets[oof_mask], selected_oof[oof_mask])
    if not fit_validation:
        empty = np.empty(0, dtype=np.float64)
        return CandidateResult(
            name="context_lightgbm_10d",
            candidate_records=candidate_records,
            selected_parameters={"rounds": selected_rounds, **fixed_parameters},
            oof_predictions_raw=selected_oof,
            oof_mask=oof_mask,
            validation_predictions_raw=empty,
            validation_predictions_calibrated=empty,
            validation_predictions_official=empty,
            calibration=calibration,
            fold_metadata=[fold.metadata for fold in folds],
            final_models={},
            preprocessors={},
        )
    final_preprocessor = fit_context_preprocessor(
        frame.loc[train], continuous_columns, industry_column="industry"
    )
    x_train = final_preprocessor.transform(
        frame.loc[train], scale_continuous=False
    )
    x_validation = final_preprocessor.transform(
        frame.loc[validation], scale_continuous=False
    )
    final_train_set = lgb.Dataset(
        x_train,
        label=targets[train],
        feature_name=final_preprocessor.feature_names,
        free_raw_data=True,
        params={"feature_pre_filter": False},
    )
    final_model = lgb.train(
        parameters,
        final_train_set,
        num_boost_round=selected_rounds,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    validation_raw = np.asarray(
        final_model.predict(x_validation, num_iteration=selected_rounds),
        dtype=np.float64,
    )
    return CandidateResult(
        name="context_lightgbm_10d",
        candidate_records=candidate_records,
        selected_parameters={"rounds": selected_rounds, **fixed_parameters},
        oof_predictions_raw=selected_oof,
        oof_mask=oof_mask,
        validation_predictions_raw=validation_raw,
        validation_predictions_calibrated=calibration.calibrated(validation_raw),
        validation_predictions_official=calibration.official(validation_raw),
        calibration=calibration,
        fold_metadata=[fold.metadata for fold in folds],
        final_models={"model": final_model},
        preprocessors={"preprocessor": final_preprocessor},
    )


def _component_prediction_frame(
    frame: pd.DataFrame,
    result: CandidateResult,
    *,
    validation_mask: np.ndarray,
    prediction_column: str,
    key_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = frame.loc[result.oof_mask, key_columns].copy()
    oof[prediction_column] = result.oof_predictions_raw[result.oof_mask]
    validation = frame.loc[validation_mask, key_columns].copy()
    validation[prediction_column] = result.validation_predictions_raw
    return oof, validation


def run_decomposed_ridge(
    frame: pd.DataFrame,
    *,
    market_feature_columns: list[str],
    industry_feature_columns: list[str],
    alpha_feature_columns: list[str],
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    lambda_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
    prediction_years: tuple[int, ...] = (2020, 2021, 2022),
    fit_validation: bool = True,
    locked_lambdas: dict[str, float] | None = None,
) -> CandidateResult:
    """Fit market, industry, and alpha Ridge components with exact recovery."""
    train, validation, targets = _validate_model_masks(
        frame,
        "target_10d",
        train_mask,
        validation_mask,
        require_validation=fit_validation,
    )
    selected = train | validation
    valid = selected & np.isfinite(targets)
    base_columns = [
        "date",
        "stock_code",
        "industry",
        "exit_date_10d",
        "split_10d",
    ]
    market_table = (
        frame.loc[
            valid & frame["market_target_10d"].notna().to_numpy(),
            [*base_columns, *market_feature_columns, "market_target_10d"],
        ]
        .sort_values(["date", "stock_code"])
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    market_table["stock_code"] = "__MARKET__"
    market_table["industry"] = "MARKET"
    industry_valid = (
        valid
        & frame["industry_target_10d"].notna().to_numpy()
        & frame["industry_target_count"].gt(0).to_numpy()
        & ~_normalized_industries(frame["industry"]).eq(UNKNOWN_INDUSTRY).to_numpy()
    )
    industry_table = (
        frame.loc[
            industry_valid,
            [
                *base_columns,
                *industry_feature_columns,
                "industry_target_10d",
                "industry_target_count",
            ],
        ]
        .sort_values(["date", "industry", "stock_code"])
        .drop_duplicates(["date", "industry"])
        .reset_index(drop=True)
    )
    industry_table["stock_code"] = industry_table["industry"]
    alpha_table = frame.loc[
        valid & frame["alpha_target_10d"].notna().to_numpy(),
        [*base_columns, *alpha_feature_columns, "alpha_target_10d"],
    ].reset_index(drop=True)

    def masks(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (
            table["split_10d"].eq("train").to_numpy(),
            table["split_10d"].eq("validation").to_numpy(),
        )

    market_train, market_validation = masks(market_table)
    industry_train, industry_validation = masks(industry_table)
    alpha_train, alpha_validation = masks(alpha_table)
    market_result = _run_ridge_table(
        market_table,
        name="market_ridge",
        continuous_columns=market_feature_columns,
        target_column="market_target_10d",
        train_mask=market_train,
        validation_mask=market_validation,
        lambda_grid=(float(locked_lambdas["market"]),)
        if locked_lambdas is not None
        else lambda_grid,
        prediction_years=prediction_years,
        fit_validation=fit_validation,
    )
    industry_result = _run_ridge_table(
        industry_table,
        name="industry_ridge",
        continuous_columns=industry_feature_columns,
        target_column="industry_target_10d",
        train_mask=industry_train,
        validation_mask=industry_validation,
        lambda_grid=(float(locked_lambdas["industry"]),)
        if locked_lambdas is not None
        else lambda_grid,
        prediction_years=prediction_years,
        sample_weight_column="industry_target_count",
        fit_validation=fit_validation,
    )
    alpha_result = _run_ridge_table(
        alpha_table,
        name="alpha_ridge",
        continuous_columns=alpha_feature_columns,
        target_column="alpha_target_10d",
        train_mask=alpha_train,
        validation_mask=alpha_validation,
        lambda_grid=(float(locked_lambdas["alpha"]),)
        if locked_lambdas is not None
        else lambda_grid,
        prediction_years=prediction_years,
        fit_validation=fit_validation,
    )

    market_oof, market_validation_frame = _component_prediction_frame(
        market_table,
        market_result,
        validation_mask=market_validation,
        prediction_column="market_pred",
        key_columns=["date"],
    )
    industry_oof, industry_validation_frame = _component_prediction_frame(
        industry_table,
        industry_result,
        validation_mask=industry_validation,
        prediction_column="industry_pred",
        key_columns=["date", "industry"],
    )
    alpha_oof, alpha_validation_frame = _component_prediction_frame(
        alpha_table,
        alpha_result,
        validation_mask=alpha_validation,
        prediction_column="alpha_pred",
        key_columns=["date", "stock_code"],
    )

    def assemble(
        stock_mask: np.ndarray,
        market_predictions: pd.DataFrame,
        industry_predictions: pd.DataFrame,
        alpha_predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        components = frame.loc[
            stock_mask,
            ["date", "stock_code", "industry", "target_10d"],
        ].copy()
        components = components.merge(
            market_predictions, on="date", how="left", validate="many_to_one"
        )
        components = components.merge(
            industry_predictions,
            on=["date", "industry"],
            how="left",
            validate="many_to_one",
        )
        components = components.merge(
            alpha_predictions,
            on=["date", "stock_code"],
            how="left",
            validate="one_to_one",
        )
        unknown = _normalized_industries(components["industry"]).eq(
            UNKNOWN_INDUSTRY
        )
        components.loc[unknown, "industry_pred"] = 0.0
        required = ["market_pred", "industry_pred", "alpha_pred"]
        if components[required].isna().any().any():
            raise RuntimeError("decomposed component prediction coverage failed")
        components["final_pred_raw"] = components[required].sum(axis=1)
        reconstructed = components[required].sum(axis=1)
        if not np.allclose(
            components["final_pred_raw"], reconstructed, rtol=0, atol=1e-12
        ):
            raise RuntimeError("decomposed prediction identity failed")
        return components.sort_values(["date", "stock_code"]).reset_index(drop=True)

    oof_stock_mask = np.zeros(len(frame), dtype=bool)
    oof_years = pd.to_datetime(frame["date"]).dt.year.isin(prediction_years).to_numpy()
    oof_stock_mask = train & oof_years & np.isfinite(targets)
    validation_stock_mask = validation & np.isfinite(targets)
    oof_components = assemble(
        oof_stock_mask, market_oof, industry_oof, alpha_oof
    )
    validation_components = (
        assemble(
            validation_stock_mask,
            market_validation_frame,
            industry_validation_frame,
            alpha_validation_frame,
        )
        if fit_validation
        else None
    )
    calibration = fit_calibrator(
        oof_components["target_10d"].to_numpy(),
        oof_components["final_pred_raw"].to_numpy(),
    )
    if validation_components is None:
        validation_raw = np.empty(0, dtype=np.float64)
    else:
        validation_raw = validation_components["final_pred_raw"].to_numpy(
            dtype=np.float64
        )
        validation_components["final_pred_calibrated"] = calibration.calibrated(
            validation_raw
        )
        validation_components["final_pred_official"] = calibration.official(
            validation_raw
        )
    combined_oof = np.full(len(frame), np.nan, dtype=np.float64)
    oof_key_predictions = oof_components.set_index(["date", "stock_code"])[
        "final_pred_raw"
    ]
    selected_keys = pd.MultiIndex.from_frame(frame.loc[oof_stock_mask, ["date", "stock_code"]])
    combined_oof[oof_stock_mask] = oof_key_predictions.reindex(selected_keys).to_numpy()
    return CandidateResult(
        name="decomposed_ridge_10d",
        candidate_records=[
            *({**record, "component": "market"} for record in market_result.candidate_records),
            *({**record, "component": "industry"} for record in industry_result.candidate_records),
            *({**record, "component": "alpha"} for record in alpha_result.candidate_records),
        ],
        selected_parameters={
            "market": market_result.selected_parameters,
            "industry": industry_result.selected_parameters,
            "alpha": alpha_result.selected_parameters,
        },
        oof_predictions_raw=combined_oof,
        oof_mask=oof_stock_mask,
        validation_predictions_raw=validation_raw,
        validation_predictions_calibrated=calibration.calibrated(validation_raw),
        validation_predictions_official=calibration.official(validation_raw),
        calibration=calibration,
        fold_metadata=market_result.fold_metadata,
        final_models={
            "market": market_result.final_models["model"],
            "industry": industry_result.final_models["model"],
            "alpha": alpha_result.final_models["model"],
        }
        if fit_validation
        else {},
        preprocessors={
            "market": market_result.preprocessors["preprocessor"],
            "industry": industry_result.preprocessors["preprocessor"],
            "alpha": alpha_result.preprocessors["preprocessor"],
        }
        if fit_validation
        else {},
        validation_components=validation_components,
        oof_components=oof_components,
    )


def _predict_torch_model(model: Any, features: np.ndarray) -> np.ndarray:
    import torch

    values = np.asarray(features, dtype=np.float32)
    predictions = np.empty(len(values), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), 8192):
            end = min(start + 8192, len(values))
            batch = torch.from_numpy(np.ascontiguousarray(values[start:end]))
            predictions[start:end] = (
                model(batch).squeeze(-1).detach().cpu().numpy()
            )
    return predictions.astype(np.float64)


def _fit_fixed_residual_network(
    train_features: np.ndarray,
    train_dates: pd.Series,
    residual_targets: np.ndarray,
    predict_features: np.ndarray,
    *,
    hidden_layers: list[int],
    dropout: float,
    weight_decay: float,
    epochs: int,
    seed: int,
    huber_mad_multiplier: float,
    learning_rate: float,
    dates_per_batch: int,
    gradient_clip_norm: float,
    num_threads: int,
) -> tuple[Any, np.ndarray, float]:
    import torch

    from model.stages.training import build_mlp_model

    residuals = np.asarray(residual_targets, dtype=np.float64)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    delta = float(huber_mad_multiplier * 1.4826 * mad)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("fixed residual MLP requires positive training residual MAD")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(num_threads)
    model = build_mlp_model(
        train_features.shape[1],
        hidden_layers,
        dropout,
        output_bias=float(residuals.mean()),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_x = torch.from_numpy(
        np.ascontiguousarray(train_features, dtype=np.float32)
    )
    train_y = torch.from_numpy(
        np.ascontiguousarray(residuals, dtype=np.float32)
    )
    normalized_dates = pd.to_datetime(train_dates, errors="raise").dt.normalize()
    date_indices = [
        group.index.to_numpy(dtype=np.int64)
        for _, group in pd.DataFrame(
            {"date": normalized_dates.to_numpy(), "position": np.arange(len(train_x))}
        )
        .set_index("position", drop=True)
        .groupby("date", sort=False)
    ]
    for epoch in range(1, epochs + 1):
        model.train()
        generator = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        shuffled = generator.permutation(len(date_indices))
        for start in range(0, len(shuffled), dates_per_batch):
            selected_dates = shuffled[start : start + dates_per_batch]
            positions = np.concatenate(
                [date_indices[int(index)] for index in selected_dates]
            )
            position_tensor = torch.from_numpy(positions)
            optimizer.zero_grad(set_to_none=True)
            output = model(train_x[position_tensor]).squeeze(-1)
            loss = torch.nn.functional.huber_loss(
                output,
                train_y[position_tensor],
                reduction="mean",
                delta=delta,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
    prediction = _predict_torch_model(model, predict_features)
    return model, prediction, delta


def run_fixed_residual_mlp(
    frame: pd.DataFrame,
    *,
    continuous_columns: list[str],
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    target_column: str,
    ridge_lambda: float,
    prediction_years: tuple[int, ...] = (2020, 2021, 2022),
    name: str,
    hidden_layers: list[int] | None = None,
    dropout: float = 0.3,
    weight_decay: float = 0.001,
    epochs: int = 6,
    seed: int = 42,
    huber_mad_multiplier: float = 1.5,
    learning_rate: float = 0.001,
    dates_per_batch: int = 8,
    gradient_clip_norm: float = 1.0,
    num_threads: int = 4,
    include_industry: bool = True,
    fit_validation: bool = True,
) -> CandidateResult:
    """Fit locked Ridge plus return-only residual MLP across expanding folds."""
    layers = [128, 64, 32] if hidden_layers is None else list(hidden_layers)
    if epochs <= 0 or ridge_lambda <= 0:
        raise ValueError("fixed residual epochs and ridge_lambda must be positive")
    train, validation, targets = _validate_model_masks(
        frame,
        target_column,
        train_mask,
        validation_mask,
        require_validation=fit_validation,
    )
    folds = iter_expanding_folds(
        frame["date"],
        frame["exit_date_10d"],
        train,
        prediction_years=prediction_years,
    )
    oof_predictions = np.full(len(frame), np.nan, dtype=np.float64)
    oof_mask = np.zeros(len(frame), dtype=bool)
    fold_deltas: list[float] = []
    for fold in folds:
        preprocessor = fit_context_preprocessor(
            frame.loc[fold.train_mask],
            continuous_columns,
            industry_column="industry" if include_industry else None,
        )
        x_train = preprocessor.transform(
            frame.loc[fold.train_mask], scale_continuous=True
        )
        x_predict = preprocessor.transform(
            frame.loc[fold.predict_mask], scale_continuous=True
        )
        ridge = Ridge(
            alpha=float(ridge_lambda) * int(fold.train_mask.sum()),
            fit_intercept=True,
            solver="cholesky",
        )
        ridge.fit(x_train, targets[fold.train_mask])
        train_ridge = np.asarray(ridge.predict(x_train), dtype=np.float64)
        predict_ridge = np.asarray(ridge.predict(x_predict), dtype=np.float64)
        _, residual_prediction, delta = _fit_fixed_residual_network(
            x_train,
            frame.loc[fold.train_mask, "date"].reset_index(drop=True),
            targets[fold.train_mask] - train_ridge,
            x_predict,
            hidden_layers=layers,
            dropout=float(dropout),
            weight_decay=float(weight_decay),
            epochs=int(epochs),
            seed=int(seed),
            huber_mad_multiplier=float(huber_mad_multiplier),
            learning_rate=float(learning_rate),
            dates_per_batch=int(dates_per_batch),
            gradient_clip_norm=float(gradient_clip_norm),
            num_threads=int(num_threads),
        )
        oof_predictions[fold.predict_mask] = predict_ridge + residual_prediction
        oof_mask |= fold.predict_mask
        fold_deltas.append(delta)
    if not np.isfinite(oof_predictions[oof_mask]).all():
        raise RuntimeError("fixed residual OOF prediction coverage failed")
    calibration = fit_calibrator(targets[oof_mask], oof_predictions[oof_mask])

    common_parameters = {
        "ridge_lambda": float(ridge_lambda),
        "hidden_layers": layers,
        "dropout": float(dropout),
        "weight_decay": float(weight_decay),
        "gamma": 1.0,
        "lambda_rank": 0.0,
        "epochs": int(epochs),
        "seed": int(seed),
        "target_standardization": False,
        "include_industry": bool(include_industry),
        "huber_mad_multiplier": float(huber_mad_multiplier),
        "fold_huber_delta": fold_deltas,
    }
    if not fit_validation:
        empty = np.empty(0, dtype=np.float64)
        return CandidateResult(
            name=name,
            candidate_records=[
                {
                    "epochs": int(epochs),
                    "oof_rmse": _weighted_rmse(
                        targets[oof_mask], oof_predictions[oof_mask], None
                    ),
                }
            ],
            selected_parameters={**common_parameters, "final_huber_delta": None},
            oof_predictions_raw=oof_predictions,
            oof_mask=oof_mask,
            validation_predictions_raw=empty,
            validation_predictions_calibrated=empty,
            validation_predictions_official=empty,
            calibration=calibration,
            fold_metadata=[fold.metadata for fold in folds],
            final_models={},
            preprocessors={},
        )

    final_preprocessor = fit_context_preprocessor(
        frame.loc[train],
        continuous_columns,
        industry_column="industry" if include_industry else None,
    )
    x_train = final_preprocessor.transform(
        frame.loc[train], scale_continuous=True
    )
    x_validation = final_preprocessor.transform(
        frame.loc[validation], scale_continuous=True
    )
    final_ridge = Ridge(
        alpha=float(ridge_lambda) * int(train.sum()),
        fit_intercept=True,
        solver="cholesky",
    )
    final_ridge.fit(x_train, targets[train])
    train_ridge = np.asarray(final_ridge.predict(x_train), dtype=np.float64)
    validation_ridge = np.asarray(
        final_ridge.predict(x_validation), dtype=np.float64
    )
    final_mlp, validation_residual, final_delta = _fit_fixed_residual_network(
        x_train,
        frame.loc[train, "date"].reset_index(drop=True),
        targets[train] - train_ridge,
        x_validation,
        hidden_layers=layers,
        dropout=float(dropout),
        weight_decay=float(weight_decay),
        epochs=int(epochs),
        seed=int(seed),
        huber_mad_multiplier=float(huber_mad_multiplier),
        learning_rate=float(learning_rate),
        dates_per_batch=int(dates_per_batch),
        gradient_clip_norm=float(gradient_clip_norm),
        num_threads=int(num_threads),
    )
    validation_raw = validation_ridge + validation_residual
    return CandidateResult(
        name=name,
        candidate_records=[
            {
                "epochs": int(epochs),
                "oof_rmse": _weighted_rmse(
                    targets[oof_mask], oof_predictions[oof_mask], None
                ),
            }
        ],
        selected_parameters={
            **common_parameters,
            "final_huber_delta": float(final_delta),
        },
        oof_predictions_raw=oof_predictions,
        oof_mask=oof_mask,
        validation_predictions_raw=validation_raw,
        validation_predictions_calibrated=calibration.calibrated(validation_raw),
        validation_predictions_official=calibration.official(validation_raw),
        calibration=calibration,
        fold_metadata=[fold.metadata for fold in folds],
        final_models={"ridge": final_ridge, "mlp": final_mlp},
        preprocessors={"preprocessor": final_preprocessor},
    )


def run_regenerated_champion_oof(
    frame: pd.DataFrame,
    *,
    continuous_columns: list[str],
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    ridge_lambda: float = 1.0,
    prediction_years: tuple[int, ...] = (2020, 2021, 2022),
    hidden_layers: list[int] | None = None,
    dropout: float = 0.3,
    weight_decay: float = 0.001,
    epochs: int = 6,
    seed: int = 42,
    huber_mad_multiplier: float = 1.5,
    learning_rate: float = 0.001,
    dates_per_batch: int = 8,
    gradient_clip_norm: float = 1.0,
    num_threads: int = 4,
    fit_validation: bool = True,
) -> CandidateResult:
    """Recreate the locked factor-only champion with nested Ridge OOF residuals."""
    layers = [128, 64, 32] if hidden_layers is None else list(hidden_layers)
    train, validation, targets = _validate_model_masks(
        frame,
        "target_10d",
        train_mask,
        validation_mask,
        require_validation=fit_validation,
    )
    if ridge_lambda <= 0 or epochs <= 0:
        raise ValueError("champion ridge_lambda and epochs must be positive")

    def fit_ridge(
        fit_mask: np.ndarray,
        predict_mask: np.ndarray,
    ) -> tuple[ContextPreprocessor, Ridge, np.ndarray]:
        preprocessor = fit_context_preprocessor(
            frame.loc[fit_mask],
            continuous_columns,
            industry_column=None,
        )
        x_fit = preprocessor.transform(
            frame.loc[fit_mask], scale_continuous=True
        )
        x_predict = preprocessor.transform(
            frame.loc[predict_mask], scale_continuous=True
        )
        model = Ridge(
            alpha=float(ridge_lambda) * int(fit_mask.sum()),
            fit_intercept=True,
            solver="cholesky",
        )
        model.fit(x_fit, targets[fit_mask])
        return (
            preprocessor,
            model,
            np.asarray(model.predict(x_predict), dtype=np.float64),
        )

    outer_folds = iter_expanding_folds(
        frame["date"],
        frame["exit_date_10d"],
        train,
        prediction_years=prediction_years,
    )
    nested_predictions = np.full(len(frame), np.nan, dtype=np.float64)
    ridge_oof_predictions = np.full(len(frame), np.nan, dtype=np.float64)
    oof_mask = np.zeros(len(frame), dtype=bool)
    fold_metadata: list[dict[str, Any]] = []
    fold_deltas: list[float | None] = []
    for outer in outer_folds:
        _, _, outer_ridge = fit_ridge(outer.train_mask, outer.predict_mask)
        ridge_oof_predictions[outer.predict_mask] = outer_ridge
        prior_years = tuple(
            year for year in prediction_years if year < outer.prediction_year
        )
        metadata = dict(outer.metadata)
        if not prior_years:
            residual_prediction = np.zeros_like(outer_ridge)
            metadata["residual_training_years"] = []
            metadata["residual_train_rows"] = 0
            fold_deltas.append(None)
        else:
            inner_predictions = np.full(len(frame), np.nan, dtype=np.float64)
            inner_mask = np.zeros(len(frame), dtype=bool)
            inner_folds = iter_expanding_folds(
                frame["date"],
                frame["exit_date_10d"],
                outer.train_mask,
                prediction_years=prior_years,
            )
            for inner in inner_folds:
                _, _, prediction = fit_ridge(
                    inner.train_mask, inner.predict_mask
                )
                inner_predictions[inner.predict_mask] = prediction
                inner_mask |= inner.predict_mask
            if not np.isfinite(inner_predictions[inner_mask]).all():
                raise RuntimeError("nested champion Ridge OOF coverage failed")
            residual_preprocessor = fit_context_preprocessor(
                frame.loc[inner_mask],
                continuous_columns,
                industry_column=None,
            )
            residual_features = residual_preprocessor.transform(
                frame.loc[inner_mask], scale_continuous=True
            )
            outer_features = residual_preprocessor.transform(
                frame.loc[outer.predict_mask], scale_continuous=True
            )
            _, residual_prediction, delta = _fit_fixed_residual_network(
                residual_features,
                frame.loc[inner_mask, "date"].reset_index(drop=True),
                targets[inner_mask] - inner_predictions[inner_mask],
                outer_features,
                hidden_layers=layers,
                dropout=float(dropout),
                weight_decay=float(weight_decay),
                epochs=int(epochs),
                seed=int(seed),
                huber_mad_multiplier=float(huber_mad_multiplier),
                learning_rate=float(learning_rate),
                dates_per_batch=int(dates_per_batch),
                gradient_clip_norm=float(gradient_clip_norm),
                num_threads=int(num_threads),
            )
            metadata["residual_training_years"] = list(prior_years)
            metadata["residual_train_rows"] = int(inner_mask.sum())
            fold_deltas.append(float(delta))
        nested_predictions[outer.predict_mask] = (
            outer_ridge + residual_prediction
        )
        oof_mask |= outer.predict_mask
        fold_metadata.append(metadata)

    if not np.isfinite(nested_predictions[oof_mask]).all():
        raise RuntimeError("nested champion OOF prediction coverage failed")
    calibration = fit_calibrator(
        targets[oof_mask], nested_predictions[oof_mask]
    )

    selected_parameters = {
        "ridge_lambda": float(ridge_lambda),
        "hidden_layers": layers,
        "dropout": float(dropout),
        "weight_decay": float(weight_decay),
        "gamma": 1.0,
        "lambda_rank": 0.0,
        "epochs": int(epochs),
        "seed": int(seed),
        "target_standardization": False,
        "nested_ridge_oof_residuals": True,
        "fold_huber_delta": fold_deltas,
    }
    if not fit_validation:
        empty = np.empty(0, dtype=np.float64)
        return CandidateResult(
            name="regenerated_champion_oof",
            candidate_records=[
                {
                    "epochs": int(epochs),
                    "oof_rmse": _weighted_rmse(
                        targets[oof_mask], nested_predictions[oof_mask], None
                    ),
                }
            ],
            selected_parameters={
                **selected_parameters,
                "final_huber_delta": None,
            },
            oof_predictions_raw=nested_predictions,
            oof_mask=oof_mask,
            validation_predictions_raw=empty,
            validation_predictions_calibrated=empty,
            validation_predictions_official=empty,
            calibration=calibration,
            fold_metadata=fold_metadata,
            final_models={},
            preprocessors={},
        )

    final_ridge_preprocessor, final_ridge, validation_ridge = fit_ridge(
        train, validation
    )
    residual_preprocessor = fit_context_preprocessor(
        frame.loc[oof_mask],
        continuous_columns,
        industry_column=None,
    )
    residual_train_features = residual_preprocessor.transform(
        frame.loc[oof_mask], scale_continuous=True
    )
    residual_validation_features = residual_preprocessor.transform(
        frame.loc[validation], scale_continuous=True
    )
    final_mlp, validation_residual, final_delta = _fit_fixed_residual_network(
        residual_train_features,
        frame.loc[oof_mask, "date"].reset_index(drop=True),
        targets[oof_mask] - ridge_oof_predictions[oof_mask],
        residual_validation_features,
        hidden_layers=layers,
        dropout=float(dropout),
        weight_decay=float(weight_decay),
        epochs=int(epochs),
        seed=int(seed),
        huber_mad_multiplier=float(huber_mad_multiplier),
        learning_rate=float(learning_rate),
        dates_per_batch=int(dates_per_batch),
        gradient_clip_norm=float(gradient_clip_norm),
        num_threads=int(num_threads),
    )
    validation_raw = validation_ridge + validation_residual
    return CandidateResult(
        name="regenerated_champion_oof",
        candidate_records=[
            {
                "epochs": int(epochs),
                "oof_rmse": _weighted_rmse(
                    targets[oof_mask], nested_predictions[oof_mask], None
                ),
            }
        ],
        selected_parameters={
            **selected_parameters,
            "final_huber_delta": float(final_delta),
        },
        oof_predictions_raw=nested_predictions,
        oof_mask=oof_mask,
        validation_predictions_raw=validation_raw,
        validation_predictions_calibrated=calibration.calibrated(validation_raw),
        validation_predictions_official=calibration.official(validation_raw),
        calibration=calibration,
        fold_metadata=fold_metadata,
        final_models={"ridge": final_ridge, "mlp": final_mlp},
        preprocessors={
            "ridge": final_ridge_preprocessor,
            "residual": residual_preprocessor,
        },
    )


def stage_two_gate(
    stage_one_results: list[CandidateResult],
    champion_result: CandidateResult,
    targets: np.ndarray,
) -> dict[str, Any]:
    """Gate Stage 2 exclusively on aligned raw pooled OOF RMSE."""
    if not stage_one_results:
        raise ValueError("stage_one_results must not be empty")
    truth = np.asarray(targets, dtype=np.float64)
    champion_mask = champion_result.oof_mask
    if len(truth) != len(champion_mask):
        raise ValueError("gate targets and champion rows differ")
    champion_rmse = _weighted_rmse(
        truth[champion_mask],
        champion_result.oof_predictions_raw[champion_mask],
        None,
    )
    candidate_values: dict[str, float] = {}
    for result in stage_one_results:
        if not np.array_equal(result.oof_mask, champion_mask):
            raise ValueError("gate candidates and champion OOF masks differ")
        candidate_values[result.name] = _weighted_rmse(
            truth[champion_mask],
            result.oof_predictions_raw[champion_mask],
            None,
        )
    best_name = min(candidate_values, key=candidate_values.get)
    best_rmse = float(candidate_values[best_name])
    return {
        "metric": "raw_oof_rmse",
        "passed": bool(best_rmse < champion_rmse),
        "champion_oof_rmse": float(champion_rmse),
        "stage_one_best_model": best_name,
        "stage_one_best_oof_rmse": best_rmse,
        "stage_one_oof_rmse": candidate_values,
    }


def compare_oof_by_year(
    frame: pd.DataFrame,
    candidate: CandidateResult,
    champion: CandidateResult,
) -> dict[str, Any]:
    """Compare aligned raw OOF RMSE separately for each prediction year."""
    if not np.array_equal(candidate.oof_mask, champion.oof_mask):
        raise ValueError("candidate and champion OOF masks differ")
    mask = candidate.oof_mask
    targets = pd.to_numeric(frame["target_10d"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    years = pd.to_datetime(frame["date"], errors="raise").dt.year.to_numpy()
    diagnostics: dict[str, Any] = {}
    for year in sorted(set(years[mask])):
        selected = mask & (years == year)
        candidate_rmse = _weighted_rmse(
            targets[selected],
            candidate.oof_predictions_raw[selected],
            None,
        )
        champion_rmse = _weighted_rmse(
            targets[selected],
            champion.oof_predictions_raw[selected],
            None,
        )
        diagnostics[str(int(year))] = {
            "rows": int(selected.sum()),
            "candidate_rmse": candidate_rmse,
            "champion_rmse": champion_rmse,
            "rmse_difference": float(candidate_rmse - champion_rmse),
            "improved": bool(candidate_rmse < champion_rmse),
        }
    return {
        "years": diagnostics,
        "all_years_improved": bool(
            diagnostics and all(item["improved"] for item in diagnostics.values())
        ),
    }


def run_decomposed_residual_mlp(
    frame: pd.DataFrame,
    *,
    base_result: CandidateResult,
    alpha_feature_columns: list[str],
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    prediction_years: tuple[int, ...] = (2020, 2021, 2022),
    epochs: int = 6,
    fit_validation: bool = True,
) -> CandidateResult:
    """Keep market/industry Ridge predictions and add an Alpha residual MLP."""
    if (
        base_result.oof_components is None
        or base_result.name != "decomposed_ridge_10d"
        or (fit_validation and base_result.validation_components is None)
    ):
        raise ValueError("base_result must be a completed decomposed Ridge result")
    alpha_parameters = base_result.selected_parameters.get("alpha")
    if not isinstance(alpha_parameters, dict) or "lambda" not in alpha_parameters:
        raise ValueError("decomposed Ridge has no selected alpha lambda")
    train = np.asarray(train_mask, dtype=bool)
    validation = np.asarray(validation_mask, dtype=bool)
    alpha_values = pd.to_numeric(
        frame["alpha_target_10d"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    alpha_train = train & np.isfinite(alpha_values)
    alpha_validation = validation & np.isfinite(alpha_values)
    alpha_result = run_fixed_residual_mlp(
        frame,
        continuous_columns=alpha_feature_columns,
        train_mask=alpha_train,
        validation_mask=alpha_validation,
        target_column="alpha_target_10d",
        ridge_lambda=float(alpha_parameters["lambda"]),
        prediction_years=prediction_years,
        name="alpha_residual_mlp",
        epochs=epochs,
        fit_validation=fit_validation,
    )
    oof_alpha = frame.loc[
        alpha_result.oof_mask, ["date", "stock_code"]
    ].copy()
    oof_alpha["alpha_pred_enhanced"] = alpha_result.oof_predictions_raw[
        alpha_result.oof_mask
    ]
    validation_alpha = frame.loc[
        alpha_validation, ["date", "stock_code"]
    ].copy()
    validation_alpha["alpha_pred_enhanced"] = (
        alpha_result.validation_predictions_raw
    )

    def replace_alpha(
        base_components: pd.DataFrame,
        enhanced_alpha: pd.DataFrame,
    ) -> pd.DataFrame:
        result = base_components.drop(
            columns=[
                column
                for column in (
                    "alpha_pred",
                    "final_pred_raw",
                    "final_pred_calibrated",
                    "final_pred_official",
                )
                if column in base_components.columns
            ]
        ).merge(
            enhanced_alpha,
            on=["date", "stock_code"],
            how="left",
            validate="one_to_one",
        )
        if result["alpha_pred_enhanced"].isna().any():
            raise RuntimeError("enhanced alpha prediction coverage failed")
        result = result.rename(columns={"alpha_pred_enhanced": "alpha_pred"})
        result["final_pred_raw"] = (
            result["market_pred"]
            + result["industry_pred"]
            + result["alpha_pred"]
        )
        if not np.allclose(
            result["final_pred_raw"],
            result["market_pred"]
            + result["industry_pred"]
            + result["alpha_pred"],
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError("enhanced decomposition identity failed")
        return result.sort_values(["date", "stock_code"]).reset_index(drop=True)

    oof_components = replace_alpha(base_result.oof_components, oof_alpha)
    validation_components = (
        replace_alpha(base_result.validation_components, validation_alpha)
        if fit_validation
        else None
    )
    calibration = fit_calibrator(
        oof_components["target_10d"].to_numpy(dtype=np.float64),
        oof_components["final_pred_raw"].to_numpy(dtype=np.float64),
    )
    if validation_components is None:
        validation_raw = np.empty(0, dtype=np.float64)
    else:
        validation_raw = validation_components["final_pred_raw"].to_numpy(
            dtype=np.float64
        )
        validation_components["final_pred_calibrated"] = calibration.calibrated(
            validation_raw
        )
        validation_components["final_pred_official"] = calibration.official(
            validation_raw
        )
    combined_oof = np.full(len(frame), np.nan, dtype=np.float64)
    oof_mask = np.zeros(len(frame), dtype=bool)
    oof_keys = pd.MultiIndex.from_frame(
        oof_components[["date", "stock_code"]]
    )
    frame_keys = pd.MultiIndex.from_frame(frame[["date", "stock_code"]])
    positions = frame_keys.get_indexer(oof_keys)
    if np.any(positions < 0):
        raise RuntimeError("enhanced OOF keys are missing from source data")
    combined_oof[positions] = oof_components["final_pred_raw"].to_numpy()
    oof_mask[positions] = True
    return CandidateResult(
        name="decomposed_residual_mlp_10d",
        candidate_records=alpha_result.candidate_records,
        selected_parameters={
            "market": base_result.selected_parameters["market"],
            "industry": base_result.selected_parameters["industry"],
            "alpha_residual": alpha_result.selected_parameters,
        },
        oof_predictions_raw=combined_oof,
        oof_mask=oof_mask,
        validation_predictions_raw=validation_raw,
        validation_predictions_calibrated=calibration.calibrated(validation_raw),
        validation_predictions_official=calibration.official(validation_raw),
        calibration=calibration,
        fold_metadata=alpha_result.fold_metadata,
        final_models={
            "market": base_result.final_models["market"],
            "industry": base_result.final_models["industry"],
            "alpha_ridge": alpha_result.final_models["ridge"],
            "alpha_mlp": alpha_result.final_models["mlp"],
        }
        if fit_validation
        else {},
        preprocessors={
            "market": base_result.preprocessors["market"],
            "industry": base_result.preprocessors["industry"],
            "alpha": alpha_result.preprocessors["preprocessor"],
        }
        if fit_validation
        else {},
        validation_components=validation_components,
        oof_components=oof_components,
    )


def _resolve_path(config_path: Path, raw_path: str) -> Path:
    return (Path(config_path).resolve().parent / raw_path).resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _verified_joblib_load(path: Path) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        return joblib.load(path)


def _stable_unique(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _load_context_development(
    context_path: Path,
    schema_path: Path,
    *,
    maximum_date: str = "2023-12-31",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not context_path.exists() or not schema_path.exists():
        raise FileNotFoundError("prepare-context must run before context training")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("context_dataset_sha256") != _file_sha256(context_path):
        raise ValueError("context dataset hash does not match its schema")
    if schema.get("development_end") != "2023-12-31":
        raise ValueError("context dataset must end at 2023-12-31")
    if schema.get("uses_entry_tradeable") is not False:
        raise ValueError("context dataset must not use entry_tradeable")
    maximum = pd.Timestamp(maximum_date).normalize()
    if maximum > pd.Timestamp("2023-12-31"):
        raise ValueError("context maximum_date cannot exceed 2023-12-31")
    data = pd.read_parquet(
        context_path,
        filters=[("date", "<=", maximum.to_pydatetime())],
    )
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    data["stock_code"] = data["stock_code"].astype("string").str.strip()
    if (
        data.empty
        or data["date"].max() > maximum
        or data.duplicated(["date", "stock_code"]).any()
    ):
        raise ValueError("context development keys or date boundary are invalid")
    forbidden = [column for column in data.columns if column.startswith("entry_")]
    if forbidden:
        raise ValueError(f"context features contain entry fields: {forbidden}")
    return data.sort_values(["date", "stock_code"]).reset_index(drop=True), schema


def _load_champion_validation(
    champion_run: Path,
    validation_keys: pd.DataFrame,
    validation_targets: np.ndarray,
) -> pd.DataFrame:
    prediction_path = champion_run / "predictions.parquet"
    manifest_path = champion_run / "manifest.json"
    if not prediction_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"champion run is incomplete: {champion_run}")
    champion = pd.read_parquet(prediction_path)
    required = {"date", "stock_code", "horizon", "y_true", "y_pred"}
    missing = sorted(required.difference(champion.columns))
    if missing:
        raise ValueError(f"champion predictions are missing columns: {missing}")
    champion["date"] = pd.to_datetime(
        champion["date"], errors="raise"
    ).dt.normalize()
    champion["stock_code"] = champion["stock_code"].astype("string").str.strip()
    if champion["stock_code"].isna().any() or champion["stock_code"].eq("").any():
        raise ValueError("champion predictions contain missing stock codes")
    champion = champion.loc[
        champion["horizon"].eq(10)
        & champion["date"].between("2023-01-01", "2023-12-31"),
        ["date", "stock_code", "y_true", "y_pred"],
    ].sort_values(["date", "stock_code"]).reset_index(drop=True)
    keys = validation_keys.sort_values(["date", "stock_code"]).reset_index(drop=True)
    if not champion[["date", "stock_code"]].equals(keys):
        raise ValueError("champion and context validation keys differ")
    if not np.allclose(
        champion["y_true"].to_numpy(dtype=np.float64),
        np.asarray(validation_targets, dtype=np.float64),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("champion and context validation targets differ")
    return champion


def _result_prediction_tables(
    data: pd.DataFrame,
    result: CandidateResult,
    validation_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = pd.to_numeric(data["target_10d"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    oof = data.loc[result.oof_mask, ["date", "stock_code"]].copy()
    oof["split"] = "oof"
    oof["horizon"] = 10
    oof["y_true"] = target[result.oof_mask]
    oof["y_pred_raw"] = result.oof_predictions_raw[result.oof_mask]
    oof["y_pred_calibrated"] = result.calibration.calibrated(
        result.oof_predictions_raw[result.oof_mask]
    )

    validation_keys = data.loc[
        validation_mask, ["date", "stock_code", "target_10d"]
    ].sort_values(["date", "stock_code"])
    if result.validation_components is None:
        raw_by_key = pd.DataFrame(
            {
                "date": data.loc[validation_mask, "date"].to_numpy(),
                "stock_code": data.loc[validation_mask, "stock_code"].to_numpy(),
                "y_pred_raw": result.validation_predictions_raw,
                "y_pred_calibrated": result.validation_predictions_calibrated,
            }
        )
    else:
        components = result.validation_components
        raw_by_key = components[
            [
                "date",
                "stock_code",
                "final_pred_raw",
                "final_pred_calibrated",
            ]
        ].rename(
            columns={
                "final_pred_raw": "y_pred_raw",
                "final_pred_calibrated": "y_pred_calibrated",
            }
        )
    validation = validation_keys.merge(
        raw_by_key,
        on=["date", "stock_code"],
        how="left",
        validate="one_to_one",
    )
    if validation[["y_pred_raw", "y_pred_calibrated"]].isna().any().any():
        raise RuntimeError("candidate validation prediction coverage failed")
    validation = validation.rename(columns={"target_10d": "y_true"})
    validation["split"] = "validation"
    validation["horizon"] = 10
    columns = [
        "date",
        "stock_code",
        "split",
        "horizon",
        "y_true",
        "y_pred_raw",
        "y_pred_calibrated",
    ]
    predictions = pd.concat([oof[columns], validation[columns]], ignore_index=True)
    predictions = predictions.sort_values(
        ["split", "date", "stock_code"]
    ).reset_index(drop=True)
    return predictions, validation[columns].sort_values(
        ["date", "stock_code"]
    ).reset_index(drop=True)


def _save_model_artifacts(
    candidate_path: Path,
    result: CandidateResult,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for name, preprocessor in result.preprocessors.items():
        filename = f"preprocessor_{name}.joblib"
        joblib.dump(preprocessor, candidate_path / filename, compress=3)
        _verified_joblib_load(candidate_path / filename)
        artifacts[f"preprocessor_{name}"] = filename
    for name, model in result.final_models.items():
        if hasattr(model, "save_model") and model.__class__.__module__.startswith(
            "lightgbm"
        ):
            filename = f"model_{name}.txt"
            model_text = model.model_to_string()
            (candidate_path / filename).write_text(model_text, encoding="utf-8")
            import lightgbm as lgb

            lgb.Booster(
                model_str=(candidate_path / filename).read_text(encoding="utf-8")
            )
        elif model.__class__.__module__.startswith("torch"):
            import torch

            filename = f"model_{name}.pt"
            torch.save(model.state_dict(), candidate_path / filename)
            torch.load(candidate_path / filename, map_location="cpu", weights_only=True)
        else:
            filename = f"model_{name}.joblib"
            joblib.dump(model, candidate_path / filename, compress=3)
            _verified_joblib_load(candidate_path / filename)
        artifacts[f"model_{name}"] = filename
    joblib.dump(result.calibration, candidate_path / "calibrator.joblib", compress=3)
    loaded_calibration = _verified_joblib_load(
        candidate_path / "calibrator.joblib"
    )
    if loaded_calibration != result.calibration:
        raise RuntimeError("reloaded calibrator differs from the fitted calibrator")
    artifacts["calibrator"] = "calibrator.joblib"
    return artifacts


def _load_model_artifacts(
    candidate_path: Path,
    result: CandidateResult,
    artifact_map: dict[str, str],
) -> tuple[dict[str, ContextPreprocessor], dict[str, Any], CalibrationResult]:
    """Reload every published object, including reconstructing PyTorch models."""
    loaded_preprocessors = {
        name: _verified_joblib_load(
            candidate_path / artifact_map[f"preprocessor_{name}"]
        )
        for name in result.preprocessors
    }
    loaded_models: dict[str, Any] = {}
    for name, original in result.final_models.items():
        path = candidate_path / artifact_map[f"model_{name}"]
        if original.__class__.__module__.startswith("lightgbm"):
            import lightgbm as lgb

            loaded_models[name] = lgb.Booster(
                model_str=path.read_text(encoding="utf-8")
            )
        elif original.__class__.__module__.startswith("torch"):
            import torch

            from model.stages.training import build_mlp_model

            if name == "alpha_mlp":
                parameters = result.selected_parameters["alpha_residual"]
                preprocessor = loaded_preprocessors["alpha"]
            else:
                parameters = result.selected_parameters
                preprocessor = loaded_preprocessors["preprocessor"]
            model = build_mlp_model(
                len(preprocessor.feature_names),
                list(parameters["hidden_layers"]),
                float(parameters["dropout"]),
                output_bias=0.0,
            )
            model.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )
            model.eval()
            loaded_models[name] = model
        else:
            loaded_models[name] = _verified_joblib_load(path)
    loaded_calibration = _verified_joblib_load(
        candidate_path / artifact_map["calibrator"]
    )
    if not isinstance(loaded_calibration, CalibrationResult):
        raise RuntimeError("reloaded calibrator has an unexpected type")
    return loaded_preprocessors, loaded_models, loaded_calibration


def _verify_reloaded_validation_predictions(
    candidate_path: Path,
    data: pd.DataFrame,
    result: CandidateResult,
    validation_mask: np.ndarray,
    artifact_map: dict[str, str],
) -> dict[str, Any]:
    """Recompute 2023 predictions exclusively from the published artifacts."""
    preprocessors, models, calibration = _load_model_artifacts(
        candidate_path, result, artifact_map
    )
    validation = data.loc[validation_mask].sort_values(
        ["date", "stock_code"]
    ).reset_index(drop=True)

    def transformed(name: str, frame: pd.DataFrame, *, scale: bool) -> np.ndarray:
        return preprocessors[name].transform(frame, scale_continuous=scale)

    if result.name == "context_ridge_10d":
        raw = np.asarray(
            models["model"].predict(
                transformed("preprocessor", validation, scale=True)
            ),
            dtype=np.float64,
        )
    elif result.name == "context_lightgbm_10d":
        raw = np.asarray(
            models["model"].predict(
                transformed("preprocessor", validation, scale=False)
            ),
            dtype=np.float64,
        )
    elif result.name == "context_residual_mlp_10d":
        features = transformed("preprocessor", validation, scale=True)
        raw = np.asarray(models["ridge"].predict(features), dtype=np.float64)
        raw += _predict_torch_model(models["mlp"], features)
    elif result.name in {
        "decomposed_ridge_10d",
        "decomposed_residual_mlp_10d",
    }:
        market_rows = (
            validation.sort_values(["date", "stock_code"])
            .drop_duplicates("date")
            .reset_index(drop=True)
        )
        market_rows["industry"] = "MARKET"
        market_prediction = pd.DataFrame(
            {
                "date": market_rows["date"],
                "market_pred": models["market"].predict(
                    transformed("market", market_rows, scale=True)
                ),
            }
        )
        known_industry = ~_normalized_industries(validation["industry"]).eq(
            UNKNOWN_INDUSTRY
        )
        industry_rows = (
            validation.loc[known_industry]
            .sort_values(["date", "industry", "stock_code"])
            .drop_duplicates(["date", "industry"])
            .reset_index(drop=True)
        )
        industry_prediction = pd.DataFrame(
            {
                "date": industry_rows["date"],
                "industry": industry_rows["industry"],
                "industry_pred": models["industry"].predict(
                    transformed("industry", industry_rows, scale=True)
                ),
            }
        )
        alpha_features = transformed("alpha", validation, scale=True)
        if result.name == "decomposed_ridge_10d":
            alpha_prediction = np.asarray(
                models["alpha"].predict(alpha_features), dtype=np.float64
            )
        else:
            alpha_prediction = np.asarray(
                models["alpha_ridge"].predict(alpha_features), dtype=np.float64
            )
            alpha_prediction += _predict_torch_model(
                models["alpha_mlp"], alpha_features
            )
        components = validation[["date", "stock_code", "industry"]].copy()
        components["alpha_pred"] = alpha_prediction
        components = components.merge(
            market_prediction, on="date", how="left", validate="many_to_one"
        ).merge(
            industry_prediction,
            on=["date", "industry"],
            how="left",
            validate="many_to_one",
        )
        unknown = _normalized_industries(components["industry"]).eq(
            UNKNOWN_INDUSTRY
        )
        components.loc[unknown, "industry_pred"] = 0.0
        if components[["market_pred", "industry_pred", "alpha_pred"]].isna().any().any():
            raise RuntimeError("reloaded decomposition has missing predictions")
        raw = (
            components["market_pred"]
            + components["industry_pred"]
            + components["alpha_pred"]
        ).to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"unsupported artifact verification model: {result.name}")

    expected_raw = np.asarray(result.validation_predictions_raw, dtype=np.float64)
    calibrated = calibration.calibrated(raw)
    expected_calibrated = np.asarray(
        result.validation_predictions_calibrated, dtype=np.float64
    )
    raw_difference = float(np.max(np.abs(raw - expected_raw)))
    calibrated_difference = float(
        np.max(np.abs(calibrated - expected_calibrated))
    )
    raw_matches = bool(
        np.allclose(raw, expected_raw, rtol=1e-6, atol=1e-8)
    )
    calibrated_matches = bool(
        np.allclose(
            calibrated, expected_calibrated, rtol=1e-6, atol=1e-8
        )
    )
    if not raw_matches or not calibrated_matches:
        raise RuntimeError(
            "reloaded artifacts do not reproduce validation predictions: "
            f"raw={raw_difference}, calibrated={calibrated_difference}"
        )
    component_identity = None
    if result.validation_components is not None:
        component_identity = float(
            np.max(
                np.abs(
                    result.validation_components["final_pred_raw"].to_numpy(
                        dtype=np.float64
                    )
                    - result.validation_components[
                        ["market_pred", "industry_pred", "alpha_pred"]
                    ].sum(axis=1).to_numpy(dtype=np.float64)
                )
            )
        )
        if component_identity > 1e-12:
            raise RuntimeError("published decomposition identity failed")
    return {
        "objects_reloaded": True,
        "validation_recomputed": True,
        "max_abs_raw_prediction_difference": raw_difference,
        "max_abs_calibrated_prediction_difference": calibrated_difference,
        "prediction_tolerance": {"rtol": 1e-6, "atol": 1e-8},
        "raw_predictions_match": raw_matches,
        "calibrated_predictions_match": calibrated_matches,
        "component_identity_max_abs_difference": component_identity,
    }


def _candidate_acceptance(
    candidate_metrics: dict[str, Any],
    champion_metrics: dict[str, Any],
    comparison: dict[str, Any],
    calibration: CalibrationResult,
    min_prediction_std_ratio: float,
    oof_yearly_comparison: dict[str, Any],
) -> dict[str, Any]:
    rmse_improved = candidate_metrics["rmse"] < champion_metrics["rmse"]
    mae_not_worse = candidate_metrics["mae"] <= champion_metrics["mae"]
    robust_candidate = candidate_metrics["robust_rmse_abs_target_le_50pct"]
    robust_champion = champion_metrics["robust_rmse_abs_target_le_50pct"]
    not_outlier_only = (
        robust_candidate is not None
        and robust_champion is not None
        and robust_candidate < robust_champion
    )
    std_ratio = candidate_metrics["prediction_std"] / candidate_metrics["target_std"]
    nonconstant = bool(std_ratio >= min_prediction_std_ratio)
    hac_upper = comparison["daily_mse_difference_hac"]["ci95_upper"]
    oof_years_all_improved = bool(
        oof_yearly_comparison["all_years_improved"]
    )
    strong = bool(
        rmse_improved
        and mae_not_worse
        and comparison["mse_skill"] is not None
        and comparison["mse_skill"] > 0
        and hac_upper < 0
        and not_outlier_only
        and nonconstant
        and calibration.slope > 0
        and oof_years_all_improved
    )
    tentative = bool(
        rmse_improved
        and mae_not_worse
        and oof_years_all_improved
        and not strong
    )
    return {
        "status": "strong_accept" if strong else "tentative" if tentative else "reject",
        "rmse_improved": rmse_improved,
        "mae_not_worse": mae_not_worse,
        "not_outlier_only": not_outlier_only,
        "prediction_std_ratio": float(std_ratio),
        "nonconstant": nonconstant,
        "positive_oof_calibration_slope": bool(calibration.slope > 0),
        "hac_ci_upper_below_zero": bool(hac_upper < 0),
        "oof_years_all_improved": oof_years_all_improved,
    }


def _publish_candidate(
    candidate_path: Path,
    data: pd.DataFrame,
    result: CandidateResult,
    validation_mask: np.ndarray,
    champion_validation: pd.DataFrame,
    champion_oof: CandidateResult,
    schema: dict[str, Any],
    *,
    hac_lag: int,
    min_prediction_std_ratio: float,
    test_contamination_disclosure: str,
) -> dict[str, Any]:
    candidate_path.mkdir(parents=True, exist_ok=False)
    predictions, validation = _result_prediction_tables(
        data, result, validation_mask
    )
    predictions.to_parquet(
        candidate_path / "predictions.parquet",
        index=False,
        compression="zstd",
    )
    official_column = (
        "y_pred_calibrated" if result.calibration.use_calibrated else "y_pred_raw"
    )
    official = validation[["date", "stock_code", "y_true"]].copy()
    official["y_pred"] = validation[official_column].to_numpy()
    champion = champion_validation[
        ["date", "stock_code", "y_true", "y_pred"]
    ].copy()
    raw_frame = validation[["date", "stock_code", "y_true"]].copy()
    raw_frame["y_pred"] = validation["y_pred_raw"].to_numpy()
    calibrated_frame = validation[["date", "stock_code", "y_true"]].copy()
    calibrated_frame["y_pred"] = validation["y_pred_calibrated"].to_numpy()
    metrics = {
        "raw": return_metrics(raw_frame),
        "calibrated": return_metrics(calibrated_frame),
        "official": return_metrics(official),
    }
    champion_metrics = return_metrics(champion)
    comparison = compare_to_champion(official, champion, hac_lag=hac_lag)
    oof_yearly_comparison = compare_oof_by_year(
        data, result, champion_oof
    )
    acceptance = _candidate_acceptance(
        metrics["official"],
        champion_metrics,
        comparison,
        result.calibration,
        min_prediction_std_ratio,
        oof_yearly_comparison,
    )
    if result.validation_components is not None:
        validation_components = result.validation_components.copy()
        if "final_pred_calibrated" not in validation_components:
            validation_components["final_pred_calibrated"] = (
                result.calibration.calibrated(
                    validation_components["final_pred_raw"].to_numpy()
                )
            )
        validation_components[
            [
                "date",
                "stock_code",
                "market_pred",
                "industry_pred",
                "alpha_pred",
                "final_pred_raw",
                "final_pred_calibrated",
            ]
        ].to_parquet(
            candidate_path / "component_predictions.parquet",
            index=False,
            compression="zstd",
        )
    feature_schema = {
        "industry_column": schema["industry_column"]
        if "industry_column" in schema
        else "industry",
        "factor_feature_columns": schema["factor_feature_columns"],
        "market_feature_columns": schema["market_feature_columns"],
        "industry_feature_columns": schema["industry_feature_columns"],
        "continuous_feature_columns": schema["continuous_feature_columns"],
        "model_feature_names": {
            name: preprocessor.feature_names
            for name, preprocessor in result.preprocessors.items()
        },
    }
    _write_json(candidate_path / "feature_schema.json", feature_schema)
    summary_columns = [
        column
        for column in schema["continuous_feature_columns"]
        if column.endswith("__missing")
        or column.endswith("_coverage")
        or column.endswith("_fallback")
    ]
    context_summary = {
        "rows": int(len(data)),
        "dates": int(data["date"].nunique()),
        "feature_diagnostics": {
            column: {
                "mean": float(data[column].mean()),
                "min": float(data[column].min()),
                "max": float(data[column].max()),
            }
            for column in summary_columns
        },
    }
    _write_json(
        candidate_path / "context_feature_summary.json", context_summary
    )
    artifact_map = _save_model_artifacts(candidate_path, result)
    artifact_verification = _verify_reloaded_validation_predictions(
        candidate_path,
        data,
        result,
        validation_mask,
        artifact_map,
    )
    artifact_sha256 = {
        path.name: _file_sha256(path)
        for path in sorted(candidate_path.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "model_name": result.name,
        "horizon": 10,
        "selection_metric": "raw_pooled_oof_rmse",
        "candidate_records": result.candidate_records,
        "selected_parameters": result.selected_parameters,
        "folds": result.fold_metadata,
        "calibration": result.calibration.to_dict(),
        "official_prediction": official_column,
        "validation_metrics": metrics,
        "champion_metrics": champion_metrics,
        "comparison_to_champion": comparison,
        "oof_yearly_comparison": oof_yearly_comparison,
        "acceptance": acceptance,
        "artifacts": artifact_map,
        "artifact_sha256": artifact_sha256,
        "artifact_verification": artifact_verification,
        "development_end": schema["development_end"],
        "context_dataset_sha256": schema["context_dataset_sha256"],
        "context_sources": schema.get("sources", {}),
        "uses_entry_tradeable": False,
        "test_contamination_disclosure": test_contamination_disclosure,
        "prediction_rows": int(len(predictions)),
    }
    _write_json(candidate_path / "manifest.json", manifest)
    return manifest


def _frozen_oof_reproduction_difference(
    selection_data: pd.DataFrame,
    frozen: CandidateResult,
    development_data: pd.DataFrame,
    finalized: CandidateResult,
) -> float:
    """Verify final runners reproduce the OOF predictions frozen before 2023."""
    expected = selection_data.loc[
        frozen.oof_mask, ["date", "stock_code"]
    ].copy()
    expected["expected"] = frozen.oof_predictions_raw[frozen.oof_mask]
    actual = development_data.loc[
        finalized.oof_mask, ["date", "stock_code"]
    ].copy()
    actual["actual"] = finalized.oof_predictions_raw[finalized.oof_mask]
    compared = expected.merge(
        actual,
        on=["date", "stock_code"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not compared["_merge"].eq("both").all():
        raise RuntimeError("frozen and finalized OOF keys differ")
    difference = float(
        np.max(np.abs(compared["expected"] - compared["actual"]))
    )
    if not np.allclose(
        compared["expected"], compared["actual"], rtol=1e-6, atol=1e-8
    ):
        raise RuntimeError(
            f"finalized OOF differs from frozen OOF: {difference}"
        )
    return difference


def run_context_experiment(
    config: dict[str, Any],
    config_path: Path,
    *,
    horizons: list[int],
    run_id: str,
) -> Path:
    """Run the complete gated 10-day Context/Decomposition experiment."""
    from model.stages.training import validate_run_id

    if horizons != [10]:
        raise ValueError("context_decomposition only supports horizon 10")
    validate_run_id(run_id)
    config_path = Path(config_path).resolve()
    context_path = _resolve_path(
        config_path, config["paths"]["context_dataset"]
    )
    schema_path = _resolve_path(config_path, config["paths"]["context_schema"])
    runs_dir = _resolve_path(config_path, config["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    final_run_path = runs_dir / run_id
    if final_run_path.exists():
        raise FileExistsError(f"run already exists: {final_run_path}")
    selection_data, schema = _load_context_development(
        context_path, schema_path, maximum_date="2022-12-31"
    )
    selection_target = pd.to_numeric(
        selection_data["target_10d"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    selection_train = (
        selection_data["split_10d"]
        .eq("train")
        .fillna(False)
        .to_numpy(dtype=bool)
        & np.isfinite(selection_target)
    )
    no_validation = np.zeros(len(selection_data), dtype=bool)
    if not selection_train.any():
        raise ValueError("context experiment has no pre-2023 training rows")
    direct_features = list(schema["continuous_feature_columns"])
    factor_features = list(schema["factor_feature_columns"])
    market_features = list(schema["market_feature_columns"])
    industry_features = _stable_unique(
        [*market_features, *schema["industry_feature_columns"]]
    )
    alpha_features = direct_features
    model_config = config["models"]["context_decomposition"]
    lambda_grid = tuple(float(value) for value in model_config["lambda_grid"])
    rounds_grid = tuple(int(value) for value in model_config["rounds_grid"])
    prediction_years = tuple(
        int(value) for value in model_config["prediction_years"]
    )
    lightgbm_parameters = {
        "objective": model_config["lightgbm_objective"],
        "metric": model_config["lightgbm_metric"],
        "learning_rate": float(model_config["lightgbm_learning_rate"]),
        "num_leaves": int(model_config["lightgbm_num_leaves"]),
        "max_depth": int(model_config["lightgbm_max_depth"]),
        "min_data_in_leaf": int(model_config["lightgbm_min_data_in_leaf"]),
        "bagging_fraction": float(model_config["lightgbm_bagging_fraction"]),
        "bagging_freq": int(model_config["lightgbm_bagging_freq"]),
        "feature_fraction": float(model_config["lightgbm_feature_fraction"]),
        "lambda_l2": float(model_config["lightgbm_lambda_l2"]),
        "lambda_l1": float(model_config["lightgbm_lambda_l1"]),
        "seed": int(model_config["lightgbm_seed"]),
        "data_random_seed": int(model_config["lightgbm_seed"]),
        "feature_fraction_seed": int(model_config["lightgbm_seed"]),
        "bagging_seed": int(model_config["lightgbm_seed"]),
        "num_threads": int(model_config["lightgbm_num_threads"]),
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    print("context experiment: Stage 1 OOF", flush=True)
    frozen_context_ridge = run_context_ridge(
        selection_data,
        continuous_columns=direct_features,
        train_mask=selection_train,
        validation_mask=no_validation,
        lambda_grid=lambda_grid,
        prediction_years=prediction_years,
        fit_validation=False,
    )
    frozen_context_lightgbm = run_context_lightgbm(
        selection_data,
        continuous_columns=direct_features,
        train_mask=selection_train,
        validation_mask=no_validation,
        rounds_grid=rounds_grid,
        prediction_years=prediction_years,
        fixed_parameters=lightgbm_parameters,
        fit_validation=False,
    )
    frozen_decomposed_ridge = run_decomposed_ridge(
        selection_data,
        market_feature_columns=market_features,
        industry_feature_columns=industry_features,
        alpha_feature_columns=alpha_features,
        train_mask=selection_train,
        validation_mask=no_validation,
        lambda_grid=lambda_grid,
        prediction_years=prediction_years,
        fit_validation=False,
    )
    frozen_champion_oof = run_regenerated_champion_oof(
        selection_data,
        continuous_columns=factor_features,
        train_mask=selection_train,
        validation_mask=no_validation,
        ridge_lambda=1.0,
        prediction_years=prediction_years,
        hidden_layers=list(model_config["mlp_hidden_layers"]),
        dropout=float(model_config["mlp_dropout"]),
        weight_decay=float(model_config["mlp_weight_decay"]),
        epochs=int(model_config["mlp_epochs"]),
        seed=int(model_config["mlp_seed"]),
        huber_mad_multiplier=float(model_config["mlp_huber_mad_multiplier"]),
        learning_rate=float(model_config["mlp_learning_rate"]),
        dates_per_batch=int(model_config["mlp_dates_per_batch"]),
        gradient_clip_norm=float(model_config["mlp_gradient_clip_norm"]),
        num_threads=int(model_config["mlp_num_threads"]),
        fit_validation=False,
    )
    frozen_stage_one = [
        frozen_context_ridge,
        frozen_context_lightgbm,
        frozen_decomposed_ridge,
    ]
    gate = stage_two_gate(
        frozen_stage_one, frozen_champion_oof, selection_target
    )
    frozen_candidates = list(frozen_stage_one)
    if gate["passed"]:
        print("context experiment: Stage 2 gate passed", flush=True)
        frozen_candidates.append(
            run_fixed_residual_mlp(
                selection_data,
                continuous_columns=direct_features,
                train_mask=selection_train,
                validation_mask=no_validation,
                target_column="target_10d",
                ridge_lambda=float(
                    frozen_context_ridge.selected_parameters["lambda"]
                ),
                prediction_years=prediction_years,
                name="context_residual_mlp_10d",
                hidden_layers=list(model_config["mlp_hidden_layers"]),
                dropout=float(model_config["mlp_dropout"]),
                weight_decay=float(model_config["mlp_weight_decay"]),
                epochs=int(model_config["mlp_epochs"]),
                seed=int(model_config["mlp_seed"]),
                huber_mad_multiplier=float(
                    model_config["mlp_huber_mad_multiplier"]
                ),
                learning_rate=float(model_config["mlp_learning_rate"]),
                dates_per_batch=int(model_config["mlp_dates_per_batch"]),
                gradient_clip_norm=float(
                    model_config["mlp_gradient_clip_norm"]
                ),
                num_threads=int(model_config["mlp_num_threads"]),
                fit_validation=False,
            )
        )
        frozen_candidates.append(
            run_decomposed_residual_mlp(
                selection_data,
                base_result=frozen_decomposed_ridge,
                alpha_feature_columns=alpha_features,
                train_mask=selection_train,
                validation_mask=no_validation,
                prediction_years=prediction_years,
                epochs=int(model_config["mlp_epochs"]),
                fit_validation=False,
            )
        )
    else:
        print("context experiment: Stage 2 gate rejected", flush=True)

    print("context experiment: candidates frozen; loading 2023", flush=True)
    data, full_schema = _load_context_development(
        context_path, schema_path, maximum_date="2023-12-31"
    )
    if full_schema != schema:
        raise RuntimeError("context schema changed between selection and validation")
    target = pd.to_numeric(data["target_10d"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    train_mask = (
        data["split_10d"].eq("train").fillna(False).to_numpy(dtype=bool)
        & np.isfinite(target)
    )
    validation_mask = (
        data["split_10d"]
        .eq("validation")
        .fillna(False)
        .to_numpy(dtype=bool)
        & np.isfinite(target)
    )
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("context experiment has no train or validation rows")

    context_ridge = run_context_ridge(
        data,
        continuous_columns=direct_features,
        train_mask=train_mask,
        validation_mask=validation_mask,
        lambda_grid=(
            float(frozen_context_ridge.selected_parameters["lambda"]),
        ),
        prediction_years=prediction_years,
    )
    context_lightgbm = run_context_lightgbm(
        data,
        continuous_columns=direct_features,
        train_mask=train_mask,
        validation_mask=validation_mask,
        rounds_grid=(
            int(frozen_context_lightgbm.selected_parameters["rounds"]),
        ),
        prediction_years=prediction_years,
        fixed_parameters=lightgbm_parameters,
    )
    locked_lambdas = {
        component: float(
            frozen_decomposed_ridge.selected_parameters[component]["lambda"]
        )
        for component in ("market", "industry", "alpha")
    }
    decomposed_ridge = run_decomposed_ridge(
        data,
        market_feature_columns=market_features,
        industry_feature_columns=industry_features,
        alpha_feature_columns=alpha_features,
        train_mask=train_mask,
        validation_mask=validation_mask,
        lambda_grid=lambda_grid,
        prediction_years=prediction_years,
        locked_lambdas=locked_lambdas,
    )
    candidates = [context_ridge, context_lightgbm, decomposed_ridge]
    if gate["passed"]:
        candidates.append(
            run_fixed_residual_mlp(
                data,
                continuous_columns=direct_features,
                train_mask=train_mask,
                validation_mask=validation_mask,
                target_column="target_10d",
                ridge_lambda=float(
                    frozen_context_ridge.selected_parameters["lambda"]
                ),
                prediction_years=prediction_years,
                name="context_residual_mlp_10d",
                hidden_layers=list(model_config["mlp_hidden_layers"]),
                dropout=float(model_config["mlp_dropout"]),
                weight_decay=float(model_config["mlp_weight_decay"]),
                epochs=int(model_config["mlp_epochs"]),
                seed=int(model_config["mlp_seed"]),
                huber_mad_multiplier=float(
                    model_config["mlp_huber_mad_multiplier"]
                ),
                learning_rate=float(model_config["mlp_learning_rate"]),
                dates_per_batch=int(model_config["mlp_dates_per_batch"]),
                gradient_clip_norm=float(
                    model_config["mlp_gradient_clip_norm"]
                ),
                num_threads=int(model_config["mlp_num_threads"]),
            )
        )
        candidates.append(
            run_decomposed_residual_mlp(
                data,
                base_result=decomposed_ridge,
                alpha_feature_columns=alpha_features,
                train_mask=train_mask,
                validation_mask=validation_mask,
                prediction_years=prediction_years,
                epochs=int(model_config["mlp_epochs"]),
            )
        )

    frozen_by_name = {result.name: result for result in frozen_candidates}
    oof_reproduction = {
        result.name: _frozen_oof_reproduction_difference(
            selection_data,
            frozen_by_name[result.name],
            data,
            result,
        )
        for result in candidates
    }
    champion_oof = run_regenerated_champion_oof(
        data,
        continuous_columns=factor_features,
        train_mask=train_mask,
        validation_mask=np.zeros(len(data), dtype=bool),
        ridge_lambda=1.0,
        prediction_years=prediction_years,
        hidden_layers=list(model_config["mlp_hidden_layers"]),
        dropout=float(model_config["mlp_dropout"]),
        weight_decay=float(model_config["mlp_weight_decay"]),
        epochs=int(model_config["mlp_epochs"]),
        seed=int(model_config["mlp_seed"]),
        huber_mad_multiplier=float(model_config["mlp_huber_mad_multiplier"]),
        learning_rate=float(model_config["mlp_learning_rate"]),
        dates_per_batch=int(model_config["mlp_dates_per_batch"]),
        gradient_clip_norm=float(model_config["mlp_gradient_clip_norm"]),
        num_threads=int(model_config["mlp_num_threads"]),
        fit_validation=False,
    )

    context_config = config["context"]
    champion_run = _resolve_path(
        config_path, context_config["champion_run"]
    )
    validation_keys = data.loc[
        validation_mask, ["date", "stock_code"]
    ].sort_values(["date", "stock_code"]).reset_index(drop=True)
    champion_validation = _load_champion_validation(
        champion_run,
        validation_keys,
        target[validation_mask],
    )
    temporary_path = Path(
        tempfile.mkdtemp(dir=runs_dir, prefix=f".{run_id}.", suffix=".tmp")
    )
    try:
        shutil.copy2(config_path, temporary_path / "config_snapshot.toml")
        candidate_manifests: dict[str, Any] = {}
        for result in candidates:
            print(f"publishing candidate: {result.name}", flush=True)
            candidate_manifests[result.name] = _publish_candidate(
                temporary_path / result.name,
                data,
                result,
                validation_mask,
                champion_validation,
                champion_oof,
                schema,
                hac_lag=int(context_config["hac_lag"]),
                min_prediction_std_ratio=float(
                    context_config["min_prediction_std_ratio"]
                ),
                test_contamination_disclosure=context_config[
                    "test_contamination_disclosure"
                ],
            )
        top_manifest = {
            "schema_version": 1,
            "status": "completed",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": "context_decomposition",
            "horizon": 10,
            "development_end": "2023-12-31",
            "stage_two_gate": gate,
            "regenerated_champion_oof": {
                "selected_parameters": frozen_champion_oof.selected_parameters,
                "folds": frozen_champion_oof.fold_metadata,
                "candidate_records": frozen_champion_oof.candidate_records,
            },
            "candidate_freeze": {
                "selection_data_end": "2022-12-31",
                "validation_loaded_after_freeze": True,
                "frozen_candidates": list(frozen_by_name),
                "oof_reproduction_max_abs_difference": oof_reproduction,
            },
            "candidates": list(candidate_manifests),
            "candidate_acceptance": {
                name: manifest["acceptance"]
                for name, manifest in candidate_manifests.items()
            },
            "uses_entry_tradeable": False,
            "context_dataset_sha256": _file_sha256(context_path),
            "context_schema_sha256": _file_sha256(schema_path),
            "champion_run": str(champion_run),
            "champion_predictions_sha256": _file_sha256(
                champion_run / "predictions.parquet"
            ),
            "test_contamination_disclosure": context_config[
                "test_contamination_disclosure"
            ],
        }
        _write_json(temporary_path / "manifest.json", top_manifest)
        os.rename(temporary_path, final_run_path)
    except BaseException:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise
    print(f"context experiment published: {final_run_path}", flush=True)
    return final_run_path


def _validate_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "stock_code", "y_true", "y_pred"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction frame is missing columns: {missing}")
    result = frame[["date", "stock_code", "y_true", "y_pred"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["stock_code"] = result["stock_code"].astype("string").str.strip()
    result["y_true"] = pd.to_numeric(result["y_true"], errors="coerce")
    result["y_pred"] = pd.to_numeric(result["y_pred"], errors="coerce")
    if (
        result.empty
        or result[["date", "stock_code"]].isna().any().any()
        or result["stock_code"].eq("").any()
        or result.duplicated(["date", "stock_code"]).any()
        or not np.isfinite(result[["y_true", "y_pred"]].to_numpy()).all()
    ):
        raise ValueError("prediction frame keys and values must be valid")
    return result.sort_values(["date", "stock_code"]).reset_index(drop=True)


def _rank_deciles(values: pd.Series) -> np.ndarray:
    ranks = values.rank(method="average", ascending=False).to_numpy(dtype=np.float64)
    return np.clip(np.ceil(10.0 * ranks / len(values)), 1, 10)


def _ndcg_at_k(group: pd.DataFrame, k: int) -> float:
    relevance = group["y_true"].rank(method="average", pct=True)
    work = group.assign(_relevance=relevance)
    predicted = work.sort_values(
        ["y_pred", "stock_code"],
        ascending=[False, True],
        kind="mergesort",
    ).head(k)
    ideal = work.sort_values(
        ["y_true", "stock_code"],
        ascending=[False, True],
        kind="mergesort",
    ).head(k)
    discounts = np.log2(np.arange(2, len(predicted) + 2, dtype=np.float64))
    predicted_gain = np.exp2(predicted["_relevance"].to_numpy()) - 1.0
    ideal_gain = np.exp2(ideal["_relevance"].to_numpy()) - 1.0
    ideal_dcg = float(np.sum(ideal_gain / discounts))
    if ideal_dcg <= 0:
        return 0.0
    return float(np.sum(predicted_gain / discounts) / ideal_dcg)


def return_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute raw-return accuracy, calibration, and cross-sectional metrics."""
    data = _validate_prediction_frame(frame)
    truth = data["y_true"].to_numpy(dtype=np.float64)
    prediction = data["y_pred"].to_numpy(dtype=np.float64)
    errors = prediction - truth
    squared_errors = np.square(errors)
    mse = float(np.mean(squared_errors))
    target_ss = float(np.sum(np.square(truth - float(truth.mean()))))
    calibration = fit_calibrator(truth, prediction)
    daily_rank_ic: list[float] = []
    group_errors: list[float] = []
    top100_targets: list[float] = []
    ndcg_values: list[float] = []
    for _, group in data.groupby("date", sort=False):
        true_ranks = group["y_true"].rank(method="average").to_numpy()
        predicted_ranks = group["y_pred"].rank(method="average").to_numpy()
        if np.ptp(true_ranks) > 0 and np.ptp(predicted_ranks) > 0:
            correlation = float(np.corrcoef(true_ranks, predicted_ranks)[0, 1])
            if np.isfinite(correlation):
                daily_rank_ic.append(correlation)
        if len(group) >= 10:
            group_errors.extend(
                np.abs(
                    _rank_deciles(group["y_true"])
                    - _rank_deciles(group["y_pred"])
                ).tolist()
            )
        selection_size = min(100, len(group))
        selected = group.sort_values(
            ["y_pred", "stock_code"],
            ascending=[False, True],
            kind="mergesort",
        ).head(selection_size)
        top100_targets.append(float(selected["y_true"].mean()))
        ndcg_values.append(_ndcg_at_k(group, selection_size))
    rank_values = np.asarray(daily_rank_ic, dtype=np.float64)
    rank_mean = float(rank_values.mean()) if len(rank_values) else np.nan
    rank_std = (
        float(rank_values.std(ddof=1)) if len(rank_values) > 1 else 0.0
    )
    robust = np.abs(truth) <= 0.50
    extreme = ~robust
    total_sse = float(squared_errors.sum())
    return {
        "rows": int(len(data)),
        "dates": int(data["date"].nunique()),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(errors))),
        "r_squared": float(1.0 - squared_errors.sum() / target_ss)
        if target_ss > 0
        else None,
        "prediction_std": float(np.std(prediction, ddof=1)),
        "target_std": float(np.std(truth, ddof=1)),
        "direction_accuracy": float(np.mean((prediction > 0) == (truth > 0))),
        "calibration_intercept": float(calibration.intercept),
        "calibration_slope": float(calibration.slope),
        "mean_daily_rank_ic": rank_mean,
        "daily_rank_ic_std": rank_std,
        "rank_icir": float(rank_mean / rank_std) if rank_std > 0 else None,
        "rank_icir_annualized": None,
        "rank_icir_annualization_note": (
            "Not annualized because daily 10-day targets overlap."
        ),
        "positive_rank_ic_rate": float(np.mean(rank_values > 0))
        if len(rank_values)
        else None,
        "rank_ic_valid_dates": int(len(rank_values)),
        "group_mae": float(np.mean(group_errors)) if group_errors else None,
        "top100_mean_target": float(np.mean(top100_targets)),
        "ndcg_at_100": float(np.mean(ndcg_values)),
        "robust_rmse_abs_target_le_50pct": float(
            np.sqrt(np.mean(squared_errors[robust]))
        )
        if robust.any()
        else None,
        "extreme_row_count": int(extreme.sum()),
        "extreme_mse_contribution": float(squared_errors[extreme].sum() / total_sse)
        if total_sse > 0
        else 0.0,
    }


def hac_mean_test(values: np.ndarray, *, lag: int = 10) -> dict[str, Any]:
    """Return a Bartlett Newey-West test for the mean of a time series."""
    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or len(observations) < 2:
        raise ValueError("HAC values must be a one-dimensional sample of size >= 2")
    if not np.isfinite(observations).all() or lag < 0:
        raise ValueError("HAC values must be finite and lag nonnegative")
    n = len(observations)
    used_lag = min(int(lag), n - 1)
    centered = observations - float(observations.mean())
    long_run_variance = float(np.dot(centered, centered) / n)
    for offset in range(1, used_lag + 1):
        weight = 1.0 - offset / (used_lag + 1.0)
        autocovariance = float(
            np.dot(centered[offset:], centered[:-offset]) / n
        )
        long_run_variance += 2.0 * weight * autocovariance
    variance_of_mean = max(long_run_variance / n, 0.0)
    standard_error = float(np.sqrt(variance_of_mean))
    mean = float(observations.mean())
    if standard_error > 0:
        t_value: float | None = float(mean / standard_error)
        lower = float(mean - 1.959963984540054 * standard_error)
        upper = float(mean + 1.959963984540054 * standard_error)
    else:
        t_value = None
        lower = mean
        upper = mean
    return {
        "observations": int(n),
        "lag": int(used_lag),
        "mean": mean,
        "standard_error": standard_error,
        "t_value": t_value,
        "ci95_lower": lower,
        "ci95_upper": upper,
    }


def compare_to_champion(
    candidate: pd.DataFrame,
    champion: pd.DataFrame,
    *,
    hac_lag: int = 10,
) -> dict[str, Any]:
    """Compare aligned candidate and champion predictions by daily MSE."""
    candidate_data = _validate_prediction_frame(candidate)
    champion_data = _validate_prediction_frame(champion)
    keys = ["date", "stock_code"]
    if not candidate_data[keys].equals(champion_data[keys]):
        raise ValueError("candidate and champion prediction keys differ")
    if not np.allclose(
        candidate_data["y_true"],
        champion_data["y_true"],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("candidate and champion targets differ")
    work = candidate_data[keys].copy()
    work["candidate_squared_error"] = np.square(
        candidate_data["y_pred"].to_numpy()
        - candidate_data["y_true"].to_numpy()
    )
    work["champion_squared_error"] = np.square(
        champion_data["y_pred"].to_numpy()
        - champion_data["y_true"].to_numpy()
    )
    daily = work.groupby("date", sort=True)[
        ["candidate_squared_error", "champion_squared_error"]
    ].mean()
    daily["difference"] = (
        daily["candidate_squared_error"] - daily["champion_squared_error"]
    )
    candidate_mse = float(work["candidate_squared_error"].mean())
    champion_mse = float(work["champion_squared_error"].mean())
    month = work.assign(month=work["date"].dt.to_period("M")).groupby(
        "month", sort=True
    )[["candidate_squared_error", "champion_squared_error"]].mean()
    monthly_difference = {
        str(period): float(
            np.sqrt(row["candidate_squared_error"])
            - np.sqrt(row["champion_squared_error"])
        )
        for period, row in month.iterrows()
    }
    return {
        "candidate_mse": candidate_mse,
        "champion_mse": champion_mse,
        "mse_skill": float(1.0 - candidate_mse / champion_mse)
        if champion_mse > 0
        else None,
        "candidate_better_date_rate": float(
            daily["difference"].lt(0).mean()
        ),
        "daily_mse_difference_hac": hac_mean_test(
            daily["difference"].to_numpy(), lag=hac_lag
        ),
        "monthly_rmse_difference": monthly_difference,
    }
