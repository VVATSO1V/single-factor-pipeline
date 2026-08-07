"""Immutable training runs for CSI1000 10-day rank models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Any, Callable
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

from rank_model import __version__
from rank_model.stages.dataset import (
    EXIT_DATE_COLUMN,
    SPLIT_COLUMN,
    file_sha256,
    load_rank_dataset,
)
from rank_model.stages.preprocessing import (
    RankPreprocessor,
    UNKNOWN_INDUSTRY,
    _normalized_industries,
    equal_date_weights,
    predicted_percentiles,
)
from rank_model.stages.ranking import (
    lightgbm_relevance,
    pairwise_logistic_loss,
    sample_date_pairs,
    sorted_group_layout,
)


KEY_COLUMNS = ["date", "stock_code"]
RANK_TARGET_COLUMN = "rank_target_10d"
MODEL_NAMES = (
    "ridge_rank_regression",
    "xgboost_rank_regression",
    "lightgbm_rank_regression",
    "mlp_rank_regression",
    "xgboost_pairwise_rank",
    "lightgbm_lambdarank",
    "mlp_pairwise_rank",
)
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_JOBLIB_NUMPY_SHAPE_WARNING = (
    "Setting the shape on a NumPy array has been deprecated in NumPy 2.5.\n"
    "As an alternative, you can create a new view using np.reshape "
    "(with copy=False if needed)."
)
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 100.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 10.0,
    "alpha": 0.1,
    "seed": 42,
}
XGB_BOOSTING_ROUNDS = 81
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 5,
    "min_data_in_leaf": 1000,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "lambda_l2": 10.0,
    "lambda_l1": 0.1,
    "deterministic": True,
    "force_col_wise": True,
    "seed": 42,
    "verbosity": -1,
}
LGB_BOOSTING_ROUNDS = 21
XGB_PAIRWISE_PARAMS = {
    **XGB_PARAMS,
    "objective": "rank:pairwise",
    "eval_metric": "ndcg@100",
}
LGB_LAMBDARANK_PARAMS = {
    **LGB_PARAMS,
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [100],
    "label_gain": list(range(100)),
    "lambdarank_truncation_level": 100,
}
NATIVE_TREE_RELOAD_TOLERANCE = 1e-6
MLP_HIDDEN_LAYERS = [128, 64, 32]
MLP_BATCH_SIZE = 8192
MLP_EPOCHS = 12
MLP_GRADIENT_CLIP_NORM = 1.0
MLP_SEED = 42
MLP_DROPOUT = 0.1
MLP_LEARNING_RATE = 0.001
MLP_WEIGHT_DECAY = 0.0001
MLP_PAIRWISE_PAIRS_PER_STOCK = 8
MLP_PAIRWISE_ADJACENT_FRACTION = 0.5
MLP_PAIRWISE_DATES_PER_BATCH = 8
MLP_RELOAD_TOLERANCE = 1e-6
DEFAULT_RELOAD_TOLERANCE = 1e-12
FIXED_SEED = 42


@dataclass
class TrainingOutcome:
    score_validation: np.ndarray
    model_objects: dict[str, Any]
    metadata: dict[str, Any]


Trainer = Callable[
    [pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]], TrainingOutcome
]


def _finite_label_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if RANK_TARGET_COLUMN not in frame:
        raise ValueError(f"training data is missing {RANK_TARGET_COLUMN}")
    labels = pd.to_numeric(frame[RANK_TARGET_COLUMN], errors="coerce")
    return frame.loc[np.isfinite(labels.to_numpy())].copy()


def _timed_prediction(predict: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    started = perf_counter()
    scores = np.asarray(predict(), dtype="float64")
    return scores, float(perf_counter() - started)


def _fixed_parameters(
    model_name: str,
    **parameters: Any,
) -> dict[str, Any]:
    """Return the JSON-safe immutable model contract recorded in every run."""
    return {
        "model_name": model_name,
        "seed": FIXED_SEED,
        "reload_tolerance": DEFAULT_RELOAD_TOLERANCE,
        **parameters,
    }


def train_ridge_rank_regression(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    schema: dict[str, Any],
    params: dict[str, Any],
) -> TrainingOutcome:
    """Fit the fixed Ridge baseline using equal total weight per date."""
    train = _finite_label_rows(train)
    if train.empty:
        raise ValueError("ridge training requires at least one finite rank label")
    if validation.empty:
        raise ValueError("ridge validation frame is empty")
    try:
        continuous_columns = schema["continuous_feature_columns"]
        industry_column = schema["industry_column"]
        regularization = float(params["lambda"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("ridge requires schema features and a numeric lambda") from error
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("ridge lambda must be finite and non-negative")

    weights = equal_date_weights(train["date"])
    preprocessor = RankPreprocessor.fit(
        train,
        continuous_columns=continuous_columns,
        industry_column=industry_column,
    )
    x_train = preprocessor.transform(train, scale_continuous=True)
    x_validation = preprocessor.transform(validation, scale_continuous=True)
    y_train = train[RANK_TARGET_COLUMN].to_numpy(dtype="float64")
    alpha = regularization * float(weights.sum())
    model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
    model.fit(x_train, y_train, sample_weight=weights)
    score_validation, validation_inference_seconds = _timed_prediction(
        lambda: model.predict(x_validation)
    )
    if not np.isfinite(score_validation).all():
        raise ValueError("ridge produced non-finite validation scores")
    return TrainingOutcome(
        score_validation=np.asarray(score_validation, dtype="float64"),
        model_objects={"preprocessor": preprocessor, "model": model},
        metadata={
            "lambda": regularization,
            "alpha": alpha,
            "solver": "cholesky",
            "validation_inference_seconds": validation_inference_seconds,
            "fixed_parameters": _fixed_parameters(
                "ridge_rank_regression",
                objective="squared_error",
                solver="cholesky",
                fit_intercept=True,
                date_weighting="equal_total_per_date",
                **{"lambda": regularization},
            ),
        },
    )


def _tree_training_inputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    schema: dict[str, Any],
) -> tuple[RankPreprocessor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = _finite_label_rows(train)
    if train.empty:
        raise ValueError("tree training requires at least one finite rank label")
    if validation.empty:
        raise ValueError("tree validation frame is empty")
    try:
        continuous_columns = schema["continuous_feature_columns"]
        industry_column = schema["industry_column"]
    except (KeyError, TypeError) as error:
        raise ValueError("tree training requires schema features") from error
    preprocessor = RankPreprocessor.fit(
        train,
        continuous_columns=continuous_columns,
        industry_column=industry_column,
    )
    return (
        preprocessor,
        preprocessor.transform(train, scale_continuous=False),
        preprocessor.transform(validation, scale_continuous=False),
        train[RANK_TARGET_COLUMN].to_numpy(dtype="float64"),
        equal_date_weights(train["date"]),
    )


def train_xgboost_rank_regression(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    schema: dict[str, Any],
    params: dict[str, Any],
) -> TrainingOutcome:
    """Fit the fixed XGBoost rank-regression baseline."""
    del params
    preprocessor, x_train, x_validation, y_train, weights = _tree_training_inputs(
        train, validation, schema
    )
    model = xgb.train(
        XGB_PARAMS,
        xgb.DMatrix(x_train, label=y_train, weight=weights),
        num_boost_round=XGB_BOOSTING_ROUNDS,
    )
    score_validation, validation_inference_seconds = _timed_prediction(
        lambda: model.predict(xgb.DMatrix(x_validation))
    )
    if not np.isfinite(score_validation).all():
        raise ValueError("xgboost produced non-finite validation scores")
    return TrainingOutcome(
        score_validation=np.asarray(score_validation, dtype="float64"),
        model_objects={"preprocessor": preprocessor, "model": model},
        metadata={
            "objective": XGB_PARAMS["objective"],
            "boosting_rounds": XGB_BOOSTING_ROUNDS,
            "validation_inference_seconds": validation_inference_seconds,
            "fixed_parameters": _fixed_parameters(
                "xgboost_rank_regression",
                **XGB_PARAMS,
                boosting_rounds=XGB_BOOSTING_ROUNDS,
                date_weighting="equal_total_per_date",
            ),
        },
    )


def train_lightgbm_rank_regression(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    schema: dict[str, Any],
    params: dict[str, Any],
) -> TrainingOutcome:
    """Fit the fixed LightGBM rank-regression baseline."""
    del params
    preprocessor, x_train, x_validation, y_train, weights = _tree_training_inputs(
        train, validation, schema
    )
    model = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(x_train, label=y_train, weight=weights),
        num_boost_round=LGB_BOOSTING_ROUNDS,
    )
    score_validation, validation_inference_seconds = _timed_prediction(
        lambda: model.predict(x_validation)
    )
    if not np.isfinite(score_validation).all():
        raise ValueError("lightgbm produced non-finite validation scores")
    return TrainingOutcome(
        score_validation=np.asarray(score_validation, dtype="float64"),
        model_objects={"preprocessor": preprocessor, "model": model},
        metadata={
            "objective": LGB_PARAMS["objective"],
            "boosting_rounds": LGB_BOOSTING_ROUNDS,
            "validation_inference_seconds": validation_inference_seconds,
            "fixed_parameters": _fixed_parameters(
                "lightgbm_rank_regression",
                **LGB_PARAMS,
                boosting_rounds=LGB_BOOSTING_ROUNDS,
                date_weighting="equal_total_per_date",
            ),
        },
    )


def _native_ranking_training_inputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    schema: dict[str, Any],
) -> tuple[
    RankPreprocessor,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.Series,
    np.ndarray,
]:
    train = _finite_label_rows(train)
    if train.empty:
        raise ValueError("native ranking requires at least one finite rank label")
    if validation.empty:
        raise ValueError("native ranking validation frame is empty")
    try:
        continuous_columns = schema["continuous_feature_columns"]
        industry_column = schema["industry_column"]
    except (KeyError, TypeError) as error:
        raise ValueError("native ranking requires schema features") from error

    train_layout, train_group_sizes = sorted_group_layout(train)
    validation_with_position = validation.copy()
    validation_with_position["_rank_row_position"] = np.arange(len(validation))
    validation_layout, validation_group_sizes = sorted_group_layout(
        validation_with_position
    )
    preprocessor = RankPreprocessor.fit(
        train,
        continuous_columns=continuous_columns,
        industry_column=industry_column,
    )
    query_ids = pd.factorize(train_layout["date"], sort=False)[0].astype("int32")
    group_weights = np.ones(len(train_group_sizes), dtype="float64")
    validation_positions = validation_layout["_rank_row_position"].to_numpy(
        dtype="int64"
    )
    return (
        preprocessor,
        preprocessor.transform(train_layout, scale_continuous=False),
        preprocessor.transform(validation_layout, scale_continuous=False),
        train_layout[RANK_TARGET_COLUMN].to_numpy(dtype="float64"),
        train_group_sizes,
        validation_group_sizes,
        query_ids,
        group_weights,
        train_layout["date"],
        validation_positions,
    )


def _restore_validation_order(scores: np.ndarray, positions: np.ndarray) -> np.ndarray:
    restored = np.empty(len(scores), dtype="float64")
    restored[positions] = scores
    return restored


def train_xgboost_pairwise_rank(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    schema: dict[str, Any],
    params: dict[str, Any],
) -> TrainingOutcome:
    """Fit fixed XGBoost pairwise ranking with one query per trading date."""
    del params
    (
        preprocessor,
        x_train,
        x_validation,
        y_train,
        train_group_sizes,
        _validation_group_sizes,
        query_ids,
        group_weights,
        _train_dates,
        validation_positions,
    ) = _native_ranking_training_inputs(train, validation, schema)
    del train_group_sizes
    model = xgb.train(
        XGB_PAIRWISE_PARAMS,
        xgb.DMatrix(x_train, label=y_train, weight=group_weights, qid=query_ids),
        num_boost_round=XGB_BOOSTING_ROUNDS,
    )
    sorted_scores, validation_inference_seconds = _timed_prediction(
        lambda: model.predict(xgb.DMatrix(x_validation))
    )
    score_validation = _restore_validation_order(sorted_scores, validation_positions)
    if not np.isfinite(score_validation).all():
        raise ValueError("xgboost pairwise ranking produced non-finite validation scores")
    return TrainingOutcome(
        score_validation=score_validation,
        model_objects={"preprocessor": preprocessor, "model": model},
        metadata={
            "objective": XGB_PAIRWISE_PARAMS["objective"],
            "boosting_rounds": XGB_BOOSTING_ROUNDS,
            "grouping": "date",
            "reload_tolerance": NATIVE_TREE_RELOAD_TOLERANCE,
            "validation_inference_seconds": validation_inference_seconds,
            "fixed_parameters": _fixed_parameters(
                "xgboost_pairwise_rank",
                **XGB_PAIRWISE_PARAMS,
                boosting_rounds=XGB_BOOSTING_ROUNDS,
                grouping="date",
                date_weighting="equal_total_per_date",
                reload_tolerance=NATIVE_TREE_RELOAD_TOLERANCE,
            ),
        },
    )


def train_lightgbm_lambdarank(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    schema: dict[str, Any],
    params: dict[str, Any],
) -> TrainingOutcome:
    """Fit fixed LightGBM LambdaRank with one group per trading date."""
    del params
    (
        preprocessor,
        x_train,
        x_validation,
        y_train,
        train_group_sizes,
        _validation_group_sizes,
        _query_ids,
        _group_weights,
        train_dates,
        validation_positions,
    ) = _native_ranking_training_inputs(train, validation, schema)
    model = lgb.train(
        LGB_LAMBDARANK_PARAMS,
        lgb.Dataset(
            x_train,
            label=lightgbm_relevance(pd.Series(y_train)),
            weight=equal_date_weights(train_dates),
            group=train_group_sizes,
        ),
        num_boost_round=LGB_BOOSTING_ROUNDS,
    )
    sorted_scores, validation_inference_seconds = _timed_prediction(
        lambda: model.predict(x_validation)
    )
    score_validation = _restore_validation_order(sorted_scores, validation_positions)
    if not np.isfinite(score_validation).all():
        raise ValueError("LightGBM LambdaRank produced non-finite validation scores")
    return TrainingOutcome(
        score_validation=score_validation,
        model_objects={"preprocessor": preprocessor, "model": model},
        metadata={
            "objective": LGB_LAMBDARANK_PARAMS["objective"],
            "boosting_rounds": LGB_BOOSTING_ROUNDS,
            "grouping": "date",
            "reload_tolerance": NATIVE_TREE_RELOAD_TOLERANCE,
            "validation_inference_seconds": validation_inference_seconds,
            "fixed_parameters": _fixed_parameters(
                "lightgbm_lambdarank",
                **LGB_LAMBDARANK_PARAMS,
                boosting_rounds=LGB_BOOSTING_ROUNDS,
                grouping="date",
                relevance_transform="floor(rank_target_10d * 100), clipped to [0, 99]",
                date_weighting="equal_total_per_date",
                reload_tolerance=NATIVE_TREE_RELOAD_TOLERANCE,
            ),
        },
    )
def _build_mlp_rank_regression_model(input_dim: int, output_bias: float) -> Any:
    """Build the fixed ReLU network used by the MLP rank regressor."""
    import torch

    layers: list[torch.nn.Module] = []
    previous_width = input_dim
    for width in MLP_HIDDEN_LAYERS:
        layers.extend(
            (
                torch.nn.Linear(previous_width, width),
                torch.nn.ReLU(),
                torch.nn.Dropout(MLP_DROPOUT),
            )
        )
        previous_width = width
    output_layer = torch.nn.Linear(previous_width, 1)
    torch.nn.init.zeros_(output_layer.weight)
    torch.nn.init.constant_(output_layer.bias, output_bias)
    layers.append(output_layer)
    return torch.nn.Sequential(*layers)


def _predict_mlp_rank_regression(model: Any, features: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.inference_mode():
        predictions = model(
            torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        ).squeeze(-1)
    return predictions.cpu().numpy().astype("float64", copy=False)


def train_mlp_rank_regression(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    schema: dict[str, Any],
    params: dict[str, Any],
) -> TrainingOutcome:
    """Fit the fixed deterministic MLP with equal total weight per date."""
    import torch

    del params
    train = _finite_label_rows(train)
    if train.empty:
        raise ValueError("MLP training requires at least one finite rank label")
    if validation.empty:
        raise ValueError("MLP validation frame is empty")
    try:
        continuous_columns = schema["continuous_feature_columns"]
        industry_column = schema["industry_column"]
    except (KeyError, TypeError) as error:
        raise ValueError("MLP training requires schema features") from error

    preprocessor = RankPreprocessor.fit(
        train,
        continuous_columns=continuous_columns,
        industry_column=industry_column,
    )
    x_train = preprocessor.transform(train, scale_continuous=True)
    x_validation = preprocessor.transform(validation, scale_continuous=True)
    y_train = train[RANK_TARGET_COLUMN].to_numpy(dtype="float32")
    row_weights = equal_date_weights(train["date"]).astype("float32")

    np.random.seed(MLP_SEED)
    torch.manual_seed(MLP_SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = _build_mlp_rank_regression_model(
        x_train.shape[1], float(np.mean(y_train))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY
    )
    features = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
    targets = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))
    weights = torch.from_numpy(np.ascontiguousarray(row_weights, dtype=np.float32))
    total_weight = weights.sum()

    model.train()
    for _ in range(MLP_EPOCHS):
        for start in range(0, len(features), MLP_BATCH_SIZE):
            end = min(start + MLP_BATCH_SIZE, len(features))
            prediction = model(features[start:end]).squeeze(-1)
            weighted_loss = (
                weights[start:end] * torch.square(prediction - targets[start:end])
            ).sum()
            weighted_loss = weighted_loss / total_weight
            optimizer.zero_grad(set_to_none=True)
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MLP_GRADIENT_CLIP_NORM)
            optimizer.step()

    score_validation, validation_inference_seconds = _timed_prediction(
        lambda: _predict_mlp_rank_regression(model, x_validation)
    )
    if not np.isfinite(score_validation).all():
        raise ValueError("MLP produced non-finite validation scores")
    return TrainingOutcome(
        score_validation=score_validation,
        model_objects={
            "preprocessor": preprocessor,
            "model": model,
            "model_type": "pytorch_mlp",
            "architecture": {
                "model_name": "mlp_rank_regression",
                "input_dim": int(x_train.shape[1]),
                "hidden_layers": MLP_HIDDEN_LAYERS,
                "activation": "relu",
                "dropout": MLP_DROPOUT,
                "output_bias": float(np.mean(y_train)),
            },
        },
        metadata={
            "hidden_layers": MLP_HIDDEN_LAYERS,
            "epochs": MLP_EPOCHS,
            "loss": "mse",
            "optimizer": "adamw",
            "learning_rate": MLP_LEARNING_RATE,
            "weight_decay": MLP_WEIGHT_DECAY,
            "batch_size": MLP_BATCH_SIZE,
            "gradient_clip_norm": MLP_GRADIENT_CLIP_NORM,
            "seed": MLP_SEED,
            "reload_tolerance": MLP_RELOAD_TOLERANCE,
            "validation_inference_seconds": validation_inference_seconds,
            "fixed_parameters": _fixed_parameters(
                "mlp_rank_regression",
                objective="rank_regression",
                hidden_layers=MLP_HIDDEN_LAYERS,
                activation="relu",
                dropout=MLP_DROPOUT,
                loss="mse",
                optimizer="adamw",
                learning_rate=MLP_LEARNING_RATE,
                weight_decay=MLP_WEIGHT_DECAY,
                batch_size=MLP_BATCH_SIZE,
                epochs=MLP_EPOCHS,
                gradient_clip_norm=MLP_GRADIENT_CLIP_NORM,
                deterministic_algorithms=True,
                date_weighting="equal_total_per_date",
                reload_tolerance=MLP_RELOAD_TOLERANCE,
            ),
        },
    )


def _build_mlp_pairwise_rank_model(input_dim: int) -> Any:
    """Build the fixed dropout MLP used by pairwise ranking."""
    import torch

    layers: list[torch.nn.Module] = []
    previous_width = input_dim
    for width in MLP_HIDDEN_LAYERS:
        layers.extend(
            (
                torch.nn.Linear(previous_width, width),
                torch.nn.ReLU(),
                torch.nn.Dropout(MLP_DROPOUT),
            )
        )
        previous_width = width
    output_layer = torch.nn.Linear(previous_width, 1)
    torch.nn.init.zeros_(output_layer.weight)
    torch.nn.init.zeros_(output_layer.bias)
    layers.append(output_layer)
    return torch.nn.Sequential(*layers)


def _pairwise_date_batch_ranges(dates: pd.Series) -> list[tuple[int, int, np.ndarray]]:
    """Return contiguous row ranges containing at most eight whole date groups."""
    date_codes, unique_dates = pd.factorize(dates, sort=False)
    ranges: list[tuple[int, int, np.ndarray]] = []
    for start_code in range(0, len(unique_dates), MLP_PAIRWISE_DATES_PER_BATCH):
        batch_codes = np.arange(
            start_code,
            min(start_code + MLP_PAIRWISE_DATES_PER_BATCH, len(unique_dates)),
            dtype="int64",
        )
        positions = np.flatnonzero(np.isin(date_codes, batch_codes))
        if positions.size == 0:
            raise ValueError("pairwise date batching produced an empty batch")
        if not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
            raise ValueError("pairwise date batches require contiguous date rows")
        ranges.append((int(positions[0]), int(positions[-1]) + 1, batch_codes))
    return ranges


def train_mlp_pairwise_rank(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    schema: dict[str, Any],
    params: dict[str, Any],
) -> TrainingOutcome:
    """Fit the fixed date-batched MLP with deterministic same-date pairs."""
    import torch

    del params
    train = _finite_label_rows(train)
    if train.empty:
        raise ValueError("pairwise MLP training requires at least one finite rank label")
    if validation.empty:
        raise ValueError("MLP validation frame is empty")
    try:
        continuous_columns = schema["continuous_feature_columns"]
        industry_column = schema["industry_column"]
    except (KeyError, TypeError) as error:
        raise ValueError("pairwise MLP training requires schema features") from error

    train, _ = sorted_group_layout(train)
    preprocessor = RankPreprocessor.fit(
        train,
        continuous_columns=continuous_columns,
        industry_column=industry_column,
    )
    x_train = preprocessor.transform(train, scale_continuous=True)
    x_validation = preprocessor.transform(validation, scale_continuous=True)
    targets = train[RANK_TARGET_COLUMN].to_numpy(dtype="float64")
    left, right, direction = sample_date_pairs(
        train["date"],
        targets,
        MLP_PAIRWISE_PAIRS_PER_STOCK,
        MLP_PAIRWISE_ADJACENT_FRACTION,
        MLP_SEED,
    )
    date_codes, unique_dates = pd.factorize(train["date"], sort=False)
    pair_date_codes = date_codes[left]
    pair_counts_by_date = {
        pd.Timestamp(date).date().isoformat(): int(np.sum(pair_date_codes == code))
        for code, date in enumerate(unique_dates)
    }

    np.random.seed(MLP_SEED)
    torch.manual_seed(MLP_SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = _build_mlp_pairwise_rank_model(x_train.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY
    )
    features = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
    left_tensor = torch.from_numpy(np.ascontiguousarray(left, dtype=np.int64))
    right_tensor = torch.from_numpy(np.ascontiguousarray(right, dtype=np.int64))
    direction_tensor = torch.from_numpy(
        np.ascontiguousarray(direction, dtype=np.float32)
    )
    batch_ranges = _pairwise_date_batch_ranges(train["date"])
    training_date_count = len(unique_dates)

    model.train()
    for _ in range(MLP_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        for batch_start, batch_end, batch_codes in batch_ranges:
            pair_mask = np.isin(pair_date_codes, batch_codes)
            batch_left = left_tensor[pair_mask] - batch_start
            batch_right = right_tensor[pair_mask] - batch_start
            batch_direction = direction_tensor[pair_mask]
            scores = model(features[batch_start:batch_end]).squeeze(-1)
            date_losses = []
            batch_pair_codes = pair_date_codes[pair_mask]
            for date_code in batch_codes:
                date_pair_mask = torch.from_numpy(batch_pair_codes == date_code)
                date_losses.append(
                    pairwise_logistic_loss(
                        scores,
                        batch_left[date_pair_mask],
                        batch_right[date_pair_mask],
                        batch_direction[date_pair_mask],
                    )
                )
            loss = torch.stack(date_losses).sum() / training_date_count
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MLP_GRADIENT_CLIP_NORM)
        optimizer.step()

    score_validation, validation_inference_seconds = _timed_prediction(
        lambda: _predict_mlp_rank_regression(model, x_validation)
    )
    if not np.isfinite(score_validation).all():
        raise ValueError("pairwise MLP produced non-finite validation scores")
    broad_pairs = MLP_PAIRWISE_PAIRS_PER_STOCK - int(
        MLP_PAIRWISE_PAIRS_PER_STOCK * MLP_PAIRWISE_ADJACENT_FRACTION
    )
    adjacent_pairs = MLP_PAIRWISE_PAIRS_PER_STOCK - broad_pairs
    return TrainingOutcome(
        score_validation=score_validation,
        model_objects={
            "preprocessor": preprocessor,
            "model": model,
            "model_type": "pytorch_mlp",
            "architecture": {
                "model_name": "mlp_pairwise_rank",
                "input_dim": int(x_train.shape[1]),
                "hidden_layers": MLP_HIDDEN_LAYERS,
                "activation": "relu",
                "dropout": MLP_DROPOUT,
                "output_bias": 0.0,
            },
        },
        metadata={
            "hidden_layers": MLP_HIDDEN_LAYERS,
            "dropout": MLP_DROPOUT,
            "epochs": MLP_EPOCHS,
            "loss": "pairwise_logistic",
            "optimizer": "adamw",
            "learning_rate": MLP_LEARNING_RATE,
            "weight_decay": MLP_WEIGHT_DECAY,
            "gradient_clip_norm": MLP_GRADIENT_CLIP_NORM,
            "optimizer_steps_per_epoch": 1,
            "global_date_loss_normalization": True,
            "seed": MLP_SEED,
            "dates_per_batch": MLP_PAIRWISE_DATES_PER_BATCH,
            "pairs_per_stock": MLP_PAIRWISE_PAIRS_PER_STOCK,
            "adjacent_fraction": MLP_PAIRWISE_ADJACENT_FRACTION,
            "broad_pairs_per_stock": broad_pairs,
            "adjacent_pairs_per_stock": adjacent_pairs,
            "pair_count": int(len(left)),
            "pair_counts_by_date": pair_counts_by_date,
            "reload_tolerance": MLP_RELOAD_TOLERANCE,
            "validation_inference_seconds": validation_inference_seconds,
            "fixed_parameters": _fixed_parameters(
                "mlp_pairwise_rank",
                objective="pairwise_ranking",
                hidden_layers=MLP_HIDDEN_LAYERS,
                activation="relu",
                dropout=MLP_DROPOUT,
                loss="pairwise_logistic",
                optimizer="adamw",
                learning_rate=MLP_LEARNING_RATE,
                weight_decay=MLP_WEIGHT_DECAY,
                epochs=MLP_EPOCHS,
                gradient_clip_norm=MLP_GRADIENT_CLIP_NORM,
                dates_per_batch=MLP_PAIRWISE_DATES_PER_BATCH,
                pairs_per_stock=MLP_PAIRWISE_PAIRS_PER_STOCK,
                adjacent_fraction=MLP_PAIRWISE_ADJACENT_FRACTION,
                broad_pairs_per_stock=broad_pairs,
                adjacent_pairs_per_stock=adjacent_pairs,
                deterministic_algorithms=True,
                grouping="date",
                date_weighting="equal_total_per_date",
                reload_tolerance=MLP_RELOAD_TOLERANCE,
            ),
        },
    )


MODEL_REGISTRY: dict[str, Trainer] = {
    "ridge_rank_regression": train_ridge_rank_regression,
    "xgboost_rank_regression": train_xgboost_rank_regression,
    "lightgbm_rank_regression": train_lightgbm_rank_regression,
    "mlp_rank_regression": train_mlp_rank_regression,
    "xgboost_pairwise_rank": train_xgboost_pairwise_rank,
    "lightgbm_lambdarank": train_lightgbm_lambdarank,
    "mlp_pairwise_rank": train_mlp_pairwise_rank,
}


def validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must contain only letters, numbers, dots, hyphens, and underscores"
        )


def _select_training_and_validation(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"date", SPLIT_COLUMN, EXIT_DATE_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"rank dataset is missing split-contract columns: {missing}")
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    exits = pd.to_datetime(frame[EXIT_DATE_COLUMN], errors="raise").dt.normalize()
    splits = frame[SPLIT_COLUMN].astype("string")
    allowed_start = pd.Timestamp("2019-01-01")
    validation_end = pd.Timestamp("2023-12-31")
    if dates.lt(allowed_start).any() or dates.gt(validation_end).any():
        raise ValueError(
            "rank dataset contains dates outside the 2019-2023 development window"
        )
    if splits.isna().any() or splits.str.strip().eq("").any():
        raise ValueError("development rows require non-blank assigned split values")
    invalid_splits = ~splits.isin(("train", "validation"))
    if invalid_splits.any():
        values = sorted(splits.loc[invalid_splits].dropna().unique().tolist())
        raise ValueError(f"development rows contain invalid assigned split values: {values}")
    if exits.isna().any():
        raise ValueError("development rows require valid assigned exit dates")

    training_start = pd.Timestamp("2019-01-01")
    training_end = pd.Timestamp("2022-12-31")
    validation_start = pd.Timestamp("2023-01-01")
    train_mask = splits.eq("train")
    validation_mask = splits.eq("validation")
    if not dates.loc[train_mask].between(training_start, training_end).all():
        raise ValueError("training split dates must stay within 2019-01-01 through 2022-12-31")
    if not exits.loc[train_mask].between(training_start, training_end).all():
        raise ValueError("training exit dates must stay within 2019-01-01 through 2022-12-31")
    if not dates.loc[validation_mask].between(validation_start, validation_end).all():
        raise ValueError("validation split dates must stay within 2023-01-01 through 2023-12-31")
    if not exits.loc[validation_mask].between(validation_start, validation_end).all():
        raise ValueError("validation exit dates must stay within 2023-01-01 through 2023-12-31")

    normalized = frame.copy()
    normalized["date"] = dates
    normalized[EXIT_DATE_COLUMN] = exits
    normalized[SPLIT_COLUMN] = splits
    train = normalized.loc[train_mask].copy()
    validation = normalized.loc[validation_mask].copy()
    train = _finite_label_rows(train)
    if train.empty:
        raise ValueError("no finite 2019-2022 rank labels are available for training")
    if validation.empty:
        raise ValueError("no 2023 rows are available for validation prediction")
    return train, validation


def _unknown_industry_rate(frame: pd.DataFrame, preprocessor: RankPreprocessor) -> float:
    normalized = _normalized_industries(frame[preprocessor.industry_column])
    known = set(preprocessor.industry_categories) - {UNKNOWN_INDUSTRY}
    return float((~normalized.isin(known)).mean())


def _missing_feature_rates(frame: pd.DataFrame, schema: dict[str, Any]) -> tuple[float, dict[str, float]]:
    continuous = schema.get("continuous_feature_columns")
    if not isinstance(continuous, list):
        raise ValueError("rank schema is missing continuous feature columns")
    indicators = [
        column
        for column in continuous
        if isinstance(column, str) and "missing" in column.lower()
    ]
    rates = {
        column: float(pd.to_numeric(frame[column], errors="raise").gt(0.0).mean())
        for column in indicators
    }
    aggregate = float(np.mean(list(rates.values()))) if rates else 0.0
    return aggregate, rates


def _artifact_bytes(directory: Path) -> int:
    return int(
        sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    )


def _artifact_payload_bytes(directory: Path) -> int:
    """Measure persisted run artifacts excluding the self-describing manifest."""
    return int(
        sum(
            path.stat().st_size
            for path in directory.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        )
    )


def _prediction_frame(validation: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    missing = sorted(set(KEY_COLUMNS).difference(validation.columns))
    if missing:
        raise ValueError(f"validation data is missing prediction keys: {missing}")
    if len(scores) != len(validation):
        raise ValueError("validation score count does not match validation rows")
    if validation.duplicated(KEY_COLUMNS).any():
        raise ValueError("validation data contains duplicate prediction keys")
    result = validation.loc[:, KEY_COLUMNS].copy()
    result["split"] = "validation"
    result["horizon"] = 10
    result["target_10d"] = pd.to_numeric(validation.get("target_10d"), errors="coerce")
    result[RANK_TARGET_COLUMN] = pd.to_numeric(
        validation.get(RANK_TARGET_COLUMN), errors="coerce"
    )
    result["score_raw"] = np.asarray(scores, dtype="float64")
    result["pred_rank_pct"] = predicted_percentiles(
        result["score_raw"].to_numpy(), result["date"]
    )
    result["pred_rank_position"] = result.groupby("date", sort=False)[
        "score_raw"
    ].rank(method="average", ascending=True)
    return result


def _write_run_bundle(
    temporary_run: Path,
    config_path: Path,
    schema: dict[str, Any],
    outcome: TrainingOutcome,
    predictions: pd.DataFrame,
) -> None:
    preprocessor = outcome.model_objects.get("preprocessor")
    model = outcome.model_objects.get("model")
    if not isinstance(preprocessor, RankPreprocessor) or model is None:
        raise ValueError("registered trainer did not provide a preprocessor and model")
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    shutil.copyfile(config_path, temporary_run / "config_snapshot.toml")
    feature_schema = {
        **schema,
        "transformed_feature_columns": preprocessor.feature_names,
    }
    (temporary_run / "feature_schema.json").write_text(
        json.dumps(feature_schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    preprocessor_path = temporary_run / "preprocessor.joblib"
    predictions_path = temporary_run / "predictions_10d.parquet"
    joblib.dump(preprocessor, preprocessor_path)
    if isinstance(model, xgb.Booster):
        model.save_model(temporary_run / "model.json")
    elif isinstance(model, lgb.Booster):
        model.save_model(temporary_run / "model.txt")
    elif outcome.model_objects.get("model_type") == "pytorch_mlp":
        import torch

        architecture = outcome.model_objects.get("architecture")
        if not isinstance(architecture, dict):
            raise ValueError("PyTorch model is missing architecture metadata")
        (temporary_run / "model_architecture.json").write_text(
            json.dumps(architecture, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        torch.save(model.state_dict(), temporary_run / "model_state_dict.pt")
    else:
        joblib.dump(model, temporary_run / "model.joblib")
    predictions.to_parquet(predictions_path, index=False)


def _load_persisted_model(temporary_run: Path) -> Any:
    xgboost_path = temporary_run / "model.json"
    lightgbm_path = temporary_run / "model.txt"
    architecture_path = temporary_run / "model_architecture.json"
    if xgboost_path.exists():
        model = xgb.Booster()
        model.load_model(xgboost_path)
        return model
    if lightgbm_path.exists():
        return lgb.Booster(model_file=str(lightgbm_path))
    if architecture_path.exists():
        import torch

        architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
        try:
            input_dim = int(architecture["input_dim"])
            hidden_layers = architecture["hidden_layers"]
            output_bias = float(architecture["output_bias"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid PyTorch model architecture") from error
        model_name = architecture.get("model_name", "mlp_rank_regression")
        if hidden_layers != MLP_HIDDEN_LAYERS:
            raise ValueError("unexpected PyTorch MLP architecture")
        if model_name == "mlp_rank_regression":
            if architecture.get("dropout") != MLP_DROPOUT:
                raise ValueError("unexpected rank-regression PyTorch MLP architecture")
            model = _build_mlp_rank_regression_model(input_dim, output_bias)
        elif model_name == "mlp_pairwise_rank":
            if architecture.get("dropout") != MLP_DROPOUT or output_bias != 0.0:
                raise ValueError("unexpected pairwise PyTorch MLP architecture")
            model = _build_mlp_pairwise_rank_model(input_dim)
        else:
            raise ValueError("unexpected PyTorch MLP model name")
        model.load_state_dict(
            torch.load(
                temporary_run / "model_state_dict.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        return model
    return joblib.load(temporary_run / "model.joblib")


def _predict_model(model: Any, features: np.ndarray) -> np.ndarray:
    if isinstance(model, xgb.Booster):
        return model.predict(xgb.DMatrix(features))
    if model.__class__.__module__.startswith("torch"):
        return _predict_mlp_rank_regression(model, features)
    return model.predict(features)


def _verify_reloaded_predictions(
    temporary_run: Path,
    validation: pd.DataFrame,
    expected_scores: np.ndarray,
    tolerance: float = 1e-12,
) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=rf"^{re.escape(_JOBLIB_NUMPY_SHAPE_WARNING)}$",
            category=DeprecationWarning,
            module=r"^joblib\.numpy_pickle$",
        )
        preprocessor = joblib.load(temporary_run / "preprocessor.joblib")
        model = _load_persisted_model(temporary_run)
    reloaded_scores = _predict_model(
        model,
        preprocessor.transform(
            validation,
            scale_continuous=not isinstance(model, (xgb.Booster, lgb.Booster)),
        ),
    )
    if not np.allclose(
        reloaded_scores,
        expected_scores,
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError("reloaded model predictions do not match staged predictions")
    persisted = pd.read_parquet(temporary_run / "predictions_10d.parquet")
    if not np.allclose(
        persisted["score_raw"].to_numpy(dtype="float64"),
        expected_scores,
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError("staged prediction artifact does not match model predictions")


def _write_manifest(
    temporary_run: Path,
    config_path: Path,
    schema: dict[str, Any],
    model_name: str,
    run_id: str,
    outcome: TrainingOutcome,
    predictions: pd.DataFrame,
) -> None:
    outcome.metadata["artifact_payload_bytes"] = _artifact_payload_bytes(temporary_run)
    outcome.metadata["artifact_bytes"] = _artifact_bytes(temporary_run)
    fixed_parameters = outcome.metadata.get("fixed_parameters")
    if not isinstance(fixed_parameters, dict) or not fixed_parameters:
        raise ValueError("registered trainer did not record fixed parameters")
    required_provenance = (
        "training_seconds",
        "validation_inference_seconds",
        "artifact_payload_bytes",
        "artifact_bytes",
        "source_dataset_sha256",
        "source_schema_sha256",
        "code_version",
        "seed",
        "training_dates",
        "feature_count",
        "training_unknown_industry_rate",
        "validation_unknown_industry_rate",
        "training_missing_feature_rate",
        "validation_missing_feature_rate",
        "training_missing_feature_rates",
        "validation_missing_feature_rates",
    )
    missing = [field for field in required_provenance if field not in outcome.metadata]
    if missing:
        raise ValueError(f"run provenance is missing fields: {missing}")
    manifest_path = temporary_run / "manifest.json"
    for _ in range(10):
        manifest = {
            "status": "completed",
            "run_id": run_id,
            "model_name": model_name,
            "training_rows": int(outcome.metadata["training_rows"]),
            "validation_rows": int(len(predictions)),
            "validation_dates": int(predictions["date"].nunique()),
            "metadata": outcome.metadata,
            "fixed_parameters": fixed_parameters,
            "config_sha256": file_sha256(config_path),
            "rank_dataset_sha256": schema.get("parquet_sha256"),
            "predictions_10d_sha256": file_sha256(
                temporary_run / "predictions_10d.parquet"
            ),
            **{field: outcome.metadata[field] for field in required_provenance},
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_bytes = _artifact_bytes(temporary_run)
        if outcome.metadata["artifact_bytes"] == artifact_bytes:
            return
        outcome.metadata["artifact_bytes"] = artifact_bytes
    raise RuntimeError("manifest artifact byte total did not stabilize")


def train_registered_model(
    config: dict[str, Any],
    config_path: Path,
    model_name: str,
    run_id: str,
) -> Path:
    """Train one registered model and atomically publish an immutable run."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model name: {model_name}")
    trainer = MODEL_REGISTRY.get(model_name)
    if trainer is None:
        raise NotImplementedError(f"model is not registered in this task: {model_name}")
    validate_run_id(run_id)
    config_path = Path(config_path).resolve()
    try:
        paths = config["paths"]
        runs_directory = (config_path.parent / paths["runs_dir"]).resolve()
    except (KeyError, TypeError) as error:
        raise ValueError("training config requires paths.runs_dir") from error
    run_directory = runs_directory / run_id
    if run_directory.exists():
        raise FileExistsError(f"run already exists and is immutable: {run_directory}")

    try:
        dataset_path = (config_path.parent / paths["rank_dataset"]).resolve()
        schema_path = (config_path.parent / paths["rank_schema"]).resolve()
        params = config["models"][model_name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"training config is missing inputs for {model_name}") from error
    dataset, schema = load_rank_dataset(dataset_path, schema_path)
    train, validation = _select_training_and_validation(dataset)
    training_started = perf_counter()
    outcome = trainer(train, validation, schema, params)
    training_elapsed = float(perf_counter() - training_started)
    outcome.metadata["training_rows"] = int(len(train))
    validation_inference_seconds = float(
        outcome.metadata.get("validation_inference_seconds", 0.0)
    )
    if not np.isfinite(validation_inference_seconds) or validation_inference_seconds < 0.0:
        raise ValueError("trainer recorded an invalid validation inference duration")
    preprocessor = outcome.model_objects.get("preprocessor")
    if not isinstance(preprocessor, RankPreprocessor):
        raise ValueError("registered trainer did not provide a RankPreprocessor")
    training_missing_feature_rate, training_missing_feature_rates = _missing_feature_rates(
        train, schema
    )
    validation_missing_feature_rate, validation_missing_feature_rates = _missing_feature_rates(
        validation, schema
    )
    outcome.metadata.update(
        {
            "training_seconds": max(training_elapsed - validation_inference_seconds, 0.0),
            "source_dataset_sha256": schema.get("source_dataset_sha256"),
            "source_schema_sha256": schema.get("source_schema_sha256"),
            "code_version": __version__,
            "seed": FIXED_SEED,
            "training_dates": int(train["date"].nunique()),
            "feature_count": int(len(preprocessor.feature_names)),
            "training_unknown_industry_rate": _unknown_industry_rate(train, preprocessor),
            "validation_unknown_industry_rate": _unknown_industry_rate(validation, preprocessor),
            "training_missing_feature_rate": training_missing_feature_rate,
            "validation_missing_feature_rate": validation_missing_feature_rate,
            "training_missing_feature_rates": training_missing_feature_rates,
            "validation_missing_feature_rates": validation_missing_feature_rates,
        }
    )
    fixed_parameters = outcome.metadata.get("fixed_parameters")
    if not isinstance(fixed_parameters, dict):
        raise ValueError("registered trainer did not record fixed parameters")
    reload_tolerance = fixed_parameters.get("reload_tolerance")
    if (
        not isinstance(reload_tolerance, (int, float))
        or not np.isfinite(reload_tolerance)
        or reload_tolerance < 0.0
    ):
        raise ValueError("registered trainer recorded an invalid reload tolerance")
    outcome.metadata["reload_tolerance"] = float(reload_tolerance)
    predictions = _prediction_frame(validation, outcome.score_validation)

    runs_directory.mkdir(parents=True, exist_ok=True)
    temporary_run = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_directory))
    try:
        _write_run_bundle(
            temporary_run,
            config_path,
            schema,
            outcome,
            predictions,
        )
        _verify_reloaded_predictions(
            temporary_run,
            validation,
            outcome.score_validation,
            tolerance=float(fixed_parameters["reload_tolerance"]),
        )
        _write_manifest(
            temporary_run,
            config_path,
            schema,
            model_name,
            run_id,
            outcome,
            predictions,
        )
        temporary_run.rename(run_directory)
    finally:
        if temporary_run.exists():
            shutil.rmtree(temporary_run)
    return run_directory
