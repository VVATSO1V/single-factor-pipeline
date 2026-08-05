"""Immutable training runs for CSI1000 10-day rank models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

from rank_model.stages.dataset import file_sha256, load_rank_dataset
from rank_model.stages.preprocessing import (
    RankPreprocessor,
    equal_date_weights,
    predicted_percentiles,
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
MLP_HIDDEN_LAYERS = [128, 64, 32]
MLP_BATCH_SIZE = 8192
MLP_EPOCHS = 12
MLP_GRADIENT_CLIP_NORM = 1.0
MLP_SEED = 42


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
    score_validation = model.predict(x_validation)
    if not np.isfinite(score_validation).all():
        raise ValueError("ridge produced non-finite validation scores")
    return TrainingOutcome(
        score_validation=np.asarray(score_validation, dtype="float64"),
        model_objects={"preprocessor": preprocessor, "model": model},
        metadata={"lambda": regularization, "alpha": alpha, "solver": "cholesky"},
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
    score_validation = model.predict(xgb.DMatrix(x_validation))
    if not np.isfinite(score_validation).all():
        raise ValueError("xgboost produced non-finite validation scores")
    return TrainingOutcome(
        score_validation=np.asarray(score_validation, dtype="float64"),
        model_objects={"preprocessor": preprocessor, "model": model},
        metadata={
            "objective": XGB_PARAMS["objective"],
            "boosting_rounds": XGB_BOOSTING_ROUNDS,
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
    score_validation = model.predict(x_validation)
    if not np.isfinite(score_validation).all():
        raise ValueError("lightgbm produced non-finite validation scores")
    return TrainingOutcome(
        score_validation=np.asarray(score_validation, dtype="float64"),
        model_objects={"preprocessor": preprocessor, "model": model},
        metadata={
            "objective": LGB_PARAMS["objective"],
            "boosting_rounds": LGB_BOOSTING_ROUNDS,
        },
    )


def _build_mlp_rank_regression_model(input_dim: int, output_bias: float) -> Any:
    """Build the fixed ReLU network used by the MLP rank regressor."""
    import torch

    layers: list[torch.nn.Module] = []
    previous_width = input_dim
    for width in MLP_HIDDEN_LAYERS:
        layers.extend((torch.nn.Linear(previous_width, width), torch.nn.ReLU()))
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
    optimizer = torch.optim.AdamW(model.parameters())
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

    score_validation = _predict_mlp_rank_regression(model, x_validation)
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
                "output_bias": float(np.mean(y_train)),
            },
        },
        metadata={
            "hidden_layers": MLP_HIDDEN_LAYERS,
            "epochs": MLP_EPOCHS,
            "loss": "mse",
            "optimizer": "adamw",
            "batch_size": MLP_BATCH_SIZE,
            "gradient_clip_norm": MLP_GRADIENT_CLIP_NORM,
            "seed": MLP_SEED,
        },
    )


MODEL_REGISTRY: dict[str, Trainer] = {
    "ridge_rank_regression": train_ridge_rank_regression,
    "xgboost_rank_regression": train_xgboost_rank_regression,
    "lightgbm_rank_regression": train_lightgbm_rank_regression,
    "mlp_rank_regression": train_mlp_rank_regression,
}


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must contain only letters, numbers, dots, hyphens, and underscores"
        )


def _select_training_and_validation(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "date" not in frame:
        raise ValueError("rank dataset is missing date")
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    allowed_start = pd.Timestamp("2019-01-01")
    validation_end = pd.Timestamp("2023-12-31")
    if dates.lt(allowed_start).any() or dates.gt(validation_end).any():
        raise ValueError(
            "rank dataset contains dates outside the 2019-2023 development window"
        )
    normalized = frame.copy()
    normalized["date"] = dates
    train = normalized.loc[dates.dt.year.between(2019, 2022)].copy()
    validation = normalized.loc[dates.dt.year.eq(2023)].copy()
    train = _finite_label_rows(train)
    if train.empty:
        raise ValueError("no finite 2019-2022 rank labels are available for training")
    if validation.empty:
        raise ValueError("no 2023 rows are available for validation prediction")
    return train, validation


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
        if hidden_layers != MLP_HIDDEN_LAYERS:
            raise ValueError("unexpected PyTorch MLP architecture")
        model = _build_mlp_rank_regression_model(input_dim, output_bias)
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
    tolerance = (
        1e-6 if model.__class__.__module__.startswith("torch") else 1e-12
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
    manifest = {
        "run_id": run_id,
        "model_name": model_name,
        "training_rows": int(outcome.metadata["training_rows"]),
        "validation_rows": int(len(predictions)),
        "validation_dates": int(predictions["date"].nunique()),
        "metadata": outcome.metadata,
        "config_sha256": file_sha256(config_path),
        "rank_dataset_sha256": schema.get("parquet_sha256"),
    }
    (temporary_run / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    _validate_run_id(run_id)
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
    outcome = trainer(train, validation, schema, params)
    outcome.metadata["training_rows"] = int(len(train))
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
        _verify_reloaded_predictions(temporary_run, validation, outcome.score_validation)
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
