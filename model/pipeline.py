"""Unified command-line entry for model data and modeling stages."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tomllib
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.toml"
REQUIRED_SECTIONS = (
    "project",
    "paths",
    "factors",
    "target",
    "split",
    "execution",
    "preprocessing",
    "context",
    "models",
)
REQUIRED_PATHS = (
    "market_panel",
    "trading_calendar",
    "factor_wide",
    "target",
    "model_dataset",
    "model_schema",
    "sample_index",
    "split_summary",
    "context_dataset",
    "context_schema",
    "runs_dir",
)
SPLIT_NAMES = ("train", "validation", "test", "final_train")


def resolve_config_path(config_path: Path, raw_path: str) -> Path:
    """Resolve a configured path relative to the configuration file."""
    return (Path(config_path).resolve().parent / raw_path).resolve()


def _parse_iso_date(value: object, label: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must use YYYY-MM-DD: {value!r}") from exc


def validate_config(config: dict[str, Any], path: Path) -> None:
    """Validate structural settings before any stage reads or writes data."""
    for section in REQUIRED_SECTIONS:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"{path}: missing [{section}] section")

    project = config["project"]
    for key in (
        "name",
        "index_code",
        "start_date",
        "end_date",
        "industry_source",
        "env_path",
    ):
        if not project.get(key):
            raise ValueError(f"{path}: [project].{key} is required")
    project_start = _parse_iso_date(project["start_date"], "project.start_date")
    project_end = _parse_iso_date(project["end_date"], "project.end_date")
    if project_start > project_end:
        raise ValueError(f"{path}: project start_date is after end_date")

    path_settings = config["paths"]
    for key in REQUIRED_PATHS:
        if not path_settings.get(key):
            raise ValueError(f"{path}: [paths].{key} is required")

    raw_factor_paths = config["factors"].get("paths")
    if not isinstance(raw_factor_paths, list) or not raw_factor_paths:
        raise ValueError(f"{path}: [factors].paths must be a non-empty list")
    if not all(isinstance(item, str) and item for item in raw_factor_paths):
        raise ValueError(f"{path}: every factor path must be a non-empty string")
    resolved_factors = [
        resolve_config_path(path, raw_path) for raw_path in raw_factor_paths
    ]
    if len(set(resolved_factors)) != len(resolved_factors):
        raise ValueError(f"{path}: [factors].paths contains duplicates")

    horizons = config["target"].get("horizons")
    if horizons != [1, 5, 10]:
        raise ValueError(
            f"{path}: target horizons must be exactly [1, 5, 10]"
        )
    if config["target"].get("type") != "absolute_open_to_open_return":
        raise ValueError(f"{path}: unsupported target.type")
    if config["target"].get("entry_price") != "post_open":
        raise ValueError(f"{path}: target.entry_price must be post_open")

    split = config["split"]
    parsed_periods: dict[str, tuple[date, date]] = {}
    for name in SPLIT_NAMES:
        period = split.get(name)
        if not isinstance(period, dict):
            raise ValueError(f"{path}: missing [split.{name}] section")
        start = _parse_iso_date(period.get("start"), f"split.{name}.start")
        end = _parse_iso_date(period.get("end"), f"split.{name}.end")
        if start > end:
            raise ValueError(f"{path}: split.{name} start is after end")
        parsed_periods[name] = (start, end)
    for previous, current in (("train", "validation"), ("validation", "test")):
        if parsed_periods[previous][1] >= parsed_periods[current][0]:
            raise ValueError(f"{path}: {previous} and {current} overlap")
    if parsed_periods["final_train"] != (
        parsed_periods["train"][0],
        parsed_periods["validation"][1],
    ):
        raise ValueError(
            f"{path}: final_train must span train start through validation end"
        )

    min_listing_days = config["execution"].get("min_listing_days")
    if (
        not isinstance(min_listing_days, int)
        or isinstance(min_listing_days, bool)
        or min_listing_days <= 0
    ):
        raise ValueError(f"{path}: execution.min_listing_days must be positive")

    preprocessing = config["preprocessing"]
    mad_width = preprocessing.get("mad_width")
    if (
        not isinstance(mad_width, (int, float))
        or isinstance(mad_width, bool)
        or mad_width <= 0
    ):
        raise ValueError(f"{path}: preprocessing.mad_width must be positive")
    for key in ("date_chunk_size", "expected_cross_section_size"):
        value = preprocessing.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{path}: preprocessing.{key} must be positive")
    if preprocessing.get("dtype") != "float32":
        raise ValueError(f"{path}: preprocessing.dtype must be float32")

    ridge = config["models"].get("ridge")
    if not isinstance(ridge, dict):
        raise ValueError(f"{path}: missing [models.ridge] section")
    lambda_grid = ridge.get("lambda_grid")
    if not isinstance(lambda_grid, list) or not lambda_grid:
        raise ValueError(
            f"{path}: models.ridge.lambda_grid must be a non-empty list"
        )
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < 0
        for value in lambda_grid
    ):
        raise ValueError(
            f"{path}: models.ridge.lambda_grid must contain nonnegative numbers"
        )
    if any(
        current <= previous
        for previous, current in zip(lambda_grid, lambda_grid[1:])
    ):
        raise ValueError(
            f"{path}: models.ridge.lambda_grid must be strictly increasing"
        )
    if ridge.get("solver") != "cholesky":
        raise ValueError(f"{path}: models.ridge.solver must be cholesky")
    tolerance = ridge.get("rank_ic_tolerance")
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or tolerance < 0
    ):
        raise ValueError(
            f"{path}: models.ridge.rank_ic_tolerance must be nonnegative"
        )

    xgboost = config["models"].get("xgboost")
    if not isinstance(xgboost, dict):
        raise ValueError(f"{path}: missing [models.xgboost] section")
    for key, expected in (
        ("objective", "reg:squarederror"),
        ("eval_metric", "rmse"),
        ("tree_method", "hist"),
    ):
        if xgboost.get(key) != expected:
            raise ValueError(
                f"{path}: models.xgboost.{key} must be {expected}"
            )
    for key in ("max_boost_rounds", "early_stopping_rounds", "max_bin"):
        value = xgboost.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(
                f"{path}: models.xgboost.{key} must be positive"
            )
    if xgboost["early_stopping_rounds"] >= xgboost["max_boost_rounds"]:
        raise ValueError(
            f"{path}: XGBoost early stopping must be below max rounds"
        )
    for key in ("learning_rate", "subsample", "colsample_bytree"):
        value = xgboost.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 < value <= 1
        ):
            raise ValueError(
                f"{path}: models.xgboost.{key} must be in (0, 1]"
            )
    for key in ("reg_lambda", "reg_alpha", "gamma"):
        value = xgboost.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(
                f"{path}: models.xgboost.{key} must be nonnegative"
            )
    for key in ("seed", "nthread"):
        value = xgboost.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(
                f"{path}: models.xgboost.{key} must be a nonnegative integer"
            )
    xgboost_tolerance = xgboost.get("rank_ic_tolerance")
    if (
        not isinstance(xgboost_tolerance, (int, float))
        or isinstance(xgboost_tolerance, bool)
        or xgboost_tolerance < 0
    ):
        raise ValueError(
            f"{path}: models.xgboost.rank_ic_tolerance must be nonnegative"
        )
    candidates = xgboost.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(
            f"{path}: models.xgboost.candidates must be a non-empty list"
        )
    candidate_names: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(
                f"{path}: XGBoost candidate {index} must be a table"
            )
        name = candidate.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{path}: XGBoost candidate {index} needs a name"
            )
        candidate_names.append(name)
        max_depth = candidate.get("max_depth")
        if (
            not isinstance(max_depth, int)
            or isinstance(max_depth, bool)
            or not 1 <= max_depth <= 16
        ):
            raise ValueError(
                f"{path}: XGBoost {name}.max_depth must be in [1, 16]"
            )
        min_child_weight = candidate.get("min_child_weight")
        if (
            not isinstance(min_child_weight, (int, float))
            or isinstance(min_child_weight, bool)
            or min_child_weight < 0
        ):
            raise ValueError(
                f"{path}: XGBoost {name}.min_child_weight must be nonnegative"
            )
    if len(candidate_names) != len(set(candidate_names)):
        raise ValueError(f"{path}: XGBoost candidate names must be unique")

    lightgbm = config["models"].get("lightgbm")
    if not isinstance(lightgbm, dict):
        raise ValueError(f"{path}: missing [models.lightgbm] section")
    for key, expected in (
        ("objective", "regression"),
        ("metric", "rmse"),
        ("boosting_type", "gbdt"),
    ):
        if lightgbm.get(key) != expected:
            raise ValueError(
                f"{path}: models.lightgbm.{key} must be {expected}"
            )
    for key in (
        "max_boost_rounds",
        "early_stopping_rounds",
        "bagging_freq",
        "max_bin",
    ):
        value = lightgbm.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(
                f"{path}: models.lightgbm.{key} must be positive"
            )
    if lightgbm["early_stopping_rounds"] >= lightgbm["max_boost_rounds"]:
        raise ValueError(
            f"{path}: LightGBM early stopping must be below max rounds"
        )
    for key in ("learning_rate", "bagging_fraction", "feature_fraction"):
        value = lightgbm.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 < value <= 1
        ):
            raise ValueError(
                f"{path}: models.lightgbm.{key} must be in (0, 1]"
            )
    for key in ("lambda_l2", "lambda_l1", "min_gain_to_split"):
        value = lightgbm.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(
                f"{path}: models.lightgbm.{key} must be nonnegative"
            )
    for key in ("seed", "num_threads"):
        value = lightgbm.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(
                f"{path}: models.lightgbm.{key} must be a nonnegative integer"
            )
    for key in ("deterministic", "force_col_wise"):
        if lightgbm.get(key) is not True:
            raise ValueError(
                f"{path}: models.lightgbm.{key} must be true"
            )
    lightgbm_tolerance = lightgbm.get("rank_ic_tolerance")
    if (
        not isinstance(lightgbm_tolerance, (int, float))
        or isinstance(lightgbm_tolerance, bool)
        or lightgbm_tolerance < 0
    ):
        raise ValueError(
            f"{path}: models.lightgbm.rank_ic_tolerance must be nonnegative"
        )
    lightgbm_candidates = lightgbm.get("candidates")
    if not isinstance(lightgbm_candidates, list) or not lightgbm_candidates:
        raise ValueError(
            f"{path}: models.lightgbm.candidates must be a non-empty list"
        )
    lightgbm_names: list[str] = []
    for index, candidate in enumerate(lightgbm_candidates):
        if not isinstance(candidate, dict):
            raise ValueError(
                f"{path}: LightGBM candidate {index} must be a table"
            )
        name = candidate.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{path}: LightGBM candidate {index} needs a name"
            )
        lightgbm_names.append(name)
        num_leaves = candidate.get("num_leaves")
        max_depth = candidate.get("max_depth")
        min_data_in_leaf = candidate.get("min_data_in_leaf")
        if (
            not isinstance(num_leaves, int)
            or isinstance(num_leaves, bool)
            or num_leaves < 2
        ):
            raise ValueError(
                f"{path}: LightGBM {name}.num_leaves must be at least 2"
            )
        if (
            not isinstance(max_depth, int)
            or isinstance(max_depth, bool)
            or not 1 <= max_depth <= 16
        ):
            raise ValueError(
                f"{path}: LightGBM {name}.max_depth must be in [1, 16]"
            )
        if num_leaves > 2**max_depth:
            raise ValueError(
                f"{path}: LightGBM {name}.num_leaves exceeds 2^max_depth"
            )
        if (
            not isinstance(min_data_in_leaf, int)
            or isinstance(min_data_in_leaf, bool)
            or min_data_in_leaf <= 0
        ):
            raise ValueError(
                f"{path}: LightGBM {name}.min_data_in_leaf must be positive"
            )
    if len(lightgbm_names) != len(set(lightgbm_names)):
        raise ValueError(f"{path}: LightGBM candidate names must be unique")

    mlp = config["models"].get("mlp")
    if not isinstance(mlp, dict):
        raise ValueError(f"{path}: missing [models.mlp] section")
    for key, expected in (
        ("framework", "pytorch"),
        ("loss", "mse"),
        ("optimizer", "adamw"),
        ("activation", "relu"),
        ("device", "cpu"),
    ):
        if mlp.get(key) != expected:
            raise ValueError(f"{path}: models.mlp.{key} must be {expected}")
    for key in (
        "batch_size",
        "max_epochs",
        "early_stopping_patience",
        "lr_scheduler_patience",
    ):
        value = mlp.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{path}: models.mlp.{key} must be positive")
    if mlp["lr_scheduler_patience"] >= mlp["early_stopping_patience"]:
        raise ValueError(
            f"{path}: MLP scheduler patience must be below early stopping"
        )
    for key in (
        "learning_rate",
        "early_stopping_min_delta",
        "min_learning_rate",
        "gradient_clip_norm",
    ):
        value = mlp.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{path}: models.mlp.{key} must be positive")
    if mlp["min_learning_rate"] >= mlp["learning_rate"]:
        raise ValueError(
            f"{path}: MLP minimum learning rate must be below initial rate"
        )
    lr_factor = mlp.get("lr_scheduler_factor")
    if (
        not isinstance(lr_factor, (int, float))
        or isinstance(lr_factor, bool)
        or not 0 < lr_factor < 1
    ):
        raise ValueError(
            f"{path}: models.mlp.lr_scheduler_factor must be in (0, 1)"
        )
    for key in ("seed", "num_threads"):
        value = mlp.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(
                f"{path}: models.mlp.{key} must be a nonnegative integer"
            )
    if mlp.get("target_standardization") is not False:
        raise ValueError(
            f"{path}: models.mlp.target_standardization must be false"
        )
    mlp_tolerance = mlp.get("rank_ic_tolerance")
    if (
        not isinstance(mlp_tolerance, (int, float))
        or isinstance(mlp_tolerance, bool)
        or mlp_tolerance < 0
    ):
        raise ValueError(
            f"{path}: models.mlp.rank_ic_tolerance must be nonnegative"
        )
    mlp_candidates = mlp.get("candidates")
    if not isinstance(mlp_candidates, list) or not mlp_candidates:
        raise ValueError(
            f"{path}: models.mlp.candidates must be a non-empty list"
        )
    mlp_names: list[str] = []
    for index, candidate in enumerate(mlp_candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"{path}: MLP candidate {index} must be a table")
        name = candidate.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}: MLP candidate {index} needs a name")
        mlp_names.append(name)
        hidden_layers = candidate.get("hidden_layers")
        if (
            not isinstance(hidden_layers, list)
            or not hidden_layers
            or any(
                not isinstance(width, int)
                or isinstance(width, bool)
                or not 1 <= width <= 4096
                for width in hidden_layers
            )
        ):
            raise ValueError(
                f"{path}: MLP {name}.hidden_layers must be positive integers"
            )
        dropout = candidate.get("dropout")
        if (
            not isinstance(dropout, (int, float))
            or isinstance(dropout, bool)
            or not 0 <= dropout < 1
        ):
            raise ValueError(
                f"{path}: MLP {name}.dropout must be in [0, 1)"
            )
        weight_decay = candidate.get("weight_decay")
        if (
            not isinstance(weight_decay, (int, float))
            or isinstance(weight_decay, bool)
            or weight_decay < 0
        ):
            raise ValueError(
                f"{path}: MLP {name}.weight_decay must be nonnegative"
            )
    if len(mlp_names) != len(set(mlp_names)):
        raise ValueError(f"{path}: MLP candidate names must be unique")

    residual = config["models"].get("residual_rank_mlp")
    if not isinstance(residual, dict):
        raise ValueError(
            f"{path}: missing [models.residual_rank_mlp] section"
        )
    positive_integer_keys = (
        "dates_per_batch",
        "pairs_per_stock",
        "max_epochs",
        "early_stopping_patience",
        "lr_scheduler_patience",
    )
    for key in positive_integer_keys:
        value = residual.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(
                f"{path}: models.residual_rank_mlp.{key} must be positive"
            )
    if (
        residual["lr_scheduler_patience"]
        >= residual["early_stopping_patience"]
    ):
        raise ValueError(
            f"{path}: residual rank MLP scheduler patience must be below "
            "early stopping"
        )
    for key in (
        "ridge_lambda",
        "top_group_weight",
        "second_group_weight",
        "huber_mad_multiplier",
        "learning_rate",
        "early_stopping_min_delta",
        "min_learning_rate",
        "gradient_clip_norm",
    ):
        value = residual.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(
                f"{path}: models.residual_rank_mlp.{key} must be positive"
            )
    if residual["min_learning_rate"] >= residual["learning_rate"]:
        raise ValueError(
            f"{path}: residual rank MLP minimum learning rate must be below "
            "initial rate"
        )
    for key in ("adjacent_pair_fraction", "lr_scheduler_factor"):
        value = residual.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 < value < 1
        ):
            raise ValueError(
                f"{path}: models.residual_rank_mlp.{key} must be in (0, 1)"
            )
    residual_tolerance = residual.get("rank_ic_tolerance")
    if (
        not isinstance(residual_tolerance, (int, float))
        or isinstance(residual_tolerance, bool)
        or residual_tolerance < 0
    ):
        raise ValueError(
            f"{path}: models.residual_rank_mlp.rank_ic_tolerance must be "
            "nonnegative"
        )
    for key in ("gamma_grid", "rank_lambda_grid"):
        values = residual.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value < 0
                for value in values
            )
            or len(values) != len(set(values))
            or 0.0 not in values
        ):
            raise ValueError(
                f"{path}: models.residual_rank_mlp.{key} must be a unique "
                "nonnegative grid containing zero"
            )
    if any(value > 1 for value in residual["gamma_grid"]):
        raise ValueError(
            f"{path}: residual rank MLP gamma values must not exceed 1"
        )
    for key in ("seed", "num_threads"):
        value = residual.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(
                f"{path}: models.residual_rank_mlp.{key} must be a "
                "nonnegative integer"
            )
    stability_seeds = residual.get("stability_seeds")
    if (
        not isinstance(stability_seeds, list)
        or len(stability_seeds) != 3
        or len(stability_seeds) != len(set(stability_seeds))
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in stability_seeds
        )
        or residual["seed"] not in stability_seeds
    ):
        raise ValueError(
            f"{path}: models.residual_rank_mlp.stability_seeds must contain "
            "three unique nonnegative integers including seed"
        )
    if residual.get("target_standardization") is not False:
        raise ValueError(
            f"{path}: residual rank MLP target_standardization must be false"
        )
    residual_candidates = residual.get("candidates")
    if not isinstance(residual_candidates, list) or not residual_candidates:
        raise ValueError(
            f"{path}: residual rank MLP candidates must be a non-empty list"
        )
    residual_names: list[str] = []
    for index, candidate in enumerate(residual_candidates):
        if not isinstance(candidate, dict):
            raise ValueError(
                f"{path}: residual rank MLP candidate {index} must be a table"
            )
        name = candidate.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{path}: residual rank MLP candidate {index} needs a name"
            )
        residual_names.append(name)
        hidden_layers = candidate.get("hidden_layers")
        if (
            not isinstance(hidden_layers, list)
            or not hidden_layers
            or any(
                not isinstance(width, int)
                or isinstance(width, bool)
                or not 1 <= width <= 4096
                for width in hidden_layers
            )
        ):
            raise ValueError(
                f"{path}: residual rank MLP {name}.hidden_layers must be "
                "positive integers"
            )
        dropout = candidate.get("dropout")
        if (
            not isinstance(dropout, (int, float))
            or isinstance(dropout, bool)
            or not 0 <= dropout < 1
        ):
            raise ValueError(
                f"{path}: residual rank MLP {name}.dropout must be in [0, 1)"
            )
        weight_decay = candidate.get("weight_decay")
        if (
            not isinstance(weight_decay, (int, float))
            or isinstance(weight_decay, bool)
            or weight_decay < 0
        ):
            raise ValueError(
                f"{path}: residual rank MLP {name}.weight_decay must be "
                "nonnegative"
            )
    if len(residual_names) != len(set(residual_names)):
        raise ValueError(
            f"{path}: residual rank MLP candidate names must be unique"
        )
    _validate_context_settings(config, path)


def _validate_context_settings(config: dict[str, Any], path: Path) -> None:
    context = config["context"]
    if context.get("development_end") != "2023-12-31":
        raise ValueError(f"{path}: context.development_end must be 2023-12-31")
    if context.get("windows") != [1, 5, 20]:
        raise ValueError(f"{path}: context.windows must be [1, 5, 20]")
    for key in ("industry_shrink_k", "cap_coverage_threshold"):
        value = context.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{path}: context.{key} must be positive")
    if context["cap_coverage_threshold"] > 1:
        raise ValueError(
            f"{path}: context.cap_coverage_threshold must not exceed 1"
        )
    for key in ("champion_run", "test_contamination_disclosure"):
        if not isinstance(context.get(key), str) or not context[key].strip():
            raise ValueError(f"{path}: context.{key} is required")
    for key in ("hac_lag",):
        value = context.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{path}: context.{key} must be nonnegative")
    ratio = context.get("min_prediction_std_ratio")
    if (
        not isinstance(ratio, (int, float))
        or isinstance(ratio, bool)
        or not 0 < ratio < 1
    ):
        raise ValueError(
            f"{path}: context.min_prediction_std_ratio must be in (0, 1)"
        )

    model = config["models"].get("context_decomposition")
    if not isinstance(model, dict):
        raise ValueError(
            f"{path}: missing [models.context_decomposition] section"
        )
    if model.get("lambda_grid") != [0.1, 1.0, 10.0]:
        raise ValueError(
            f"{path}: context decomposition lambda_grid must be [0.1, 1, 10]"
        )
    if model.get("rounds_grid") != [10, 20, 40, 80, 120, 200]:
        raise ValueError(
            f"{path}: context decomposition rounds_grid is invalid"
        )
    if model.get("prediction_years") != [2020, 2021, 2022]:
        raise ValueError(
            f"{path}: context decomposition prediction_years is invalid"
        )
    exact_values = {
        "lightgbm_objective": "regression",
        "lightgbm_metric": "rmse",
        "lightgbm_num_leaves": 31,
        "lightgbm_max_depth": 5,
        "lightgbm_min_data_in_leaf": 1000,
        "lightgbm_seed": 42,
        "mlp_hidden_layers": [128, 64, 32],
        "mlp_dropout": 0.3,
        "mlp_weight_decay": 0.001,
        "mlp_epochs": 6,
        "mlp_seed": 42,
        "mlp_huber_mad_multiplier": 1.5,
    }
    for key, expected in exact_values.items():
        if model.get(key) != expected:
            raise ValueError(
                f"{path}: models.context_decomposition.{key} must be {expected}"
            )
    positive_numbers = (
        "lightgbm_learning_rate",
        "lightgbm_bagging_fraction",
        "lightgbm_feature_fraction",
        "lightgbm_lambda_l2",
        "mlp_learning_rate",
        "mlp_dates_per_batch",
        "mlp_gradient_clip_norm",
        "mlp_num_threads",
    )
    for key in positive_numbers:
        value = model.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(
                f"{path}: models.context_decomposition.{key} must be positive"
            )
    for key in ("lightgbm_lambda_l1",):
        value = model.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(
                f"{path}: models.context_decomposition.{key} must be nonnegative"
            )
    for key in ("lightgbm_bagging_freq", "lightgbm_num_threads"):
        value = model.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"{path}: models.context_decomposition.{key} must be nonnegative"
            )


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate TOML settings."""
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    validate_config(config, config_path)
    return config


def make_parser() -> argparse.ArgumentParser:
    """Create the public command parser."""
    parser = argparse.ArgumentParser(
        description="Build and run the CSI1000 multi-factor model pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="TOML configuration path.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Validate configuration and inputs.")
    fetch_parser = subparsers.add_parser(
        "fetch-market",
        help="Explicitly download Ricequant market data.",
    )
    fetch_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing market source files.",
    )
    subparsers.add_parser(
        "prepare-data",
        help="Build all local model data products.",
    )
    subparsers.add_parser(
        "prepare-context",
        help="Build the local development-only 10-day context dataset.",
    )
    train_parser = subparsers.add_parser(
        "train",
        help="Train and select a registered model.",
    )
    train_parser.add_argument(
        "--model",
        choices=(
            "ridge",
            "xgboost",
            "lightgbm",
            "mlp",
            "residual_rank_mlp",
            "context_decomposition",
        ),
        required=True,
        help="Registered model name.",
    )
    train_parser.add_argument(
        "--horizon",
        choices=("1", "5", "10", "all"),
        required=True,
        help="Return horizon in trading days, or all supported horizons.",
    )
    train_parser.add_argument(
        "--run-id",
        required=True,
        help="Unique output directory name under model/runs.",
    )
    return parser


def _configured_path(
    config: dict[str, Any],
    config_path: Path,
    name: str,
) -> Path:
    return resolve_config_path(config_path, config["paths"][name])


def command_fetch_market(
    config: dict[str, Any],
    config_path: Path,
    *,
    force: bool,
) -> None:
    """Download market sources only after an explicit online command."""
    from model.stages import market

    project = config["project"]
    panel_path = _configured_path(config, config_path, "market_panel")
    calendar_path = _configured_path(config, config_path, "trading_calendar")
    existing = [path for path in (panel_path, calendar_path) if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"market source already exists: {joined}; use --force to overwrite"
        )
    panel, calendar = market.build_market_panel(
        env_path=resolve_config_path(config_path, project["env_path"]),
        start_date=project["start_date"],
        end_date=project["end_date"],
        index_code=project["index_code"],
        industry_source=project["industry_source"],
    )
    market.write_market_outputs(
        panel,
        calendar,
        panel_path,
        calendar_path,
    )


def _factor_paths(config: dict[str, Any], config_path: Path) -> list[Path]:
    return [
        resolve_config_path(config_path, raw_path)
        for raw_path in config["factors"]["paths"]
    ]


def _split_periods(
    config: dict[str, Any],
) -> dict[str, tuple[str, str]]:
    return {
        name: (
            config["split"][name]["start"],
            config["split"][name]["end"],
        )
        for name in ("train", "validation", "test")
    }


def _read_csv_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            columns = next(csv.reader(handle))
        except StopIteration as exc:
            raise ValueError(f"{path} is empty") from exc
    if len(columns) != len(set(columns)):
        raise ValueError(f"{path} contains duplicate columns")
    return columns


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_doctor(
    config: dict[str, Any],
    config_path: Path,
) -> list[str]:
    """Validate environment, source contracts, and existing artifact lineage."""
    from model.stages.features import infer_factor_name

    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11 or newer is required")
    messages = [f"Python {sys.version.split()[0]}"]
    for module_name in (
        "pandas",
        "numpy",
        "pyarrow",
        "rqdatac",
        "sklearn",
        "joblib",
        "xgboost",
        "lightgbm",
        "torch",
    ):
        if importlib.util.find_spec(module_name) is None:
            raise RuntimeError(f"missing Python dependency: {module_name}")
    messages.append(
        "dependencies: pandas, numpy, pyarrow, rqdatac, sklearn, joblib, "
        "xgboost, lightgbm, torch"
    )

    paths = {
        name: _configured_path(config, config_path, name)
        for name in REQUIRED_PATHS
    }
    factor_paths = _factor_paths(config, config_path)
    factor_names: list[str] = []
    required_factor_columns = {"date", "stock_code", "factor_value"}
    for factor_path in factor_paths:
        if not factor_path.exists():
            raise FileNotFoundError(f"factor file not found: {factor_path}")
        columns = set(_read_csv_columns(factor_path))
        missing = sorted(required_factor_columns.difference(columns))
        if missing:
            raise ValueError(
                f"{factor_path} is missing required columns: {missing}"
            )
        factor_names.append(infer_factor_name(factor_path))
    if len(factor_names) != len(set(factor_names)):
        duplicates = sorted(
            {
                name
                for name in factor_names
                if factor_names.count(name) > 1
            }
        )
        raise ValueError(f"duplicate inferred factor names: {duplicates}")
    messages.append(f"{len(factor_names)} factor files: ready")

    source_contracts = {
        paths["market_panel"]: {
            "date",
            "stock_code",
            "in_universe",
            "post_open",
            "post_close",
            "raw_close",
            "limit_up",
            "limit_down",
            "is_st",
            "is_suspended",
            "industry",
            "market_cap",
        },
        paths["trading_calendar"]: {"date"},
    }
    for source_path, required_columns in source_contracts.items():
        if not source_path.exists():
            raise FileNotFoundError(f"market source not found: {source_path}")
        columns = set(_read_csv_columns(source_path))
        missing = sorted(required_columns.difference(columns))
        if missing:
            raise ValueError(
                f"{source_path} is missing required columns: {missing}"
            )
    messages.append("market panel and trading calendar: ready")

    schema_path = paths["model_schema"]
    dataset_path = paths["model_dataset"]
    schema: dict[str, Any] | None = None
    if schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if schema.get("feature_columns") != factor_names:
            raise ValueError(
                f"{schema_path} feature_columns do not match config factors"
            )
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"schema exists but model dataset is missing: {dataset_path}"
            )
        expected_hash = schema.get("parquet_sha256")
        actual_hash = _file_sha256(dataset_path)
        if expected_hash != actual_hash:
            raise ValueError(
                "model dataset hash does not match schema: "
                f"expected={expected_hash} actual={actual_hash}"
            )
        messages.append("model dataset and schema hash: matched")
    elif dataset_path.exists():
        raise FileNotFoundError(
            f"model dataset exists but schema is missing: {schema_path}"
        )
    else:
        messages.append("model dataset: not built")

    sample_path = paths["sample_index"]
    summary_path = paths["split_summary"]
    if sample_path.exists() != summary_path.exists():
        missing_path = summary_path if sample_path.exists() else sample_path
        raise FileNotFoundError(f"incomplete sample-index bundle: {missing_path}")
    if sample_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("uses_entry_tradeable") is not False:
            raise ValueError(
                f"{summary_path} must set uses_entry_tradeable to false"
            )
        expected_split_periods = {
            name: {
                "start": config["split"][name]["start"],
                "end": config["split"][name]["end"],
            }
            for name in ("train", "validation", "test")
        }
        if summary.get("split_periods") != expected_split_periods:
            raise ValueError(
                f"{summary_path} split_periods do not match config"
            )
        expected_final_train = {
            "start": config["split"]["final_train"]["start"],
            "end": config["split"]["final_train"]["end"],
        }
        if summary.get("final_train_period") != expected_final_train:
            raise ValueError(
                f"{summary_path} final_train_period does not match config"
            )
        if schema is not None and summary.get(
            "source_model_dataset_sha256"
        ) != schema.get("parquet_sha256"):
            raise ValueError(
                f"{summary_path} references a different model dataset"
            )
        actual_sample_hash = _file_sha256(sample_path)
        if summary.get("sample_index_sha256") != actual_sample_hash:
            raise ValueError(
                "sample index hash does not match split summary: "
                f"{sample_path}"
            )
        messages.append("sample index and split summary hash: matched")
    else:
        messages.append("sample index: not built")

    context_dataset = paths["context_dataset"]
    context_schema = paths["context_schema"]
    if context_dataset.exists() != context_schema.exists():
        missing_path = context_schema if context_dataset.exists() else context_dataset
        raise FileNotFoundError(f"incomplete context bundle: {missing_path}")
    if context_dataset.exists():
        published_context = json.loads(
            context_schema.read_text(encoding="utf-8")
        )
        if published_context.get("development_end") != "2023-12-31":
            raise ValueError(
                f"{context_schema} development_end must be 2023-12-31"
            )
        if published_context.get("uses_entry_tradeable") is not False:
            raise ValueError(
                f"{context_schema} must set uses_entry_tradeable to false"
            )
        if published_context.get("context_dataset_sha256") != _file_sha256(
            context_dataset
        ):
            raise ValueError("context dataset hash does not match its schema")
        messages.append("context dataset and schema hash: matched")
    else:
        messages.append("context dataset: not built")
    return messages


def command_doctor(config: dict[str, Any], config_path: Path) -> None:
    for message in run_doctor(config, config_path):
        print(f"[ok] {message}")


def command_prepare_data(
    config: dict[str, Any],
    config_path: Path,
) -> None:
    """Build every local data artifact without accessing Ricequant."""
    from model.stages import dataset, features, labels

    paths = {
        name: _configured_path(config, config_path, name)
        for name in REQUIRED_PATHS
    }
    factor_paths = _factor_paths(config, config_path)
    required_inputs = [
        paths["market_panel"],
        paths["trading_calendar"],
        *factor_paths,
    ]
    missing = [path for path in required_inputs if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "prepare-data requires existing local inputs:\n"
            f"{formatted}\n"
            "Run fetch-market only when the market sources are missing."
        )

    print("stage 1/4: build factor-wide table", flush=True)
    features.build_factor_table(
        paths["market_panel"],
        factor_paths,
        paths["factor_wide"],
    )
    print("stage 2/4: build return targets", flush=True)
    labels.build_target_table(
        paths["market_panel"],
        paths["trading_calendar"],
        paths["factor_wide"],
        paths["target"],
    )
    print("stage 3/4: build raw model dataset", flush=True)
    dataset.build_model_dataset(
        paths["factor_wide"],
        paths["target"],
        paths["market_panel"],
        paths["trading_calendar"],
        paths["model_dataset"],
        paths["model_schema"],
        min_listing_days=config["execution"]["min_listing_days"],
    )
    print("stage 4/4: build sample time index", flush=True)
    final_train = config["split"]["final_train"]
    dataset.build_sample_index(
        paths["model_dataset"],
        paths["model_schema"],
        paths["trading_calendar"],
        paths["sample_index"],
        paths["split_summary"],
        split_periods=_split_periods(config),
        final_train_period=(
            final_train["start"],
            final_train["end"],
        ),
    )
    print("prepare-data completed", flush=True)


def command_prepare_context(
    config: dict[str, Any],
    config_path: Path,
) -> None:
    """Build point-in-time context and decomposed labels from local files."""
    from model.stages.context import build_context_dataset

    required_names = (
        "model_dataset",
        "model_schema",
        "sample_index",
        "split_summary",
        "market_panel",
        "trading_calendar",
    )
    paths = {
        name: _configured_path(config, config_path, name)
        for name in (*required_names, "context_dataset", "context_schema")
    }
    missing = [paths[name] for name in required_names if not paths[name].exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "prepare-context requires existing local inputs:\n" + formatted
        )
    context = config["context"]
    preprocessing = config["preprocessing"]
    build_context_dataset(
        paths["model_dataset"],
        paths["model_schema"],
        paths["sample_index"],
        paths["split_summary"],
        paths["market_panel"],
        paths["trading_calendar"],
        paths["context_dataset"],
        paths["context_schema"],
        development_end=context["development_end"],
        mad_width=float(preprocessing["mad_width"]),
        expected_cross_section_size=int(
            preprocessing["expected_cross_section_size"]
        ),
        date_chunk_size=int(preprocessing["date_chunk_size"]),
        windows=tuple(int(value) for value in context["windows"]),
        shrink_k=float(context["industry_shrink_k"]),
        cap_coverage_threshold=float(context["cap_coverage_threshold"]),
    )


def parse_horizons(value: str) -> list[int]:
    """Map the public CLI value to supported model horizons."""
    if value == "all":
        return [1, 5, 10]
    if value in {"1", "5", "10"}:
        return [int(value)]
    raise ValueError(f"unsupported horizon: {value}")


def command_train(
    config: dict[str, Any],
    config_path: Path,
    *,
    model: str,
    horizon: str,
    run_id: str,
) -> None:
    """Dispatch a registered model through the unified training entry."""
    if model == "context_decomposition" and horizon != "10":
        raise ValueError("context_decomposition only supports horizon 10")
    from model.stages.training import (
        train_lightgbm_run,
        train_mlp_run,
        train_residual_rank_mlp_run,
        train_ridge_run,
        train_xgboost_run,
    )
    from model.stages.context_training import run_context_experiment

    trainers = {
        "ridge": train_ridge_run,
        "xgboost": train_xgboost_run,
        "lightgbm": train_lightgbm_run,
        "mlp": train_mlp_run,
        "residual_rank_mlp": train_residual_rank_mlp_run,
        "context_decomposition": run_context_experiment,
    }
    try:
        trainer = trainers[model]
    except KeyError as exc:
        raise ValueError(f"unsupported model: {model}") from exc
    trainer(
        config,
        config_path,
        horizons=parse_horizons(horizon),
        run_id=run_id,
    )


def main() -> None:
    args = make_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.command == "fetch-market":
        command_fetch_market(config, config_path, force=args.force)
        return
    if args.command == "prepare-data":
        command_prepare_data(config, config_path)
        return
    if args.command == "prepare-context":
        command_prepare_context(config, config_path)
        return
    if args.command == "train":
        command_train(
            config,
            config_path,
            model=args.model,
            horizon=args.horizon,
            run_id=args.run_id,
        )
        return
    command_doctor(config, config_path)


if __name__ == "__main__":
    main()
