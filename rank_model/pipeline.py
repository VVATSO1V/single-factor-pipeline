"""Public command-line entry point for the rank-model pipeline."""

from __future__ import annotations

import argparse
from datetime import date
import importlib
import json
from pathlib import Path
import tomllib
from typing import Any

import numpy as np
import pandas as pd

from rank_model.stages.dataset import (
    EXIT_DATE_COLUMN,
    FORBIDDEN_FEATURE_COLUMNS,
    KEY_COLUMNS,
    MAXIMUM_DEVELOPMENT_END,
    MAX_TARGET_EXIT_CALENDAR_DAYS,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    _is_forbidden_feature,
    _load_source_schema,
    _normalize_keys,
    _normalize_target,
    _read_source_dataset,
    _source_columns,
    _validate_date_sizes,
    _validate_source_dates,
    build_rank_dataset,
    file_sha256,
    load_rank_dataset,
)
from rank_model.stages.evaluation import compare_runs, evaluate_predictions, write_evaluation
from rank_model.stages.finalization import (
    build_frozen_spec,
    freeze_refit_dataset,
    rank_dataset_logical_sha256,
    refit_frozen_candidate,
    validate_frozen_dataset,
    validate_frozen_files_before_read,
    write_frozen_spec,
)
from rank_model.stages.training import MODEL_NAMES, train_registered_model, validate_run_id
from rank_model.stages.locked_test import (
    LOCKED_CAP_COVERAGE_THRESHOLD,
    LOCKED_MAD_WIDTH,
    LOCKED_MODEL_NAMES,
    LOCKED_WINDOWS,
    compare_locked_test_models,
    evaluate_locked_test_model,
    predict_locked_test_model,
    prepare_locked_test_dataset,
)
from rank_model.stages.strategy import (
    backtest_locked_strategy,
    compare_strategy_runs,
    load_strategy_settings,
)
from rank_model.stages.staggered_strategy import (
    STAGGERED_CONFIG_CONTRACT,
    backtest_staggered_strategy,
    compare_staggered_strategy_runs,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.toml"
RUNTIME_DEPENDENCIES = (
    "numpy",
    "pandas",
    "scipy",
    "pyarrow",
    "sklearn",
    "joblib",
    "xgboost",
    "lightgbm",
    "torch",
)
TRAINING_START = pd.Timestamp("2019-01-01")
TRAINING_END = pd.Timestamp("2022-12-31")
VALIDATION_START = pd.Timestamp("2023-01-01")
VALIDATION_END = pd.Timestamp("2023-12-31")


def resolve_config_path(config_path: Path, raw_path: str) -> Path:
    """Resolve a configured path relative to its TOML file."""
    return (config_path.resolve().parent / raw_path).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the minimal configuration required by this foundation stage."""
    path = Path(config_path).resolve()
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if not isinstance(config.get("project"), dict):
        raise ValueError(f"{path} is missing [project]")
    if not isinstance(config.get("paths"), dict):
        raise ValueError(f"{path} is missing [paths]")
    development_end = config["project"].get("development_end")
    if not isinstance(development_end, str):
        raise ValueError(f"{path} requires project.development_end")
    try:
        parsed_end = date.fromisoformat(development_end)
    except ValueError as error:
        raise ValueError("project.development_end must use YYYY-MM-DD") from error
    if parsed_end != date(2023, 12, 31):
        raise ValueError("project.development_end must be 2023-12-31")
    for name in (
        "source_dataset",
        "source_schema",
        "rank_dataset",
        "rank_schema",
        "rank_label_coverage",
        "runs_dir",
    ):
        if not isinstance(config["paths"].get(name), str):
            raise ValueError(f"{path} requires paths.{name}")
    locked_test = config.get("locked_test")
    if not isinstance(locked_test, dict):
        raise ValueError(f"{path} is missing [locked_test]")
    if locked_test.get("start") != "2024-01-01" or locked_test.get("end") != "2025-12-31":
        raise ValueError(
            f"{path}: locked_test must span 2024-01-01 through 2025-12-31"
        )
    if locked_test.get("expected_cross_section_size") != 1000:
        raise ValueError(f"{path}: locked_test expected_cross_section_size must be 1000")
    windows = locked_test.get("windows")
    try:
        mad_width = float(locked_test["mad_width"])
        date_chunk_size = int(locked_test["date_chunk_size"])
        cap_coverage_threshold = float(locked_test["cap_coverage_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: locked_test numeric settings are invalid") from error
    if mad_width <= 0 or date_chunk_size <= 0 or not 0 < cap_coverage_threshold <= 1:
        raise ValueError(f"{path}: locked_test numeric settings are out of range")
    if (
        mad_width != LOCKED_MAD_WIDTH
        or windows != list(LOCKED_WINDOWS)
        or cap_coverage_threshold != LOCKED_CAP_COVERAGE_THRESHOLD
    ):
        raise ValueError(
            f"{path}: locked_test feature semantics must remain "
            f"mad_width={LOCKED_MAD_WIDTH}, windows={list(LOCKED_WINDOWS)}, "
            f"cap_coverage_threshold={LOCKED_CAP_COVERAGE_THRESHOLD}"
        )
    load_strategy_settings(config)
    _validate_staggered_strategy_paths(config, path)
    return config


def command_prepare(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Publish the 10-day percentile-label dataset."""
    paths = _validate_configured_paths(config, config_path)
    return build_rank_dataset(
        paths["source_dataset"],
        paths["source_schema"],
        paths["rank_dataset"],
        paths["rank_schema"],
        paths["rank_label_coverage"],
        date.fromisoformat(config["project"]["development_end"]),
    )


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(directory.resolve(strict=False))
    except ValueError:
        return False
    return True


def _is_model_data_path(path: Path) -> bool:
    """Accept a source only when it is directly beneath this worktree's model/data."""
    expected = (PACKAGE_DIR.parent / "model" / "data").resolve()
    return path.resolve().parent == expected


def _is_inside_model_data(path: Path) -> bool:
    """Return whether a resolved path is a descendant of any model/data directory."""
    parts = [part.lower() for part in path.resolve(strict=False).parts]
    return any(
        parts[index : index + 2] == ["model", "data"]
        for index in range(len(parts) - 1)
    )


def _validate_configured_paths(config: dict[str, Any], config_path: Path) -> dict[str, Path]:
    paths = config["paths"]
    resolved = {
        name: resolve_config_path(config_path, paths[name])
        for name in (
            "source_dataset",
            "source_schema",
            "rank_dataset",
            "rank_schema",
            "rank_label_coverage",
            "runs_dir",
        )
    }
    for name in ("source_dataset", "source_schema"):
        if not _is_model_data_path(resolved[name]):
            raise ValueError(f"paths.{name} must resolve directly under model/data")
    for name in (
        "rank_dataset",
        "rank_schema",
        "rank_label_coverage",
        "runs_dir",
    ):
        if not _is_within(resolved[name], PACKAGE_DIR):
            raise ValueError(f"paths.{name} must resolve under rank_model")
        if _is_inside_model_data(resolved[name]):
            raise ValueError(f"paths.{name} must not resolve inside model/data")
    return resolved


def _validate_finalization_paths(
    config: dict[str, Any], config_path: Path
) -> dict[str, Path]:
    try:
        raw_paths = config["paths"]
        resolved = {
            name: resolve_config_path(config_path, raw_paths[name])
            for name in ("frozen_spec", "final_runs_dir")
        }
    except (KeyError, TypeError) as error:
        raise ValueError(
            "freeze/refit requires paths.frozen_spec and paths.final_runs_dir"
        ) from error
    for name, path in resolved.items():
        if not _is_within(path, PACKAGE_DIR) or _is_inside_model_data(path):
            raise ValueError(
                f"paths.{name} must resolve under rank_model outside model/data"
            )
    return resolved


def _validate_locked_test_paths(
    config: dict[str, Any], config_path: Path
) -> dict[str, Path]:
    source_names = (
        "full_model_dataset",
        "full_model_schema",
        "full_sample_index",
        "full_split_summary",
        "market_panel",
        "trading_calendar",
    )
    output_names = (
        "locked_test_dataset",
        "locked_test_schema",
        "locked_test_label_coverage",
        "locked_test_runs_dir",
        "locked_test_comparison",
    )
    required = (*source_names, *output_names)
    try:
        resolved = {
            name: resolve_config_path(config_path, config["paths"][name])
            for name in required
        }
    except (KeyError, TypeError) as error:
        raise ValueError("locked-test paths are incomplete") from error
    for name in source_names:
        if not _is_model_data_path(resolved[name]):
            raise ValueError(f"paths.{name} must resolve directly under model/data")
    for name in output_names:
        if not _is_within(resolved[name], PACKAGE_DIR) or _is_inside_model_data(
            resolved[name]
        ):
            raise ValueError(
                f"paths.{name} must resolve under rank_model outside model/data"
            )
    return resolved


def _validate_strategy_paths(
    config: dict[str, Any], config_path: Path
) -> dict[str, Path]:
    source_names = ("market_panel", "trading_calendar")
    sealed_names = (
        "locked_test_schema",
        "locked_test_runs_dir",
        "frozen_spec",
        "locked_test_conclusion",
        "locked_test_comparison",
    )
    output_names = ("strategy_runs_dir", "strategy_comparison")
    required = (*source_names, *sealed_names, *output_names)
    try:
        resolved = {
            name: resolve_config_path(config_path, config["paths"][name])
            for name in required
        }
    except (KeyError, TypeError) as error:
        raise ValueError("strategy paths are incomplete") from error
    for name in source_names:
        if not _is_model_data_path(resolved[name]):
            raise ValueError(f"paths.{name} must resolve directly under model/data")
    for name in (*sealed_names, *output_names):
        if not _is_within(resolved[name], PACKAGE_DIR) or _is_inside_model_data(
            resolved[name]
        ):
            raise ValueError(
                f"paths.{name} must resolve under rank_model outside model/data"
            )
    expected_runs_directory = (PACKAGE_DIR / "strategy_runs").resolve()
    if resolved["strategy_runs_dir"] != expected_runs_directory:
        raise ValueError(
            "paths.strategy_runs_dir must resolve to rank_model/strategy_runs"
        )
    expected_comparison = (PACKAGE_DIR / "strategy_comparison.csv").resolve()
    if resolved["strategy_comparison"] != expected_comparison:
        raise ValueError(
            "paths.strategy_comparison must resolve to "
            "rank_model/strategy_comparison.csv"
        )
    return resolved


def _validate_staggered_strategy_paths(
    config: dict[str, Any], config_path: Path
) -> dict[str, Path]:
    """Resolve the fixed 10-day strategy outputs without aliasing daily artifacts."""
    resolved = _validate_strategy_paths(config, config_path)
    output_names = (
        "strategy_10d_runs_dir",
        "strategy_10d_comparison",
        "daily_vs_10d_comparison",
    )
    try:
        staggered = config["strategy_10d"]
        staggered_outputs = {
            name: resolve_config_path(config_path, config["paths"][name])
            for name in output_names
        }
    except (KeyError, TypeError) as error:
        raise ValueError("staggered strategy paths or settings are incomplete") from error
    if (
        not isinstance(staggered, dict)
        or set(staggered) != set(STAGGERED_CONFIG_CONTRACT)
        or any(
            type(staggered[name]) is not type(expected)
            or staggered[name] != expected
            for name, expected in STAGGERED_CONFIG_CONTRACT.items()
        )
    ):
        raise ValueError(
            "strategy_10d must keep horizon=10, offset_count=10, "
            "last_complete_signal=2025-12-16, retry_blocked_sells_daily=true, "
            "and retry_failed_buys_daily=false"
        )
    expected = {
        "strategy_10d_runs_dir": (PACKAGE_DIR / "strategy_10d_runs").resolve(),
        "strategy_10d_comparison": (
            PACKAGE_DIR / "strategy_10d_comparison.csv"
        ).resolve(),
        "daily_vs_10d_comparison": (
            PACKAGE_DIR / "daily_vs_10d_comparison.csv"
        ).resolve(),
    }
    daily_artifacts = {
        resolved["strategy_runs_dir"],
        resolved["strategy_comparison"],
        resolved["strategy_comparison"].with_suffix(".manifest.json"),
    }
    for name, output in staggered_outputs.items():
        if not _is_within(output, PACKAGE_DIR) or _is_inside_model_data(output):
            raise ValueError(
                f"paths.{name} must resolve under rank_model outside model/data"
            )
        if output != expected[name]:
            raise ValueError(f"paths.{name} must resolve to {expected[name]}")
        if output in daily_artifacts:
            raise ValueError(f"paths.{name} must not alias a daily strategy artifact")
    if len(set(staggered_outputs.values())) != len(staggered_outputs):
        raise ValueError("staggered strategy outputs must be distinct")
    resolved.update(staggered_outputs)
    return resolved


def command_prepare_test(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Publish the sealed local 2024-2025 rank-model dataset."""
    paths = _validate_configured_paths(config, config_path)
    paths.update(_validate_finalization_paths(config, config_path))
    paths.update(_validate_locked_test_paths(config, config_path))
    existing = [
        str(paths[name])
        for name in (
            "locked_test_dataset",
            "locked_test_schema",
            "locked_test_label_coverage",
        )
        if paths[name].exists()
    ]
    if existing:
        raise FileExistsError(f"locked-test dataset is immutable: {existing}")
    settings = config["locked_test"]
    return prepare_locked_test_dataset(
        model_dataset_path=paths["full_model_dataset"],
        model_schema_path=paths["full_model_schema"],
        sample_index_path=paths["full_sample_index"],
        split_summary_path=paths["full_split_summary"],
        market_panel_path=paths["market_panel"],
        trading_calendar_path=paths["trading_calendar"],
        development_schema_path=paths["rank_schema"],
        output_dataset_path=paths["locked_test_dataset"],
        output_schema_path=paths["locked_test_schema"],
        output_coverage_path=paths["locked_test_label_coverage"],
        test_start=pd.Timestamp(settings["start"]),
        test_end=pd.Timestamp(settings["end"]),
        mad_width=float(settings["mad_width"]),
        expected_cross_section_size=int(settings["expected_cross_section_size"]),
        date_chunk_size=int(settings["date_chunk_size"]),
        windows=tuple(int(value) for value in settings["windows"]),
        cap_coverage_threshold=float(settings["cap_coverage_threshold"]),
    )


def command_predict_test(
    config: dict[str, Any], config_path: Path, model_name: str
) -> Path:
    """Generate static 2024-2025 predictions from one final frozen model."""
    paths = _validate_configured_paths(config, config_path)
    paths.update(_validate_finalization_paths(config, config_path))
    paths.update(_validate_locked_test_paths(config, config_path))
    return predict_locked_test_model(
        model_name,
        test_dataset_path=paths["locked_test_dataset"],
        test_schema_path=paths["locked_test_schema"],
        development_schema_path=paths["rank_schema"],
        trading_calendar_path=paths["trading_calendar"],
        frozen_spec_path=paths["frozen_spec"],
        final_runs_directory=paths["final_runs_dir"],
        output_runs_directory=paths["locked_test_runs_dir"],
        expected_cross_section_size=int(
            config["locked_test"]["expected_cross_section_size"]
        ),
    )


def command_evaluate_test(
    config: dict[str, Any], config_path: Path, model_name: str
) -> Path:
    """Evaluate one completed locked-test prediction run."""
    paths = _validate_locked_test_paths(config, config_path)
    return evaluate_locked_test_model(
        model_name,
        output_runs_directory=paths["locked_test_runs_dir"],
        hac_lag=10,
        top_k=100,
    )


def command_compare_test(
    config: dict[str, Any], config_path: Path
) -> pd.DataFrame:
    """Write one metric-only comparison for every frozen model."""
    paths = _validate_locked_test_paths(config, config_path)
    return compare_locked_test_models(
        output_runs_directory=paths["locked_test_runs_dir"],
        output_path=paths["locked_test_comparison"],
        model_names=LOCKED_MODEL_NAMES,
    )


def _strategy_source_paths(paths: dict[str, Path], model_name: str) -> dict[str, Path]:
    locked_run_directory = paths["locked_test_runs_dir"] / model_name
    return {
        "prediction": locked_run_directory / "predictions_10d.parquet",
        "prediction_manifest": locked_run_directory / "manifest.json",
        "locked_test_schema": paths["locked_test_schema"],
        "market_panel": paths["market_panel"],
        "trading_calendar": paths["trading_calendar"],
        "frozen_models": paths["frozen_spec"],
        "locked_test_conclusion": paths["locked_test_conclusion"],
        "locked_test_comparison": paths["locked_test_comparison"],
    }


def command_backtest_strategy(
    config: dict[str, Any], config_path: Path, model_name: str
) -> Path:
    """Run and publish one immutable frozen-model strategy report."""
    paths = _validate_strategy_paths(config, config_path)
    return backtest_locked_strategy(
        model_name,
        destination=paths["strategy_runs_dir"] / model_name,
        settings=load_strategy_settings(config),
        source_paths=_strategy_source_paths(paths, model_name),
    )


def command_compare_strategy(
    config: dict[str, Any], config_path: Path
) -> pd.DataFrame:
    """Publish one descriptive comparison for all frozen-model strategies."""
    paths = _validate_strategy_paths(config, config_path)
    return compare_strategy_runs(
        paths["strategy_runs_dir"],
        paths["strategy_comparison"],
        LOCKED_MODEL_NAMES,
    )


def command_backtest_strategy_10d(
    config: dict[str, Any], config_path: Path, model_name: str
) -> Path:
    """Run and publish one immutable staggered ten-day strategy report."""
    paths = _validate_staggered_strategy_paths(config, config_path)
    return backtest_staggered_strategy(
        model_name,
        destination=paths["strategy_10d_runs_dir"] / model_name,
        publication_root=paths["strategy_10d_runs_dir"],
        settings=load_strategy_settings(config),
        source_paths=_strategy_source_paths(paths, model_name),
    )


def command_compare_strategy_10d(
    config: dict[str, Any], config_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Publish the frozen five-model 10-day and daily-policy comparisons."""
    paths = _validate_staggered_strategy_paths(config, config_path)
    source_paths_by_model = {
        model_name: _strategy_source_paths(paths, model_name)
        for model_name in LOCKED_MODEL_NAMES
    }
    return compare_staggered_strategy_runs(
        paths["strategy_10d_runs_dir"],
        paths["strategy_10d_comparison"],
        paths["strategy_comparison"],
        paths["daily_vs_10d_comparison"],
        source_paths_by_model,
        LOCKED_MODEL_NAMES,
        settings=load_strategy_settings(config),
    )


def command_freeze(
    config: dict[str, Any],
    config_path: Path,
    validation_runs_directory: Path | None = None,
) -> Path:
    """Freeze the selected, completed 2023 validation specifications."""
    paths = _validate_configured_paths(config, config_path)
    paths.update(_validate_finalization_paths(config, config_path))
    candidates = config.get("finalization", {}).get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("config requires [[finalization.candidates]]")
    runs_directory = (
        Path(validation_runs_directory).resolve()
        if validation_runs_directory is not None
        else paths["runs_dir"]
    )
    specification = build_frozen_spec(
        candidates,
        runs_directory,
        date.fromisoformat(config["project"]["training_start"]),
        date.fromisoformat(config["project"]["development_end"]),
    )
    dataset, schema = load_rank_dataset(paths["rank_dataset"], paths["rank_schema"])
    if file_sha256(paths["source_dataset"]) != schema.get("source_dataset_sha256"):
        raise ValueError("current source dataset hash does not match the rank schema")
    if file_sha256(paths["source_schema"]) != schema.get("source_schema_sha256"):
        raise ValueError("current source schema hash does not match the rank schema")
    specification = freeze_refit_dataset(
        specification,
        dataset,
        schema,
        paths["rank_dataset"],
        paths["rank_schema"],
    )
    return write_frozen_spec(specification, paths["frozen_spec"])


def command_refit(
    config: dict[str, Any], config_path: Path, model_name: str
) -> Path:
    """Refit one frozen candidate on all eligible 2019-2023 observations."""
    paths = _validate_configured_paths(config, config_path)
    paths.update(_validate_finalization_paths(config, config_path))
    frozen_spec = json.loads(paths["frozen_spec"].read_text(encoding="utf-8"))
    if frozen_spec.get("schema_version") != 1:
        raise ValueError("unsupported frozen model specification")
    matches = [
        candidate
        for candidate in frozen_spec.get("candidates", [])
        if candidate.get("model_name") == model_name
    ]
    if len(matches) != 1:
        raise ValueError(f"frozen specification has no unique candidate: {model_name}")
    data_hashes = frozen_spec.get("data_hashes")
    if not isinstance(data_hashes, dict):
        raise ValueError("frozen specification is missing data hashes")
    refit_dataset_contract = frozen_spec.get("refit_dataset")
    if not isinstance(refit_dataset_contract, dict):
        raise ValueError("frozen specification is missing the exact refit dataset")
    validate_frozen_files_before_read(
        paths["rank_dataset"], paths["rank_schema"], refit_dataset_contract
    )
    candidate = matches[0]
    dataset, schema = load_rank_dataset(paths["rank_dataset"], paths["rank_schema"])
    validate_frozen_dataset(dataset, schema, data_hashes)
    if rank_dataset_logical_sha256(dataset, schema) != refit_dataset_contract.get(
        "logical_sha256"
    ):
        raise ValueError("loaded refit dataset does not match its frozen logical hash")
    dates = pd.to_datetime(dataset["date"], errors="raise").dt.normalize()
    if (
        len(dataset) != refit_dataset_contract.get("row_count")
        or dates.nunique() != refit_dataset_contract.get("date_count")
        or dates.min().date().isoformat() != refit_dataset_contract.get("date_min")
        or dates.max().date().isoformat() != refit_dataset_contract.get("date_max")
    ):
        raise ValueError("loaded refit dataset does not match its frozen dimensions")
    window = frozen_spec.get("development_window")
    if not isinstance(window, dict):
        raise ValueError("frozen specification is missing its development window")
    try:
        params = config["models"][model_name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"config is missing model parameters: {model_name}") from error
    return refit_frozen_candidate(
        dataset,
        schema,
        candidate,
        config_path,
        paths["frozen_spec"],
        paths["final_runs_dir"],
        model_params=params,
        development_start=date.fromisoformat(window["signal_start"]),
        development_end=date.fromisoformat(window["signal_end"]),
    )


def _validated_run_directory(runs_directory: Path, run_id: str) -> Path:
    """Resolve a final run path and reject any run-ID link or junction escape."""
    resolved_runs_directory = runs_directory.resolve(strict=False)
    run_directory = (resolved_runs_directory / run_id).resolve(strict=False)
    if (
        not _is_within(run_directory, resolved_runs_directory)
        or not _is_within(run_directory, PACKAGE_DIR)
        or _is_inside_model_data(run_directory)
    ):
        raise ValueError("run directory must resolve under rank_model outside model/data")
    return run_directory


def command_train(
    config: dict[str, Any], config_path: Path, model_name: str, run_id: str
) -> Path:
    """Train only after validating every configured output target."""
    paths = _validate_configured_paths(config, config_path)
    _validated_run_directory(paths["runs_dir"], run_id)
    return train_registered_model(config, config_path, model_name, run_id)


def command_evaluate(
    config: dict[str, Any], config_path: Path, run_id: str, split: str
) -> Path:
    """Write one evaluation bundle under the validated immutable run directory."""
    paths = _validate_configured_paths(config, config_path)
    run_directory = _validated_run_directory(paths["runs_dir"], run_id)
    predictions = pd.read_parquet(run_directory / "predictions_10d.parquet")
    if "split" not in predictions:
        raise ValueError("prediction artifact is missing split")
    bundle = evaluate_predictions(predictions.loc[predictions["split"].eq(split)].copy())
    write_evaluation(bundle, run_directory)
    return run_directory


def command_compare(
    config: dict[str, Any], config_path: Path, run_ids: list[str]
) -> pd.DataFrame:
    """Write a comparison table only beneath the validated run root."""
    paths = _validate_configured_paths(config, config_path)
    runs_directory = paths["runs_dir"]
    return compare_runs(
        [runs_directory / run_id for run_id in run_ids],
        runs_directory / "rank_model_comparison.csv",
    )


def _validate_split_boundaries(frame: pd.DataFrame) -> None:
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    exits = pd.to_datetime(frame[EXIT_DATE_COLUMN], errors="raise").dt.normalize()
    split = frame[SPLIT_COLUMN].astype("string").str.strip().str.lower()
    assigned = split.notna()
    if split.loc[assigned].eq("").any():
        raise ValueError("source dataset contains blank assigned split values")
    valid = split.loc[assigned].isin(("train", "validation"))
    if not valid.all():
        invalid = sorted(split.loc[assigned].loc[~valid].unique().tolist())
        raise ValueError(f"source dataset contains invalid split values: {invalid}")
    if exits.loc[assigned].isna().any():
        raise ValueError("assigned development rows require exit dates")
    known_exit = exits.notna()
    exit_days = (exits - dates).dt.days
    invalid_exit_direction = known_exit & (
        exit_days.le(0) | exit_days.gt(MAX_TARGET_EXIT_CALENDAR_DAYS)
    )
    if invalid_exit_direction.any():
        raise ValueError(
            "exit_date_10d must be 1 through "
            f"{MAX_TARGET_EXIT_CALENDAR_DAYS} calendar days after its signal date"
        )
    training = split.eq("train").fillna(False)
    validation = split.eq("validation").fillna(False)
    unassigned = split.isna()
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    missing_target = ~np.isfinite(target.to_numpy(dtype="float64"))
    boundary_purge = (
        dates.between(TRAINING_START, TRAINING_END)
        & exits.gt(TRAINING_END)
    ) | (
        dates.between(VALIDATION_START, VALIDATION_END)
        & exits.gt(VALIDATION_END)
    )
    unexplained_unassigned = unassigned & ~missing_target & ~boundary_purge
    if unexplained_unassigned.any():
        raise ValueError(
            "unassigned source row must have a missing target or an exit date "
            "outside its train/validation boundary"
        )
    if not dates.loc[training].between(TRAINING_START, TRAINING_END).all():
        raise ValueError("training dates must stay within 2019-01-01 through 2022-12-31")
    if not exits.loc[training].between(TRAINING_START, TRAINING_END).all():
        raise ValueError("training exit dates must stay within 2019-01-01 through 2022-12-31")
    if not dates.loc[validation].between(VALIDATION_START, VALIDATION_END).all():
        raise ValueError("validation dates must stay within 2023-01-01 through 2023-12-31")
    if not exits.loc[validation].between(VALIDATION_START, VALIDATION_END).all():
        raise ValueError("validation exit dates must stay within 2023-01-01 through 2023-12-31")


def _validate_source_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    continuous_columns: list[str],
) -> None:
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"source dataset is missing schema features: {missing}")
    forbidden = sorted(
        column
        for column in feature_columns
        if column.lower() in FORBIDDEN_FEATURE_COLUMNS or _is_forbidden_feature(column)
    )
    if forbidden:
        raise ValueError(f"model feature list contains forbidden features: {forbidden}")
    values = frame[continuous_columns].to_numpy(dtype="float64", copy=False)
    if not np.isfinite(values).all():
        raise ValueError("schema-declared continuous features are not finite")


def _validate_published_bundle(paths: dict[str, Path], source_schema: Path) -> str:
    rank_dataset = paths["rank_dataset"]
    rank_schema = paths["rank_schema"]
    if not rank_dataset.exists() and not rank_schema.exists():
        return "absent (run prepare to publish the rank dataset)"
    if not rank_dataset.exists() or not rank_schema.exists():
        raise ValueError("published rank dataset and schema must either both exist or both be absent")
    _, published_schema = load_rank_dataset(rank_dataset, rank_schema)
    if published_schema.get("source_dataset_sha256") != file_sha256(paths["source_dataset"]):
        raise ValueError("published rank schema source dataset hash does not match")
    if published_schema.get("source_schema_sha256") != file_sha256(source_schema):
        raise ValueError("published rank schema source schema hash does not match")
    return f"verified ({published_schema['parquet_sha256']})"


def command_doctor(config: dict[str, Any], config_path: Path) -> dict[str, str]:
    """Validate source, boundaries, dependencies, and output containment read-only."""
    development_end = date.fromisoformat(config["project"]["development_end"])
    if development_end > MAXIMUM_DEVELOPMENT_END:
        raise ValueError("project.development_end must be no later than 2023-12-31")
    paths = _validate_configured_paths(config, config_path)
    source_schema = paths["source_schema"]
    source_dataset = paths["source_dataset"]
    source_contract = _load_source_schema(source_schema, source_dataset)
    dataset, feature_columns, continuous_columns, _industry_column = _read_source_dataset(
        source_dataset, source_contract
    )
    dataset = _normalize_keys(dataset, source_dataset)
    _validate_source_dates(dataset, source_dataset, development_end)
    _validate_date_sizes(dataset, source_dataset)
    dataset = _normalize_target(dataset, source_dataset)
    _validate_source_features(dataset, feature_columns, continuous_columns)
    _validate_split_boundaries(dataset)
    for dependency in RUNTIME_DEPENDENCIES:
        importlib.import_module(dependency)
    published = _validate_published_bundle(paths, source_schema)
    return {
        "source": f"verified ({source_dataset})",
        "source_hash": "verified",
        "development_end": development_end.isoformat(),
        "source_dates": f"verified ({dataset['date'].nunique()} dates; 1000 keys/date)",
        "keys": "verified (unique and non-null)",
        "features": f"verified ({len(feature_columns)} features; continuous values finite)",
        "forbidden_features": "verified (absent)",
        "split_boundaries": "verified (train 2019-2022; validation 2023; exits contained)",
        "runtime_dependencies": f"verified ({len(RUNTIME_DEPENDENCIES)} imports)",
        "paths": "verified (source under model/data; outputs under rank_model)",
        "published_hashes": published,
    }


def parse_run_ids(raw_run_ids: str) -> list[str]:
    """Parse a comma-delimited run list without silently dropping empty IDs."""
    run_ids = [run_id.strip() for run_id in raw_run_ids.split(",")]
    if not run_ids or any(not run_id for run_id in run_ids):
        raise ValueError("run IDs must not contain blank components")
    for run_id in run_ids:
        validate_run_id(run_id)
    return run_ids


def parse_run_id(raw_run_id: str) -> str:
    """Validate one immutable run ID before it can be resolved as a path."""
    validate_run_id(raw_run_id)
    return raw_run_id


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSI1000 10-day rank-model pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Build the rank-label dataset.")
    train_parser = subparsers.add_parser("train", help="Train one registered rank model.")
    train_parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    train_parser.add_argument("--run-id", required=True)
    subparsers.add_parser("doctor", help="Validate source and pipeline contracts read-only.")
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate one completed run.")
    evaluate_parser.add_argument("--run-id", required=True)
    evaluate_parser.add_argument("--split", required=True, choices=("validation",))
    compare_parser = subparsers.add_parser("compare", help="Compare completed runs.")
    compare_parser.add_argument("--run-ids", required=True)
    freeze_parser = subparsers.add_parser(
        "freeze", help="Freeze selected completed validation specifications."
    )
    freeze_parser.add_argument("--validation-runs-dir", type=Path)
    refit_parser = subparsers.add_parser(
        "refit", help="Refit one frozen candidate on 2019-2023."
    )
    refit_parser.add_argument(
        "--model",
        required=True,
        choices=(
            "ridge_rank_regression",
            "xgboost_rank_regression",
            "lightgbm_rank_regression",
            "lightgbm_lambdarank",
            "mlp_top100_hybrid_rank",
        ),
    )
    subparsers.add_parser(
        "prepare-test", help="Build the sealed 2024-2025 locked-test dataset."
    )
    predict_test_parser = subparsers.add_parser(
        "predict-test", help="Run one frozen final model on 2024-2025."
    )
    predict_test_parser.add_argument(
        "--model", required=True, choices=LOCKED_MODEL_NAMES
    )
    evaluate_test_parser = subparsers.add_parser(
        "evaluate-test", help="Evaluate one completed locked-test prediction."
    )
    evaluate_test_parser.add_argument(
        "--model", required=True, choices=LOCKED_MODEL_NAMES
    )
    subparsers.add_parser(
        "compare-test", help="Compare all completed locked-test model reports."
    )
    strategy_parser = subparsers.add_parser(
        "backtest-strategy", help="Run one frozen model's static strategy."
    )
    strategy_parser.add_argument(
        "--model", required=True, choices=LOCKED_MODEL_NAMES
    )
    subparsers.add_parser(
        "compare-strategy", help="Compare all five completed strategy runs."
    )
    strategy_10d_parser = subparsers.add_parser(
        "backtest-strategy-10d",
        help="Run one frozen model's staggered ten-day strategy.",
    )
    strategy_10d_parser.add_argument(
        "--model", required=True, choices=LOCKED_MODEL_NAMES
    )
    subparsers.add_parser(
        "compare-strategy-10d",
        help="Compare all five staggered ten-day strategy runs.",
    )
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    if args.command == "doctor":
        try:
            checks = command_doctor(config, config_path)
        except (FileNotFoundError, ValueError) as error:
            parser = make_parser()
            parser.error(str(error))
        for name, result in checks.items():
            print(f"doctor {name}: {result}")
        return
    if args.command == "train":
        try:
            run_directory = command_train(config, config_path, args.model, args.run_id)
        except (NotImplementedError, ValueError, FileExistsError) as error:
            parser = make_parser()
            parser.error(str(error))
        print(f"rank model run written: {run_directory}")
        return
    if args.command == "evaluate":
        try:
            run_id = parse_run_id(args.run_id)
            run_directory = command_evaluate(
                config, config_path, run_id, args.split
            )
        except (FileNotFoundError, ValueError, FileExistsError) as error:
            parser = make_parser()
            parser.error(str(error))
        print(f"rank model evaluation written: {run_directory}")
        return
    if args.command == "compare":
        try:
            run_ids = parse_run_ids(args.run_ids)
            comparison = command_compare(config, config_path, run_ids)
        except (FileNotFoundError, ValueError) as error:
            parser = make_parser()
            parser.error(str(error))
        print(f"rank model comparison written: rows={len(comparison)}")
        return
    if args.command == "freeze":
        try:
            frozen_path = command_freeze(
                config, config_path, args.validation_runs_dir
            )
        except (FileNotFoundError, ValueError, FileExistsError) as error:
            parser = make_parser()
            parser.error(str(error))
        print(f"frozen model specification written: {frozen_path}")
        return
    if args.command == "refit":
        try:
            run_directory = command_refit(config, config_path, args.model)
        except (FileNotFoundError, ValueError, FileExistsError) as error:
            parser = make_parser()
            parser.error(str(error))
        print(f"final model refit written: {run_directory}")
        return
    if args.command == "prepare-test":
        try:
            result = command_prepare_test(config, config_path)
        except (FileNotFoundError, ValueError, FileExistsError) as error:
            parser = make_parser()
            parser.error(str(error))
        print(
            "locked-test rank dataset written: "
            f"rows={result['row_count']} dates={result['date_count']}"
        )
        return
    if args.command == "predict-test":
        try:
            directory = command_predict_test(config, config_path, args.model)
        except (FileNotFoundError, ValueError, FileExistsError) as error:
            parser = make_parser()
            parser.error(str(error))
        print(f"locked-test predictions written: {directory}")
        return
    if args.command == "evaluate-test":
        try:
            directory = command_evaluate_test(config, config_path, args.model)
        except (FileNotFoundError, ValueError, FileExistsError) as error:
            parser = make_parser()
            parser.error(str(error))
        print(f"locked-test evaluation written: {directory}")
        return
    if args.command == "compare-test":
        try:
            comparison = command_compare_test(config, config_path)
        except (FileNotFoundError, ValueError, FileExistsError) as error:
            parser = make_parser()
            parser.error(str(error))
        print(f"locked-test comparison written: rows={len(comparison)}")
        return
    if args.command == "backtest-strategy":
        try:
            directory = command_backtest_strategy(
                config, config_path, args.model
            )
        except (FileNotFoundError, ValueError, FileExistsError) as error:
            parser = make_parser()
            parser.error(str(error))
        print(f"strategy run written: {directory}")
        return
    if args.command == "compare-strategy":
        try:
            comparison = command_compare_strategy(config, config_path)
        except (FileNotFoundError, ValueError, FileExistsError) as error:
            parser = make_parser()
            parser.error(str(error))
        print(f"strategy comparison written: rows={len(comparison)}")
        return
    if args.command == "backtest-strategy-10d":
        try:
            directory = command_backtest_strategy_10d(
                config, config_path, args.model
            )
        except (FileNotFoundError, ValueError, FileExistsError) as error:
            parser = make_parser()
            parser.error(str(error))
        print(f"staggered 10-day strategy run written: {directory}")
        return
    if args.command == "compare-strategy-10d":
        try:
            comparison, daily_vs = command_compare_strategy_10d(
                config, config_path
            )
        except (FileNotFoundError, ValueError, FileExistsError) as error:
            parser = make_parser()
            parser.error(str(error))
        print(
            "staggered 10-day comparisons written: "
            f"models={len(comparison)} daily_vs_rows={len(daily_vs)}"
        )
        return
    if args.command != "prepare":
        raise NotImplementedError(f"{args.command} is not wired in this task")
    try:
        result = command_prepare(config, config_path)
    except (FileNotFoundError, ValueError) as error:
        parser = make_parser()
        parser.error(str(error))
    print(
        "rank dataset written: "
        f"rows={result['row_count']} dates={result['date_count']} "
        f"path={result['output_dataset']}"
    )


if __name__ == "__main__":
    main()
