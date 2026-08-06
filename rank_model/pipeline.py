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
    SPLIT_COLUMN,
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
from rank_model.stages.training import MODEL_NAMES, train_registered_model, validate_run_id


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
    """Accept a source only when it is directly beneath a model/data directory."""
    parent = path.resolve().parent
    return parent.name == "data" and parent.parent.name == "model"


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
    training = split.eq("train").fillna(False)
    validation = split.eq("validation").fillna(False)
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
    return parser


def main() -> None:
    args = make_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
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
