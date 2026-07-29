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
    for module_name in ("pandas", "numpy", "pyarrow", "rqdatac"):
        if importlib.util.find_spec(module_name) is None:
            raise RuntimeError(f"missing Python dependency: {module_name}")
    messages.append("dependencies: pandas, numpy, pyarrow, rqdatac")

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
    command_doctor(config, config_path)


if __name__ == "__main__":
    main()
