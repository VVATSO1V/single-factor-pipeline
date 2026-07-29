"""Unified command-line entry for model data and modeling stages."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import tomllib
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.toml"
COMMANDS = ("doctor", "fetch-market", "prepare-data")
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
ALLOWED_HORIZONS = {1, 5, 10}


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
    if (
        not isinstance(horizons, list)
        or not horizons
        or len(set(horizons)) != len(horizons)
        or not set(horizons).issubset(ALLOWED_HORIZONS)
    ):
        raise ValueError(
            f"{path}: target horizons must be unique values from [1, 5, 10]"
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


def main() -> None:
    args = make_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.command == "fetch-market":
        command_fetch_market(config, config_path, force=args.force)
        return
    raise RuntimeError(f"{args.command} stage is not migrated yet")


if __name__ == "__main__":
    main()
