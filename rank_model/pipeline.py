"""Public command-line entry point for the rank-model pipeline."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import tomllib
from typing import Any

from rank_model.stages.dataset import build_rank_dataset
from rank_model.stages.training import MODEL_NAMES, train_registered_model


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.toml"


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
    paths = config["paths"]
    return build_rank_dataset(
        resolve_config_path(config_path, paths["source_dataset"]),
        resolve_config_path(config_path, paths["source_schema"]),
        resolve_config_path(config_path, paths["rank_dataset"]),
        resolve_config_path(config_path, paths["rank_schema"]),
        resolve_config_path(config_path, paths["rank_label_coverage"]),
        date.fromisoformat(config["project"]["development_end"]),
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSI1000 10-day rank-model pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Build the rank-label dataset.")
    train_parser = subparsers.add_parser("train", help="Train one registered rank model.")
    train_parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    train_parser.add_argument("--run-id", required=True)
    for command in ("doctor", "evaluate", "compare"):
        subparsers.add_parser(command, help=f"Reserved for a later stage: {command}.")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.command == "train":
        try:
            run_directory = train_registered_model(
                config, config_path, args.model, args.run_id
            )
        except (NotImplementedError, ValueError, FileExistsError) as error:
            parser = make_parser()
            parser.error(str(error))
        print(f"rank model run written: {run_directory}")
        return
    if args.command != "prepare":
        raise NotImplementedError(f"{args.command} is not wired in this task")
    result = command_prepare(config, config_path)
    print(
        "rank dataset written: "
        f"rows={result['row_count']} dates={result['date_count']} "
        f"path={result['output_dataset']}"
    )


if __name__ == "__main__":
    main()
