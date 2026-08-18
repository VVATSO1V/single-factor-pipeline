"""Independent latest-data forward strategy entrypoint."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Any, Mapping

import pandas as pd

from model.stages import features
from model.stages.market import build_market_panel, init_rqdatac, write_market_outputs
from rank_model.stages.dataset import file_sha256
from rank_model.stages.forward import (
    FIVE_FROZEN_MODELS,
    build_forward_frame,
    last_complete_signal,
    predict_frozen_models,
    resolve_as_of_date,
)
from rank_model.stages.forward_strategy import publish_forward_run, strategy_settings


def _read_toml(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        value = tomllib.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain a TOML table: {path}")
    return value


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_configs(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = _read_toml(config_path)
    project = config.get("project")
    paths = config.get("paths")
    if not isinstance(project, Mapping) or not isinstance(paths, Mapping):
        raise ValueError("forward config requires [project] and [paths]")
    model_config_path = _resolve(config_path.parent, str(paths["model_config"]))
    model_config = _read_toml(model_config_path)
    return config, model_config, model_config_path


def _model_paths(model_config: Mapping[str, Any], model_config_path: Path) -> tuple[list[Path], Path, Path]:
    paths = model_config.get("paths")
    factors = model_config.get("factors")
    if not isinstance(paths, Mapping) or not isinstance(factors, Mapping):
        raise ValueError("model config requires [paths] and [factors]")
    raw_factor_paths = factors.get("paths")
    if not isinstance(raw_factor_paths, list) or len(raw_factor_paths) != 17:
        raise ValueError("model config must declare exactly 17 factor paths")
    factor_paths = [_resolve(model_config_path.parent, str(path)) for path in raw_factor_paths]
    market_panel = _resolve(model_config_path.parent, str(paths["market_panel"]))
    trading_calendar = _resolve(model_config_path.parent, str(paths["trading_calendar"]))
    return factor_paths, market_panel, trading_calendar


def _forward_paths(config: Mapping[str, Any], config_path: Path) -> dict[str, Path]:
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("forward config is missing [paths]")
    names = (
        "market_panel",
        "trading_calendar",
        "frozen_spec",
        "final_runs_dir",
        "forward_runs_dir",
    )
    resolved = {name: _resolve(config_path.parent, str(paths[name])) for name in names}
    if resolved["forward_runs_dir"].name != "forward_strategy_runs":
        raise ValueError("forward_runs_dir must be rank_model/forward_strategy_runs")
    return resolved


def _automatic_run_id(output_root: Path) -> str:
    """Return a readable local-time run id that does not overwrite a run."""
    timestamp = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d_%H%M%S")
    stem = f"forward_{timestamp}"
    candidate = stem
    suffix = 1
    while (Path(output_root) / candidate).exists():
        candidate = f"{stem}_{suffix:02d}"
        suffix += 1
    return candidate


def _date_from_csv(path: Path) -> pd.Timestamp:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=["date"], dtype={"date": "string"})
    if frame.empty:
        raise ValueError(f"date source is empty: {path}")
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    return pd.Timestamp(dates.max()).normalize()


def _common_local_as_of(
    factor_paths: list[Path], market_panel: Path, trading_calendar: Path
) -> pd.Timestamp:
    sources = {
        str(path): _date_from_csv(path)
        for path in [*factor_paths, market_panel, trading_calendar]
    }
    unique = set(sources.values())
    if len(unique) != 1:
        details = ", ".join(
            f"{Path(path).name}={value.date().isoformat()}"
            for path, value in sources.items()
        )
        raise ValueError(f"local inputs do not share one as-of date: {details}")
    return next(iter(unique))


def _latest_rq_date(env_path: Path, history_start: str) -> pd.Timestamp:
    rq = init_rqdatac(env_path)
    today = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    endpoint = (today - pd.Timedelta(days=1)).date().isoformat()
    dates = pd.DatetimeIndex(
        pd.to_datetime(rq.get_trading_dates(history_start, endpoint), errors="raise")
    ).normalize()
    if len(dates) == 0:
        raise ValueError("RiceQuant returned no completed trading dates")
    return pd.Timestamp(dates.max()).normalize()


def _backup_inputs(paths: list[Path]) -> tuple[Path, dict[Path, Path | None]]:
    backup_root = Path(tempfile.mkdtemp(prefix="forward-input-backup-"))
    records: dict[Path, Path | None] = {}
    for index, path in enumerate(paths):
        path = Path(path)
        if path.exists():
            backup = backup_root / f"{index}-{path.name}"
            shutil.copy2(path, backup)
            records[path] = backup
        else:
            records[path] = None
    return backup_root, records


def _restore_inputs(backup_root: Path, records: Mapping[Path, Path | None]) -> None:
    for path, backup in records.items():
        if backup is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, path)
    shutil.rmtree(backup_root, ignore_errors=True)


def _run_factor_builders(
    factor_paths: list[Path],
    *,
    root: Path,
    start_date: str,
    end_date: str,
) -> None:
    for position, factor_path in enumerate(factor_paths, start=1):
        script = factor_path.parent.parent / "build_factor.py"
        if not script.is_file():
            raise FileNotFoundError(f"factor builder is missing: {script}")
        print(
            f"forward data update: factor {position}/{len(factor_paths)} "
            f"{factor_path.parent.parent.name}",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--start-date",
                start_date,
                "--end-date",
                end_date,
            ],
            cwd=root,
            check=True,
        )
        if not factor_path.is_file():
            raise FileNotFoundError(f"factor builder did not publish: {factor_path}")


def _refresh_inputs(
    config: Mapping[str, Any],
    config_path: Path,
    model_config: Mapping[str, Any],
    model_config_path: Path,
    *,
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    factor_paths, market_panel_path, trading_calendar_path = _model_paths(
        model_config, model_config_path
    )
    input_paths = [*factor_paths, market_panel_path, trading_calendar_path]
    backup_root, records = _backup_inputs(input_paths)
    project = config["project"]
    root = config_path.parent.parent
    start_date = str(project["history_start"])
    end_date = as_of.date().isoformat()
    try:
        _run_factor_builders(
            factor_paths,
            root=root,
            start_date=start_date,
            end_date=end_date,
        )
        # build_market_panel returns data rather than writing it, so publish the
        # refreshed market sources only after the whole download succeeds.
        panel, calendar = build_market_panel(
            env_path=_resolve(config_path.parent, str(project["env_path"])),
            start_date=start_date,
            end_date=end_date,
            index_code=str(project["index_code"]),
            industry_source=str(project["industry_source"]),
        )
        write_market_outputs(panel, calendar, market_panel_path, trading_calendar_path)
        observed = _common_local_as_of(
            factor_paths, market_panel_path, trading_calendar_path
        )
        if observed != as_of:
            raise ValueError(
                f"refreshed inputs end at {observed.date().isoformat()}, "
                f"expected {as_of.date().isoformat()}"
            )
    except Exception:
        _restore_inputs(backup_root, records)
        raise
    else:
        shutil.rmtree(backup_root, ignore_errors=True)
    return {
        "mode": "update-data",
        "history_start": start_date,
        "as_of_date": as_of.date().isoformat(),
        "factor_count": len(factor_paths),
        "factor_paths": [str(path) for path in factor_paths],
        "market_panel": str(market_panel_path),
        "trading_calendar": str(trading_calendar_path),
    }


def _source_hashes(
    factor_paths: list[Path],
    factor_wide_path: Path,
    market_panel_path: Path,
    trading_calendar_path: Path,
    frozen_spec_path: Path,
    final_runs_dir: Path,
) -> dict[str, str]:
    hashes = {
        "factor_wide": file_sha256(factor_wide_path),
        "market_panel": file_sha256(market_panel_path),
        "trading_calendar": file_sha256(trading_calendar_path),
        "frozen_spec": file_sha256(frozen_spec_path),
    }
    for factor_path in factor_paths:
        hashes[f"factor:{factor_path.parent.parent.name}"] = file_sha256(factor_path)
    for model_name in FIVE_FROZEN_MODELS:
        manifest_path = final_runs_dir / model_name / "manifest.json"
        hashes[f"model_manifest:{model_name}"] = file_sha256(manifest_path)
    return hashes


def run_forward(
    config_path: Path,
    *,
    run_id: str | None,
    update_data: bool,
) -> Path:
    config, model_config, model_config_path = _load_configs(config_path)
    paths = _forward_paths(config, config_path)
    if run_id is None:
        run_id = _automatic_run_id(paths["forward_runs_dir"])
    factor_paths, market_panel_path, trading_calendar_path = _model_paths(
        model_config, model_config_path
    )
    project = config["project"]
    if update_data:
        as_of = _latest_rq_date(
            _resolve(config_path.parent, str(project["env_path"])),
            str(project["history_start"]),
        )
        update_metadata = _refresh_inputs(
            config,
            config_path,
            model_config,
            model_config_path,
            as_of=as_of,
        )
    else:
        as_of = _common_local_as_of(
            factor_paths, market_panel_path, trading_calendar_path
        )
        update_metadata = {
            "mode": "local-only",
            "history_start": str(project["history_start"]),
            "as_of_date": as_of.date().isoformat(),
            "factor_count": len(factor_paths),
            "factor_paths": [str(path) for path in factor_paths],
            "market_panel": str(market_panel_path),
            "trading_calendar": str(trading_calendar_path),
        }

    calendar = pd.read_csv(trading_calendar_path, dtype={"date": "string"})
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise").dt.normalize()
    as_of = resolve_as_of_date(calendar, as_of)
    signal_end = last_complete_signal(calendar, as_of, horizon=10)
    # Keep prediction rows through as_of so the simulation can value every
    # official date while excluding the final incomplete signals from schedules.
    start = pd.Timestamp(str(project["strategy_start"])).normalize()
    market = pd.read_csv(market_panel_path, low_memory=False)

    run_parent = paths["forward_runs_dir"].resolve()
    if (run_parent / run_id).exists():
        raise FileExistsError(f"forward run already exists: {run_parent / run_id}")
    staging_root = Path(tempfile.mkdtemp(prefix="forward-prep-"))
    try:
        factor_wide_path = staging_root / "factor_wide.csv"
        features.build_factor_table(
            market_panel_path,
            factor_paths,
            factor_wide_path,
        )
        factor_wide = pd.read_csv(factor_wide_path, low_memory=False)
        feature_schema_path = (
            paths["final_runs_dir"] / FIVE_FROZEN_MODELS[0] / "feature_schema.json"
        )
        feature_schema = json.loads(feature_schema_path.read_text(encoding="utf-8"))
        forward_section = config.get("forward", {})
        strategy_section = config.get("strategy", {})
        frame = build_forward_frame(
            factor_wide,
            market,
            calendar,
            start=start,
            end=as_of,
            feature_schema=feature_schema,
            mad_width=float(forward_section.get("mad_width", 3.0)),
            expected_cross_section_size=int(
                strategy_section.get("expected_cross_section_size", 1000)
            ),
            date_chunk_size=int(forward_section.get("date_chunk_size", 60)),
            windows=tuple(int(value) for value in forward_section.get("context_windows", (1, 5, 20))),
            cap_coverage_threshold=float(
                forward_section.get("cap_coverage_threshold", 0.95)
            ),
        )
        predictions, model_metadata = predict_frozen_models(
            frame,
            final_runs_dir=paths["final_runs_dir"],
            frozen_spec_path=paths["frozen_spec"],
        )
        settings = strategy_settings(config, start=start, end=as_of)
        source_hashes = _source_hashes(
            factor_paths,
            factor_wide_path,
            market_panel_path,
            trading_calendar_path,
            paths["frozen_spec"],
            paths["final_runs_dir"],
        )
        metadata = {
            **update_metadata,
            "as_of_date": as_of.date().isoformat(),
            "last_complete_signal": signal_end.date().isoformat(),
            "strategy_start": start.date().isoformat(),
            "model_metadata": model_metadata,
            "factor_update_manifest": update_metadata,
        }
        return publish_forward_run(
            run_id,
            feature_frame=frame,
            predictions=predictions,
            market=market,
            calendar=calendar,
            settings=settings,
            output_root=run_parent,
            metadata=metadata,
            source_hashes=source_hashes,
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update inputs, score frozen rank models, and run forward strategy."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional output run id; defaults to forward_YYYYMMDD_HHMMSS.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--update-data", action="store_true")
    mode.add_argument("--local-only", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    try:
        output = run_forward(
            Path(args.config).resolve(),
            run_id=args.run_id,
            update_data=bool(args.update_data),
        )
    except (FileNotFoundError, ValueError, FileExistsError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"forward strategy failed: {error}") from error
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    print(
        "forward strategy run written: "
        f"as_of_date={metadata['as_of_date']} "
        f"last_complete_signal={metadata['last_complete_signal']} "
        f"models={len(FIVE_FROZEN_MODELS)} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
