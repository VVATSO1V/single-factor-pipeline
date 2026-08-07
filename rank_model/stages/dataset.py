"""Validated publication of the isolated 10-day rank-label dataset."""

from __future__ import annotations

import csv
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["date", "stock_code"]
TARGET_COLUMN = "target_10d"
RANK_TARGET_COLUMN = "rank_target_10d"
SPLIT_COLUMN = "split_10d"
EXIT_DATE_COLUMN = "exit_date_10d"
FORBIDDEN_FEATURE_COLUMNS = {
    "date",
    "stock_code",
    "target_10d",
    "rank_target_10d",
    "market_target_10d",
    "industry_target_10d",
    "alpha_target_10d",
    "exit_date_10d",
    "split_10d",
    "entry_tradeable",
}
MAXIMUM_DEVELOPMENT_END = date(2023, 12, 31)
MAX_TARGET_EXIT_CALENDAR_DAYS = 31
FUTURE_STATUS_TIME_PATTERN = re.compile(
    r"t(?:_|\+|_?plus_?)?0*([1-9][0-9]*)"
)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile_rank_from_returns(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    finite = np.isfinite(numeric.to_numpy())
    count = int(finite.sum())
    if count < 2:
        raise ValueError("a rank date requires at least two finite returns")
    output = pd.Series(np.nan, index=values.index, dtype="float64")
    average_rank = numeric.loc[finite].rank(method="average", ascending=True)
    output.loc[finite] = (average_rank - 1.0) / float(count - 1)
    return output


def _load_source_schema(source_schema: Path, source_dataset: Path) -> dict[str, Any]:
    if not source_schema.exists():
        raise FileNotFoundError(source_schema)
    if not source_dataset.exists():
        raise FileNotFoundError(source_dataset)
    schema = json.loads(source_schema.read_text(encoding="utf-8"))
    expected_hash = schema.get("context_dataset_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"{source_schema} has no valid context_dataset_sha256")
    actual_hash = file_sha256(source_dataset)
    if actual_hash != expected_hash:
        raise ValueError(
            "source hash does not match source schema: "
            f"expected={expected_hash} actual={actual_hash}"
        )
    return schema


def _source_columns(
    schema: dict[str, Any],
) -> tuple[list[str], list[str], list[str], str]:
    key_columns = schema.get("key_columns")
    if key_columns != KEY_COLUMNS:
        raise ValueError("source schema key_columns must be ['date', 'stock_code']")
    factor_features = schema.get("factor_feature_columns")
    continuous_features = schema.get("continuous_feature_columns")
    industry_column = schema.get("industry_column")
    if not isinstance(factor_features, list) or not factor_features:
        raise ValueError("source schema requires factor_feature_columns")
    if not isinstance(continuous_features, list) or not continuous_features:
        raise ValueError("source schema requires continuous_feature_columns")
    if not isinstance(industry_column, str) or not industry_column:
        raise ValueError("source schema requires industry_column")
    if not all(
        isinstance(column, str) and column
        for column in [*factor_features, *continuous_features]
    ):
        raise ValueError("source schema feature groups must contain non-empty strings")
    if not set(factor_features).issubset(continuous_features):
        raise ValueError(
            "source schema factor_feature_columns must be included in "
            "continuous_feature_columns"
        )
    feature_columns = [*continuous_features, industry_column]
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("source schema feature columns contain duplicates")
    forbidden = sorted(
        column for column in feature_columns if _is_forbidden_feature(column)
    )
    if forbidden:
        raise ValueError(f"source schema contains forbidden features: {forbidden}")
    return key_columns, feature_columns, list(continuous_features), industry_column


def _is_forbidden_feature(column: str) -> bool:
    normalized = column.lower()
    return (
        normalized in FORBIDDEN_FEATURE_COLUMNS
        or normalized.startswith(("target_", "split_", "exit_"))
        or normalized.startswith(("entry_", "future_", "next_"))
        or "t+1" in normalized
        or "t_plus_" in normalized
        or (
            "status" in normalized
            and FUTURE_STATUS_TIME_PATTERN.search(normalized) is not None
        )
    )


def _read_source_dataset(
    source_dataset: Path,
    schema: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], list[str], str]:
    import pyarrow.parquet as pq

    key_columns, feature_columns, continuous_columns, industry_column = _source_columns(
        schema
    )
    target_columns = schema.get("target_columns")
    sample_columns = schema.get("sample_columns")
    if not isinstance(target_columns, list) or TARGET_COLUMN not in target_columns:
        raise ValueError(f"source schema must declare {TARGET_COLUMN}")
    if not isinstance(sample_columns, list) or not {
        SPLIT_COLUMN,
        EXIT_DATE_COLUMN,
    }.issubset(sample_columns):
        raise ValueError(
            f"source schema must declare {SPLIT_COLUMN} and {EXIT_DATE_COLUMN}"
        )
    required_columns = [
        *key_columns,
        *feature_columns,
        TARGET_COLUMN,
        SPLIT_COLUMN,
        EXIT_DATE_COLUMN,
    ]
    if len(set(required_columns)) != len(required_columns):
        raise ValueError("source schema feature_columns overlap source contract columns")
    available_columns = set(pq.read_schema(source_dataset).names)
    missing = sorted(set(required_columns).difference(available_columns))
    if missing:
        raise ValueError(f"{source_dataset} is missing required columns: {missing}")
    frame = pd.read_parquet(source_dataset, columns=required_columns)
    return frame, feature_columns, continuous_columns, industry_column


def _normalize_keys(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["stock_code"] = result["stock_code"].astype("string").str.strip()
    if result[KEY_COLUMNS].isna().any().any() or result["stock_code"].eq("").any():
        raise ValueError(f"{path} contains missing date or stock_code keys")
    duplicated = result.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        sample = result.loc[duplicated, KEY_COLUMNS].head(5).to_dict("records")
        raise ValueError(f"{path} contains duplicate keys: {sample}")
    return result


def _validate_source_dates(
    frame: pd.DataFrame,
    source_dataset: Path,
    development_end: date,
) -> None:
    if development_end != MAXIMUM_DEVELOPMENT_END:
        raise ValueError("development_end must be 2023-12-31")
    latest = frame["date"].max()
    if latest.date() > MAXIMUM_DEVELOPMENT_END:
        raise ValueError(
            f"{source_dataset} contains dates after 2023-12-31: "
            f"{latest.strftime('%Y-%m-%d')}"
        )


def _validate_date_sizes(
    frame: pd.DataFrame,
    source_dataset: Path,
) -> None:
    sizes = frame.groupby("date", sort=False).size()
    invalid = sizes.loc[sizes.ne(1000)]
    if not invalid.empty:
        sample = {
            timestamp.strftime("%Y-%m-%d"): int(size)
            for timestamp, size in invalid.head(5).items()
        }
        raise ValueError(
            f"{source_dataset} must contain exactly 1000 source rows per date: "
            f"{sample}"
        )


def _normalize_target(frame: pd.DataFrame, source_dataset: Path) -> pd.DataFrame:
    result = frame.copy()
    original = result[TARGET_COLUMN]
    numeric = pd.to_numeric(original, errors="coerce")
    invalid = original.notna() & numeric.isna()
    if invalid.any():
        sample = original.loc[invalid].astype(str).head(5).tolist()
        raise ValueError(f"{source_dataset} has non-numeric {TARGET_COLUMN} values: {sample}")
    result[TARGET_COLUMN] = numeric.astype("float64").replace([np.inf, -np.inf], np.nan)
    return result


def _validate_rank_ordering(values: pd.Series, ranks: pd.Series, timestamp: pd.Timestamp) -> None:
    finite = np.isfinite(values.to_numpy())
    ordered = pd.DataFrame(
        {"return": values.loc[finite], "rank": ranks.loc[finite]}
    ).sort_values("return", kind="stable")
    if ordered["rank"].diff().lt(0).any():
        raise ValueError(
            "return ordering and rank ordering disagree on "
            f"{timestamp.strftime('%Y-%m-%d')}"
        )


def _coverage_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for timestamp, group in frame.groupby("date", sort=True):
        values = group[TARGET_COLUMN]
        valid = np.isfinite(values.to_numpy())
        finite_values = values.loc[valid]
        rows.append(
            {
                "date": timestamp.strftime("%Y-%m-%d"),
                "total_rows": int(len(group)),
                "valid_targets": int(valid.sum()),
                "coverage": float(valid.mean()),
                "tie_rows": int(finite_values.duplicated(keep=False).sum()),
            }
        )
    return rows


def _temporary_path(path: Path, suffix: str = ".tmp") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=suffix,
    )
    os.close(descriptor)
    return Path(temporary_name)


def _publish_bundle(
    temporary_paths: list[Path],
    output_paths: list[Path],
) -> None:
    """Publish a bundle or restore every pre-existing file after a failure."""
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for output_path in output_paths:
            if output_path.exists():
                backup_path = _temporary_path(output_path, suffix=".bak")
                backup_path.unlink()
                output_path.replace(backup_path)
                backups.append((output_path, backup_path))
        for temporary_path, output_path in zip(temporary_paths, output_paths):
            temporary_path.replace(output_path)
            published.append(output_path)
    except Exception:
        for output_path in reversed(published):
            if output_path.exists():
                output_path.unlink()
        for output_path, backup_path in reversed(backups):
            if backup_path.exists():
                backup_path.replace(output_path)
        raise
    finally:
        for backup_path in (backup for _, backup in backups):
            if backup_path.exists():
                backup_path.unlink()


def _write_bundle(
    frame: pd.DataFrame,
    output_dataset: Path,
    output_schema: Path,
    coverage_path: Path,
    schema: dict[str, Any],
    coverage_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    temporary_dataset = _temporary_path(output_dataset)
    temporary_schema = _temporary_path(output_schema)
    temporary_coverage = _temporary_path(coverage_path)
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata[b"rank_model_dataset_schema"] = json.dumps(
            schema, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        pq.write_table(
            table.replace_schema_metadata(metadata),
            temporary_dataset,
            compression="zstd",
        )
        output_hash = file_sha256(temporary_dataset)
        published_schema = {**schema, "parquet_sha256": output_hash}
        temporary_schema.write_text(
            json.dumps(published_schema, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with temporary_coverage.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["date", "total_rows", "valid_targets", "coverage", "tie_rows"],
            )
            writer.writeheader()
            writer.writerows(coverage_rows)
        _publish_bundle(
            [temporary_dataset, temporary_schema, temporary_coverage],
            [output_dataset, output_schema, coverage_path],
        )
        return published_schema
    finally:
        for temporary_path in (
            temporary_dataset,
            temporary_schema,
            temporary_coverage,
        ):
            if temporary_path.exists():
                temporary_path.unlink()


def build_rank_dataset(
    source_dataset: Path,
    source_schema: Path,
    output_dataset: Path,
    output_schema: Path,
    coverage_path: Path,
    development_end: date,
) -> dict[str, Any]:
    """Build and atomically publish 10-day percentile labels from source returns."""
    source_dataset = Path(source_dataset)
    source_schema = Path(source_schema)
    output_dataset = Path(output_dataset)
    output_schema = Path(output_schema)
    coverage_path = Path(coverage_path)

    source_contract = _load_source_schema(source_schema, source_dataset)
    (
        dataset,
        feature_columns,
        continuous_columns,
        industry_column,
    ) = _read_source_dataset(source_dataset, source_contract)
    dataset = _normalize_keys(dataset, source_dataset)
    _validate_source_dates(dataset, source_dataset, development_end)
    _validate_date_sizes(dataset, source_dataset)
    dataset = _normalize_target(dataset, source_dataset)
    dataset[RANK_TARGET_COLUMN] = np.nan
    for timestamp, index in dataset.groupby("date", sort=False).groups.items():
        ranks = percentile_rank_from_returns(dataset.loc[index, TARGET_COLUMN])
        _validate_rank_ordering(dataset.loc[index, TARGET_COLUMN], ranks, timestamp)
        dataset.loc[index, RANK_TARGET_COLUMN] = ranks

    output_columns = [
        *KEY_COLUMNS,
        *feature_columns,
        TARGET_COLUMN,
        RANK_TARGET_COLUMN,
        SPLIT_COLUMN,
        EXIT_DATE_COLUMN,
    ]
    dataset = dataset[output_columns].sort_values(KEY_COLUMNS).reset_index(drop=True)
    coverage_rows = _coverage_rows(dataset)
    schema: dict[str, Any] = {
        "schema_version": 1,
        "source_dataset": str(source_dataset.resolve()),
        "source_dataset_sha256": file_sha256(source_dataset),
        "source_schema_sha256": file_sha256(source_schema),
        "key_columns": KEY_COLUMNS,
        "feature_columns": feature_columns,
        "continuous_feature_columns": continuous_columns,
        "industry_column": industry_column,
        "target_column": TARGET_COLUMN,
        "rank_target_column": RANK_TARGET_COLUMN,
        "split_column": SPLIT_COLUMN,
        "exit_date_column": EXIT_DATE_COLUMN,
        "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "target_formula": "(average_rank(target_10d) - 1) / (finite_count - 1)",
        "development_end": development_end.isoformat(),
        "row_count": int(len(dataset)),
        "date_count": int(dataset["date"].nunique()),
        "output_columns": output_columns,
    }
    published_schema = _write_bundle(
        dataset,
        output_dataset,
        output_schema,
        coverage_path,
        schema,
        coverage_rows,
    )
    return {
        "row_count": published_schema["row_count"],
        "date_count": published_schema["date_count"],
        "rank_target_column": RANK_TARGET_COLUMN,
        "output_dataset": str(output_dataset.resolve()),
        "output_schema": str(output_schema.resolve()),
        "coverage_path": str(coverage_path.resolve()),
        "parquet_sha256": published_schema["parquet_sha256"],
    }


def load_rank_dataset(
    dataset_path: Path,
    schema_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a published rank dataset only after verifying its output hash."""
    dataset_path = Path(dataset_path)
    schema_path = Path(schema_path)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if not schema_path.exists():
        raise FileNotFoundError(schema_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    expected_hash = schema.get("parquet_sha256")
    actual_hash = file_sha256(dataset_path)
    if not isinstance(expected_hash, str) or actual_hash != expected_hash:
        raise ValueError(
            "rank dataset hash does not match rank schema: "
            f"expected={expected_hash} actual={actual_hash}"
        )
    frame = pd.read_parquet(dataset_path, columns=schema.get("output_columns"))
    if list(frame.columns) != schema.get("output_columns"):
        raise ValueError("rank dataset columns do not match rank schema")
    return frame, schema
