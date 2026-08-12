"""Freeze selected specifications and refit them on the full development window."""

from __future__ import annotations

from datetime import date
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from rank_model.stages.dataset import (
    EXIT_DATE_COLUMN,
    MAX_TARGET_EXIT_CALENDAR_DAYS,
    SPLIT_COLUMN,
    file_sha256,
)
from rank_model.stages.preprocessing import RankPreprocessor
from rank_model.stages.ranking import sample_top100_pairs, sorted_group_layout
from rank_model.stages.training import (
    MODEL_REGISTRY,
    RANK_TARGET_COLUMN,
    TrainingOutcome,
    _artifact_bytes,
    _artifact_payload_bytes,
    _build_mlp_model,
    _date_batch_loss_scale,
    _date_batch_ranges,
    _fixed_parameters,
    _hybrid_date_loss,
    _hybrid_mlp_parameters,
    _load_persisted_model,
    _missing_feature_rates,
    _predict_model,
    _unknown_industry_rate,
    validate_run_id,
    _write_lightgbm_model,
)


FREEZE_SCHEMA_VERSION = 1
REQUIRED_FINAL_MODELS = (
    "ridge_rank_regression",
    "xgboost_rank_regression",
    "lightgbm_rank_regression",
    "lightgbm_lambdarank",
    "mlp_top100_hybrid_rank",
)
_SHA_FIELDS = (
    "rank_dataset_sha256",
    "source_dataset_sha256",
    "source_schema_sha256",
)
_SCHEMA_LOCATION_FIELDS = {
    "source_dataset",
    "parquet_sha256",
    "transformed_feature_columns",
}


def rank_schema_contract_sha256(schema: dict[str, Any]) -> str:
    """Hash the logical rank-data contract while excluding storage locations."""
    contract = {
        key: value
        for key, value in schema.items()
        if key not in _SCHEMA_LOCATION_FIELDS
    }
    payload = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rank_label_audit_sha256(frame: pd.DataFrame) -> str:
    """Hash ordered 2023 keys and raw/rank labels independent of Parquet bytes."""
    required = {"date", "stock_code", "target_10d", RANK_TARGET_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"validation label audit is missing columns: {missing}")
    audit = frame.loc[:, ["date", "stock_code", "target_10d", RANK_TARGET_COLUMN]].copy()
    audit["date"] = pd.to_datetime(audit["date"], errors="raise").dt.strftime("%Y-%m-%d")
    audit["stock_code"] = audit["stock_code"].astype("string")
    audit = audit.sort_values(["date", "stock_code"], kind="mergesort")
    if audit.duplicated(["date", "stock_code"]).any():
        raise ValueError("validation label audit keys are duplicated")
    row_hashes = pd.util.hash_pandas_object(audit, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(row_hashes.astype("<u8", copy=False).tobytes()).hexdigest()


def rank_dataset_logical_sha256(
    frame: pd.DataFrame, schema: dict[str, Any]
) -> str:
    """Hash every published value plus ordered column names and dtypes."""
    columns = schema.get("output_columns")
    if not isinstance(columns, list) or list(frame.columns) != columns:
        raise ValueError("rank dataset columns do not match its logical hash contract")
    descriptor = json.dumps(
        {
            "columns": columns,
            "dtypes": [str(frame[column].dtype) for column in columns],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    row_hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="uint64")
    digest = hashlib.sha256(descriptor)
    digest.update(row_hashes.astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def validate_candidate_configuration(candidates: list[dict[str, Any]]) -> None:
    """Require the exact five selected specifications and fixed MLP epoch count."""
    if not isinstance(candidates, list):
        raise ValueError("finalization candidates must be a list")
    model_names = [candidate.get("model_name") for candidate in candidates]
    run_ids = [candidate.get("run_id") for candidate in candidates]
    if len(candidates) != len(REQUIRED_FINAL_MODELS) or set(model_names) != set(
        REQUIRED_FINAL_MODELS
    ):
        raise ValueError("finalization requires the exact five selected model types")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("finalization candidate run IDs must be unique")
    for candidate in candidates:
        model_name = candidate["model_name"]
        if model_name == "mlp_top100_hybrid_rank":
            if candidate.get("final_epochs") != 12:
                raise ValueError("final hybrid MLP must use exactly 12 epochs")
        elif "final_epochs" in candidate:
            raise ValueError("only the final hybrid MLP may configure final_epochs")


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def build_frozen_spec(
    candidates: list[dict[str, Any]],
    validation_runs_directory: Path,
    development_start: date,
    development_end: date,
    required_models: tuple[str, ...] | None = REQUIRED_FINAL_MODELS,
) -> dict[str, Any]:
    """Build an auditable contract from completed validation runs."""
    if development_start != date(2019, 1, 1) or development_end != date(2023, 12, 31):
        raise ValueError(
            "final refit development window must be exactly 2019-01-01 through 2023-12-31"
        )
    if not candidates:
        raise ValueError("at least one finalization candidate is required")
    if required_models is not None:
        if tuple(required_models) != REQUIRED_FINAL_MODELS:
            raise ValueError("unsupported required final model contract")
        validate_candidate_configuration(candidates)
    runs_directory = Path(validation_runs_directory).resolve()
    frozen_candidates: list[dict[str, Any]] = []
    common_hashes: dict[str, str] | None = None

    for configured in candidates:
        try:
            run_id = configured["run_id"]
            expected_model = configured["model_name"]
            role = configured["role"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "each finalization candidate requires run_id, model_name, and role"
            ) from error
        if not all(isinstance(value, str) and value for value in (run_id, expected_model, role)):
            raise ValueError("candidate run_id, model_name, and role must be non-empty strings")
        validate_run_id(run_id)

        run_directory = (runs_directory / run_id).resolve()
        try:
            run_directory.relative_to(runs_directory)
        except ValueError as error:
            raise ValueError("candidate run_id escapes the validation runs directory") from error
        manifest_path = resolve_run_artifact(run_directory, run_directory / "manifest.json")
        feature_schema_path = resolve_run_artifact(
            run_directory, run_directory / "feature_schema.json"
        )
        manifest = _load_json_object(manifest_path)
        feature_schema = _load_json_object(feature_schema_path)
        if manifest.get("status") != "completed":
            raise ValueError(f"candidate run is not completed: {run_id}")
        evaluation = manifest.get("evaluation")
        if not isinstance(evaluation, dict) or evaluation.get("status") != "completed":
            raise ValueError(f"candidate run has no completed validation evaluation: {run_id}")
        if manifest.get("run_id") != run_id or manifest.get("model_name") != expected_model:
            raise ValueError(f"candidate identity does not match its manifest: {run_id}")

        hashes = {field: manifest.get(field) for field in _SHA_FIELDS}
        if not all(
            isinstance(value, str) and len(value) == 64 for value in hashes.values()
        ):
            raise ValueError(f"candidate run has invalid data hashes: {run_id}")
        if common_hashes is None:
            common_hashes = hashes
        elif hashes != common_hashes:
            raise ValueError("all frozen candidates must use the same rank dataset and source hashes")

        fixed_parameters = manifest.get("fixed_parameters")
        transformed_features = feature_schema.get("transformed_feature_columns")
        if not isinstance(fixed_parameters, dict) or not fixed_parameters:
            raise ValueError(f"candidate has no fixed parameters: {run_id}")
        if fixed_parameters.get("model_name") != expected_model:
            raise ValueError(f"candidate fixed parameters name does not match: {run_id}")
        if (
            not isinstance(transformed_features, list)
            or not transformed_features
            or not all(isinstance(column, str) and column for column in transformed_features)
        ):
            raise ValueError(f"candidate has no transformed feature contract: {run_id}")

        final_training: dict[str, Any] = {
            "uses_validation": False,
            "uses_early_stopping": False,
        }
        if "final_epochs" in configured:
            epochs = configured["final_epochs"]
            if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
                raise ValueError("final_epochs must be a positive integer")
            final_training["epochs"] = epochs

        predictions_path = resolve_run_artifact(
            run_directory, run_directory / "predictions_10d.parquet"
        )
        if not predictions_path.exists():
            raise FileNotFoundError(predictions_path)
        if file_sha256(predictions_path) != manifest.get("predictions_10d_sha256"):
            raise ValueError(f"candidate prediction artifact hash does not match: {run_id}")
        validation_labels = pd.read_parquet(
            predictions_path,
            columns=["date", "stock_code", "split", "target_10d", RANK_TARGET_COLUMN],
        )
        if not validation_labels["split"].eq("validation").all():
            raise ValueError(f"candidate predictions contain a non-validation row: {run_id}")
        validation_dates = pd.to_datetime(validation_labels["date"], errors="raise")
        if not validation_dates.between("2023-01-01", "2023-12-31").all():
            raise ValueError(f"candidate validation predictions must stay within 2023: {run_id}")
        if len(validation_labels) != manifest.get("validation_rows"):
            raise ValueError(f"candidate validation row count does not match manifest: {run_id}")
        if validation_dates.nunique() != manifest.get("validation_dates"):
            raise ValueError(f"candidate validation date count does not match manifest: {run_id}")
        label_audit = rank_label_audit_sha256(validation_labels)
        schema_contract = rank_schema_contract_sha256(feature_schema)
        if frozen_candidates:
            if label_audit != frozen_candidates[0]["validation_label_audit_sha256"]:
                raise ValueError("all frozen candidates must have the same validation labels")
            if schema_contract != frozen_candidates[0]["rank_schema_contract_sha256"]:
                raise ValueError("all frozen candidates must have the same rank schema contract")

        frozen_candidates.append(
            {
                "role": role,
                "model_name": expected_model,
                "source_run_id": run_id,
                "source_manifest_sha256": file_sha256(manifest_path),
                "feature_schema_sha256": file_sha256(feature_schema_path),
                "transformed_feature_columns": transformed_features,
                "fixed_parameters": fixed_parameters,
                "final_training": final_training,
                "rank_schema_contract_sha256": schema_contract,
                "validation_label_audit_sha256": label_audit,
            }
        )

    assert common_hashes is not None
    return {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "development_window": {
            "signal_start": development_start.isoformat(),
            "signal_end": development_end.isoformat(),
            "maximum_target_exit": development_end.isoformat(),
        },
        "data_hashes": {
            **common_hashes,
            "rank_schema_contract_sha256": frozen_candidates[0]["rank_schema_contract_sha256"],
            "validation_label_audit_sha256": frozen_candidates[0]["validation_label_audit_sha256"],
            "validation_label_rows": int(len(validation_labels)),
            "validation_label_dates": int(pd.to_datetime(validation_labels["date"]).nunique()),
        },
        "candidate_count": len(frozen_candidates),
        "candidates": frozen_candidates,
    }


def resolve_run_artifact(run_directory: Path, artifact_path: Path) -> Path:
    """Resolve one existing run artifact and reject links outside the run."""
    run_root = Path(run_directory).resolve(strict=True)
    artifact = Path(artifact_path).resolve(strict=True)
    try:
        artifact.relative_to(run_root)
    except ValueError as error:
        raise ValueError(f"run artifact escapes its run directory: {artifact}") from error
    if not artifact.is_file():
        raise ValueError(f"run artifact is not a regular file: {artifact}")
    return artifact


def write_frozen_spec(specification: dict[str, Any], output_path: Path) -> Path:
    """Publish a frozen contract without allowing silent replacement."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(specification, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise FileExistsError(f"frozen specification already exists: {output}") from error
    return output


def freeze_refit_dataset(
    specification: dict[str, Any],
    dataset: pd.DataFrame,
    schema: dict[str, Any],
    dataset_path: Path,
    schema_path: Path,
) -> dict[str, Any]:
    """Attach the exact physical and logical 2019-2023 refit bundle contract."""
    window = specification.get("development_window", {})
    start = pd.Timestamp(window.get("signal_start"))
    end = pd.Timestamp(window.get("signal_end"))
    dates = pd.to_datetime(dataset["date"], errors="raise").dt.normalize()
    if dates.lt(start).any() or dates.gt(end).any():
        raise ValueError("refit dataset contains dates outside the frozen development window")
    if schema.get("development_end") != end.date().isoformat():
        raise ValueError("refit schema development_end does not match the freeze")
    data_hashes = specification.get("data_hashes")
    if not isinstance(data_hashes, dict):
        raise ValueError("frozen specification is missing validation data hashes")
    validate_frozen_dataset(dataset, schema, data_hashes)
    result = json.loads(json.dumps(specification))
    result["refit_dataset"] = {
        "dataset_sha256": file_sha256(dataset_path),
        "dataset_bytes": Path(dataset_path).stat().st_size,
        "schema_contract_sha256": rank_schema_contract_sha256(schema),
        "logical_sha256": rank_dataset_logical_sha256(dataset, schema),
        "row_count": int(len(dataset)),
        "date_count": int(dates.nunique()),
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
    }
    return result


def validate_frozen_files_before_read(
    dataset_path: Path,
    schema_path: Path,
    contract: dict[str, Any],
) -> None:
    """Reject a changed refit bundle before pandas parses any data rows."""
    dataset = Path(dataset_path)
    schema = Path(schema_path)
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    if not schema.exists():
        raise FileNotFoundError(schema)
    if dataset.stat().st_size != contract.get("dataset_bytes"):
        raise ValueError("frozen refit dataset size does not match")
    if file_sha256(dataset) != contract.get("dataset_sha256"):
        raise ValueError("frozen refit dataset hash does not match")
    schema_value = _load_json_object(schema)
    if rank_schema_contract_sha256(schema_value) != contract.get(
        "schema_contract_sha256"
    ):
        raise ValueError("frozen refit schema logical contract does not match")


def select_final_development_rows(
    frame: pd.DataFrame,
    development_start: date,
    development_end: date,
) -> pd.DataFrame:
    """Select finite labels whose signals and target exits stay in 2019-2023."""
    required = {
        "date",
        "stock_code",
        SPLIT_COLUMN,
        EXIT_DATE_COLUMN,
        RANK_TARGET_COLUMN,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"rank dataset is missing final-refit columns: {missing}")
    start = pd.Timestamp(development_start)
    end = pd.Timestamp(development_end)
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    exits = pd.to_datetime(frame[EXIT_DATE_COLUMN], errors="coerce").dt.normalize()
    if dates.lt(start).any() or dates.gt(end).any():
        raise ValueError("rank dataset contains dates outside the frozen development window")

    splits = frame[SPLIT_COLUMN].astype("string").str.strip().str.lower()
    assigned = splits.notna()
    if splits.loc[assigned].eq("").any():
        raise ValueError("development rows contain blank assigned split values")
    invalid_splits = assigned & ~splits.isin(("train", "validation"))
    if invalid_splits.any():
        raise ValueError("development rows contain invalid split values")

    labels = pd.to_numeric(frame[RANK_TARGET_COLUMN], errors="coerce")
    known_exit = exits.notna()
    exit_days = (exits - dates).dt.days
    invalid_exit = known_exit & (
        exit_days.le(0) | exit_days.gt(MAX_TARGET_EXIT_CALENDAR_DAYS)
    )
    if invalid_exit.any():
        raise ValueError(
            "exit_date_10d must be 1 through "
            f"{MAX_TARGET_EXIT_CALENDAR_DAYS} calendar days after its signal date"
        )
    old_training_boundary = pd.Timestamp("2022-12-31")
    reclaimed_boundary = (
        splits.isna()
        & np.isfinite(labels.to_numpy(dtype="float64"))
        & dates.le(old_training_boundary)
        & exits.gt(old_training_boundary)
        & exits.le(end)
    )
    unexplained_unassigned = (
        splits.isna()
        & np.isfinite(labels.to_numpy(dtype="float64"))
        & exits.le(end)
        & ~reclaimed_boundary
    )
    if unexplained_unassigned.any():
        raise ValueError(
            "final refit found an unexplained unassigned row with a finite label"
        )
    eligible = (
        np.isfinite(labels.to_numpy(dtype="float64"))
        & known_exit
        & exits.le(end)
        & (splits.isin(("train", "validation")) | reclaimed_boundary)
    )
    selected = frame.loc[eligible].copy()
    if selected.empty:
        raise ValueError("no eligible 2019-2023 labels remain for final refit")
    selected["date"] = dates.loc[eligible]
    selected[EXIT_DATE_COLUMN] = exits.loc[eligible]
    selected[SPLIT_COLUMN] = splits.loc[eligible]
    if selected.duplicated(["date", "stock_code"]).any():
        raise ValueError("final-refit development keys are not unique")
    return selected


def validate_frozen_dataset(
    dataset: pd.DataFrame,
    schema: dict[str, Any],
    frozen_data_hashes: dict[str, Any],
) -> None:
    """Accept exact rank bytes or a rebuilt bundle with the same logical contract."""
    for field in ("source_dataset_sha256", "source_schema_sha256"):
        if schema.get(field) != frozen_data_hashes.get(field):
            raise ValueError(f"rank schema {field} does not match the frozen source")
    expected_contract = frozen_data_hashes.get("rank_schema_contract_sha256")
    if rank_schema_contract_sha256(schema) != expected_contract:
        raise ValueError("rank schema does not match the frozen logical contract")
    exact_rank_bytes = schema.get("parquet_sha256") == frozen_data_hashes.get(
        "rank_dataset_sha256"
    )
    if not exact_rank_bytes:
        split = dataset[SPLIT_COLUMN].astype("string").str.strip().str.lower()
        validation = dataset.loc[split.eq("validation")]
        if len(validation) != frozen_data_hashes.get("validation_label_rows"):
            raise ValueError("rebuilt validation label row count does not match the freeze")
        if validation["date"].nunique() != frozen_data_hashes.get("validation_label_dates"):
            raise ValueError("rebuilt validation label date count does not match the freeze")
        if rank_label_audit_sha256(validation) != frozen_data_hashes.get(
            "validation_label_audit_sha256"
        ):
            raise ValueError("rebuilt validation label audit does not match the freeze")


def _write_model_artifacts(directory: Path, outcome: TrainingOutcome) -> None:
    preprocessor = outcome.model_objects.get("preprocessor")
    model = outcome.model_objects.get("model")
    if not isinstance(preprocessor, RankPreprocessor) or model is None:
        raise ValueError("final trainer did not provide a preprocessor and model")
    joblib.dump(preprocessor, directory / "preprocessor.joblib")
    if isinstance(model, xgb.Booster):
        model.save_model(directory / "model.json")
    elif isinstance(model, lgb.Booster):
        _write_lightgbm_model(model, directory / "model.txt")
    elif outcome.model_objects.get("model_type") == "pytorch_mlp":
        import torch

        architecture = outcome.model_objects.get("architecture")
        if not isinstance(architecture, dict):
            raise ValueError("final PyTorch model is missing architecture metadata")
        (directory / "model_architecture.json").write_text(
            json.dumps(architecture, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        torch.save(model.state_dict(), directory / "model_state_dict.pt")
    else:
        joblib.dump(model, directory / "model.joblib")


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: file_sha256(path)
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "manifest.json"
    }


def _check_frozen_data_contract(
    schema: dict[str, Any], candidate: dict[str, Any]
) -> None:
    fixed = candidate.get("fixed_parameters")
    if not isinstance(fixed, dict) or not fixed:
        raise ValueError("frozen candidate has no fixed parameters")
    if fixed.get("model_name") != candidate.get("model_name"):
        raise ValueError("frozen candidate model identity is inconsistent")
    expected_features = candidate.get("transformed_feature_columns")
    if not isinstance(expected_features, list) or not expected_features:
        raise ValueError("frozen candidate has no transformed feature contract")


def train_final_mlp_top100_hybrid_rank(
    development: pd.DataFrame,
    schema: dict[str, Any],
    params: dict[str, Any],
    *,
    epochs: int,
) -> TrainingOutcome:
    """Fit the hybrid MLP for a fixed epoch count without validation selection."""
    import torch

    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("final hybrid epochs must be a positive integer")
    settings = _hybrid_mlp_parameters(params)
    development, _ = sorted_group_layout(development)
    if development.empty:
        raise ValueError("final hybrid MLP development frame is empty")
    continuous_columns = schema.get("continuous_feature_columns")
    industry_column = schema.get("industry_column")
    if not isinstance(continuous_columns, list) or not isinstance(industry_column, str):
        raise ValueError("final hybrid MLP requires schema features")

    preprocessor = RankPreprocessor.fit(
        development,
        continuous_columns=continuous_columns,
        industry_column=industry_column,
    )
    x_train = preprocessor.transform(development, scale_continuous=True)
    y_train = development[RANK_TARGET_COLUMN].to_numpy(dtype="float64")
    date_codes, unique_dates = pd.factorize(development["date"], sort=False)
    batch_ranges = _date_batch_ranges(development["date"], settings["dates_per_batch"])

    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = _build_mlp_model(
        x_train.shape[1],
        hidden_layers=settings["hidden_layers"],
        dropout=settings["dropout"],
        output_bias=float(np.mean(y_train)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    features = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
    targets = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))
    date_code_tensor = torch.from_numpy(np.ascontiguousarray(date_codes, dtype=np.int64))
    history: list[dict[str, Any]] = []
    optimizer_steps = 0

    for epoch in range(1, epochs + 1):
        left, right, direction = sample_top100_pairs(
            development["date"],
            y_train,
            settings["boundary_pairs_per_positive"],
            settings["broad_pairs_per_positive"],
            seed=settings["seed"] + epoch,
        )
        pair_date_codes = date_codes[left]
        left_tensor = torch.from_numpy(np.ascontiguousarray(left, dtype=np.int64))
        right_tensor = torch.from_numpy(np.ascontiguousarray(right, dtype=np.int64))
        direction_tensor = torch.from_numpy(
            np.ascontiguousarray(direction, dtype=np.float32)
        )
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_pairwise = 0.0
        epoch_date_count = 0
        model.train()
        for batch_start, batch_end, batch_codes in batch_ranges:
            pair_mask = np.isin(pair_date_codes, batch_codes)
            batch_scores = model(features[batch_start:batch_end]).squeeze(-1)
            batch_loss, batch_mse, batch_pairwise = _hybrid_date_loss(
                batch_scores,
                targets[batch_start:batch_end],
                date_code_tensor[batch_start:batch_end],
                left_tensor[pair_mask] - batch_start,
                right_tensor[pair_mask] - batch_start,
                direction_tensor[pair_mask],
                pairwise_weight=settings["pairwise_weight"],
            )
            date_count = len(batch_codes)
            optimizer.zero_grad(set_to_none=True)
            (
                batch_loss
                * _date_batch_loss_scale(date_count, settings["dates_per_batch"])
            ).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["gradient_clip_norm"]
            )
            optimizer.step()
            optimizer_steps += 1
            epoch_date_count += date_count
            epoch_loss += float(batch_loss.detach()) * date_count
            epoch_mse += float(batch_mse) * date_count
            epoch_pairwise += float(batch_pairwise) * date_count
        if epoch_date_count != len(unique_dates):
            raise ValueError("final hybrid MLP did not cover every development date")
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / epoch_date_count,
                "train_rank_mse": epoch_mse / epoch_date_count,
                "train_top100_pairwise": epoch_pairwise / epoch_date_count,
            }
        )

    model.eval()
    with torch.inference_mode():
        audit_scores = model(features[: min(len(features), 256)]).squeeze(-1)
    return TrainingOutcome(
        score_validation=audit_scores.cpu().numpy().astype("float64", copy=False),
        model_objects={
            "preprocessor": preprocessor,
            "model": model,
            "model_type": "pytorch_mlp",
            "architecture": {
                "model_name": "mlp_top100_hybrid_rank",
                "input_dim": int(x_train.shape[1]),
                "hidden_layers": settings["hidden_layers"],
                "activation": "relu",
                "dropout": settings["dropout"],
                "output_bias": float(np.mean(y_train)),
            },
        },
        metadata={
            "completed_epochs": epochs,
            "optimizer_step_count": optimizer_steps,
            "training_history": history,
            "fixed_parameters": _fixed_parameters(
                "mlp_top100_hybrid_rank",
                objective="top100_hybrid_ranking",
                hidden_layers=settings["hidden_layers"],
                activation="relu",
                dropout=settings["dropout"],
                optimizer="adamw",
                learning_rate=settings["learning_rate"],
                weight_decay=settings["weight_decay"],
                rank_loss="equal_date_mse",
                pairwise_loss="equal_date_top100_logistic",
                pairwise_weight=settings["pairwise_weight"],
                boundary_pairs_per_positive=settings["boundary_pairs_per_positive"],
                broad_pairs_per_positive=settings["broad_pairs_per_positive"],
                dates_per_batch=settings["dates_per_batch"],
                max_epochs=settings["max_epochs"],
                patience=settings["patience"],
                min_delta=settings["min_delta"],
                gradient_clip_norm=settings["gradient_clip_norm"],
                validation_selection="mean_daily_ndcg_at_100",
                top_k=settings["top_k"],
                deterministic_algorithms=True,
                date_weighting="equal_total_per_date",
                reload_tolerance=1e-6,
            ),
        },
    )


def _fit_candidate(
    development: pd.DataFrame,
    schema: dict[str, Any],
    candidate: dict[str, Any],
    model_params: dict[str, Any],
) -> TrainingOutcome:
    model_name = candidate["model_name"]
    trainer = MODEL_REGISTRY.get(model_name)
    if trainer is None:
        raise ValueError(f"frozen model is not registered: {model_name}")
    audit_sample = development.iloc[: min(len(development), 256)].copy()
    if model_name == "mlp_top100_hybrid_rank":
        epochs = candidate.get("final_training", {}).get("epochs")
        outcome = train_final_mlp_top100_hybrid_rank(
            development, schema, model_params, epochs=epochs
        )
        implemented_parameters = outcome.metadata.get("fixed_parameters")
    else:
        outcome = trainer(development, audit_sample, schema, model_params)
        implemented_parameters = outcome.metadata.get("fixed_parameters")
    if implemented_parameters != candidate["fixed_parameters"]:
        raise ValueError(
            f"current {model_name} implementation does not match frozen parameters"
        )
    preprocessor = outcome.model_objects.get("preprocessor")
    if (
        not isinstance(preprocessor, RankPreprocessor)
        or preprocessor.feature_names != candidate["transformed_feature_columns"]
    ):
        raise ValueError(f"current {model_name} feature contract does not match the freeze")
    return outcome


def refit_frozen_candidate(
    dataset: pd.DataFrame,
    schema: dict[str, Any],
    candidate: dict[str, Any],
    config_path: Path,
    freeze_path: Path,
    final_runs_directory: Path,
    *,
    model_params: dict[str, Any],
    development_start: date,
    development_end: date,
) -> Path:
    """Refit one frozen candidate and atomically publish an immutable final run."""
    _check_frozen_data_contract(schema, candidate)
    model_name = candidate["model_name"]
    output_directory = Path(final_runs_directory) / model_name
    if output_directory.exists():
        raise FileExistsError(f"final run already exists and is immutable: {output_directory}")
    config_path = Path(config_path)
    freeze_path = Path(freeze_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not freeze_path.exists():
        raise FileNotFoundError(freeze_path)

    development = select_final_development_rows(
        dataset, development_start, development_end
    )
    started = perf_counter()
    outcome = _fit_candidate(development, schema, candidate, model_params)
    training_seconds = float(perf_counter() - started)

    final_runs_directory = Path(final_runs_directory)
    final_runs_directory.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{model_name}.", dir=final_runs_directory)
    )
    try:
        shutil.copyfile(config_path, temporary / "config_snapshot.toml")
        shutil.copyfile(freeze_path, temporary / "frozen_models_snapshot.json")
        preprocessor = outcome.model_objects["preprocessor"]
        feature_schema = {
            **schema,
            "transformed_feature_columns": preprocessor.feature_names,
        }
        (temporary / "feature_schema.json").write_text(
            json.dumps(feature_schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_model_artifacts(temporary, outcome)

        reloaded_preprocessor = joblib.load(temporary / "preprocessor.joblib")
        reloaded_model = _load_persisted_model(temporary)
        audit_sample = development.iloc[: min(len(development), 256)].copy()
        expected = np.asarray(outcome.score_validation, dtype="float64")
        actual = _predict_model(
            reloaded_model,
            reloaded_preprocessor.transform(
                audit_sample,
                scale_continuous=not isinstance(reloaded_model, (xgb.Booster, lgb.Booster)),
            ),
        )
        tolerance = float(candidate["fixed_parameters"]["reload_tolerance"])
        if not np.allclose(actual, expected, rtol=tolerance, atol=tolerance):
            raise ValueError("reloaded final model predictions do not match staged predictions")

        missing_rate, missing_rates = _missing_feature_rates(development, schema)
        maximum_exit = pd.to_datetime(development[EXIT_DATE_COLUMN]).max()
        trainer_metadata = {
            key: value
            for key, value in outcome.metadata.items()
            if key
            in {
                "completed_epochs",
                "optimizer_step_count",
                "training_history",
            }
        }
        manifest = {
            "status": "completed",
            "purpose": "final_refit_2019_2023",
            "model_name": model_name,
            "role": candidate["role"],
            "source_validation_run_id": candidate["source_run_id"],
            "source_validation_manifest_sha256": candidate["source_manifest_sha256"],
            "feature_schema_sha256_at_freeze": candidate["feature_schema_sha256"],
            "frozen_spec_sha256": file_sha256(freeze_path),
            "rank_dataset_sha256": schema.get("parquet_sha256"),
            "source_dataset_sha256": schema.get("source_dataset_sha256"),
            "source_schema_sha256": schema.get("source_schema_sha256"),
            "fixed_parameters": candidate["fixed_parameters"],
            "final_training": candidate["final_training"],
            "uses_validation": False,
            "uses_early_stopping": False,
            "training_rows": int(len(development)),
            "training_dates": int(development["date"].nunique()),
            "signal_start": pd.Timestamp(development["date"].min()).date().isoformat(),
            "signal_end": pd.Timestamp(development["date"].max()).date().isoformat(),
            "maximum_target_exit": maximum_exit.date().isoformat(),
            "feature_count": len(preprocessor.feature_names),
            "unknown_industry_rate": _unknown_industry_rate(development, preprocessor),
            "missing_feature_rate": missing_rate,
            "missing_feature_rates": missing_rates,
            "training_seconds": training_seconds,
            "reload_audit_rows": int(len(audit_sample)),
            "reload_tolerance": tolerance,
            "trainer_metadata": trainer_metadata,
            "artifact_sha256": _artifact_hashes(temporary),
            "artifact_payload_bytes": _artifact_payload_bytes(temporary),
        }
        manifest["artifact_bytes_before_manifest"] = _artifact_bytes(temporary)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_directory
