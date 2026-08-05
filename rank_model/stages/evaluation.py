"""Descriptive, per-date evaluation for 10-day rank-model predictions."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


KEY_COLUMNS = ["date", "stock_code"]
_REQUIRED_COLUMNS = {
    *KEY_COLUMNS,
    "target_10d",
    "rank_target_10d",
    "score_raw",
}
_DECILE_COUNT = 10
_DECISION_FIELDS = {"accept", "reject", "champion"}


@dataclass(frozen=True)
class EvaluationBundle:
    """Tables and summary derived from one prediction artifact."""

    predictions: pd.DataFrame
    daily_metrics: pd.DataFrame
    monthly_metrics: pd.DataFrame
    yearly_metrics: pd.DataFrame
    decile_returns: pd.DataFrame
    top100_detail: pd.DataFrame
    summary: dict[str, Any]


def _finite(values: pd.Series) -> np.ndarray:
    return np.isfinite(pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64"))


def _normalize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_REQUIRED_COLUMNS.difference(predictions.columns))
    if missing:
        raise ValueError(f"predictions are missing required columns: {missing}")
    result = predictions.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["stock_code"] = result["stock_code"].astype("string").str.strip()
    if result.empty:
        raise ValueError("predictions are empty")
    if result[KEY_COLUMNS].isna().any().any() or result["stock_code"].eq("").any():
        raise ValueError("prediction keys must be present")
    if result.duplicated(KEY_COLUMNS).any():
        raise ValueError("predictions contain duplicate date,stock_code keys")
    for column in ("target_10d", "rank_target_10d", "score_raw"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _predicted_percentiles(scores: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=scores.index, dtype="float64")
    finite = _finite(scores)
    count = int(finite.sum())
    if count == 1:
        result.loc[finite] = 0.5
    elif count > 1:
        ranks = scores.loc[finite].rank(method="average", ascending=True)
        result.loc[finite] = (ranks - 1.0) / (count - 1.0)
    return result


def _ordered_top(frame: pd.DataFrame, score_column: str, count: int) -> pd.DataFrame:
    return frame.sort_values(
        [score_column, "stock_code"],
        ascending=[False, True],
        kind="mergesort",
    ).head(count)


def _ordered_bottom(frame: pd.DataFrame, score_column: str, count: int) -> pd.DataFrame:
    return frame.sort_values(
        [score_column, "stock_code"],
        ascending=[True, True],
        kind="mergesort",
    ).head(count)


def _ndcg_at_k(frame: pd.DataFrame, top_k: int) -> float:
    """NDCG using the continuous rank percentile as relevance."""
    if frame.empty:
        return np.nan
    count = min(int(top_k), len(frame))
    predicted = _ordered_top(frame, "score_raw", count)
    ideal = frame.sort_values(
        ["rank_target_10d", "stock_code"],
        ascending=[False, True],
        kind="mergesort",
    ).head(count)
    discounts = np.log2(np.arange(2, count + 2, dtype="float64"))
    predicted_gain = np.exp2(predicted["rank_target_10d"].to_numpy()) - 1.0
    ideal_gain = np.exp2(ideal["rank_target_10d"].to_numpy()) - 1.0
    ideal_dcg = float(np.sum(ideal_gain / discounts))
    if ideal_dcg <= 0.0:
        return 0.0
    return float(np.sum(predicted_gain / discounts) / ideal_dcg)


def hac_mean_test(values: np.ndarray, *, lag: int = 10) -> dict[str, Any]:
    """Return a Bartlett Newey-West test for one daily metric series."""
    observations = np.asarray(values, dtype="float64")
    if observations.ndim != 1 or len(observations) < 2:
        raise ValueError("HAC values must be a one-dimensional sample of size >= 2")
    if not np.isfinite(observations).all() or lag < 0:
        raise ValueError("HAC values must be finite and lag nonnegative")
    count = len(observations)
    used_lag = min(int(lag), count - 1)
    centered = observations - float(observations.mean())
    long_run_variance = float(np.dot(centered, centered) / count)
    for offset in range(1, used_lag + 1):
        weight = 1.0 - offset / (used_lag + 1.0)
        autocovariance = float(np.dot(centered[offset:], centered[:-offset]) / count)
        long_run_variance += 2.0 * weight * autocovariance
    standard_error = float(np.sqrt(max(long_run_variance / count, 0.0)))
    mean = float(observations.mean())
    if standard_error > 0.0:
        t_value: float | None = float(mean / standard_error)
        lower = float(mean - 1.959963984540054 * standard_error)
        upper = float(mean + 1.959963984540054 * standard_error)
    else:
        t_value = None
        lower = mean
        upper = mean
    return {
        "observations": int(count),
        "lag": used_lag,
        "mean": mean,
        "standard_error": standard_error,
        "t_value": t_value,
        "ci95_lower": lower,
        "ci95_upper": upper,
    }


def _optional_hac(values: pd.Series, lag: int) -> dict[str, Any] | None:
    finite_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy("float64")
    if len(finite_values) < 2:
        return None
    return hac_mean_test(finite_values, lag=lag)


def _daily_metrics(group: pd.DataFrame, top_k: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    date = group["date"].iloc[0]
    score_finite = _finite(group["score_raw"])
    score_data = group.loc[score_finite].copy()
    score_data["_pred_rank_pct"] = _predicted_percentiles(score_data["score_raw"])
    score_count = len(score_data)
    unique_score_count = int(score_data["score_raw"].nunique())
    rank_data = score_data.loc[_finite(score_data["rank_target_10d"])].copy()
    valid_target_count = len(rank_data)
    metric: dict[str, Any] = {
        "date": date,
        "valid_target_count": valid_target_count,
        "valid_score_count": score_count,
        "unique_score_count": unique_score_count,
        "tie_ratio": float(1.0 - unique_score_count / score_count)
        if score_count
        else np.nan,
        "constant_score_date": bool(score_count >= 2 and unique_score_count == 1),
    }

    if valid_target_count >= 2 and unique_score_count > 1:
        rank_ic = spearmanr(rank_data["score_raw"], rank_data["rank_target_10d"]).statistic
        tau = kendalltau(rank_data["score_raw"], rank_data["rank_target_10d"]).statistic
        metric["rank_ic"] = float(rank_ic) if np.isfinite(rank_ic) else np.nan
        metric["kendall_tau"] = float(tau) if np.isfinite(tau) else np.nan
    else:
        metric["rank_ic"] = np.nan
        metric["kendall_tau"] = np.nan
    metric["pairwise_accuracy"] = (
        float((metric["kendall_tau"] + 1.0) / 2.0)
        if np.isfinite(metric["kendall_tau"])
        else np.nan
    )
    if valid_target_count:
        rank_error = rank_data["_pred_rank_pct"] - rank_data["rank_target_10d"]
        metric["rank_mae"] = float(rank_error.abs().mean())
        metric["rank_rmse"] = float(np.sqrt(np.square(rank_error).mean()))
        selection_size = min(top_k, valid_target_count)
        predicted_top = _ordered_top(rank_data, "score_raw", selection_size)
        realized_top = _ordered_top(rank_data, "rank_target_10d", selection_size)
        predicted_keys = set(predicted_top["stock_code"])
        realized_keys = set(realized_top["stock_code"])
        overlap = len(predicted_keys.intersection(realized_keys))
        union = len(predicted_keys.union(realized_keys))
        metric["ndcg_100"] = _ndcg_at_k(rank_data, selection_size)
        metric["precision_100"] = float(overlap / selection_size)
        metric["recall_100"] = float(overlap / selection_size)
        metric["jaccard_100"] = float(overlap / union) if union else np.nan
    else:
        for name in ("rank_mae", "rank_rmse", "ndcg_100", "precision_100", "recall_100", "jaccard_100"):
            metric[name] = np.nan

    raw_data = score_data.loc[_finite(score_data["target_10d"])].copy()
    raw_selection_size = min(top_k, len(raw_data))
    if raw_selection_size:
        raw_top = _ordered_top(raw_data, "score_raw", raw_selection_size)
        raw_bottom = _ordered_bottom(raw_data, "score_raw", raw_selection_size)
        metric["top100_mean_return"] = float(raw_top["target_10d"].mean())
        metric["top100_median_return"] = float(raw_top["target_10d"].median())
        metric["bottom100_mean_return"] = float(raw_bottom["target_10d"].mean())
        metric["top100_valid_return_count"] = int(len(raw_top))
    else:
        raw_top = raw_data.copy()
        for name in ("top100_mean_return", "top100_median_return", "bottom100_mean_return"):
            metric[name] = np.nan
        metric["top100_valid_return_count"] = 0
    universe_returns = group.loc[_finite(group["target_10d"]), "target_10d"]
    metric["universe_mean_return"] = (
        float(universe_returns.mean()) if len(universe_returns) else np.nan
    )
    metric["top100_excess_return"] = (
        metric["top100_mean_return"] - metric["universe_mean_return"]
        if np.isfinite(metric["top100_mean_return"])
        and np.isfinite(metric["universe_mean_return"])
        else np.nan
    )
    metric["top_bottom_spread"] = (
        metric["top100_mean_return"] - metric["bottom100_mean_return"]
        if np.isfinite(metric["top100_mean_return"])
        and np.isfinite(metric["bottom100_mean_return"])
        else np.nan
    )

    decile_rows: list[dict[str, Any]] = []
    if score_count:
        decile_data = _ordered_top(score_data, "score_raw", score_count).copy()
        decile_data["decile"] = (
            np.floor(np.arange(score_count, dtype="float64") * _DECILE_COUNT / score_count)
            .astype("int64")
            + 1
        )
    else:
        decile_data = score_data.assign(decile=pd.Series(dtype="int64"))
    for decile in range(1, _DECILE_COUNT + 1):
        members = decile_data.loc[decile_data["decile"].eq(decile)]
        returns = members.loc[_finite(members["target_10d"]), "target_10d"]
        count = int(len(returns))
        decile_rows.append(
            {
                "date": date,
                "decile": decile,
                "score_count": int(len(members)),
                "valid_return_count": count,
                "mean_return": float(returns.mean()) if count else np.nan,
                "median_return": float(returns.median()) if count else np.nan,
            }
        )
        metric[f"decile_{decile}_valid_return_count"] = count
        metric[f"decile_{decile}_mean_return"] = decile_rows[-1]["mean_return"]

    detail_count = min(top_k, score_count)
    detail = _ordered_top(score_data, "score_raw", detail_count).copy()
    detail.insert(0, "top_k", detail_count)
    return metric, pd.DataFrame(decile_rows), detail


def _period_metrics(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    work = daily.copy()
    if period == "month":
        work["month"] = work["date"].dt.to_period("M").astype(str)
        key = "month"
    else:
        work["year"] = work["date"].dt.year
        key = "year"
    numeric = work.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    numeric = [column for column in numeric if column not in {"date", key}]
    result = work.groupby(key, sort=True)[numeric].mean().reset_index()
    result.insert(1, "valid_daily_observations", work.groupby(key, sort=True).size().to_numpy())
    return result


def _mean_or_none(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.mean()) if len(finite) else None


def _summary(daily: pd.DataFrame, hac_lag: int, top_k: int, rows: int) -> dict[str, Any]:
    rank_ic = pd.to_numeric(daily["rank_ic"], errors="coerce").dropna()
    summary: dict[str, Any] = {
        "rows": int(rows),
        "dates": int(len(daily)),
        "top_k": int(top_k),
        "hac_lag": int(hac_lag),
        "rank_ic_valid_dates": int(len(rank_ic)),
        "rank_ic_invalid_dates": int(len(daily) - len(rank_ic)),
        "constant_score_dates": int(daily["constant_score_date"].sum()),
        "mean_rank_ic": _mean_or_none(daily["rank_ic"]),
        "median_rank_ic": float(rank_ic.median()) if len(rank_ic) else None,
        "std_rank_ic": float(rank_ic.std(ddof=1)) if len(rank_ic) > 1 else None,
        "positive_rank_ic_rate": float((rank_ic > 0.0).mean()) if len(rank_ic) else None,
        "rank_ic_q05": float(rank_ic.quantile(0.05)) if len(rank_ic) else None,
        "rank_ic_q25": float(rank_ic.quantile(0.25)) if len(rank_ic) else None,
        "rank_ic_q75": float(rank_ic.quantile(0.75)) if len(rank_ic) else None,
        "rank_ic_q95": float(rank_ic.quantile(0.95)) if len(rank_ic) else None,
        "rank_icir": (
            float(rank_ic.mean() / rank_ic.std(ddof=1))
            if len(rank_ic) > 1 and rank_ic.std(ddof=1) > 0.0
            else None
        ),
        "rank_icir_annualized": None,
        "rank_icir_annualization_note": "Not annualized because daily 10-day targets overlap.",
    }
    for column in (
        "kendall_tau",
        "pairwise_accuracy",
        "rank_mae",
        "rank_rmse",
        "ndcg_100",
        "precision_100",
        "recall_100",
        "jaccard_100",
        "top100_mean_return",
        "top100_median_return",
        "universe_mean_return",
        "top100_excess_return",
        "bottom100_mean_return",
        "top_bottom_spread",
        "top100_valid_return_count",
        "unique_score_count",
        "tie_ratio",
        "valid_target_count",
    ):
        summary[f"mean_{column}"] = _mean_or_none(daily[column])
    summary["rank_ic_hac"] = _optional_hac(daily["rank_ic"], hac_lag)
    summary["top100_excess_return_hac"] = _optional_hac(
        daily["top100_excess_return"], hac_lag
    )
    summary["top_bottom_spread_hac"] = _optional_hac(daily["top_bottom_spread"], hac_lag)
    return summary


def evaluate_predictions(
    predictions: pd.DataFrame,
    hac_lag: int = 10,
    top_k: int = 100,
) -> EvaluationBundle:
    """Evaluate a prediction frame without dropping any prediction keys."""
    if hac_lag < 0:
        raise ValueError("hac_lag must be nonnegative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    result = _normalize_predictions(predictions)
    metric_rows: list[dict[str, Any]] = []
    deciles: list[pd.DataFrame] = []
    details: list[pd.DataFrame] = []
    ordered = result.sort_values(KEY_COLUMNS, kind="mergesort")
    for _, group in ordered.groupby("date", sort=True):
        metric, decile, detail = _daily_metrics(group, top_k)
        metric_rows.append(metric)
        deciles.append(decile)
        details.append(detail)
    daily = pd.DataFrame(metric_rows).sort_values("date").reset_index(drop=True)
    decile_returns = pd.concat(deciles, ignore_index=True)
    top_detail = pd.concat(details, ignore_index=True)
    return EvaluationBundle(
        predictions=result,
        daily_metrics=daily,
        monthly_metrics=_period_metrics(daily, "month"),
        yearly_metrics=_period_metrics(daily, "year"),
        decile_returns=decile_returns,
        top100_detail=top_detail,
        summary=_summary(daily, hac_lag, top_k, len(result)),
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _completed_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "manifest.json"
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid manifest: {path}") from error
    if manifest.get("status") != "completed":
        raise ValueError(f"run is not completed: {run_dir}")
    return manifest


def write_evaluation(bundle: EvaluationBundle, run_dir: Path) -> None:
    """Write an evaluation exactly once into an immutable completed run."""
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = _completed_manifest(destination)
    if manifest is None:
        _write_json(
            destination / "manifest.json",
            {"run_id": destination.name, "status": "completed"},
        )
    predictions_path = destination / "predictions_10d.parquet"
    if predictions_path.exists():
        existing = _normalize_predictions(pd.read_parquet(predictions_path))
        if not existing[KEY_COLUMNS].equals(bundle.predictions[KEY_COLUMNS]):
            raise ValueError("evaluation prediction keys do not match the immutable run")
    else:
        _write_parquet(predictions_path, bundle.predictions)
    outputs = {
        "metrics_summary.json": lambda path: _write_json(path, bundle.summary),
        "daily_metrics.csv": lambda path: _write_csv(path, bundle.daily_metrics),
        "monthly_metrics.csv": lambda path: _write_csv(path, bundle.monthly_metrics),
        "yearly_metrics.csv": lambda path: _write_csv(path, bundle.yearly_metrics),
        "decile_returns.csv": lambda path: _write_csv(path, bundle.decile_returns),
        "top100_detail.parquet": lambda path: _write_parquet(path, bundle.top100_detail),
    }
    existing_outputs = [name for name in outputs if (destination / name).exists()]
    if existing_outputs:
        raise FileExistsError(f"evaluation outputs already exist: {existing_outputs}")
    for name, writer in outputs.items():
        writer(destination / name)


def compare_runs(run_dirs: list[Path], output_path: Path) -> pd.DataFrame:
    """Write descriptive run summaries side by side for completed runs only."""
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in run_dirs:
        run_dir = Path(raw_path)
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        manifest = _completed_manifest(run_dir)
        if manifest is None:
            raise ValueError(f"run is missing manifest: {run_dir}")
        run_id = manifest.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"run manifest has no valid run_id: {run_dir}")
        if run_id in seen:
            raise ValueError(f"duplicate run_id: {run_id}")
        seen.add(run_id)
        summary_path = run_dir / "metrics_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid metrics summary: {summary_path}") from error
        if not isinstance(summary, dict):
            raise ValueError(f"metrics summary must be an object: {summary_path}")
        rows.append(
            {
                "run_id": run_id,
                "model_name": manifest.get("model_name"),
                **{key: value for key, value in summary.items() if key not in _DECISION_FIELDS},
            }
        )
    comparison = pd.DataFrame(rows)
    _write_csv(Path(output_path), comparison)
    return comparison
