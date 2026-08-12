"""Shared date-group layout helpers for native ranking models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


def _validated_ranking_arrays(
    dates: pd.Series,
    scores: np.ndarray,
    targets: np.ndarray,
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    normalized_dates = pd.to_datetime(dates, errors="raise").dt.normalize()
    score_values = np.asarray(scores, dtype="float64")
    target_values = np.asarray(targets, dtype="float64")
    if not (
        len(normalized_dates) == len(score_values) == len(target_values)
    ):
        raise ValueError("ranking dates, scores, and targets must have equal length")
    if normalized_dates.isna().any():
        raise ValueError("ranking dates must be non-null")
    if not np.isfinite(score_values).all() or not np.isfinite(target_values).all():
        raise ValueError("ranking scores and targets must be finite")
    return normalized_dates, score_values, target_values


def sample_top100_pairs(
    dates: pd.Series,
    targets: np.ndarray,
    boundary_pairs_per_positive: int,
    broad_pairs_per_positive: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample deterministic lower-ranked opponents for each daily Top100 stock."""
    normalized_dates = pd.to_datetime(dates, errors="raise").dt.normalize()
    target_values = np.asarray(targets, dtype="float64")
    if len(normalized_dates) != len(target_values):
        raise ValueError("Top100 pair dates and targets must have equal length")
    if normalized_dates.isna().any():
        raise ValueError("Top100 pair dates must be non-null")
    if not np.isfinite(target_values).all():
        raise ValueError("Top100 pair targets must be finite")
    for name, count in (
        ("boundary_pairs_per_positive", boundary_pairs_per_positive),
        ("broad_pairs_per_positive", broad_pairs_per_positive),
    ):
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"{name} must be a positive integer")

    rng = np.random.default_rng(seed)
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    date_codes, unique_dates = pd.factorize(normalized_dates, sort=False)
    for date_code, date in enumerate(unique_dates):
        positions = np.flatnonzero(date_codes == date_code)
        date_targets = target_values[positions]
        positives = positions[date_targets >= 0.9]
        boundary = positions[(date_targets >= 0.7) & (date_targets < 0.9)]
        broad = positions[date_targets < 0.9]
        if positives.size == 0:
            raise ValueError(f"Top100 pair date {date.date()} has no positive stocks")
        if boundary.size == 0:
            raise ValueError(f"Top100 pair date {date.date()} has no boundary opponents")
        if broad.size == 0:
            raise ValueError(f"Top100 pair date {date.date()} has no broad opponents")
        for anchor in positives:
            opponents = np.concatenate(
                (
                    rng.choice(
                        boundary, size=boundary_pairs_per_positive, replace=True
                    ),
                    rng.choice(broad, size=broad_pairs_per_positive, replace=True),
                )
            ).astype("int64", copy=False)
            left_parts.append(
                np.full(len(opponents), anchor, dtype="int64")
            )
            right_parts.append(opponents)

    left = np.concatenate(left_parts) if left_parts else np.empty(0, dtype="int64")
    right = (
        np.concatenate(right_parts) if right_parts else np.empty(0, dtype="int64")
    )
    direction = np.sign(target_values[left] - target_values[right]).astype("int8")
    if np.any(left == right) or np.any(direction != 1):
        raise ValueError("Top100 pair sampling produced an invalid directed pair")
    return left, right, direction


def mean_daily_ndcg_at_k(
    dates: pd.Series,
    stock_codes: pd.Series,
    scores: np.ndarray,
    targets: np.ndarray,
    top_k: int,
) -> float:
    """Return equal-date mean NDCG using the evaluator's continuous gain."""
    normalized_dates, score_values, target_values = _validated_ranking_arrays(
        dates, scores, targets
    )
    stock_values = pd.Series(stock_codes, dtype="string")
    if len(stock_values) != len(score_values):
        raise ValueError("ranking stock codes must have equal length")
    if stock_values.isna().any():
        raise ValueError("ranking stock codes must be non-null")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    frame = pd.DataFrame(
        {
            "date": normalized_dates,
            "stock_code": stock_values,
            "score": score_values,
            "target": target_values,
        }
    )
    values: list[float] = []
    for _, group in frame.groupby("date", sort=False, observed=True):
        count = min(top_k, len(group))
        discounts = np.log2(np.arange(2, count + 2, dtype="float64"))
        predicted = group.sort_values(
            ["score", "stock_code"],
            ascending=[False, True],
            kind="mergesort",
        ).head(count)
        ideal = group.sort_values(
            ["target", "stock_code"],
            ascending=[False, True],
            kind="mergesort",
        ).head(count)
        predicted_gain = np.exp2(predicted["target"].to_numpy()) - 1.0
        ideal_gain = np.exp2(ideal["target"].to_numpy()) - 1.0
        ideal_dcg = float(np.sum(ideal_gain / discounts))
        if ideal_dcg <= 0.0:
            raise ValueError("daily NDCG ideal gain must be positive")
        values.append(float(np.sum(predicted_gain / discounts) / ideal_dcg))
    if not values:
        raise ValueError("daily NDCG requires at least one date")
    return float(np.mean(values))


def mean_daily_spearman(
    dates: pd.Series,
    scores: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Return equal-date mean Spearman rank correlation."""
    normalized_dates, score_values, target_values = _validated_ranking_arrays(
        dates, scores, targets
    )
    values: list[float] = []
    date_codes, unique_dates = pd.factorize(normalized_dates, sort=False)
    for date_code, date in enumerate(unique_dates):
        mask = date_codes == date_code
        correlation = float(spearmanr(score_values[mask], target_values[mask]).statistic)
        if not np.isfinite(correlation):
            raise ValueError(f"daily Spearman is non-finite for {date.date()}")
        values.append(correlation)
    if not values:
        raise ValueError("daily Spearman requires at least one date")
    return float(np.mean(values))


def sorted_group_layout(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a stable date/stock ordered frame and one group size per date."""
    required_columns = {"date", "stock_code"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"ranking data is missing group keys: {missing}")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    if result["date"].isna().any() or result["stock_code"].isna().any():
        raise ValueError("ranking group keys must be non-null")
    result = result.sort_values("stock_code", kind="mergesort")
    result = result.sort_values("date", kind="mergesort")
    result = result.reset_index(drop=True)
    group_sizes = (
        result.groupby("date", sort=False, observed=True).size().to_numpy(dtype="int64")
    )
    if int(group_sizes.sum()) != len(result):
        raise ValueError("ranking group sizes do not cover every row")
    return result, group_sizes


def lightgbm_relevance(rank_target: pd.Series) -> np.ndarray:
    """Map rank percentiles to LightGBM's fixed 100-level relevance labels."""
    values = pd.to_numeric(rank_target, errors="raise").to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("LightGBM ranking labels must be finite")
    return np.clip(np.floor(values * 100.0), 0, 99).astype("int32")


def sample_date_pairs(
    dates: pd.Series,
    targets: np.ndarray,
    pairs_per_stock: int,
    adjacent_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample deterministic broad and target-neighbor pairs within each date."""
    normalized_dates = pd.to_datetime(dates, errors="raise").dt.normalize()
    target_values = np.asarray(targets, dtype="float64")
    if len(normalized_dates) != len(target_values):
        raise ValueError("pair sampling dates and targets must have equal length")
    if normalized_dates.isna().any():
        raise ValueError("pair sampling dates must be non-null")
    if not np.isfinite(target_values).all():
        raise ValueError("pair sampling targets must be finite")
    if not isinstance(pairs_per_stock, int) or pairs_per_stock <= 0:
        raise ValueError("pairs_per_stock must be a positive integer")
    if not np.isfinite(adjacent_fraction) or not 0.0 <= adjacent_fraction <= 1.0:
        raise ValueError("adjacent_fraction must be finite and between zero and one")

    adjacent_pairs = pairs_per_stock * adjacent_fraction
    if not float(adjacent_pairs).is_integer():
        raise ValueError("pairs_per_stock times adjacent_fraction must be an integer")
    adjacent_count = int(adjacent_pairs)
    broad_count = pairs_per_stock - adjacent_count
    rng = np.random.default_rng(seed)
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []

    date_codes, _ = pd.factorize(normalized_dates, sort=False)
    for date_code in range(int(date_codes.max()) + 1):
        date_positions = np.flatnonzero(date_codes == date_code)
        date_targets = target_values[date_positions]
        for local_anchor, anchor in enumerate(date_positions):
            valid_local = np.flatnonzero(date_targets != date_targets[local_anchor])
            if valid_local.size == 0:
                raise ValueError(
                    "pair sampling date has no valid opponent for every anchor"
                )
            valid_positions = date_positions[valid_local]
            broad = rng.choice(valid_positions, size=broad_count, replace=True)
            if adjacent_count:
                distances = np.abs(
                    date_targets[valid_local] - date_targets[local_anchor]
                )
                nearby_count = max(
                    1, int(np.ceil(valid_positions.size * adjacent_fraction))
                )
                nearby_order = np.argsort(distances, kind="mergesort")
                nearby_positions = valid_positions[nearby_order[:nearby_count]]
                adjacent = rng.choice(
                    nearby_positions, size=adjacent_count, replace=True
                )
            else:
                adjacent = np.empty(0, dtype="int64")
            opponents = np.concatenate((broad, adjacent)).astype("int64", copy=False)
            left_parts.append(np.full(pairs_per_stock, anchor, dtype="int64"))
            right_parts.append(opponents)

    left = np.concatenate(left_parts) if left_parts else np.empty(0, dtype="int64")
    right = (
        np.concatenate(right_parts) if right_parts else np.empty(0, dtype="int64")
    )
    direction = np.sign(target_values[left] - target_values[right]).astype("int8")
    if np.any(left == right) or np.any(direction == 0):
        raise ValueError("pair sampling produced a self-pair or target tie")
    return left, right, direction


def pairwise_logistic_loss(
    scores: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Return the mean logistic loss for target-directed score margins."""
    signed_margin = direction * (scores[left] - scores[right])
    return torch.nn.functional.softplus(-signed_margin).mean()
