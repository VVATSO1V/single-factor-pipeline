"""Immutable configuration and input seals for static strategy execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from rank_model.stages.dataset import file_sha256


LOCKED_MODEL_NAMES = (
    "ridge_rank_regression",
    "xgboost_rank_regression",
    "lightgbm_rank_regression",
    "lightgbm_lambdarank",
    "mlp_top100_hybrid_rank",
)


STRATEGY_START = pd.Timestamp("2024-01-01")
STRATEGY_END = pd.Timestamp("2025-12-31")
STRATEGY_TOP_K = 100
STRATEGY_CROSS_SECTION_SIZE = 1000
STRATEGY_MIN_LISTING_DAYS = 120
STRATEGY_INITIAL_NAV = 1.0
STRATEGY_COMMISSION_BPS = 1.0
STRATEGY_SLIPPAGE_BPS = 5.0
STRATEGY_SELL_STAMP_DUTY_BPS = 5.0
STRATEGY_LIMIT_TOLERANCE = 1e-8
STRATEGY_ANNUALIZATION_DAYS = 252


@dataclass(frozen=True)
class StrategySettings:
    start: pd.Timestamp
    end: pd.Timestamp
    top_k: int
    expected_cross_section_size: int
    min_listing_days: int
    initial_nav: float
    commission_rate: float
    slippage_rate: float
    sell_stamp_duty_rate: float
    limit_tolerance: float
    annualization_days: int

    @property
    def buy_cost_rate(self) -> float:
        return round(self.commission_rate + self.slippage_rate, 12)

    @property
    def sell_cost_rate(self) -> float:
        return round(
            self.commission_rate + self.slippage_rate + self.sell_stamp_duty_rate,
            12,
        )


def _exact_number(settings: Mapping[str, Any], name: str, expected: float) -> float:
    try:
        value = float(settings[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"strategy {name} must be {expected}") from error
    if value != expected:
        raise ValueError(f"strategy {name} must be {expected}")
    return value


def _exact_integer(settings: Mapping[str, Any], name: str, expected: int) -> int:
    value = settings.get(name)
    if isinstance(value, bool) or value != expected:
        raise ValueError(f"strategy {name} must be {expected}")
    return expected


def load_strategy_settings(config: Mapping[str, Any]) -> StrategySettings:
    """Load the single permitted static-strategy configuration."""
    settings = config.get("strategy")
    if not isinstance(settings, Mapping):
        raise ValueError("config is missing [strategy]")
    if settings.get("start") != STRATEGY_START.strftime("%Y-%m-%d"):
        raise ValueError("strategy start must be 2024-01-01")
    if settings.get("end") != STRATEGY_END.strftime("%Y-%m-%d"):
        raise ValueError("strategy end must be 2025-12-31")

    return StrategySettings(
        start=STRATEGY_START,
        end=STRATEGY_END,
        top_k=_exact_integer(settings, "top_k", STRATEGY_TOP_K),
        expected_cross_section_size=_exact_integer(
            settings, "expected_cross_section_size", STRATEGY_CROSS_SECTION_SIZE
        ),
        min_listing_days=_exact_integer(
            settings, "min_listing_days", STRATEGY_MIN_LISTING_DAYS
        ),
        initial_nav=_exact_number(settings, "initial_nav", STRATEGY_INITIAL_NAV),
        commission_rate=_exact_number(
            settings, "commission_bps", STRATEGY_COMMISSION_BPS
        ) / 10_000,
        slippage_rate=_exact_number(settings, "slippage_bps", STRATEGY_SLIPPAGE_BPS)
        / 10_000,
        sell_stamp_duty_rate=_exact_number(
            settings, "sell_stamp_duty_bps", STRATEGY_SELL_STAMP_DUTY_BPS
        )
        / 10_000,
        limit_tolerance=_exact_number(
            settings, "limit_tolerance", STRATEGY_LIMIT_TOLERANCE
        ),
        annualization_days=_exact_integer(
            settings, "annualization_days", STRATEGY_ANNUALIZATION_DAYS
        ),
    )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"locked-test conclusion has invalid {label} hash")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"locked-test conclusion has invalid {label} hash") from error
    return value


def validate_locked_test_conclusion(
    *,
    conclusion: Mapping[str, Any],
    frozen_spec_path: Path,
    comparison_path: Path,
    prediction_paths: Mapping[str, Path],
) -> None:
    """Verify that the sealed conclusion still binds every strategy input."""
    if conclusion.get("schema_version") != 1:
        raise ValueError("locked-test conclusion schema version must be 1")
    if conclusion.get("period") != {
        "start": STRATEGY_START.strftime("%Y-%m-%d"),
        "end": STRATEGY_END.strftime("%Y-%m-%d"),
    }:
        raise ValueError("locked-test conclusion period must be 2024-01-01 through 2025-12-31")
    if conclusion.get("selection_policy") != "no_test_based_selection":
        raise ValueError("locked-test conclusion selection policy is invalid")
    if conclusion.get("retuning_allowed") is not False:
        raise ValueError("locked-test conclusion must prohibit retuning")
    if tuple(conclusion.get("strategy_models", ())) != LOCKED_MODEL_NAMES:
        raise ValueError("locked-test conclusion must contain exactly the five frozen models")
    if any(field in conclusion for field in ("winner", "recommendation")):
        raise ValueError("locked-test conclusion must not select a winner or recommendation")

    metrics = conclusion.get("locked_test_metrics")
    if not isinstance(metrics, list) or len(metrics) != len(LOCKED_MODEL_NAMES):
        raise ValueError("locked-test conclusion must contain five test metric rows")
    if tuple(row.get("model_name") for row in metrics if isinstance(row, Mapping)) != LOCKED_MODEL_NAMES:
        raise ValueError("locked-test conclusion metric rows must cover frozen models")

    artifact_hashes = conclusion.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping):
        raise ValueError("locked-test conclusion is missing artifact hashes")
    expected_frozen_hash = _require_sha256(artifact_hashes.get("frozen_models"), "frozen spec")
    expected_comparison_hash = _require_sha256(
        artifact_hashes.get("locked_test_comparison"), "comparison"
    )
    prediction_hashes = artifact_hashes.get("predictions")
    if not isinstance(prediction_hashes, Mapping) or set(prediction_hashes) != set(
        LOCKED_MODEL_NAMES
    ):
        raise ValueError("locked-test conclusion must contain prediction hashes for five models")

    if file_sha256(Path(frozen_spec_path)) != expected_frozen_hash:
        raise ValueError("locked-test conclusion frozen spec hash does not match")
    if file_sha256(Path(comparison_path)) != expected_comparison_hash:
        raise ValueError("locked-test conclusion comparison hash does not match")
    if set(prediction_paths) != set(LOCKED_MODEL_NAMES):
        raise ValueError("locked-test conclusion requires prediction paths for five models")
    for model_name in LOCKED_MODEL_NAMES:
        expected_prediction_hash = _require_sha256(
            prediction_hashes[model_name], f"{model_name} prediction"
        )
        if file_sha256(Path(prediction_paths[model_name])) != expected_prediction_hash:
            raise ValueError(f"locked-test conclusion prediction hash does not match: {model_name}")
