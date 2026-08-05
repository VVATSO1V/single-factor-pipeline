from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from rank_model.stages.ranking import pairwise_logistic_loss, sample_date_pairs
from rank_model.stages.training import (
    MLP_DROPOUT,
    MLP_EPOCHS,
    MLP_GRADIENT_CLIP_NORM,
    MLP_LEARNING_RATE,
    MLP_WEIGHT_DECAY,
    MODEL_REGISTRY,
    train_mlp_pairwise_rank,
    train_registered_model,
)


class MlpPairwiseRankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = {
            "continuous_feature_columns": ["factor_a", "factor_b"],
            "industry_column": "industry",
        }
        dates = pd.to_datetime(
            ["2019-01-02", "2020-01-02", "2021-01-04", "2022-01-03", "2023-01-03"]
        )
        self.dataset = self._frame(dates)
        self.train = self.dataset.loc[self.dataset["date"].dt.year.lt(2023)].copy()
        self.validation = self.dataset.loc[
            self.dataset["date"].dt.year.eq(2023)
        ].copy()

    @staticmethod
    def _frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
        rows_per_date = 10
        positions = np.tile(np.arange(rows_per_date), len(dates))
        signals = positions.astype("float64") / (rows_per_date - 1)
        return pd.DataFrame(
            {
                "date": np.repeat(dates, rows_per_date),
                "stock_code": [f"S{index:04d}" for index in range(len(positions))],
                "factor_a": signals,
                "factor_b": np.cos(signals * np.pi),
                "industry": np.where(positions % 2 == 0, "A", "B"),
                "target_10d": signals,
                "rank_target_10d": signals,
                "split_10d": np.where(
                    np.repeat(dates.year, rows_per_date) == 2023,
                    "validation",
                    "train",
                ),
                "exit_date_10d": np.repeat(dates, rows_per_date),
            }
        )

    def test_sampler_is_same_date_deterministic_and_has_eight_pairs_per_anchor(
        self,
    ) -> None:
        dates = pd.Series(
            np.repeat(pd.to_datetime(["2020-01-02", "2020-01-03"]), 10)
        )
        targets = np.tile(np.linspace(0.0, 1.0, 10), 2)

        first = sample_date_pairs(dates, targets, 8, 0.5, 42)
        second = sample_date_pairs(dates, targets, 8, 0.5, 42)
        changed = sample_date_pairs(dates, targets, 8, 0.5, 43)
        left, right, direction = first

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])
        self.assertTrue(
            not np.array_equal(right, changed[1]),
            "changing the seed must change at least one sampled opponent",
        )
        np.testing.assert_array_equal(
            np.bincount(left, minlength=len(dates)), np.full(len(dates), 8)
        )
        self.assertTrue(np.all(dates.iloc[left].to_numpy() == dates.iloc[right].to_numpy()))
        self.assertTrue(np.all(left != right))
        self.assertTrue(np.all(targets[left] != targets[right]))
        self.assertTrue(np.all(np.isin(direction, [-1, 1])))
        np.testing.assert_array_equal(
            direction, np.sign(targets[left] - targets[right]).astype("int8")
        )
        for anchor in range(len(dates)):
            opponents = right[left == anchor]
            self.assertEqual(len(opponents[:4]), 4)
            self.assertEqual(len(opponents[4:]), 4)
            date_positions = np.flatnonzero(dates.to_numpy() == dates.iloc[anchor])
            valid = date_positions[targets[date_positions] != targets[anchor]]
            distances = np.abs(targets[valid] - targets[anchor])
            adjacent_pool = valid[
                np.argsort(distances, kind="mergesort")[: int(np.ceil(len(valid) * 0.5))]
            ]
            self.assertTrue(np.all(np.isin(opponents[:4], valid)))
            self.assertTrue(np.all(np.isin(opponents[4:], adjacent_pool)))

    def test_sampler_replaces_within_a_small_same_date_pool(self) -> None:
        dates = pd.Series(pd.to_datetime(["2020-01-02", "2020-01-02"]))
        targets = np.array([0.0, 1.0])

        left, right, _ = sample_date_pairs(dates, targets, 8, 0.5, 42)

        np.testing.assert_array_equal(np.bincount(left, minlength=2), [8, 8])
        np.testing.assert_array_equal(right[:8], np.full(8, 1))
        np.testing.assert_array_equal(right[8:], np.zeros(8, dtype="int64"))

    def test_sampler_rejects_a_date_without_a_valid_opponent(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid opponent"):
            sample_date_pairs(
                pd.Series(pd.to_datetime(["2020-01-02", "2020-01-02"])),
                np.array([0.5, 0.5]),
                8,
                0.5,
                42,
            )

    def test_pairwise_logistic_loss_rewards_correct_ordering(self) -> None:
        good = pairwise_logistic_loss(
            torch.tensor([2.0, 0.0]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([1.0]),
        )
        bad = pairwise_logistic_loss(
            torch.tensor([0.0, 2.0]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([1.0]),
        )

        self.assertLess(float(good), float(bad))

    def test_pairwise_mlp_is_deterministic_and_publishes_reloaded_scores(self) -> None:
        first = train_mlp_pairwise_rank(self.train, self.validation, self.schema, {})
        second = train_mlp_pairwise_rank(self.train, self.validation, self.schema, {})

        self.assertTrue(np.isfinite(first.score_validation).all())
        np.testing.assert_allclose(
            first.score_validation, second.score_validation, rtol=1e-6, atol=1e-6
        )
        self.assertEqual(first.metadata["epochs"], MLP_EPOCHS)
        self.assertEqual(first.metadata["pairs_per_stock"], 8)
        self.assertEqual(first.metadata["adjacent_fraction"], 0.5)
        self.assertEqual(first.metadata["dates_per_batch"], 8)
        self.assertEqual(first.metadata["pair_count"], len(self.train) * 8)
        self.assertEqual(first.metadata["broad_pairs_per_stock"], 4)
        self.assertEqual(first.metadata["adjacent_pairs_per_stock"], 4)
        self.assertEqual(first.metadata["dropout"], MLP_DROPOUT)
        self.assertEqual(first.metadata["optimizer"], "adamw")
        self.assertEqual(first.metadata["learning_rate"], MLP_LEARNING_RATE)
        self.assertEqual(first.metadata["weight_decay"], MLP_WEIGHT_DECAY)
        self.assertEqual(
            first.metadata["gradient_clip_norm"], MLP_GRADIENT_CLIP_NORM
        )
        self.assertEqual(first.metadata["optimizer_steps_per_epoch"], 1)
        self.assertTrue(first.metadata["global_date_loss_normalization"])
        self.assertIs(MODEL_REGISTRY["mlp_pairwise_rank"], train_mlp_pairwise_rank)

        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            dataset_path = root / "rank_dataset.parquet"
            schema_path = root / "rank_schema.json"
            config_path = root / "config.toml"
            columns = [
                "date",
                "stock_code",
                "factor_a",
                "factor_b",
                "industry",
                "target_10d",
                "rank_target_10d",
                "split_10d",
                "exit_date_10d",
            ]
            self.dataset.loc[:, columns].to_parquet(dataset_path, index=False)
            schema_path.write_text(
                json.dumps(
                    {
                        **self.schema,
                        "key_columns": ["date", "stock_code"],
                        "target_column": "target_10d",
                        "rank_target_column": "rank_target_10d",
                        "split_column": "split_10d",
                        "output_columns": columns,
                        "parquet_sha256": hashlib.sha256(
                            dataset_path.read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                """[paths]
rank_dataset = "rank_dataset.parquet"
rank_schema = "rank_schema.json"
runs_dir = "runs"

[models.mlp_pairwise_rank]
""",
                encoding="utf-8",
            )
            run_directory = train_registered_model(
                {
                    "paths": {
                        "rank_dataset": "rank_dataset.parquet",
                        "rank_schema": "rank_schema.json",
                        "runs_dir": "runs",
                    },
                    "models": {"mlp_pairwise_rank": {}},
                },
                config_path,
                "mlp_pairwise_rank",
                "mlp-pairwise-synthetic",
            )

            manifest = json.loads(
                (run_directory / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["metadata"]["pair_count"], len(self.train) * 8)
            self.assertEqual(manifest["metadata"]["pairs_per_stock"], 8)
            self.assertEqual(manifest["metadata"]["dropout"], MLP_DROPOUT)
            self.assertEqual(manifest["metadata"]["optimizer"], "adamw")
            self.assertEqual(
                manifest["metadata"]["learning_rate"], MLP_LEARNING_RATE
            )
            self.assertEqual(
                manifest["metadata"]["weight_decay"], MLP_WEIGHT_DECAY
            )
            self.assertEqual(
                manifest["metadata"]["gradient_clip_norm"], MLP_GRADIENT_CLIP_NORM
            )
            self.assertEqual(manifest["metadata"]["optimizer_steps_per_epoch"], 1)
            self.assertTrue(manifest["metadata"]["global_date_loss_normalization"])
            self.assertEqual(
                manifest["metadata"]["pair_counts_by_date"],
                {
                    "2019-01-02": 80,
                    "2020-01-02": 80,
                    "2021-01-04": 80,
                    "2022-01-03": 80,
                },
            )
            self.assertTrue((run_directory / "model_state_dict.pt").exists())
            architecture = json.loads(
                (run_directory / "model_architecture.json").read_text(encoding="utf-8")
            )
            self.assertEqual(architecture["dropout"], MLP_DROPOUT)
            predictions = pd.read_parquet(run_directory / "predictions_10d.parquet")
            np.testing.assert_allclose(
                predictions["score_raw"].to_numpy(dtype="float64"),
                first.score_validation,
                rtol=1e-6,
                atol=1e-6,
            )

    def test_nine_uneven_dates_use_one_global_adamw_update_per_epoch(self) -> None:
        train = self._uneven_nine_date_frame()
        validation = self._frame(pd.to_datetime(["2023-01-03"]))
        original_step = torch.optim.AdamW.step
        step_count = 0

        def counted_step(
            optimizer: torch.optim.AdamW, *args: object, **kwargs: object
        ) -> object:
            nonlocal step_count
            step_count += 1
            return original_step(optimizer, *args, **kwargs)

        with patch.object(torch.optim.AdamW, "step", counted_step):
            outcome = train_mlp_pairwise_rank(train, validation, self.schema, {})

        self.assertTrue(np.isfinite(outcome.score_validation).all())
        self.assertEqual(outcome.metadata["pair_count"], len(train) * 8)
        self.assertEqual(step_count, MLP_EPOCHS)

    @staticmethod
    def _uneven_nine_date_frame() -> pd.DataFrame:
        sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        dates = pd.bdate_range("2020-01-02", periods=len(sizes))
        date_values = np.repeat(dates, sizes)
        positions = np.concatenate([np.arange(size) for size in sizes])
        signals = positions / np.repeat(np.asarray(sizes) - 1, sizes)
        return pd.DataFrame(
            {
                "date": date_values,
                "stock_code": [
                    f"U{index:04d}" for index in range(len(date_values))
                ],
                "factor_a": signals,
                "factor_b": np.cos(signals * np.pi),
                "industry": np.where(positions % 2 == 0, "A", "B"),
                "target_10d": signals,
                "rank_target_10d": signals,
                "split_10d": "train",
                "exit_date_10d": date_values,
            }
        )


if __name__ == "__main__":
    unittest.main()
