from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from rank_model.stages.ranking import (
    mean_daily_ndcg_at_k,
    mean_daily_spearman,
    sample_top100_pairs,
)
from rank_model.stages.training import (
    _hybrid_date_loss,
    train_mlp_top100_hybrid_rank,
)


class Top100PairSamplingTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_stays_within_each_date(self) -> None:
        dates = pd.Series(
            [pd.Timestamp("2022-01-04")] * 11
            + [pd.Timestamp("2022-01-05")] * 11
        )
        targets = np.tile(np.linspace(0.0, 1.0, 11), 2)

        first = sample_top100_pairs(dates, targets, 4, 4, seed=42)
        second = sample_top100_pairs(dates, targets, 4, 4, seed=42)

        for first_part, second_part in zip(first, second, strict=True):
            np.testing.assert_array_equal(first_part, second_part)
        left, right, direction = first
        self.assertEqual(len(left), 32)
        self.assertTrue(np.all(dates.iloc[left].to_numpy() == dates.iloc[right].to_numpy()))
        self.assertTrue(np.all(targets[left] >= 0.9))
        self.assertTrue(np.all(targets[right] < 0.9))
        self.assertTrue(np.all(direction == 1))
        self.assertTrue(np.all(left != right))

        for anchor in np.unique(left):
            opponents = targets[right[left == anchor]]
            self.assertEqual(len(opponents), 8)
            self.assertTrue(np.all((opponents[:4] >= 0.7) & (opponents[:4] < 0.9)))
            self.assertTrue(np.all(opponents[4:] < 0.9))

    def test_sampling_rejects_date_without_boundary_opponents(self) -> None:
        dates = pd.Series([pd.Timestamp("2022-01-04")] * 3)
        targets = np.array([0.1, 0.6, 1.0])
        with self.assertRaisesRegex(ValueError, "boundary"):
            sample_top100_pairs(dates, targets, 1, 1, seed=42)


class ValidationMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.Series(
            [pd.Timestamp("2023-01-03")] * 5
            + [pd.Timestamp("2023-01-04")] * 5
        )
        self.stocks = pd.Series([f"S{i}" for i in range(5)] * 2)
        self.targets = np.tile(np.linspace(0.0, 1.0, 5), 2)

    def test_perfect_and_reverse_ordering(self) -> None:
        perfect_ndcg = mean_daily_ndcg_at_k(
            self.dates, self.stocks, self.targets, self.targets, top_k=2
        )
        reverse_ndcg = mean_daily_ndcg_at_k(
            self.dates, self.stocks, -self.targets, self.targets, top_k=2
        )
        self.assertAlmostEqual(perfect_ndcg, 1.0)
        self.assertLess(reverse_ndcg, perfect_ndcg)
        self.assertAlmostEqual(
            mean_daily_spearman(self.dates, self.targets, self.targets), 1.0
        )
        self.assertAlmostEqual(
            mean_daily_spearman(self.dates, -self.targets, self.targets), -1.0
        )

    def test_metrics_reject_bad_shapes_and_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal length"):
            mean_daily_spearman(self.dates.iloc[:-1], self.targets, self.targets)
        bad_scores = self.targets.copy()
        bad_scores[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            mean_daily_ndcg_at_k(
                self.dates, self.stocks, bad_scores, self.targets, top_k=2
            )


class HybridLossTests(unittest.TestCase):
    def test_loss_averages_dates_instead_of_rows_or_pairs(self) -> None:
        scores = torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0])
        targets = torch.zeros(5)
        date_codes = torch.tensor([0, 0, 1, 1, 1])
        left = torch.tensor([1, 2, 3])
        right = torch.tensor([0, 4, 4])
        direction = torch.ones(3)

        total, mse, pairwise = _hybrid_date_loss(
            scores,
            targets,
            date_codes,
            left,
            right,
            direction,
            pairwise_weight=0.1,
        )

        expected_mse = torch.tensor(1.0)
        expected_pairwise = (
            torch.nn.functional.softplus(torch.tensor(-2.0))
            + torch.nn.functional.softplus(torch.tensor(0.0))
        ) / 2.0
        torch.testing.assert_close(mse, expected_mse)
        torch.testing.assert_close(pairwise, expected_pairwise)
        torch.testing.assert_close(total, expected_mse + 0.1 * expected_pairwise)


class HybridTrainingTests(unittest.TestCase):
    @staticmethod
    def _frame(start: str, date_count: int) -> pd.DataFrame:
        rows = []
        for date_index, date in enumerate(pd.bdate_range(start, periods=date_count)):
            for stock_index in range(20):
                rank = stock_index / 19.0
                rows.append(
                    {
                        "date": date,
                        "stock_code": f"S{stock_index:02d}",
                        "feature": rank + date_index * 0.01,
                        "industry": "A" if stock_index % 2 == 0 else "B",
                        "rank_target_10d": rank,
                    }
                )
        return pd.DataFrame(rows)

    def test_training_updates_per_batch_and_is_deterministic(self) -> None:
        train = self._frame("2022-01-03", 4)
        validation = self._frame("2023-01-03", 2)
        schema = {
            "continuous_feature_columns": ["feature"],
            "industry_column": "industry",
        }
        params = {
            "hidden_layers": [8, 4],
            "dropout": 0.0,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "pairwise_weight": 0.1,
            "boundary_pairs_per_positive": 1,
            "broad_pairs_per_positive": 1,
            "dates_per_batch": 2,
            "max_epochs": 3,
            "patience": 1,
            "min_delta": 0.0,
            "gradient_clip_norm": 1.0,
            "top_k": 2,
            "seed": 42,
        }

        first = train_mlp_top100_hybrid_rank(train, validation, schema, params)
        second = train_mlp_top100_hybrid_rank(train, validation, schema, params)

        np.testing.assert_array_equal(first.score_validation, second.score_validation)
        metadata = first.metadata
        self.assertTrue(np.isfinite(first.score_validation).all())
        self.assertGreater(metadata["optimizer_step_count"], metadata["stopped_epoch"])
        self.assertGreaterEqual(metadata["best_epoch"], 1)
        self.assertLessEqual(metadata["best_epoch"], metadata["stopped_epoch"])
        self.assertEqual(len(metadata["training_history"]), metadata["stopped_epoch"])
        self.assertTrue(np.isfinite(metadata["best_validation_ndcg_100"]))
        self.assertTrue(np.isfinite(metadata["best_validation_rank_ic"]))


if __name__ == "__main__":
    unittest.main()
