from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rank_model.stages.ranking import (
    mean_daily_ndcg_at_k,
    mean_daily_spearman,
    sample_top100_pairs,
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


if __name__ == "__main__":
    unittest.main()
