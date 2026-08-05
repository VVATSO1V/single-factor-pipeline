from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rank_model.stages.preprocessing import (
    RankPreprocessor,
    equal_date_weights,
    predicted_percentiles,
)


class RankPreprocessorTests(unittest.TestCase):
    def test_train_only_industry_vocabulary_maps_unseen_values_to_unknown(self) -> None:
        train = pd.DataFrame(
            {
                "factor_a": [1.0, 2.0, 3.0],
                "log_market_cap": [10.0, 11.0, 12.0],
                "industry": ["Bank", "Tech", None],
            }
        )
        validation = pd.DataFrame(
            {
                "factor_a": [4.0, 5.0],
                "log_market_cap": [13.0, 14.0],
                "industry": ["Energy", "Bank"],
            }
        )

        preprocessor = RankPreprocessor.fit(
            train,
            continuous_columns=["factor_a", "log_market_cap"],
            industry_column="industry",
        )

        self.assertEqual(preprocessor.industry_categories, ["Bank", "Tech", "UNKNOWN"])
        matrix = preprocessor.transform(validation, scale_continuous=True)
        self.assertEqual(matrix.shape[1], 5)
        self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(matrix[0, -1], 1.0)

    def test_fit_rejects_forbidden_or_duplicate_feature_names(self) -> None:
        frame = pd.DataFrame(
            {
                "factor_a": [1.0, 2.0],
                "industry": ["Bank", "Tech"],
            }
        )

        with self.assertRaisesRegex(ValueError, "unique"):
            RankPreprocessor.fit(
                frame,
                continuous_columns=["factor_a", "factor_a"],
                industry_column="industry",
            )
        with self.assertRaisesRegex(ValueError, "forbidden"):
            RankPreprocessor.fit(
                frame.assign(target_10d=[0.0, 1.0]),
                continuous_columns=["target_10d"],
                industry_column="industry",
            )


class DateUtilityTests(unittest.TestCase):
    def test_equal_date_weights_sum_to_one_per_date(self) -> None:
        dates = pd.Series(
            ["2023-01-03", "2023-01-03", "2023-01-04", "2023-01-04", "2023-01-04", "2023-01-04"]
        )

        weights = equal_date_weights(dates)

        weighted = pd.DataFrame({"date": pd.to_datetime(dates), "weight": weights})
        np.testing.assert_allclose(weighted.groupby("date")["weight"].sum(), [1.0, 1.0])

    def test_predicted_percentiles_rank_each_date(self) -> None:
        actual = predicted_percentiles(
            np.array([3.0, 1.0, 2.0]),
            pd.Series(["2023-01-03", "2023-01-03", "2023-01-03"]),
        )

        np.testing.assert_allclose(actual, [1.0, 0.0, 0.5])


if __name__ == "__main__":
    unittest.main()
