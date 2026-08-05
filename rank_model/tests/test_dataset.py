from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from rank_model.stages.dataset import (
    build_rank_dataset,
    percentile_rank_from_returns,
)


class PercentileRankTests(unittest.TestCase):
    def test_strict_order_maps_to_zero_half_one(self) -> None:
        values = pd.Series([-0.1, 0.0, 0.2], dtype="float64")
        actual = percentile_rank_from_returns(values)
        np.testing.assert_allclose(actual.to_numpy(), [0.0, 0.5, 1.0])

    def test_ties_receive_average_positions(self) -> None:
        values = pd.Series([0.0, 0.0, 0.2], dtype="float64")
        actual = percentile_rank_from_returns(values)
        np.testing.assert_allclose(actual.to_numpy(), [0.25, 0.25, 1.0])

    def test_missing_target_remains_missing(self) -> None:
        values = pd.Series([0.0, np.nan, 0.2], dtype="float64")
        actual = percentile_rank_from_returns(values)
        np.testing.assert_allclose(actual.iloc[[0, 2]].to_numpy(), [0.0, 1.0])
        self.assertTrue(np.isnan(actual.iloc[1]))

    def test_fewer_than_two_labels_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two finite returns"):
            percentile_rank_from_returns(pd.Series([np.nan, 0.1]))


class RankDatasetTests(unittest.TestCase):
    def _write_source_bundle(self, directory: Path) -> tuple[Path, Path]:
        source_dataset = directory / "context_dataset_10d.parquet"
        source_schema = directory / "context_dataset_10d_schema.json"
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2023-01-03",
                        "2023-01-03",
                        "2023-01-03",
                        "2023-01-03",
                        "2023-01-04",
                        "2023-01-04",
                        "2023-01-04",
                        "2023-01-04",
                    ]
                ),
                "stock_code": [
                    "000001.XSHE",
                    "000002.XSHE",
                    "000003.XSHE",
                    "000004.XSHE",
                    "000001.XSHE",
                    "000002.XSHE",
                    "000003.XSHE",
                    "000004.XSHE",
                ],
                "factor_value": np.arange(8, dtype="float64"),
                "industry": ["A", "A", "B", "B", "A", "A", "B", "B"],
                "target_10d": [0.0, 0.0, 0.2, np.nan, -0.1, 0.1, 0.3, 0.5],
                "split_10d": ["validation"] * 8,
                "exit_date_10d": pd.to_datetime(["2023-01-17"] * 8),
            }
        )
        frame.to_parquet(source_dataset, index=False)
        source_schema.write_text(
            json.dumps(
                {
                    "parquet_sha256": self._sha256(source_dataset),
                    "key_columns": ["date", "stock_code"],
                    "feature_columns": ["factor_value", "industry"],
                    "expected_rows_per_date": 4,
                }
            ),
            encoding="utf-8",
        )
        return source_dataset, source_schema

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def test_build_rank_dataset_publishes_validated_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            directory = Path(temporary_name)
            source_dataset, source_schema = self._write_source_bundle(directory)
            output_dataset = directory / "rank_dataset_10d.parquet"
            output_schema = directory / "rank_dataset_10d_schema.json"
            coverage_path = directory / "rank_label_coverage.csv"

            result = build_rank_dataset(
                source_dataset,
                source_schema,
                output_dataset,
                output_schema,
                coverage_path,
                date(2023, 12, 31),
            )

            self.assertEqual(result["row_count"], 8)
            self.assertEqual(result["date_count"], 2)
            self.assertEqual(result["rank_target_column"], "rank_target_10d")
            self.assertTrue(output_dataset.exists())
            self.assertTrue(output_schema.exists())
            self.assertTrue(coverage_path.exists())

            dataset = pd.read_parquet(output_dataset)
            first_date = dataset.loc[dataset["date"] == pd.Timestamp("2023-01-03")]
            np.testing.assert_allclose(
                first_date["rank_target_10d"].iloc[:3].to_numpy(),
                [0.25, 0.25, 1.0],
            )
            self.assertTrue(pd.isna(first_date["rank_target_10d"].iloc[3]))

    def test_build_rank_dataset_rejects_source_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            directory = Path(temporary_name)
            source_dataset, source_schema = self._write_source_bundle(directory)
            schema = json.loads(source_schema.read_text(encoding="utf-8"))
            schema["parquet_sha256"] = "0" * 64
            source_schema.write_text(json.dumps(schema), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source hash"):
                build_rank_dataset(
                    source_dataset,
                    source_schema,
                    directory / "rank_dataset_10d.parquet",
                    directory / "rank_dataset_10d_schema.json",
                    directory / "rank_label_coverage.csv",
                    date(2023, 12, 31),
                )

    def test_build_rank_dataset_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            directory = Path(temporary_name)
            source_dataset, source_schema = self._write_source_bundle(directory)
            frame = pd.read_parquet(source_dataset)
            frame.loc[1, "stock_code"] = frame.loc[0, "stock_code"]
            frame.to_parquet(source_dataset, index=False)
            schema = json.loads(source_schema.read_text(encoding="utf-8"))
            schema["parquet_sha256"] = self._sha256(source_dataset)
            source_schema.write_text(json.dumps(schema), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate keys"):
                build_rank_dataset(
                    source_dataset,
                    source_schema,
                    directory / "rank_dataset_10d.parquet",
                    directory / "rank_dataset_10d_schema.json",
                    directory / "rank_label_coverage.csv",
                    date(2023, 12, 31),
                )

    def test_build_rank_dataset_rejects_post_development_dates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            directory = Path(temporary_name)
            source_dataset, source_schema = self._write_source_bundle(directory)
            frame = pd.read_parquet(source_dataset)
            frame.loc[frame["date"] == pd.Timestamp("2023-01-04"), "date"] = (
                pd.Timestamp("2024-01-02")
            )
            frame.to_parquet(source_dataset, index=False)
            schema = json.loads(source_schema.read_text(encoding="utf-8"))
            schema["parquet_sha256"] = self._sha256(source_dataset)
            source_schema.write_text(json.dumps(schema), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "after 2023-12-31"):
                build_rank_dataset(
                    source_dataset,
                    source_schema,
                    directory / "rank_dataset_10d.parquet",
                    directory / "rank_dataset_10d_schema.json",
                    directory / "rank_label_coverage.csv",
                    date(2023, 12, 31),
                )


if __name__ == "__main__":
    unittest.main()
