from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from rank_model.pipeline import parse_run_ids
from rank_model.stages.dataset import file_sha256
from rank_model.stages.evaluation import (
    compare_runs,
    evaluate_predictions,
    write_evaluation,
)


def _prediction_frame(
    *,
    score_direction: float = 1.0,
    constant_scores: bool = False,
) -> pd.DataFrame:
    rows_per_date = 1000
    dates = pd.to_datetime(["2023-01-03", "2023-01-04"])
    positions = np.tile(np.arange(rows_per_date, dtype="float64"), len(dates))
    rank_target = positions / (rows_per_date - 1)
    score = np.zeros(len(positions)) if constant_scores else score_direction * rank_target
    return pd.DataFrame(
        {
            "date": np.repeat(dates, rows_per_date),
            "stock_code": [f"S{index:04d}" for index in range(len(positions))],
            "split": "validation",
            "horizon": 10,
            "target_10d": -0.10 + 0.30 * rank_target,
            "rank_target_10d": rank_target,
            "score_raw": score,
        }
    )


class EvaluationMetricTests(unittest.TestCase):
    def test_perfect_scores_have_perfect_cross_sectional_and_top100_metrics(self) -> None:
        bundle = evaluate_predictions(_prediction_frame())

        self.assertEqual(bundle.summary["mean_rank_ic"], 1.0)
        self.assertEqual(bundle.summary["mean_kendall_tau"], 1.0)
        self.assertEqual(bundle.summary["mean_pairwise_accuracy"], 1.0)
        self.assertEqual(bundle.summary["mean_precision_100"], 1.0)
        self.assertEqual(bundle.summary["mean_recall_100"], 1.0)
        self.assertEqual(bundle.summary["mean_ndcg_100"], 1.0)
        self.assertEqual(bundle.summary["mean_rank_mae"], 0.0)
        self.assertTrue((bundle.daily_metrics["valid_target_count"] == 1000).all())

    def test_reversed_scores_have_negative_rank_ic_and_no_top100_overlap(self) -> None:
        bundle = evaluate_predictions(_prediction_frame(score_direction=-1.0))

        self.assertEqual(bundle.summary["mean_rank_ic"], -1.0)
        self.assertEqual(bundle.summary["mean_precision_100"], 0.0)
        self.assertEqual(bundle.summary["mean_recall_100"], 0.0)
        self.assertEqual(bundle.summary["mean_jaccard_100"], 0.0)

    def test_constant_scores_are_counted_and_do_not_produce_rank_ic(self) -> None:
        bundle = evaluate_predictions(_prediction_frame(constant_scores=True))

        self.assertEqual(bundle.summary["rank_ic_valid_dates"], 0)
        self.assertEqual(bundle.summary["rank_ic_invalid_dates"], 2)
        self.assertEqual(bundle.summary["constant_score_dates"], 2)
        self.assertTrue(bundle.daily_metrics["rank_ic"].isna().all())

    def test_missing_targets_preserve_prediction_keys_and_filter_each_metric(self) -> None:
        predictions = _prediction_frame()
        predictions.loc[0, "rank_target_10d"] = np.nan
        predictions.loc[999, "target_10d"] = np.nan
        bundle = evaluate_predictions(predictions)

        self.assertEqual(len(bundle.predictions), len(predictions))
        self.assertEqual(
            bundle.predictions["stock_code"].astype(str).tolist(),
            predictions["stock_code"].astype(str).tolist(),
        )
        self.assertEqual(
            bundle.predictions["date"].tolist(), predictions["date"].tolist()
        )
        self.assertEqual(bundle.daily_metrics.loc[0, "valid_target_count"], 999)
        self.assertEqual(bundle.daily_metrics.loc[0, "top100_valid_return_count"], 100)
        self.assertEqual(bundle.daily_metrics.loc[0, "decile_1_valid_return_count"], 99)

    def test_fewer_than_top_k_finite_targets_uses_the_available_target_universe(self) -> None:
        predictions = _prediction_frame().iloc[:10].copy()
        predictions.loc[predictions.index[3:], "rank_target_10d"] = np.nan
        bundle = evaluate_predictions(predictions, top_k=100)

        self.assertEqual(bundle.daily_metrics.loc[0, "valid_target_count"], 3)
        self.assertEqual(bundle.daily_metrics.loc[0, "precision_100"], 1.0)
        self.assertEqual(bundle.daily_metrics.loc[0, "recall_100"], 1.0)
        self.assertEqual(bundle.daily_metrics.loc[0, "ndcg_100"], 1.0)

    def test_rank_errors_repercentile_only_the_finite_target_subset(self) -> None:
        predictions = pd.DataFrame(
            {
                "date": pd.Timestamp("2023-01-03"),
                "stock_code": ["S0", "S1", "S2", "S3", "S4"],
                "target_10d": [0.0, 0.1, 0.2, 0.3, 0.4],
                "rank_target_10d": [np.nan, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
                "score_raw": [0.0, 1.0, 2.0, 3.0, 4.0],
            }
        )

        bundle = evaluate_predictions(predictions)

        self.assertEqual(bundle.daily_metrics.loc[0, "rank_mae"], 0.0)
        self.assertEqual(bundle.daily_metrics.loc[0, "rank_rmse"], 0.0)


class EvaluationOutputTests(unittest.TestCase):
    def test_writes_standard_reports_and_compares_completed_runs(self) -> None:
        bundle = evaluate_predictions(_prediction_frame().iloc[:20].copy(), top_k=10)
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            first = self._run_directory(root, "first", bundle.predictions)
            second = self._run_directory(root, "second", bundle.predictions)
            write_evaluation(bundle, first)
            write_evaluation(bundle, second)

            for name in (
                "predictions_10d.parquet",
                "metrics_summary.json",
                "daily_metrics.csv",
                "monthly_metrics.csv",
                "yearly_metrics.csv",
                "decile_returns.csv",
                "top100_detail.parquet",
                "manifest.json",
            ):
                self.assertTrue((first / name).exists(), name)

            comparison_path = root / "comparison.csv"
            comparison = compare_runs([first, second], comparison_path)
            self.assertEqual(comparison["run_id"].tolist(), ["first", "second"])
            self.assertTrue(comparison_path.exists())
            self.assertFalse(
                {"accept", "reject", "champion"}.intersection(comparison.columns)
            )

    def test_comparison_rejects_duplicate_or_incomplete_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            run = self._run_directory(
                root, "only", evaluate_predictions(_prediction_frame().iloc[:20]).predictions
            )
            (run / "metrics_summary.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate"):
                compare_runs([run, run], root / "comparison.csv")

            manifest_path = run / "manifest.json"
            manifest_path.write_text(
                json.dumps({"run_id": "only", "status": "training"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "completed"):
                compare_runs([run], root / "comparison.csv")

    def test_evaluation_requires_existing_completed_run_and_prediction_artifact(self) -> None:
        bundle = evaluate_predictions(_prediction_frame().iloc[:20])
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            missing = root / "missing"
            with self.assertRaises(FileNotFoundError):
                write_evaluation(bundle, missing)
            self.assertFalse(missing.exists())

            missing_manifest = root / "missing-manifest"
            missing_manifest.mkdir()
            with self.assertRaises(FileNotFoundError):
                write_evaluation(bundle, missing_manifest)

    def test_evaluation_rejects_tampered_prediction_scores(self) -> None:
        bundle = evaluate_predictions(_prediction_frame().iloc[:20])
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            run = self._run_directory(root, "tampered", bundle.predictions)
            artifact = pd.read_parquet(run / "predictions_10d.parquet")
            artifact.loc[artifact.index[0], "score_raw"] += 1.0
            artifact.to_parquet(run / "predictions_10d.parquet", index=False)

            with self.assertRaisesRegex(ValueError, "hash"):
                write_evaluation(bundle, run)

            self.assertFalse((run / "metrics_summary.json").exists())

    @staticmethod
    def _run_directory(root: Path, run_id: str, predictions: pd.DataFrame) -> Path:
        run = root / run_id
        run.mkdir()
        predictions_path = run / "predictions_10d.parquet"
        predictions.to_parquet(predictions_path, index=False)
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "model_name": "ridge",
                    "status": "completed",
                    "predictions_10d_sha256": file_sha256(predictions_path),
                }
            ),
            encoding="utf-8",
        )
        return run


class PipelineArgumentTests(unittest.TestCase):
    def test_parse_run_ids_rejects_blank_components(self) -> None:
        self.assertEqual(parse_run_ids("first, second"), ["first", "second"])
        for value in ("first,,second", ",first", "first,", "   "):
            with self.assertRaisesRegex(ValueError, "blank"):
                parse_run_ids(value)


if __name__ == "__main__":
    unittest.main()
