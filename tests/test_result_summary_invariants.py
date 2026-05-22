import unittest
from types import SimpleNamespace

import torch

from train.base.results import ResultSummaryMixin


class DummySummary(ResultSummaryMixin):
    def __init__(self, checkpoint_handler):
        self.method = "dummy"
        self.experiment_name = "summary-test"
        self.device = torch.device("cpu")
        self._run_start_time = 100.0
        self._run_end_time = 130.0
        self._max_gpu_mem_bytes = 0
        self.checkpoint_handler = checkpoint_handler
        self.global_epoch = 3
        self.params = {
            "method_metadata": {
                "alignment": {
                    "level": "controlled_adaptation",
                    "source_reproduction": False,
                }
            }
        }
        self.train_loader = None
        self.test_loader = None

    def _oracle_prior_calibrated_fallback(self) -> bool:
        return False


class TestResultSummaryInvariants(unittest.TestCase):
    def test_duration_is_total_wall_clock_not_time_to_best(self) -> None:
        checkpoint = SimpleNamespace(
            best_elapsed_seconds=12.5,
            best_metrics={"val_proxy_auc": 0.75},
            best_epoch=2,
            monitor="val_proxy_auc",
        )
        summary = DummySummary(checkpoint)._compose_result_summary()

        self.assertEqual(summary["timing"]["duration_seconds"], 30.0)
        self.assertEqual(summary["timing"]["time_to_best_seconds"], 12.5)
        self.assertEqual(summary["best"]["epoch"], 2)
        self.assertEqual(
            summary["method_metadata"]["alignment"]["level"],
            "controlled_adaptation",
        )

    def test_time_to_best_is_optional(self) -> None:
        checkpoint = SimpleNamespace(
            best_elapsed_seconds=None,
            best_metrics={"val_proxy_auc": 0.75},
            best_epoch=2,
            monitor="val_proxy_auc",
        )
        summary = DummySummary(checkpoint)._compose_result_summary()

        self.assertEqual(summary["timing"]["duration_seconds"], 30.0)
        self.assertIsNone(summary["timing"]["time_to_best_seconds"])


if __name__ == "__main__":
    unittest.main()
