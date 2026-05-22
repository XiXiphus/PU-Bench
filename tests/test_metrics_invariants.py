import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.data_utils import PUDataset
from train.metrics import evaluate_metrics


class NegativeLogitRanker(nn.Module):
    def forward(self, x):
        return x.view(-1)


class TestMetricInvariants(unittest.TestCase):
    def _loader(self) -> DataLoader:
        features = torch.tensor([[-4.0], [-3.0], [-2.0], [-1.0]])
        true_labels = torch.tensor([0, 0, 1, 1])
        pu_labels = torch.tensor([-1, -1, 1, 1])
        dataset = PUDataset(features, pu_labels, true_labels)
        return DataLoader(dataset, batch_size=4, shuffle=False)

    def test_oracle_metrics_default_to_model_decision_rule(self) -> None:
        metrics = evaluate_metrics(
            NegativeLogitRanker(),
            self._loader(),
            torch.device("cpu"),
            prior=0.5,
        )
        self.assertEqual(metrics["oracle_accuracy"], 0.5)
        self.assertEqual(metrics["oracle_recall"], 0.0)

    def test_prior_calibrated_fallback_is_explicit(self) -> None:
        metrics = evaluate_metrics(
            NegativeLogitRanker(),
            self._loader(),
            torch.device("cpu"),
            prior=0.5,
            prior_calibrated_fallback=True,
        )
        self.assertEqual(metrics["oracle_accuracy"], 1.0)
        self.assertEqual(metrics["oracle_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
