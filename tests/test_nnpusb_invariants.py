import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.data_utils import PUDataset
from train.nnpusb.trainer import NNPUSBTrainer
from train.utils.metrics import evaluate_metrics


class RawScoreModel(nn.Module):
    def forward(self, x):
        return x.view(-1)


class TestNNPUSBInvariants(unittest.TestCase):
    def test_create_criterion_rejects_positive_risk_weight(self) -> None:
        trainer = object.__new__(NNPUSBTrainer)
        trainer.params = {"weight": 1.5}
        trainer.prior = 0.3

        with self.assertRaisesRegex(ValueError, "does not reweight"):
            trainer.create_criterion()

    def test_calibrates_inference_from_all_training_raw_scores(self) -> None:
        features = torch.arange(10, dtype=torch.float32).view(-1, 1)
        true_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        pu_labels = torch.tensor([-1, -1, -1, -1, -1, 1, 1, 1, 1, -1])
        loader = DataLoader(
            PUDataset(features, pu_labels, true_labels),
            batch_size=4,
            shuffle=False,
        )

        trainer = object.__new__(NNPUSBTrainer)
        trainer.model = RawScoreModel()
        trainer.train_loader = loader
        trainer.device = torch.device("cpu")
        trainer.prior = 1.0 / 6.0

        threshold = trainer.calibrate_decision_threshold()
        self.assertEqual(threshold.item(), 5.0)
        self.assertEqual(trainer.threshold_prior, 0.5)

        metrics = evaluate_metrics(trainer.model, loader, trainer.device, trainer.prior)
        self.assertEqual(metrics["oracle_accuracy"], 1.0)
        self.assertEqual(metrics["oracle_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
