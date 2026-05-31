import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from config.method_loader import load_method_config
from train.vaepu.trainer import VAEPUTrainer


class _FakeDataset:
    def __init__(self, metadata: dict, length: int):
        self.pu_metadata = metadata
        self._length = length

    def __len__(self) -> int:
        return self._length


class TestVAEPUInvariants(unittest.TestCase):
    def test_source_sigmoid_loss_matches_author_formula_not_bce(self) -> None:
        logits = torch.tensor([-2.0, 0.0, 2.0])
        labels = torch.ones_like(logits)

        source_loss = VAEPUTrainer._source_sigmoid_loss(logits, labels)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )

        self.assertTrue(
            torch.allclose(source_loss, torch.sigmoid(-logits * labels))
        )
        self.assertFalse(torch.allclose(source_loss, bce_loss))
        self.assertAlmostEqual(float(source_loss[1]), 0.5)
        self.assertAlmostEqual(float(bce_loss[1]), 0.6931471824645996)

    def test_split_metadata_drives_benchmark_pi_weights(self) -> None:
        trainer = object.__new__(VAEPUTrainer)
        trainer.params = {}
        trainer.train_loader = SimpleNamespace(
            dataset=_FakeDataset(
                {
                    "n_total": 100,
                    "n_labeled": 5,
                    "n_pos_unlabeled": 45,
                    "n_unlabeled": 95,
                },
                length=100,
            )
        )

        VAEPUTrainer._set_pu_mixture_weights(trainer)

        self.assertAlmostEqual(trainer.pi_pl, 0.05)
        self.assertAlmostEqual(trainer.pi_pu, 0.45)
        self.assertAlmostEqual(trainer.pi_u, 0.95)
        self.assertAlmostEqual(trainer.pi_p, 0.5)
        self.assertAlmostEqual(trainer.params["pi_given"], 0.5)

    def test_source_mnist_pi_constants_are_recorded_not_applied_as_overrides(self) -> None:
        config = load_method_config("vaepu")
        mnist = config.params["source_hparams_by_dataset"]["mnist"]

        self.assertEqual(mnist["source_script"], "main.py")
        self.assertAlmostEqual(float(mnist["source_pi_pl"]), 0.01)
        self.assertAlmostEqual(float(mnist["source_pi_pu"]), 0.49)
        self.assertAlmostEqual(float(mnist["source_pi_u"]), 0.99)
        self.assertEqual(mnist["bench_prior_adapter"], "split_metadata_pi_weights")
        self.assertNotIn("pi_pl", mnist)
        self.assertNotIn("pi_pu", mnist)
        self.assertNotIn("pi_u", mnist)


if __name__ == "__main__":
    unittest.main()
