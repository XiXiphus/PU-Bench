import unittest

from train.nnpusb.trainer import NNPUSBTrainer


class TestNNPUSBInvariants(unittest.TestCase):
    def test_create_criterion_rejects_positive_risk_weight(self) -> None:
        trainer = object.__new__(NNPUSBTrainer)
        trainer.params = {"weight": 1.5}
        trainer.prior = 0.3

        with self.assertRaisesRegex(ValueError, "does not reweight"):
            trainer.create_criterion()


if __name__ == "__main__":
    unittest.main()
