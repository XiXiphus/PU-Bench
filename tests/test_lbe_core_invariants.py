import unittest

import torch

from train.lbe.core import observed_label_pretrain_loss


class TestLBECoreInvariants(unittest.TestCase):
    def test_pretrain_loss_matches_source_formula_for_regular_logits(self) -> None:
        logits = torch.tensor([-1.5, -0.2, 0.4, 1.7])
        q = torch.tensor([0.0, 1.0, 0.0, 1.0])
        proportion_labeled = q.mean()
        p = torch.sigmoid(logits)
        source_loss = -(
            q * (1 - proportion_labeled) * torch.log(p)
            + proportion_labeled * (1 - q) * torch.log(1 - p)
        ).mean()

        self.assertTrue(
            torch.allclose(
                observed_label_pretrain_loss(logits, q, proportion_labeled),
                source_loss,
            )
        )

    def test_pretrain_loss_is_finite_for_saturated_logits(self) -> None:
        logits = torch.tensor([-1000.0, 1000.0])
        q = torch.tensor([0.0, 1.0])

        loss = observed_label_pretrain_loss(logits, q)

        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
