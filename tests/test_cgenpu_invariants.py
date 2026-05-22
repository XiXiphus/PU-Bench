import unittest

import torch
from torch import nn

from train.cgenpu.trainer import CGenPUTrainer
from train.checkpointing import CheckpointBundle


class TestCGenPUInvariants(unittest.TestCase):
    def test_checkpoint_model_contains_all_cgenpu_components(self) -> None:
        trainer = object.__new__(CGenPUTrainer)
        trainer.D = nn.Linear(1, 1)
        trainer.A = nn.Linear(1, 1)
        trainer.G = nn.Linear(3, 1)
        trainer.D_opt = torch.optim.Adam(trainer.D.parameters(), lr=1e-4)
        trainer.A_opt = torch.optim.Adam(trainer.A.parameters(), lr=1e-4)
        trainer.G_opt = torch.optim.Adam(trainer.G.parameters(), lr=1e-4)

        checkpoint_model = trainer.get_checkpoint_model()

        self.assertIsInstance(checkpoint_model, CheckpointBundle)
        state = checkpoint_model.state_dict()
        self.assertIn("discriminator", state)
        self.assertIn("auxiliary", state)
        self.assertIn("generator", state)
        self.assertIn("optimizer_d", state)
        self.assertIn("optimizer_a", state)
        self.assertIn("optimizer_g", state)


if __name__ == "__main__":
    unittest.main()
