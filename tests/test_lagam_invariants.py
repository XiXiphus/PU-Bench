import unittest

import torch

from config.experiment_plan import build_plan
from config.method_loader import load_method_config
from train.lagam.trainer import LaGAMTrainer


class TestLaGAMInvariants(unittest.TestCase):
    def test_config_promotes_only_cifar10_source_recipe(self) -> None:
        config = load_method_config("lagam")

        self.assertTrue(config.params["use_source_hparams_by_dataset"])
        self.assertEqual(set(config.params["source_hparams_by_dataset"]), {"cifar10"})
        recipe = config.params["source_hparams_by_dataset"]["cifar10"]
        self.assertEqual(recipe["source_preset"], "cifar10-1")
        self.assertEqual(recipe["source_positive_label_list"], [0, 1, 8, 9])
        self.assertEqual(recipe["source_num_labeled"], 1000)
        self.assertEqual(recipe["source_num_valid"], 500)
        self.assertEqual(recipe["num_epochs"], 400)
        self.assertEqual(recipe["warmup_epoch"], 20)
        self.assertEqual(recipe["num_cluster"], 5)
        self.assertNotIn("arch", recipe)
        self.assertNotIn("normalization_mean", recipe)
        self.assertTrue(recipe["use_source_lr_schedule"])

    def test_plan_applies_lagam_cifar10_source_recipe_without_model_override(self) -> None:
        plan = build_plan(
            dataset_config_path=(
                "tests/fixtures/dataset_configs/"
                "source_hparams_lagam_cifar10_seed2_dataset.yaml"
            ),
            methods=["lagam"],
            methods_dir="config/methods",
        )
        run = plan.runs[0]

        self.assertEqual(
            run.params["source_hparams_resolved_from"],
            "source_hparams_by_dataset.cifar10",
        )
        self.assertEqual(run.params["scenario"], "single")
        self.assertEqual(run.params["n_labeled"], 1000)
        self.assertNotIn("arch", run.params)
        self.assertEqual(run.params["batch_size"], 64)
        self.assertEqual(run.params["num_epochs"], 400)
        self.assertEqual(run.params["warmup_epoch"], 20)
        self.assertEqual(run.params["num_cluster"], 5)
        self.assertEqual(run.params["lr_decay_epochs"], [250, 300, 350])
        self.assertFalse(run.params["checkpoint"]["early_stopping"]["enabled"])
        self.assertNotIn("recommended_hparams_resolved_from", run.params)

    def test_source_lr_schedule_uses_strict_source_milestones(self) -> None:
        trainer = object.__new__(LaGAMTrainer)
        trainer.params = {
            "use_source_lr_schedule": True,
            "lr": 0.001,
            "lr_decay_epochs": [250, 300, 350],
            "lr_decay_rate": 0.1,
            "cosine": False,
            "num_epochs": 400,
        }
        trainer.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=1.0)

        trainer.global_epoch = 251
        LaGAMTrainer._adjust_source_learning_rate(trainer)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 0.001)

        trainer.global_epoch = 252
        LaGAMTrainer._adjust_source_learning_rate(trainer)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 0.0001)


if __name__ == "__main__":
    unittest.main()
