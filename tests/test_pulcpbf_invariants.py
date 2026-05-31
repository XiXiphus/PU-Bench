import unittest
from types import SimpleNamespace

from config.method_loader import load_method_config
from train.pulcpbf.trainer import PULCPBFTrainer


class TestPULCPBFInvariants(unittest.TestCase):
    def test_config_does_not_promote_partial_source_recipe_evidence(self) -> None:
        config = load_method_config("pulcpbf")

        self.assertNotIn("source_hparams_by_dataset", config.params)
        self.assertIn(
            "the provided run.sh only contains fmnist_1 commands, and source main.py hard-codes a PseudoAlzheimer phase-2 wrapper",
            config.metadata["alignment"]["known_alignment_gaps"],
        )
        self.assertIn(
            "source warmup.py leaves val_dataset unset for FashionMNIST/CIFAR/STL branches while later constructing a validation DataLoader",
            config.metadata["alignment"]["known_alignment_gaps"],
        )
        self.assertIn(
            "source alpha sweep checkpoints are written into alpha-suffixed directories, but source train.py reloads only one warmup_model path during alpha_list pseudo-labeling",
            config.metadata["alignment"]["known_alignment_gaps"],
        )

    def test_phase2_resets_checkpoint_tracking_after_reinitialization(self) -> None:
        trainer = object.__new__(PULCPBFTrainer)
        events = []

        trainer.phase2_epochs = 25
        trainer.phase2_reinitialize_model = True
        trainer.current_phase = 1
        trainer.file_console = None
        trainer.checkpoint_handler = object()
        trainer.console = SimpleNamespace(
            log=lambda *args, **kwargs: events.append(("console_log", args[0]))
        )

        trainer._install_image_source_transforms = lambda: events.append(("install",))
        trainer.before_training = lambda: events.append(("before",))
        trainer.set_checkpoint_early_stopping = (
            lambda enabled, reset=False: events.append(("early", enabled, reset))
        )
        trainer._run_phase1 = lambda: events.append(("phase1",))
        trainer._generate_pseudo_labels = lambda: events.append(("pseudo",))
        trainer._reinitialize_model_for_phase2 = lambda: events.append(("reinit",))
        trainer._ensure_phase2_train_dataset = lambda: events.append(("wrap",))
        trainer._init_optimizer_phase2 = lambda: events.append(("optimizer",))
        trainer.reset_checkpoint_tracking = lambda: events.append(("reset_tracking",))
        trainer.run_stage = (
            lambda stage, epochs: events.append(("stage", stage, epochs)) or {}
        )
        trainer.finalize = lambda: events.append(("finalize",))

        PULCPBFTrainer.run(trainer)

        self.assertIn(("reset_tracking",), events)
        self.assertLess(
            events.index(("reset_tracking",)),
            events.index(("stage", "Fine-tuning", 25)),
        )
        self.assertLess(
            events.index(("reinit",)),
            events.index(("reset_tracking",)),
        )


if __name__ == "__main__":
    unittest.main()
