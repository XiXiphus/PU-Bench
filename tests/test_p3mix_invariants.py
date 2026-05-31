import unittest

import torch
from torch.utils.data import Dataset

from config.experiment_plan import build_plan
from config.method_loader import load_method_config
from train.p3mix.models import MixCNN_CIFAR10, MixLeNet
from train.p3mix.source_adapter import (
    P3MixSourceInputDataset,
    create_p3mix_source_ema,
)


class _TinyPUDataset(Dataset):
    def __init__(self):
        self.features = torch.tensor(
            [
                [[[0.0, 0.5], [1.0, 0.25]]],
                [[[0.75, 0.5], [0.25, 1.0]]],
            ],
            dtype=torch.float32,
        )
        self.pu_labels = torch.tensor([1, -1])
        self.true_labels = torch.tensor([1, 0])
        self.indices = torch.arange(2)
        self.pseudo_labels = torch.zeros(2)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.pu_labels[idx],
            self.true_labels[idx],
            self.indices[idx],
            self.pseudo_labels[idx],
        )


class TestP3MixInvariants(unittest.TestCase):
    def test_source_mix_models_preserve_mix_interface(self) -> None:
        cifar = MixCNN_CIFAR10()
        self.assertEqual(len(cifar.layers), 4)
        self.assertEqual(cifar.conv_list[0].kernel_size, (3, 3))
        self.assertEqual(cifar.conv_list[0].padding, (0, 0))
        self.assertEqual(cifar.conv_list[1].stride, (2, 2))
        self.assertEqual(cifar.conv_list[2].kernel_size, (1, 1))
        self.assertEqual(cifar.conv_list[3].out_channels, 10)
        self.assertEqual(cifar.fc1.in_features, 1960)
        self.assertEqual(cifar.final_classifier.out_features, 1)
        out = cifar(torch.zeros(2, 3, 32, 32), torch.ones(2, 3, 32, 32), 0.5, 3)
        self.assertEqual(tuple(out.shape), (2, 1))

        lenet = MixLeNet()
        self.assertEqual(len(lenet.layers), 3)
        self.assertEqual(lenet.conv1.out_channels, 6)
        self.assertEqual(lenet.conv1.padding, (2, 2))
        self.assertEqual(lenet.conv3.out_channels, 120)
        self.assertEqual(lenet.fc1.in_features, 120)
        self.assertEqual(lenet.final_classifier.out_features, 1)
        out = lenet(torch.zeros(2, 1, 28, 28), torch.ones(2, 1, 28, 28), 0.5, 2)
        self.assertEqual(tuple(out.shape), (2, 1))

    def test_cifar_source_input_adapter_matches_hmix_normalization(self) -> None:
        wrapped = P3MixSourceInputDataset(_TinyPUDataset(), "cifar10_hmix")
        x, pu, true, idx, pseudo = wrapped[0]
        self.assertTrue(torch.allclose(x, torch.tensor([[[-1.0, 0.0], [1.0, -0.5]]])))
        self.assertEqual(int(pu), 1)
        self.assertEqual(int(true), 1)
        self.assertEqual(int(idx), 0)
        self.assertEqual(float(pseudo), 0.0)
        self.assertTrue(torch.equal(wrapped.pu_labels, torch.tensor([1, -1])))

    def test_source_ema_teacher_is_independently_initialized(self) -> None:
        torch.manual_seed(7)
        student = MixCNN_CIFAR10()
        ema = create_p3mix_source_ema(MixCNN_CIFAR10, torch.device("cpu")).ema
        self.assertFalse(
            torch.allclose(student.conv_list[0].weight, ema.conv_list[0].weight)
        )
        self.assertTrue(all(not p.requires_grad for p in ema.parameters()))

    def test_config_promotes_only_cifar10_source_recipe(self) -> None:
        for method in ("p3mixc", "p3mixe"):
            with self.subTest(method=method):
                config = load_method_config(method)
                self.assertTrue(config.params["use_source_hparams_by_dataset"])
                self.assertEqual(set(config.params["source_hparams_by_dataset"]), {"cifar10"})
                self.assertIn("fashionmnist", config.params["source_partial_hparams_by_dataset"])

    def test_plan_applies_p3mix_source_recipe_only_on_cifar10(self) -> None:
        for method in ("p3mixc", "p3mixe"):
            with self.subTest(method=method, dataset="cifar10"):
                plan = build_plan(
                    dataset_config_path=(
                        "tests/fixtures/dataset_configs/"
                        "source_hparams_cifar10_seed2_dataset.yaml"
                    ),
                    methods=[method],
                    methods_dir="config/methods",
                )
                run = plan.runs[0]
                self.assertEqual(
                    run.params["source_hparams_resolved_from"],
                    "source_hparams_by_dataset.cifar10",
                )
                self.assertEqual(run.params["source_preset"], "cifar10-1")
                self.assertEqual(run.params["source_positive_label_list"], [0, 1, 8, 9])
                self.assertEqual(run.params["batch_size"], 512)
                self.assertEqual(run.params["num_epochs"], 200)
                self.assertEqual(run.params["val_iterations"], 20)
                self.assertEqual(float(run.params["alpha"]), 1.0)
                self.assertEqual(run.params["mix_layer"], 3)
                self.assertEqual(run.params["source_input_normalization"], "cifar10_hmix")
                self.assertTrue(run.params["source_loader_drop_last"])
                self.assertFalse(run.params["checkpoint"]["early_stopping"]["enabled"])

            with self.subTest(method=method, dataset="fashionmnist"):
                plan = build_plan(
                    dataset_config_path=(
                        "tests/fixtures/dataset_configs/"
                        "source_hparams_fashionmnist_seed2_dataset.yaml"
                    ),
                    methods=[method],
                    methods_dir="config/methods",
                )
                run = plan.runs[0]
                self.assertNotIn("source_hparams_resolved_from", run.params)
                self.assertNotIn("recommended_hparams_resolved_from", run.params)


if __name__ == "__main__":
    unittest.main()
