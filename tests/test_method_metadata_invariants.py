import unittest

from config.experiment_plan import (
    build_plan,
    plan_to_dict,
    resolve_method_params_for_dataset,
)
from config.method_loader import load_method_config
from config.method_loader import list_method_configs


ACTIVE_SOURCE_HPARAM_DATASETS = {
    "20news",
    "alzheimermri",
    "cifar10",
    "connect4",
    "fashionmnist",
    "imdb",
    "mnist",
    "spambase",
}

SOURCE_ALIGNED_METHOD_RECIPE_SETS = {
    "nnpu": {
        "source": {"mnist", "cifar10"},
        "recommended": {
            "20news",
            "alzheimermri",
            "connect4",
            "fashionmnist",
            "imdb",
            "spambase",
        },
    },
    "nnpusb": {
        "source": {"mnist"},
        "recommended": {
            "20news",
            "alzheimermri",
            "cifar10",
            "connect4",
            "fashionmnist",
            "imdb",
            "spambase",
        },
    },
    "vpu": {
        "source": {"cifar10", "fashionmnist"},
        "recommended": {
            "20news",
            "alzheimermri",
            "connect4",
            "imdb",
            "mnist",
            "spambase",
        },
    },
    "distpu": {
        "source": {"alzheimermri", "cifar10", "fashionmnist"},
        "recommended": {"20news", "connect4", "imdb", "mnist", "spambase"},
    },
    "pulda": {
        "source": {"cifar10"},
        "recommended": {
            "20news",
            "alzheimermri",
            "connect4",
            "fashionmnist",
            "imdb",
            "mnist",
            "spambase",
        },
    },
    "selfpu": {
        "source": {"cifar10", "mnist"},
        "recommended": {
            "20news",
            "alzheimermri",
            "connect4",
            "fashionmnist",
            "imdb",
            "spambase",
        },
    },
    "holisticpu": {
        "source": {"alzheimermri", "cifar10", "fashionmnist"},
        "recommended": {"20news", "connect4", "imdb", "mnist", "spambase"},
    },
    "puet": {
        "source": {"mnist"},
        "recommended": set(),
    },
    "vaepu": {
        "source": {"mnist"},
        "recommended": set(),
    },
    "p3mixc": {
        "source": {"cifar10"},
        "recommended": set(),
    },
    "p3mixe": {
        "source": {"cifar10"},
        "recommended": set(),
    },
}


class TestMethodMetadataInvariants(unittest.TestCase):
    def test_all_enabled_method_checkpoints_use_proxy_accuracy(self) -> None:
        for config in list_method_configs("config/methods"):
            checkpoint = config.params.get("checkpoint")
            if not checkpoint or not checkpoint.get("enabled", False):
                continue
            with self.subTest(method=config.method_key):
                self.assertEqual(checkpoint.get("monitor"), "val_proxy_acc")
                self.assertEqual(checkpoint.get("mode", "max"), "max")

    def test_source_hparam_recipes_stay_within_active_bench_datasets(self) -> None:
        for config in list_method_configs("config/methods"):
            recipes = config.params.get("source_hparams_by_dataset") or {}
            if not recipes:
                continue
            with self.subTest(method=config.method_key):
                self.assertTrue(
                    set(recipes).issubset(ACTIVE_SOURCE_HPARAM_DATASETS),
                    f"{config.method_key} has out-of-scope source recipes: "
                    f"{sorted(set(recipes) - ACTIVE_SOURCE_HPARAM_DATASETS)}",
                )

    def test_source_partial_hparam_notes_stay_within_active_bench_datasets(self) -> None:
        for config in list_method_configs("config/methods"):
            recipes = config.params.get("source_partial_hparams_by_dataset") or {}
            if not recipes:
                continue
            with self.subTest(method=config.method_key):
                self.assertTrue(
                    set(recipes).issubset(ACTIVE_SOURCE_HPARAM_DATASETS),
                    f"{config.method_key} has out-of-scope partial source notes: "
                    f"{sorted(set(recipes) - ACTIVE_SOURCE_HPARAM_DATASETS)}",
                )

    def test_validated_source_and_recommended_recipe_sets_match_master_table(
        self,
    ) -> None:
        for method, expected in SOURCE_ALIGNED_METHOD_RECIPE_SETS.items():
            config = load_method_config(method)
            source_recipes = set(config.params.get("source_hparams_by_dataset") or {})
            recommended_recipes = set(
                config.params.get("recommended_hparams_by_dataset") or {}
            )
            with self.subTest(method=method):
                self.assertEqual(source_recipes, expected["source"])
                self.assertEqual(recommended_recipes, expected["recommended"])
                self.assertNotIn("20news", source_recipes)
                if "20news" in expected["recommended"]:
                    self.assertIn("20news", recommended_recipes)
                else:
                    self.assertNotIn("20news", recommended_recipes)

    def test_bbepu_alignment_metadata_is_structured(self) -> None:
        config = load_method_config("bbepu")

        alignment = config.metadata["alignment"]
        self.assertEqual(alignment["level"], "controlled_adaptation")
        self.assertFalse(alignment["source_reproduction"])
        self.assertIn("source_references", alignment)
        self.assertIn("benchmark_adaptations", alignment)

    def test_cgenpu_alignment_metadata_is_structured(self) -> None:
        config = load_method_config("cgenpu")

        alignment = config.metadata["alignment"]
        self.assertEqual(
            alignment["level"],
            "source_adapted_kernel_benchmark_wrapper",
        )
        self.assertFalse(alignment["source_reproduction"])
        self.assertIn(
            "D, auxiliary A, then G update order for each unlabeled batch",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "PyTorch port of the TensorFlow source implementation",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "validation monitor val_proxy_acc instead of source WandB/image callback logging",
            alignment["benchmark_adaptations"],
        )
        self.assertEqual(config.params["checkpoint"]["monitor"], "val_proxy_acc")
        self.assertFalse(config.params["checkpoint"]["early_stopping"]["enabled"])

    def test_distpu_alignment_metadata_is_structured(self) -> None:
        config = load_method_config("distpu")

        alignment = config.metadata["alignment"]
        self.assertEqual(
            alignment["level"],
            "source_faithful_kernel_benchmark_wrapper",
        )
        self.assertFalse(alignment["source_reproduction"])
        self.assertIn(
            "warm-up stage with label-distribution loss plus entropy regularization",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "source train.py dataset choices cifar-10, fmnist, and alzheimer share the same optimizer/schedule defaults",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "mixup-stage pseudo-label refresh from model scores",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "PU-Bench empirical pi_unlabeled prior from dataset metadata",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "validation monitor val_proxy_acc instead of source-script test accuracy logging",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "PU-Bench recommended dataset recipes are recorded separately from source recipes when source train.py hyperparameters are unavailable",
            alignment["benchmark_adaptations"],
        )
        self.assertTrue(config.params["use_source_hparams_by_dataset"])
        self.assertEqual(
            config.params["source_hparams_by_dataset"]["alzheimermri"]["source_dataset"],
            "alzheimer",
        )
        self.assertEqual(
            config.params["source_hparams_by_dataset"]["fashionmnist"]["stages"]["mixup"]["alpha"],
            6.0,
        )
        self.assertTrue(config.params["use_recommended_hparams_by_dataset"])
        self.assertEqual(
            set(config.params["recommended_hparams_by_dataset"]),
            {"mnist", "imdb", "20news", "spambase", "connect4"},
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["mnist"]["recommended_preset"],
            "mnist_fashionmnist_source_schedule",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["imdb"]["recommended_preset"],
            "vector_source_schedule",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["20news"]["recommended_preset"],
            "vector_source_schedule",
        )
        self.assertEqual(config.params["checkpoint"]["monitor"], "val_proxy_acc")
        self.assertFalse(config.params["checkpoint"]["early_stopping"]["enabled"])

    def test_holisticpu_alignment_metadata_is_structured(self) -> None:
        config = load_method_config("holisticpu")

        alignment = config.metadata["alignment"]
        self.assertEqual(alignment["level"], "mixed_controlled_adaptation")
        self.assertFalse(alignment["source_reproduction"])
        self.assertEqual(
            alignment["modality_alignment"]["image"],
            "source_aligned_recipe_with_controlled_backbone",
        )
        self.assertEqual(
            alignment["modality_alignment"]["vector_tabular"],
            "benchmark_specific_extension",
        )
        self.assertIn(
            "source run.sh dataset recipes for source-supported image datasets",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "source run.sh dataset recipes are applied at plan time for Bench-supported source datasets",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "PU-Bench recommended dataset recipes are recorded separately from source recipes when source run.sh hyperparameters are unavailable",
            alignment["benchmark_adaptations"],
        )
        self.assertTrue(config.params["use_source_hparams_by_dataset"])
        self.assertEqual(
            set(config.params["source_hparams_by_dataset"]),
            {"cifar10", "fashionmnist", "alzheimermri"},
        )
        self.assertEqual(
            config.params["source_hparams_by_dataset"]["cifar10"]["source_dataset"],
            "cifar10_1",
        )
        self.assertFalse(
            config.params["source_hparams_by_dataset"]["fashionmnist"][
                "source_label_split_exact_match"
            ],
        )
        self.assertEqual(
            config.params["source_hparams_by_dataset"]["alzheimermri"]["eval_step"],
            100,
        )
        self.assertTrue(config.params["use_recommended_hparams_by_dataset"])
        self.assertEqual(
            set(config.params["recommended_hparams_by_dataset"]),
            {"mnist", "imdb", "20news", "spambase", "connect4"},
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["mnist"]["recommended_preset"],
            "mnist_fashionmnist_style",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["imdb"]["recommended_preset"],
            "vector_default_short_eval",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["20news"]["recommended_preset"],
            "vector_default_short_eval",
        )

    def test_lbe_alignment_metadata_is_structured(self) -> None:
        config = load_method_config("lbe")

        alignment = config.metadata["alignment"]
        self.assertEqual(alignment["level"], "controlled_adaptation")
        self.assertFalse(alignment["source_reproduction"])
        self.assertIn(
            "q=1 labeled positives as hard posterior anchors",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "PU-Bench shared controlled backbone instead of the author MLP script",
            alignment["benchmark_adaptations"],
        )

    def test_nnpu_alignment_metadata_is_structured(self) -> None:
        config = load_method_config("nnpu")

        alignment = config.metadata["alignment"]
        self.assertEqual(
            alignment["level"],
            "source_faithful_kernel_benchmark_wrapper",
        )
        self.assertFalse(alignment["source_reproduction"])
        self.assertIn(
            "PU labels +1 for labeled positives and -1 for unlabeled samples",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "risk prior is the positive fraction inside U",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "validation monitor val_proxy_acc instead of source train/test zero-one error reporting",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "PU-Bench recommended dataset recipes are recorded separately from source recipes when source hyperparameters are unavailable",
            alignment["benchmark_adaptations"],
        )
        self.assertEqual(config.params["loss"], "sigmoid")
        self.assertEqual(float(config.params["beta"]), 0.0)
        self.assertEqual(float(config.params["gamma"]), 1.0)
        self.assertTrue(config.params["use_source_hparams_by_dataset"])
        self.assertEqual(
            config.params["source_hparams_by_dataset"]["mnist"]["source_preset"],
            "exp-mnist",
        )
        self.assertEqual(
            config.params["source_hparams_by_dataset"]["cifar10"]["source_preset"],
            "exp-cifar",
        )
        self.assertTrue(config.params["use_recommended_hparams_by_dataset"])
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["fashionmnist"]["recommended_preset"],
            "fashionmnist_mnist_style",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["alzheimermri"]["recommended_preset"],
            "image_default_baseline",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["imdb"]["recommended_preset"],
            "text_embedding_default_baseline",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["20news"]["recommended_preset"],
            "text_embedding_default_baseline",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["spambase"]["recommended_preset"],
            "tabular_default_baseline",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["connect4"]["recommended_preset"],
            "tabular_default_baseline",
        )
        self.assertFalse(config.params["init_bias_from_prior"])
        self.assertEqual(config.params["checkpoint"]["monitor"], "val_proxy_acc")
        self.assertFalse(config.params["checkpoint"]["early_stopping"]["enabled"])

    def test_nnpusb_alignment_metadata_is_structured(self) -> None:
        config = load_method_config("nnpusb")

        alignment = config.metadata["alignment"]
        self.assertEqual(
            alignment["level"],
            "source_faithful_kernel_benchmark_wrapper",
        )
        self.assertFalse(alignment["source_reproduction"])
        self.assertIn(
            "selected-bias positives use the source mean-normalize then max-normalize accept-reject rule when SAR-PUSB selection is requested",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "source main_nnPUSB_mnist.py optimizer and schedule recipe for MNIST",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "risk prior is the positive fraction inside U",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "validation monitor val_proxy_acc instead of source test accuracy and quantile-evaluation logging",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "benchmark may run the estimator under SCAR or non-source SAR splits for robustness evaluation",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "selected-bias scores come from a PU-Bench PN scorer on the benchmark split instead of the source pretraining MLP on concatenated MNIST train/test",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "PU-Bench fallback learning rate, weight decay, batch size, and epoch budget for source-unsupported datasets",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "PU-Bench recommended dataset recipes are recorded separately from source recipes when source hyperparameters are unavailable",
            alignment["benchmark_adaptations"],
        )
        self.assertTrue(config.params["use_source_hparams_by_dataset"])
        self.assertEqual(
            config.params["source_hparams_by_dataset"]["mnist"]["source_preset"],
            "main_nnPUSB_mnist",
        )
        self.assertTrue(config.params["use_recommended_hparams_by_dataset"])
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["alzheimermri"]["recommended_preset"],
            "image_default_baseline",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["cifar10"]["recommended_preset"],
            "cifar10_nnpu_style",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["fashionmnist"]["recommended_preset"],
            "fashionmnist_mnist_style",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["imdb"]["recommended_preset"],
            "text_embedding_default_baseline",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["20news"]["recommended_preset"],
            "text_embedding_default_baseline",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["spambase"]["recommended_preset"],
            "tabular_default_baseline",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["connect4"]["recommended_preset"],
            "tabular_default_baseline",
        )
        self.assertEqual(float(config.params["beta"]), 0.0)
        self.assertEqual(float(config.params["gamma"]), 1.0)
        self.assertFalse(config.params["init_bias_from_prior"])
        self.assertEqual(config.params["label_scheme"]["pu_labeled_label"], 1)
        self.assertEqual(config.params["label_scheme"]["pu_unlabeled_label"], -1)
        self.assertFalse(config.params["checkpoint"]["save_model"])
        self.assertEqual(config.params["checkpoint"]["monitor"], "val_proxy_acc")
        self.assertFalse(config.params["checkpoint"]["early_stopping"]["enabled"])

    def test_pulcpbf_alignment_metadata_is_structured(self) -> None:
        config = load_method_config("pulcpbf")

        alignment = config.metadata["alignment"]
        self.assertEqual(alignment["level"], "mixed_controlled_adaptation")
        self.assertFalse(alignment["source_reproduction"])
        self.assertEqual(
            alignment["modality_alignment"]["image"],
            "partial_source_kernel_with_controlled_backbone",
        )
        self.assertEqual(
            alignment["modality_alignment"]["vector_tabular"],
            "benchmark_specific_extension",
        )
        self.assertIn(
            "the provided run.sh only contains fmnist_1 commands, and source main.py hard-codes a PseudoAlzheimer phase-2 wrapper",
            alignment["known_alignment_gaps"],
        )
        self.assertIn(
            "unlabeled phase-2 targets do not use oracle true labels",
            alignment["retained_source_components"],
        )

    def test_pulda_alignment_metadata_is_structured(self) -> None:
        config = load_method_config("pulda")

        alignment = config.metadata["alignment"]
        self.assertEqual(
            alignment["level"],
            "source_faithful_kernel_benchmark_wrapper",
        )
        self.assertFalse(alignment["source_reproduction"])
        self.assertIn(
            "fixed P/U resampling batches with P_batch_size=16 and U_batch_size=128",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "source train.py CIFAR-10-only optimizer and schedule recipe",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "PU-Bench controlled dataset splits with shared public backbones",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "PU-Bench recommended dataset recipes are recorded separately from the source CIFAR-10 recipe",
            alignment["benchmark_adaptations"],
        )
        self.assertTrue(config.params["use_source_hparams_by_dataset"])
        self.assertEqual(
            config.params["source_hparams_by_dataset"]["cifar10"]["source_dataset"],
            "cifar-10",
        )
        self.assertEqual(
            config.params["source_hparams_by_dataset"]["cifar10"]["P_batch_size"],
            16,
        )
        self.assertTrue(config.params["use_recommended_hparams_by_dataset"])
        self.assertEqual(
            set(config.params["recommended_hparams_by_dataset"]),
            {
                "mnist",
                "fashionmnist",
                "alzheimermri",
                "imdb",
                "20news",
                "spambase",
                "connect4",
            },
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["mnist"]["recommended_preset"],
            "image_cifar10_schedule_bce_warmup",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["imdb"]["recommended_preset"],
            "vector_cifar10_schedule_bce_warmup",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["20news"]["recommended_preset"],
            "vector_cifar10_schedule_bce_warmup",
        )
        self.assertEqual(config.params["checkpoint"]["monitor"], "val_proxy_acc")
        self.assertFalse(config.params["checkpoint"]["early_stopping"]["enabled"])

    def test_robustpu_alignment_metadata_is_structured(self) -> None:
        config = load_method_config("robustpu")

        alignment = config.metadata["alignment"]
        self.assertEqual(
            alignment["level"],
            "source_faithful_kernel_benchmark_wrapper",
        )
        self.assertFalse(alignment["source_reproduction"])
        self.assertIn(
            "source README 100 epoch nnPU pretraining budget with pre_lr=1e-3, pre_batch_size=128, and pre_wd=1e-4",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "Welsch self-paced weighting with separate P/U threshold schedulers",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "PU-Bench controlled dataset splits with shared public backbones",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "checkpoint monitor val_proxy_acc instead of source oracle validation accuracy to satisfy PU-Bench proxy-metric checkpoint policy",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "pretraining restores the best source-budget epoch by validation proxy accuracy; no test labels are used for model selection",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "oracle labels are diagnostics only; no train/validation/test oracle labels are used for RobustPU selection, calibration, losses, or sample weights",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "Stage2 keeps the source fixed-zero prediction threshold; on controlled public backbones it can improve ranking AUC while fixed-threshold F1/precision/recall collapse to all-negative predictions",
            alignment["known_limitations"],
        )
        self.assertEqual(config.params["backbone_policy"], "controlled")
        self.assertEqual(config.params["batch_size"], 128)
        self.assertTrue(config.params["restore_best_pretrain"])
        self.assertEqual(config.params["pre_train"]["epochs"], 100)
        self.assertEqual(float(config.params["pre_train"]["lr"]), 0.001)
        self.assertEqual(config.params["pre_train"]["batch_size"], 128)
        self.assertEqual(config.params["pre_train"]["monitor"], "val_proxy_acc")
        self.assertEqual(config.params["pre_train"]["calibration"], "none")
        self.assertEqual(config.params["main_train"]["epochs"], 100)
        self.assertEqual(config.params["main_train"]["inner_epochs"], 20)
        self.assertEqual(config.params["main_train"]["batch_size"], 64)
        self.assertEqual(config.params["main_train"]["scheduler_p"]["grow_steps"], 5)
        self.assertEqual(config.params["main_train"]["scheduler_n"]["grow_steps"], 5)
        self.assertEqual(config.params["main_train"]["scheduler_n"]["temper"], 1.3)
        self.assertEqual(config.params["checkpoint"]["monitor"], "val_proxy_acc")
        self.assertTrue(config.params["checkpoint"]["early_stopping"]["enabled"])
        self.assertEqual(config.params["checkpoint"]["early_stopping"]["patience"], 5)

    def test_selfpu_alignment_metadata_is_structured(self) -> None:
        config = load_method_config("selfpu")

        alignment = config.metadata["alignment"]
        self.assertEqual(
            alignment["level"],
            "source_faithful_self_calibrated_2s2t_benchmark_wrapper",
        )
        self.assertFalse(alignment["source_reproduction"])
        self.assertIn(
            "source 200 epoch training budget",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "self-calibrated sigmoid_eps noisy-batch reweighting",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "source nnPU negative-risk training scalar for noisy branches",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "source train_2s2t_mix.py MNIST and CIFAR-10 command recipes when the PU-Bench dataset matches",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "PU-Bench controlled dataset splits with shared public backbones",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "self-calibration meta target uses train/validation unlabeled inputs only, never test inputs",
            alignment["benchmark_adaptations"],
        )
        self.assertEqual(config.params["num_epochs"], 200)
        self.assertEqual(config.params["num_workers"], 4)
        self.assertTrue(config.params["use_source_hparams_by_dataset"])
        self.assertEqual(
            config.params["source_hparams_by_dataset"]["mnist"]["source_preset"],
            "train_2s2t_mix_mnist_soft_label",
        )
        self.assertEqual(
            config.params["source_hparams_by_dataset"]["cifar10"]["source_dataset"],
            "cifar",
        )
        self.assertTrue(config.params["use_recommended_hparams_by_dataset"])
        self.assertEqual(
            set(config.params["recommended_hparams_by_dataset"]),
            {"fashionmnist", "alzheimermri", "imdb", "20news", "spambase", "connect4"},
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["fashionmnist"]["recommended_preset"],
            "fashionmnist_mnist_source_schedule",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["alzheimermri"]["recommended_preset"],
            "image_source_schedule_memory_capped",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["imdb"]["recommended_preset"],
            "vector_source_schedule",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["20news"]["recommended_preset"],
            "vector_source_schedule",
        )
        self.assertTrue(config.params["self_calibration_enabled"])
        self.assertEqual(config.params["self_calibration_meta_source"], "val_unlabeled")
        self.assertEqual(config.params["self_calibration_gamma"], 0.0625)
        self.assertEqual(config.params["checkpoint"]["monitor"], "val_proxy_acc")
        self.assertFalse(config.params["checkpoint"]["early_stopping"]["enabled"])

    def test_vaepu_alignment_metadata_separates_full_and_partial_source_recipes(
        self,
    ) -> None:
        config = load_method_config("vaepu")

        alignment = config.metadata["alignment"]
        self.assertEqual(
            alignment["level"],
            "source_adapted_tensorflow_to_pytorch_benchmark_wrapper",
        )
        self.assertFalse(alignment["source_reproduction"])
        self.assertIn(
            "source MNIST_35_val flat MLP recipe from main.py",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "CIFAR10 and 20News source entries are alpha-table recipes, not exact source-script reproductions",
            alignment["known_limitations"],
        )
        self.assertTrue(config.params["use_source_hparams_by_dataset"])
        self.assertEqual(set(config.params["source_hparams_by_dataset"]), {"mnist"})
        self.assertEqual(
            set(config.params["source_partial_hparams_by_dataset"]),
            {"cifar10", "20news"},
        )
        self.assertEqual(
            config.params["source_hparams_by_dataset"]["mnist"]["source_dataset"],
            "MNIST_35_val",
        )
        self.assertEqual(
            config.params["source_hparams_by_dataset"]["mnist"]["num_epoch"],
            800,
        )
        self.assertEqual(
            config.params["source_partial_hparams_by_dataset"]["cifar10"][
                "source_recipe_scope"
            ],
            "source_readme_random_labeling_alpha_table_only",
        )
        self.assertEqual(
            float(config.params["source_partial_hparams_by_dataset"]["cifar10"]["alpha_gen"]),
            0.3,
        )
        self.assertEqual(
            float(config.params["source_partial_hparams_by_dataset"]["20news"]["alpha_gen"]),
            0.01,
        )
        self.assertEqual(config.params["checkpoint"]["monitor"], "val_proxy_acc")
        self.assertFalse(config.params["checkpoint"]["early_stopping"]["enabled"])

    def test_vpu_alignment_metadata_is_structured(self) -> None:
        config = load_method_config("vpu")

        alignment = config.metadata["alignment"]
        self.assertEqual(
            alignment["level"],
            "source_faithful_kernel_benchmark_wrapper",
        )
        self.assertFalse(alignment["source_reproduction"])
        self.assertIn(
            "separate labeled-positive P loader and all-training-data X loader",
            alignment["retained_source_components"],
        )
        self.assertIn(
            "binary log-softmax adapter for single-logit benchmark backbones",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "checkpoint monitor val_proxy_acc instead of source validation variational loss to satisfy PU-Bench proxy-metric checkpoint policy",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "PU-Bench recommended dataset recipes are recorded separately from source recipes when source hyperparameters are unavailable",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "unified benchmark oracle/proxy metrics rather than the source-script normalized test decision rule",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "source default batch_size=500 is retained and can be high-memory on large image datasets",
            alignment["known_resource_risks"],
        )
        self.assertTrue(config.params["use_source_hparams_by_dataset"])
        self.assertEqual(
            config.params["source_hparams_by_dataset"]["fashionmnist"]["lam"],
            0.3,
        )
        self.assertEqual(
            set(config.params["source_hparams_by_dataset"]),
            {"cifar10", "fashionmnist"},
        )
        self.assertTrue(config.params["use_recommended_hparams_by_dataset"])
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["mnist"]["recommended_preset"],
            "mnist_fashionmnist_style",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["alzheimermri"]["recommended_preset"],
            "image_fashionmnist_style_memory_capped",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["20news"]["recommended_preset"],
            "vector_source_grid_style",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["imdb"]["recommended_preset"],
            "vector_source_grid_style",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["spambase"]["recommended_preset"],
            "vector_source_grid_style",
        )
        self.assertEqual(
            config.params["recommended_hparams_by_dataset"]["connect4"]["recommended_preset"],
            "vector_source_grid_style",
        )
        self.assertEqual(config.params["checkpoint"]["monitor"], "val_proxy_acc")
        self.assertFalse(config.params["checkpoint"]["early_stopping"]["enabled"])

    def test_source_hparams_resolver_deep_merges_dataset_recipe(self) -> None:
        params = {
            "lr": 0.1,
            "stages": {
                "warm_up": {"epochs": 1, "lr": 0.2},
                "main": {"epochs": 2},
            },
            "use_source_hparams_by_dataset": True,
            "source_hparams_by_dataset": {
                "cifar10": {
                    "lr": 0.3,
                    "stages": {"warm_up": {"lr": 0.4}},
                }
            },
        }

        resolved = resolve_method_params_for_dataset(
            params,
            {"dataset_class": "CIFAR10"},
        )

        self.assertEqual(resolved["lr"], 0.3)
        self.assertEqual(resolved["stages"]["warm_up"]["epochs"], 1)
        self.assertEqual(resolved["stages"]["warm_up"]["lr"], 0.4)
        self.assertEqual(resolved["stages"]["main"]["epochs"], 2)
        self.assertEqual(
            resolved["source_hparams_resolved_from"],
            "source_hparams_by_dataset.cifar10",
        )

    def test_recommended_hparams_resolver_deep_merges_non_source_recipe(self) -> None:
        params = {
            "lr": 0.1,
            "batch_size": 16,
            "checkpoint": {"monitor": "val_proxy_acc", "mode": "max"},
            "use_source_hparams_by_dataset": True,
            "source_hparams_by_dataset": {
                "mnist": {"lr": 0.2},
            },
            "use_recommended_hparams_by_dataset": True,
            "recommended_hparams_by_dataset": {
                "fashionmnist": {
                    "lr": 0.3,
                    "checkpoint": {"mode": "min"},
                }
            },
        }

        resolved = resolve_method_params_for_dataset(
            params,
            {"dataset_class": "FashionMNIST"},
        )

        self.assertEqual(resolved["lr"], 0.3)
        self.assertEqual(resolved["batch_size"], 16)
        self.assertEqual(resolved["checkpoint"]["monitor"], "val_proxy_acc")
        self.assertEqual(resolved["checkpoint"]["mode"], "min")
        self.assertNotIn("source_hparams_resolved_from", resolved)
        self.assertEqual(
            resolved["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.fashionmnist",
        )

    def test_source_hparams_take_precedence_over_recommended_hparams(self) -> None:
        params = {
            "lr": 0.1,
            "use_source_hparams_by_dataset": True,
            "source_hparams_by_dataset": {
                "cifar10": {"lr": 0.2},
            },
            "use_recommended_hparams_by_dataset": True,
            "recommended_hparams_by_dataset": {
                "cifar10": {"lr": 0.3},
            },
        }

        resolved = resolve_method_params_for_dataset(
            params,
            {"dataset_class": "CIFAR10"},
        )

        self.assertEqual(resolved["lr"], 0.2)
        self.assertEqual(
            resolved["source_hparams_resolved_from"],
            "source_hparams_by_dataset.cifar10",
        )
        self.assertNotIn("recommended_hparams_resolved_from", resolved)

    def test_plan_applies_nnpu_source_presets_on_supported_datasets(self) -> None:
        mnist_plan = build_plan(
            dataset_config_path="config/datasets_smoke/param_sweep_mnist_seed2.yaml",
            methods=["nnpu"],
            methods_dir="config/methods",
        )
        mnist_run = mnist_plan.runs[0]
        self.assertEqual(mnist_run.params["source_preset"], "exp-mnist")
        self.assertEqual(mnist_run.params["batch_size"], 30000)
        self.assertEqual(mnist_run.params["num_epochs"], 100)
        self.assertEqual(float(mnist_run.params["lr"]), 0.001)
        self.assertEqual(
            mnist_run.params["source_hparams_resolved_from"],
            "source_hparams_by_dataset.mnist",
        )

        cifar_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_cifar10.yaml",
            methods=["nnpu"],
            methods_dir="config/methods",
        )
        cifar_run = cifar_plan.runs[0]
        self.assertEqual(cifar_run.params["source_preset"], "exp-cifar")
        self.assertEqual(cifar_run.params["batch_size"], 500)
        self.assertEqual(cifar_run.params["num_epochs"], 100)
        self.assertEqual(float(cifar_run.params["lr"]), 0.00001)
        self.assertEqual(
            cifar_run.params["source_hparams_resolved_from"],
            "source_hparams_by_dataset.cifar10",
        )

    def test_plan_applies_nnpu_recommended_fashionmnist_recipe(self) -> None:
        fashion_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_fashionmnist.yaml",
            methods=["nnpu"],
            methods_dir="config/methods",
        )
        fashion_run = fashion_plan.runs[0]

        self.assertEqual(fashion_run.params["recommended_preset"], "fashionmnist_mnist_style")
        self.assertEqual(fashion_run.params["batch_size"], 500)
        self.assertEqual(fashion_run.params["num_epochs"], 100)
        self.assertEqual(float(fashion_run.params["lr"]), 0.001)
        self.assertEqual(
            fashion_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.fashionmnist",
        )
        self.assertNotIn("source_hparams_resolved_from", fashion_run.params)

    def test_plan_applies_nnpu_recommended_alzheimermri_recipe(self) -> None:
        alzheimer_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_alzheimermri.yaml",
            methods=["nnpu"],
            methods_dir="config/methods",
        )
        alzheimer_run = alzheimer_plan.runs[0]

        self.assertEqual(alzheimer_run.params["recommended_preset"], "image_default_baseline")
        self.assertEqual(alzheimer_run.params["batch_size"], 256)
        self.assertEqual(alzheimer_run.params["num_epochs"], 40)
        self.assertEqual(float(alzheimer_run.params["lr"]), 0.0003)
        self.assertEqual(
            alzheimer_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.alzheimermri",
        )
        self.assertNotIn("source_hparams_resolved_from", alzheimer_run.params)

    def test_plan_applies_nnpu_recommended_imdb_recipe(self) -> None:
        imdb_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_imdb_sbert.yaml",
            methods=["nnpu"],
            methods_dir="config/methods",
        )
        imdb_run = imdb_plan.runs[0]

        self.assertEqual(
            imdb_run.params["recommended_preset"],
            "text_embedding_default_baseline",
        )
        self.assertEqual(imdb_run.params["batch_size"], 256)
        self.assertEqual(imdb_run.params["num_epochs"], 40)
        self.assertEqual(float(imdb_run.params["lr"]), 0.0003)
        self.assertEqual(
            imdb_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.imdb",
        )
        self.assertNotIn("source_hparams_resolved_from", imdb_run.params)

    def test_plan_applies_nnpu_recommended_20news_recipe(self) -> None:
        news_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_20news_sbert.yaml",
            methods=["nnpu"],
            methods_dir="config/methods",
        )
        news_run = news_plan.runs[0]

        self.assertEqual(
            news_run.params["recommended_preset"],
            "text_embedding_default_baseline",
        )
        self.assertEqual(news_run.params["batch_size"], 256)
        self.assertEqual(news_run.params["num_epochs"], 40)
        self.assertEqual(float(news_run.params["lr"]), 0.0003)
        self.assertEqual(
            news_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.20news",
        )
        self.assertNotIn("source_hparams_resolved_from", news_run.params)

    def test_plan_applies_nnpu_recommended_spambase_recipe(self) -> None:
        spambase_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_spambase.yaml",
            methods=["nnpu"],
            methods_dir="config/methods",
        )
        spambase_run = spambase_plan.runs[0]

        self.assertEqual(
            spambase_run.params["recommended_preset"],
            "tabular_default_baseline",
        )
        self.assertEqual(spambase_run.params["batch_size"], 256)
        self.assertEqual(spambase_run.params["num_epochs"], 40)
        self.assertEqual(float(spambase_run.params["lr"]), 0.0003)
        self.assertEqual(
            spambase_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.spambase",
        )
        self.assertNotIn("source_hparams_resolved_from", spambase_run.params)

    def test_plan_applies_nnpu_recommended_connect4_recipe(self) -> None:
        connect4_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_connect4.yaml",
            methods=["nnpu"],
            methods_dir="config/methods",
        )
        connect4_run = connect4_plan.runs[0]

        self.assertEqual(
            connect4_run.params["recommended_preset"],
            "tabular_default_baseline",
        )
        self.assertEqual(connect4_run.params["batch_size"], 256)
        self.assertEqual(connect4_run.params["num_epochs"], 40)
        self.assertEqual(float(connect4_run.params["lr"]), 0.0003)
        self.assertEqual(
            connect4_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.connect4",
        )
        self.assertNotIn("source_hparams_resolved_from", connect4_run.params)

    def test_plan_applies_nnpusb_source_mnist_recipe(self) -> None:
        mnist_plan = build_plan(
            dataset_config_path="config/datasets_smoke/param_sweep_mnist_seed2.yaml",
            methods=["nnpusb"],
            methods_dir="config/methods",
        )
        mnist_run = mnist_plan.runs[0]
        self.assertEqual(mnist_run.params["source_preset"], "main_nnPUSB_mnist")
        self.assertEqual(mnist_run.params["batch_size"], 1000)
        self.assertEqual(mnist_run.params["num_epochs"], 100)
        self.assertEqual(float(mnist_run.params["lr"]), 0.00001)
        self.assertEqual(float(mnist_run.params["weight_decay"]), 0.005)
        self.assertEqual(
            mnist_run.params["source_hparams_resolved_from"],
            "source_hparams_by_dataset.mnist",
        )

    def test_plan_applies_nnpusb_recommended_cifar10_recipe(self) -> None:
        cifar_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_cifar10.yaml",
            methods=["nnpusb"],
            methods_dir="config/methods",
        )
        cifar_run = cifar_plan.runs[0]

        self.assertEqual(cifar_run.params["recommended_preset"], "cifar10_nnpu_style")
        self.assertEqual(cifar_run.params["batch_size"], 500)
        self.assertEqual(cifar_run.params["num_epochs"], 100)
        self.assertEqual(float(cifar_run.params["lr"]), 0.00001)
        self.assertEqual(float(cifar_run.params["weight_decay"]), 0.005)
        self.assertEqual(
            cifar_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.cifar10",
        )
        self.assertNotIn("source_hparams_resolved_from", cifar_run.params)

    def test_plan_applies_nnpusb_recommended_alzheimermri_recipe(self) -> None:
        alzheimer_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_alzheimermri.yaml",
            methods=["nnpusb"],
            methods_dir="config/methods",
        )
        alzheimer_run = alzheimer_plan.runs[0]

        self.assertEqual(alzheimer_run.params["recommended_preset"], "image_default_baseline")
        self.assertEqual(alzheimer_run.params["batch_size"], 256)
        self.assertEqual(alzheimer_run.params["num_epochs"], 40)
        self.assertEqual(float(alzheimer_run.params["lr"]), 0.0003)
        self.assertEqual(float(alzheimer_run.params["weight_decay"]), 0.0001)
        self.assertEqual(
            alzheimer_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.alzheimermri",
        )
        self.assertNotIn("source_hparams_resolved_from", alzheimer_run.params)

    def test_plan_applies_nnpusb_recommended_fashionmnist_recipe(self) -> None:
        fashion_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_fashionmnist.yaml",
            methods=["nnpusb"],
            methods_dir="config/methods",
        )
        fashion_run = fashion_plan.runs[0]

        self.assertEqual(fashion_run.params["recommended_preset"], "fashionmnist_mnist_style")
        self.assertEqual(fashion_run.params["batch_size"], 1000)
        self.assertEqual(fashion_run.params["num_epochs"], 100)
        self.assertEqual(float(fashion_run.params["lr"]), 0.00001)
        self.assertEqual(float(fashion_run.params["weight_decay"]), 0.005)
        self.assertEqual(
            fashion_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.fashionmnist",
        )
        self.assertNotIn("source_hparams_resolved_from", fashion_run.params)

    def test_plan_applies_nnpusb_recommended_imdb_recipe(self) -> None:
        imdb_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_imdb_sbert.yaml",
            methods=["nnpusb"],
            methods_dir="config/methods",
        )
        imdb_run = imdb_plan.runs[0]

        self.assertEqual(
            imdb_run.params["recommended_preset"],
            "text_embedding_default_baseline",
        )
        self.assertEqual(imdb_run.params["batch_size"], 256)
        self.assertEqual(imdb_run.params["num_epochs"], 40)
        self.assertEqual(float(imdb_run.params["lr"]), 0.0003)
        self.assertEqual(float(imdb_run.params["weight_decay"]), 0.0001)
        self.assertEqual(
            imdb_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.imdb",
        )
        self.assertNotIn("source_hparams_resolved_from", imdb_run.params)

    def test_plan_applies_nnpusb_recommended_20news_recipe(self) -> None:
        news_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_20news_sbert.yaml",
            methods=["nnpusb"],
            methods_dir="config/methods",
        )
        news_run = news_plan.runs[0]

        self.assertEqual(
            news_run.params["recommended_preset"],
            "text_embedding_default_baseline",
        )
        self.assertEqual(news_run.params["batch_size"], 256)
        self.assertEqual(news_run.params["num_epochs"], 40)
        self.assertEqual(float(news_run.params["lr"]), 0.0003)
        self.assertEqual(float(news_run.params["weight_decay"]), 0.0001)
        self.assertEqual(
            news_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.20news",
        )
        self.assertNotIn("source_hparams_resolved_from", news_run.params)

    def test_plan_applies_nnpusb_recommended_spambase_recipe(self) -> None:
        spambase_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_spambase.yaml",
            methods=["nnpusb"],
            methods_dir="config/methods",
        )
        spambase_run = spambase_plan.runs[0]

        self.assertEqual(
            spambase_run.params["recommended_preset"],
            "tabular_default_baseline",
        )
        self.assertEqual(spambase_run.params["batch_size"], 256)
        self.assertEqual(spambase_run.params["num_epochs"], 40)
        self.assertEqual(float(spambase_run.params["lr"]), 0.0003)
        self.assertEqual(float(spambase_run.params["weight_decay"]), 0.0001)
        self.assertEqual(
            spambase_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.spambase",
        )
        self.assertNotIn("source_hparams_resolved_from", spambase_run.params)

    def test_plan_applies_nnpusb_recommended_connect4_recipe(self) -> None:
        connect4_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_connect4.yaml",
            methods=["nnpusb"],
            methods_dir="config/methods",
        )
        connect4_run = connect4_plan.runs[0]

        self.assertEqual(
            connect4_run.params["recommended_preset"],
            "tabular_default_baseline",
        )
        self.assertEqual(connect4_run.params["batch_size"], 256)
        self.assertEqual(connect4_run.params["num_epochs"], 40)
        self.assertEqual(float(connect4_run.params["lr"]), 0.0003)
        self.assertEqual(float(connect4_run.params["weight_decay"]), 0.0001)
        self.assertEqual(
            connect4_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.connect4",
        )
        self.assertNotIn("source_hparams_resolved_from", connect4_run.params)

    def test_plan_applies_vpu_source_readme_recipe_on_supported_datasets(self) -> None:
        fashion_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_fashionmnist.yaml",
            methods=["vpu"],
            methods_dir="config/methods",
        )
        fashion_run = fashion_plan.runs[0]
        self.assertEqual(float(fashion_run.params["lr"]), 0.0003)
        self.assertEqual(float(fashion_run.params["lam"]), 0.3)
        self.assertEqual(fashion_run.params["source_num_labeled"], 3000)
        self.assertEqual(
            fashion_run.params["source_hparams_resolved_from"],
            "source_hparams_by_dataset.fashionmnist",
        )

    def test_plan_applies_vpu_recommended_mnist_recipe(self) -> None:
        mnist_plan = build_plan(
            dataset_config_path="config/datasets_smoke/param_sweep_mnist_seed2.yaml",
            methods=["vpu"],
            methods_dir="config/methods",
        )
        mnist_run = mnist_plan.runs[0]

        self.assertEqual(mnist_run.params["recommended_preset"], "mnist_fashionmnist_style")
        self.assertEqual(mnist_run.params["batch_size"], 500)
        self.assertEqual(mnist_run.params["num_epochs"], 50)
        self.assertEqual(float(mnist_run.params["lr"]), 0.0003)
        self.assertEqual(float(mnist_run.params["lam"]), 0.3)
        self.assertEqual(
            mnist_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.mnist",
        )
        self.assertNotIn("source_hparams_resolved_from", mnist_run.params)

    def test_plan_applies_vpu_recommended_image_recipe(self) -> None:
        alzheimer_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_alzheimermri.yaml",
            methods=["vpu"],
            methods_dir="config/methods",
        )
        alzheimer_run = alzheimer_plan.runs[0]

        self.assertEqual(
            alzheimer_run.params["recommended_preset"],
            "image_fashionmnist_style_memory_capped",
        )
        self.assertEqual(alzheimer_run.params["batch_size"], 64)
        self.assertEqual(alzheimer_run.params["num_epochs"], 50)
        self.assertEqual(float(alzheimer_run.params["lr"]), 0.0003)
        self.assertEqual(float(alzheimer_run.params["lam"]), 0.3)
        self.assertEqual(
            alzheimer_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.alzheimermri",
        )
        self.assertNotIn("source_hparams_resolved_from", alzheimer_run.params)

    def test_plan_applies_vpu_recommended_vector_recipes(self) -> None:
        dataset_cases = {
            "20news": "config/datasets_typical/param_sweep_20news_sbert.yaml",
            "imdb": "config/datasets_typical/param_sweep_imdb_sbert.yaml",
            "spambase": "config/datasets_typical/param_sweep_spambase.yaml",
            "connect4": "config/datasets_typical/param_sweep_connect4.yaml",
        }

        for dataset_key, dataset_config_path in dataset_cases.items():
            with self.subTest(dataset=dataset_key):
                plan = build_plan(
                    dataset_config_path=dataset_config_path,
                    methods=["vpu"],
                    methods_dir="config/methods",
                )
                run = plan.runs[0]

                self.assertEqual(run.params["recommended_preset"], "vector_source_grid_style")
                self.assertEqual(run.params["batch_size"], 500)
                self.assertEqual(run.params["num_epochs"], 50)
                self.assertEqual(float(run.params["lr"]), 0.0003)
                self.assertEqual(float(run.params["lam"]), 0.1)
                self.assertEqual(
                    run.params["recommended_hparams_resolved_from"],
                    f"recommended_hparams_by_dataset.{dataset_key}",
                )
                self.assertNotIn("source_hparams_resolved_from", run.params)

    def test_plan_applies_distpu_source_recipe_on_supported_datasets(self) -> None:
        fashion_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_fashionmnist.yaml",
            methods=["distpu"],
            methods_dir="config/methods",
        )
        fashion_run = fashion_plan.runs[0]
        self.assertEqual(fashion_run.params["source_dataset"], "fmnist")
        self.assertEqual(fashion_run.params["source_num_labeled"], 1000)
        self.assertEqual(fashion_run.params["stages"]["warm_up"]["epochs"], 60)
        self.assertEqual(float(fashion_run.params["stages"]["mixup"]["lr"]), 0.00005)
        self.assertEqual(
            fashion_run.params["source_hparams_resolved_from"],
            "source_hparams_by_dataset.fashionmnist",
        )

        alzheimer_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_alzheimermri.yaml",
            methods=["distpu"],
            methods_dir="config/methods",
        )
        alzheimer_run = alzheimer_plan.runs[0]
        self.assertEqual(alzheimer_run.params["source_dataset"], "alzheimer")
        self.assertEqual(
            alzheimer_run.params["source_hparams_resolved_from"],
            "source_hparams_by_dataset.alzheimermri",
        )

    def test_plan_applies_distpu_recommended_recipes_on_source_unsupported_datasets(
        self,
    ) -> None:
        mnist_plan = build_plan(
            dataset_config_path="config/datasets_smoke/param_sweep_mnist_seed2.yaml",
            methods=["distpu"],
            methods_dir="config/methods",
        )
        mnist_run = mnist_plan.runs[0]
        self.assertEqual(
            mnist_run.params["recommended_preset"],
            "mnist_fashionmnist_source_schedule",
        )
        self.assertEqual(mnist_run.params["stages"]["warm_up"]["epochs"], 60)
        self.assertEqual(mnist_run.params["stages"]["mixup"]["epochs"], 60)
        self.assertEqual(float(mnist_run.params["stages"]["mixup"]["lr"]), 0.00005)
        self.assertEqual(
            mnist_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.mnist",
        )
        self.assertNotIn("source_hparams_resolved_from", mnist_run.params)

        dataset_cases = {
            "imdb": "config/datasets_typical/param_sweep_imdb_sbert.yaml",
            "20news": "config/datasets_typical/param_sweep_20news_sbert.yaml",
            "spambase": "config/datasets_typical/param_sweep_spambase.yaml",
            "connect4": "config/datasets_typical/param_sweep_connect4.yaml",
        }
        for dataset_key, dataset_config_path in dataset_cases.items():
            with self.subTest(dataset=dataset_key):
                plan = build_plan(
                    dataset_config_path=dataset_config_path,
                    methods=["distpu"],
                    methods_dir="config/methods",
                )
                run = plan.runs[0]
                self.assertEqual(run.params["recommended_preset"], "vector_source_schedule")
                self.assertEqual(run.params["batch_size"], 256)
                self.assertEqual(run.params["test_batch_size"], 128)
                self.assertEqual(run.params["stages"]["warm_up"]["epochs"], 60)
                self.assertEqual(run.params["stages"]["mixup"]["epochs"], 60)
                self.assertEqual(
                    run.params["recommended_hparams_resolved_from"],
                    f"recommended_hparams_by_dataset.{dataset_key}",
                )
                self.assertNotIn("source_hparams_resolved_from", run.params)

    def test_plan_applies_holisticpu_source_recipes_on_supported_datasets(self) -> None:
        cifar_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_cifar10.yaml",
            methods=["holisticpu"],
            methods_dir="config/methods",
        )
        cifar_run = cifar_plan.runs[0]
        self.assertEqual(cifar_run.params["source_dataset"], "cifar10_1")
        self.assertEqual(cifar_run.params["source_num_labeled"], 1000)
        self.assertEqual(float(cifar_run.params["lr"]), 0.0015)
        self.assertEqual(cifar_run.params["batch_size"], 64)
        self.assertEqual(cifar_run.params["phase1_epochs"], 15)
        self.assertEqual(cifar_run.params["phase2_epochs"], 25)
        self.assertEqual(
            cifar_run.params["source_hparams_resolved_from"],
            "source_hparams_by_dataset.cifar10",
        )

        fashion_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_fashionmnist.yaml",
            methods=["holisticpu"],
            methods_dir="config/methods",
        )
        fashion_run = fashion_plan.runs[0]
        self.assertEqual(fashion_run.params["source_dataset"], "fmnist")
        self.assertFalse(fashion_run.params["source_label_split_exact_match"])
        self.assertEqual(float(fashion_run.params["lr"]), 0.002)
        self.assertEqual(
            fashion_run.params["source_hparams_resolved_from"],
            "source_hparams_by_dataset.fashionmnist",
        )

        alzheimer_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_alzheimermri.yaml",
            methods=["holisticpu"],
            methods_dir="config/methods",
        )
        alzheimer_run = alzheimer_plan.runs[0]
        self.assertEqual(alzheimer_run.params["source_dataset"], "alzheimer")
        self.assertEqual(alzheimer_run.params["source_num_labeled"], 769)
        self.assertEqual(float(alzheimer_run.params["lr"]), 0.0005)
        self.assertEqual(alzheimer_run.params["batch_size"], 16)
        self.assertEqual(alzheimer_run.params["eval_step"], 100)
        self.assertEqual(alzheimer_run.params["phase1_epochs"], 10)
        self.assertEqual(alzheimer_run.params["phase2_epochs"], 20)
        self.assertEqual(
            alzheimer_run.params["source_hparams_resolved_from"],
            "source_hparams_by_dataset.alzheimermri",
        )

    def test_plan_applies_holisticpu_recommended_recipes_on_source_unsupported_datasets(
        self,
    ) -> None:
        mnist_plan = build_plan(
            dataset_config_path="config/datasets_smoke/param_sweep_mnist_seed2.yaml",
            methods=["holisticpu"],
            methods_dir="config/methods",
        )
        mnist_run = mnist_plan.runs[0]
        self.assertEqual(mnist_run.params["recommended_preset"], "mnist_fashionmnist_style")
        self.assertEqual(float(mnist_run.params["lr"]), 0.002)
        self.assertEqual(mnist_run.params["batch_size"], 64)
        self.assertEqual(mnist_run.params["phase1_epochs"], 15)
        self.assertEqual(mnist_run.params["phase2_epochs"], 25)
        self.assertEqual(
            mnist_run.params["recommended_hparams_resolved_from"],
            "recommended_hparams_by_dataset.mnist",
        )
        self.assertNotIn("source_hparams_resolved_from", mnist_run.params)

        dataset_cases = {
            "imdb": "config/datasets_typical/param_sweep_imdb_sbert.yaml",
            "20news": "config/datasets_typical/param_sweep_20news_sbert.yaml",
            "spambase": "config/datasets_typical/param_sweep_spambase.yaml",
            "connect4": "config/datasets_typical/param_sweep_connect4.yaml",
        }
        for dataset_key, dataset_config_path in dataset_cases.items():
            with self.subTest(dataset=dataset_key):
                plan = build_plan(
                    dataset_config_path=dataset_config_path,
                    methods=["holisticpu"],
                    methods_dir="config/methods",
                )
                run = plan.runs[0]
                self.assertEqual(run.params["recommended_preset"], "vector_default_short_eval")
                self.assertEqual(run.params["eval_step"], 128)
                self.assertEqual(float(run.params["lr"]), 0.01)
                self.assertEqual(
                    run.params["recommended_hparams_resolved_from"],
                    f"recommended_hparams_by_dataset.{dataset_key}",
                )
                self.assertNotIn("source_hparams_resolved_from", run.params)

    def test_plan_applies_pulda_source_recipe_on_supported_datasets(self) -> None:
        cifar_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_cifar10.yaml",
            methods=["pulda"],
            methods_dir="config/methods",
        )
        cifar_run = cifar_plan.runs[0]
        self.assertEqual(cifar_run.params["source_dataset"], "cifar-10")
        self.assertEqual(cifar_run.params["source_num_labeled"], 1000)
        self.assertEqual(float(cifar_run.params["warm_up_lr"]), 0.0001)
        self.assertEqual(float(cifar_run.params["lr"]), 0.001)
        self.assertEqual(cifar_run.params["P_batch_size"], 16)
        self.assertEqual(
            cifar_run.params["source_hparams_resolved_from"],
            "source_hparams_by_dataset.cifar10",
        )
        self.assertNotIn("recommended_hparams_resolved_from", cifar_run.params)

    def test_plan_applies_pulda_recommended_recipes_on_source_unsupported_datasets(
        self,
    ) -> None:
        image_cases = {
            "mnist": "config/datasets_smoke/param_sweep_mnist_seed2.yaml",
            "fashionmnist": "tests/fixtures/dataset_configs/source_hparams_fashionmnist_seed2_dataset.yaml",
            "alzheimermri": "tests/fixtures/dataset_configs/recommended_hparams_alzheimermri_seed2_dataset.yaml",
        }
        for dataset_key, dataset_config_path in image_cases.items():
            with self.subTest(dataset=dataset_key):
                plan = build_plan(
                    dataset_config_path=dataset_config_path,
                    methods=["pulda"],
                    methods_dir="config/methods",
                )
                run = plan.runs[0]
                self.assertEqual(
                    run.params["recommended_preset"],
                    "image_cifar10_schedule_bce_warmup",
                )
                self.assertEqual(run.params["warmup_loss"], "bce")
                self.assertEqual(float(run.params["warm_up_lr"]), 0.0001)
                self.assertEqual(float(run.params["lr"]), 0.001)
                self.assertEqual(run.params["warm_up_epochs"], 60)
                self.assertEqual(run.params["pu_epochs"], 60)
                self.assertEqual(run.params["P_batch_size"], 16)
                self.assertEqual(run.params["U_batch_size"], 128)
                self.assertEqual(
                    run.params["recommended_hparams_resolved_from"],
                    f"recommended_hparams_by_dataset.{dataset_key}",
                )
                self.assertNotIn("source_hparams_resolved_from", run.params)

        vector_cases = {
            "imdb": "tests/fixtures/dataset_configs/recommended_hparams_imdb_seed2_dataset.yaml",
            "20news": "config/datasets_typical/param_sweep_20news_sbert.yaml",
            "spambase": "tests/fixtures/dataset_configs/recommended_hparams_spambase_seed2_dataset.yaml",
            "connect4": "tests/fixtures/dataset_configs/recommended_hparams_connect4_seed2_dataset.yaml",
        }
        for dataset_key, dataset_config_path in vector_cases.items():
            with self.subTest(dataset=dataset_key):
                plan = build_plan(
                    dataset_config_path=dataset_config_path,
                    methods=["pulda"],
                    methods_dir="config/methods",
                )
                run = plan.runs[0]
                self.assertEqual(
                    run.params["recommended_preset"],
                    "vector_cifar10_schedule_bce_warmup",
                )
                self.assertEqual(run.params["warmup_loss"], "bce")
                self.assertEqual(float(run.params["warm_up_lr"]), 0.0001)
                self.assertEqual(float(run.params["lr"]), 0.001)
                self.assertEqual(run.params["warm_up_epochs"], 60)
                self.assertEqual(run.params["pu_epochs"], 60)
                self.assertEqual(run.params["P_batch_size"], 16)
                self.assertEqual(run.params["U_batch_size"], 128)
                self.assertEqual(
                    run.params["recommended_hparams_resolved_from"],
                    f"recommended_hparams_by_dataset.{dataset_key}",
                )
                self.assertNotIn("source_hparams_resolved_from", run.params)

    def test_plan_applies_selfpu_source_recipes_on_supported_datasets(self) -> None:
        mnist_plan = build_plan(
            dataset_config_path="config/datasets_smoke/param_sweep_mnist_seed2.yaml",
            methods=["selfpu"],
            methods_dir="config/methods",
        )
        mnist_run = mnist_plan.runs[0]
        self.assertEqual(
            mnist_run.params["source_preset"],
            "train_2s2t_mix_mnist_soft_label",
        )
        self.assertEqual(mnist_run.params["batch_size"], 256)
        self.assertEqual(mnist_run.params["num_epochs"], 200)
        self.assertEqual(float(mnist_run.params["lr"]), 0.0005)
        self.assertEqual(float(mnist_run.params["self_calibration_gamma"]), 0.0625)
        self.assertEqual(mnist_run.params["source_num_labeled"], 1000)
        self.assertEqual(
            mnist_run.params["source_hparams_resolved_from"],
            "source_hparams_by_dataset.mnist",
        )

        cifar_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_cifar10.yaml",
            methods=["selfpu"],
            methods_dir="config/methods",
        )
        cifar_run = cifar_plan.runs[0]
        self.assertEqual(
            cifar_run.params["source_preset"],
            "train_2s2t_mix_cifar_soft_label",
        )
        self.assertEqual(cifar_run.params["source_dataset"], "cifar")
        self.assertEqual(cifar_run.params["source_unlabeled"], 50000)
        self.assertTrue(cifar_run.params["mean_teacher_enabled"])
        self.assertEqual(
            cifar_run.params["source_hparams_resolved_from"],
            "source_hparams_by_dataset.cifar10",
        )

    def test_plan_applies_selfpu_recommended_recipes_on_source_unsupported_datasets(
        self,
    ) -> None:
        image_cases = {
            "fashionmnist": (
                "tests/fixtures/dataset_configs/source_hparams_fashionmnist_seed2_dataset.yaml",
                "fashionmnist_mnist_source_schedule",
                256,
            ),
            "alzheimermri": (
                "tests/fixtures/dataset_configs/recommended_hparams_alzheimermri_seed2_dataset.yaml",
                "image_source_schedule_memory_capped",
                64,
            ),
        }
        for dataset_key, (
            dataset_config_path,
            recommended_preset,
            batch_size,
        ) in image_cases.items():
            with self.subTest(dataset=dataset_key):
                plan = build_plan(
                    dataset_config_path=dataset_config_path,
                    methods=["selfpu"],
                    methods_dir="config/methods",
                )
                run = plan.runs[0]
                self.assertEqual(run.params["recommended_preset"], recommended_preset)
                self.assertEqual(run.params["batch_size"], batch_size)
                self.assertEqual(run.params["num_epochs"], 200)
                self.assertEqual(float(run.params["lr"]), 0.0005)
                self.assertEqual(float(run.params["weight_decay"]), 0.005)
                self.assertTrue(run.params["self_calibration_enabled"])
                self.assertEqual(
                    float(run.params["self_calibration_gamma"]),
                    0.0625,
                )
                self.assertTrue(run.params["mean_teacher_enabled"])
                self.assertEqual(
                    run.params["recommended_hparams_resolved_from"],
                    f"recommended_hparams_by_dataset.{dataset_key}",
                )
                self.assertNotIn("source_hparams_resolved_from", run.params)

        vector_cases = {
            "imdb": "tests/fixtures/dataset_configs/recommended_hparams_imdb_seed2_dataset.yaml",
            "20news": "config/datasets_typical/param_sweep_20news_sbert.yaml",
            "spambase": "tests/fixtures/dataset_configs/recommended_hparams_spambase_seed2_dataset.yaml",
            "connect4": "tests/fixtures/dataset_configs/recommended_hparams_connect4_seed2_dataset.yaml",
        }
        for dataset_key, dataset_config_path in vector_cases.items():
            with self.subTest(dataset=dataset_key):
                plan = build_plan(
                    dataset_config_path=dataset_config_path,
                    methods=["selfpu"],
                    methods_dir="config/methods",
                )
                run = plan.runs[0]
                self.assertEqual(run.params["recommended_preset"], "vector_source_schedule")
                self.assertEqual(run.params["batch_size"], 256)
                self.assertEqual(run.params["num_epochs"], 200)
                self.assertEqual(float(run.params["lr"]), 0.0005)
                self.assertEqual(float(run.params["weight_decay"]), 0.005)
                self.assertTrue(run.params["self_calibration_enabled"])
                self.assertEqual(run.params["self_paced_start"], 10)
                self.assertEqual(run.params["mean_teacher_start"], 50)
                self.assertEqual(
                    run.params["recommended_hparams_resolved_from"],
                    f"recommended_hparams_by_dataset.{dataset_key}",
                )
                self.assertNotIn("source_hparams_resolved_from", run.params)

    def test_plan_applies_vaepu_source_mnist_recipe_only(self) -> None:
        mnist_plan = build_plan(
            dataset_config_path="config/datasets_smoke/param_sweep_mnist_seed2.yaml",
            methods=["vaepu"],
            methods_dir="config/methods",
        )
        mnist_run = mnist_plan.runs[0]
        self.assertFalse(mnist_run.params["adaptive_config"])
        self.assertTrue(mnist_run.params["mnist_source_recipe"])
        self.assertEqual(mnist_run.params["source_dataset"], "MNIST_35_val")
        self.assertEqual(mnist_run.params["batch_size_l"], 10)
        self.assertEqual(mnist_run.params["batch_size_u"], 990)
        self.assertEqual(mnist_run.params["n_h_y"], 10)
        self.assertEqual(mnist_run.params["n_h_o"], 2)
        self.assertEqual(float(mnist_run.params["lr_pn"]), 0.00001)
        self.assertEqual(mnist_run.params["num_epoch_pre"], 100)
        self.assertEqual(mnist_run.params["num_epoch"], 800)
        self.assertEqual(
            mnist_run.params["source_hparams_resolved_from"],
            "source_hparams_by_dataset.mnist",
        )

        non_full_source_cases = {
            "cifar10": "config/datasets_typical/param_sweep_cifar10.yaml",
            "20news": "config/datasets_typical/param_sweep_20news_sbert.yaml",
        }
        for dataset_key, dataset_config_path in non_full_source_cases.items():
            with self.subTest(dataset=dataset_key):
                plan = build_plan(
                    dataset_config_path=dataset_config_path,
                    methods=["vaepu"],
                    methods_dir="config/methods",
                )
                run = plan.runs[0]
                self.assertTrue(run.params["adaptive_config"])
                self.assertNotIn("source_hparams_resolved_from", run.params)
                self.assertNotIn("source_recipe_scope", run.params)

    def test_plan_applies_puet_source_mnist_recipe_only(self) -> None:
        mnist_plan = build_plan(
            dataset_config_path="config/datasets_smoke/param_sweep_mnist_seed2.yaml",
            methods=["puet"],
            methods_dir="config/methods",
        )
        mnist_run = mnist_plan.runs[0]
        self.assertEqual(mnist_run.params["source_script"], "run_puet_simple.py")
        self.assertEqual(mnist_run.params["source_dataset"], "OpenML mnist_784_even_vs_odd")
        self.assertTrue(mnist_run.params["source_label_split_exact_match"])
        self.assertEqual(mnist_run.params["source_protocol_labeled_positives"], 1000)
        self.assertEqual(mnist_run.params["n_estimators"], 100)
        self.assertEqual(mnist_run.params["risk_estimator"], "nnPU")
        self.assertEqual(mnist_run.params["loss"], "quadratic")
        self.assertIsNone(mnist_run.params["max_depth"])
        self.assertEqual(mnist_run.params["min_samples_leaf"], 1)
        self.assertEqual(mnist_run.params["max_features"], "sqrt")
        self.assertEqual(mnist_run.params["max_candidates"], 1)
        self.assertEqual(mnist_run.params["n_jobs"], 1)
        self.assertEqual(
            mnist_run.params["source_hparams_resolved_from"],
            "source_hparams_by_dataset.mnist",
        )

        cifar_plan = build_plan(
            dataset_config_path="config/datasets_typical/param_sweep_cifar10.yaml",
            methods=["puet"],
            methods_dir="config/methods",
        )
        cifar_run = cifar_plan.runs[0]
        self.assertEqual(cifar_run.params["n_estimators"], 30)
        self.assertEqual(cifar_run.params["max_depth"], 30)
        self.assertNotIn("source_hparams_resolved_from", cifar_run.params)
        self.assertNotIn("source_script", cifar_run.params)

    def test_method_metadata_reaches_plan_and_run_params(self) -> None:
        plan = build_plan(
            dataset_config_path="config/datasets_smoke/param_sweep_mnist_seed2.yaml",
            methods=[
                "bbepu",
                "cgenpu",
                "distpu",
                "holisticpu",
                "lbe",
                "nnpu",
                "nnpusb",
                "pulcpbf",
                "selfpu",
                "vpu",
            ],
            methods_dir="config/methods",
        )

        run = plan.runs[0]
        self.assertEqual(
            run.method_metadata["alignment"]["level"],
            "controlled_adaptation",
        )
        self.assertEqual(run.params["method_metadata"], run.method_metadata)

        exported = plan_to_dict(plan)
        self.assertEqual(
            exported["runs"][0]["method_metadata"]["alignment"]["level"],
            "controlled_adaptation",
        )
        self.assertEqual(
            exported["runs"][1]["method_metadata"]["alignment"]["level"],
            "source_adapted_kernel_benchmark_wrapper",
        )
        self.assertEqual(
            exported["runs"][2]["method_metadata"]["alignment"]["level"],
            "source_faithful_kernel_benchmark_wrapper",
        )
        self.assertEqual(
            exported["runs"][3]["method_metadata"]["alignment"]["level"],
            "mixed_controlled_adaptation",
        )
        self.assertEqual(
            exported["runs"][4]["method_metadata"]["alignment"]["level"],
            "controlled_adaptation",
        )
        self.assertEqual(
            exported["runs"][5]["method_metadata"]["alignment"]["level"],
            "source_faithful_kernel_benchmark_wrapper",
        )
        self.assertEqual(
            exported["runs"][6]["method_metadata"]["alignment"]["level"],
            "source_faithful_kernel_benchmark_wrapper",
        )
        self.assertEqual(
            exported["runs"][7]["method_metadata"]["alignment"]["level"],
            "mixed_controlled_adaptation",
        )
        self.assertEqual(
            exported["runs"][8]["method_metadata"]["alignment"]["level"],
            "source_faithful_self_calibrated_2s2t_benchmark_wrapper",
        )
        self.assertEqual(
            exported["runs"][9]["method_metadata"]["alignment"]["level"],
            "source_faithful_kernel_benchmark_wrapper",
        )


if __name__ == "__main__":
    unittest.main()
