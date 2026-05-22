import unittest

from config.experiment_plan import build_plan, plan_to_dict
from config.method_loader import load_method_config


class TestMethodMetadataInvariants(unittest.TestCase):
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
        self.assertEqual(config.params["loss"], "sigmoid")
        self.assertEqual(float(config.params["beta"]), 0.0)
        self.assertEqual(float(config.params["gamma"]), 1.0)
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
            "source_aligned_recipe_with_controlled_backbone",
        )
        self.assertEqual(
            alignment["modality_alignment"]["vector_tabular"],
            "benchmark_specific_extension",
        )
        self.assertIn(
            "unlabeled phase-2 targets do not use oracle true labels",
            alignment["retained_source_components"],
        )

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
            "unified benchmark oracle/proxy metrics rather than the source-script normalized test decision rule",
            alignment["benchmark_adaptations"],
        )
        self.assertIn(
            "source default batch_size=500 is retained and can be high-memory on large image datasets",
            alignment["known_resource_risks"],
        )
        self.assertFalse(config.params["checkpoint"]["early_stopping"]["enabled"])

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
            "source_faithful_kernel_benchmark_wrapper",
        )


if __name__ == "__main__":
    unittest.main()
