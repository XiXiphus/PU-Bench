"""Static architecture contract checks for PU-Bench.

This script intentionally avoids running training. It only performs quick
registry/config consistency checks and trainer import resolution.

Usage:
  uv run python scripts/check_arch_contracts.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
import subprocess
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS_DIR = PROJECT_ROOT / "config" / "methods"
MNIST_SEED2_CONFIG = PROJECT_ROOT / "config" / "datasets_smoke" / "param_sweep_mnist_seed2.yaml"
RUN_TRAIN = PROJECT_ROOT / "run_train.py"
MOVED_METRIC_SYMBOLS = (
    "_dataset_metadata",
    "_pu_label_values_from_loader",
    "_adapt_input_for_model",
    "_model_predict",
    "evaluate_metrics",
    "evaluate_proxy_metrics",
)
REMOVED_FILES = (
    PROJECT_ROOT / "train" / "train_utils.py",
    PROJECT_ROOT / "data" / "vector_augment.py",
)
MOVED_MODULE_SYMBOLS = {
    "train.metrics": MOVED_METRIC_SYMBOLS,
    "train.data_factory": ("prepare_loaders",),
    "train.model_factory": ("infer_model_name", "select_model", "select_public_model"),
    "train.common.pu_risk": ("PULoss", "choose_loss", "pu_loss"),
    "train.mixup": ("mixup_data", "mixup_criterion"),
    "train.schedules": ("sigmoid_rampup", "linear_rampup"),
    "train.augmentations.vector": (
        "VectorWeakAugment",
        "VectorStrongAugment",
        "VectorAugPUDatasetWrapper",
    ),
    "train.checkpointing": (
        "ModelCheckpoint",
        "CheckpointBundle",
    ),
}
LEGACY_MODULE_FRAGMENTS = (
    "train_utils",
    "train.train_utils",
    "data.vector_augment",
)
BASE_TRAINER_LIFECYCLE_METHODS = (
    "setup_context",
    "configure",
    "build_data",
    "build_components",
    "create_model",
    "run_stage",
    "evaluate",
    "finalize",
)
STAGED_TRAINER_FILES = (
    PROJECT_ROOT / "train" / "distpu" / "trainer.py",
    PROJECT_ROOT / "train" / "robustpu" / "trainer.py",
    PROJECT_ROOT / "train" / "lagam" / "trainer.py",
    PROJECT_ROOT / "train" / "pulda" / "trainer.py",
    PROJECT_ROOT / "train" / "pulcpbf" / "trainer.py",
    PROJECT_ROOT / "train" / "holisticpu" / "trainer.py",
    PROJECT_ROOT / "train" / "vaepu" / "trainer.py",
    PROJECT_ROOT / "train" / "selfpu" / "trainer.py",
    PROJECT_ROOT / "train" / "bbepu" / "trainer.py",
    PROJECT_ROOT / "train" / "lbe" / "trainer.py",
)
CHECKPOINT_INTERNAL_FIELDS = {
    "best_score",
    "best_epoch",
    "best_metrics",
    "best_elapsed_seconds",
    "wait",
    "should_stop",
    "early_stopping_enabled",
}
METHOD_PRIVATE_MODEL_FRAGMENTS = {
    "holisticpu",
    "robustpu",
    "lagam",
    "p3mix",
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CheckReport:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def ok(self, message: str) -> None:
        self.passed.append(message)

    def fail(self, message: str) -> None:
        self.failed.append(message)

    def extend(self, other: "CheckReport") -> None:
        self.passed.extend(other.passed)
        self.failed.extend(other.failed)


def _load_registry_module(report: CheckReport) -> tuple[Any | None, Any | None]:
    try:
        from train import registry as registry_module

        report.ok("Imported train.registry via package import path.")
        return registry_module, None
    except Exception as exc:  # noqa: BLE001
        report.fail(
            "Failed to import train.registry via package path.\n"
            f"Exception: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc().rstrip()}"
        )
        return None, exc


def _check_registry_contracts(registry_module: Any, report: CheckReport) -> None:
    registered = registry_module.list_registered_methods()
    report.ok(f"Registry listed {len(registered)} methods.")

    if "upu" in registered:
        report.fail("Registry must not expose 'upu' as a runnable method.")
    else:
        report.ok("Registry does not expose 'upu'.")

    for method_name in registered:
        try:
            import_path = registry_module.get_trainer_import_path(method_name)
            if not isinstance(import_path, str) or "." not in import_path:
                raise ValueError(f"Invalid trainer import path: {import_path!r}")
            report.ok(f"Resolved import path for '{method_name}' -> {import_path}.")
        except Exception as exc:  # noqa: BLE001
            report.fail(
                f"Failed to resolve import path for '{method_name}'.\n"
                f"Exception: {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc().rstrip()}"
            )
            continue

        try:
            trainer_cls = registry_module.import_trainer_class(method_name)
            resolved = f"{trainer_cls.__module__}.{trainer_cls.__name__}"
            report.ok(f"Imported trainer class for '{method_name}' -> {resolved}.")
        except Exception as exc:  # noqa: BLE001
            report.fail(
                f"Failed to import trainer class for '{method_name}'.\n"
                f"Exception: {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc().rstrip()}"
            )


def _load_yaml_file(path: Path) -> tuple[Any | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}, None
    except Exception as exc:  # noqa: BLE001
        return None, (
            f"Failed to parse YAML '{path.name}'.\n"
            f"Exception: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc().rstrip()}"
        )


def _check_method_yaml_contracts(report: CheckReport) -> set[str]:
    yaml_methods: set[str] = set()

    if not METHODS_DIR.is_dir():
        report.fail(f"Methods directory is missing: {METHODS_DIR}")
        return yaml_methods

    for yaml_path in sorted(METHODS_DIR.glob("*.yaml")):
        stem = yaml_path.stem.lower()
        yaml_methods.add(stem)

        data, error = _load_yaml_file(yaml_path)
        if error is not None:
            report.fail(error)
            continue

        if not isinstance(data, dict):
            report.fail(
                f"Method YAML '{yaml_path.name}' must contain a top-level mapping, "
                f"got {type(data).__name__}."
            )
            continue

        top_level_keys = list(data)
        if stem not in data:
            report.fail(
                f"Method YAML '{yaml_path.name}' is keyed by {top_level_keys!r}, "
                f"expected top-level key '{stem}'."
            )
            continue

        if len(top_level_keys) != 1:
            report.fail(
                f"Method YAML '{yaml_path.name}' should expose exactly one top-level "
                f"method key, got {top_level_keys!r}."
            )
            continue

        if not isinstance(data[stem], dict):
            report.fail(
                f"Method YAML '{yaml_path.name}' entry '{stem}' must map to a config "
                f"dictionary, got {type(data[stem]).__name__}."
            )
            continue

        metadata = data[stem].get("metadata")
        if not isinstance(metadata, dict):
            report.fail(
                f"Method YAML '{yaml_path.name}' entry '{stem}' must declare metadata."
            )
            continue
        if str(metadata.get("method_key", "")).lower() != stem:
            report.fail(
                f"Method YAML '{yaml_path.name}' metadata.method_key must be '{stem}', "
                f"got {metadata.get('method_key')!r}."
            )
            continue
        trainer_key = str(metadata.get("trainer_key", "")).lower()
        if not trainer_key:
            report.fail(
                f"Method YAML '{yaml_path.name}' metadata.trainer_key must be declared."
            )
            continue

        report.ok(f"Validated method YAML '{yaml_path.name}' for '{stem}'.")

    return yaml_methods


def _check_registry_yaml_alignment(
    registry_methods: set[str], yaml_methods: set[str], report: CheckReport
) -> None:
    missing_yaml = sorted(registry_methods - yaml_methods)
    extra_yaml = sorted(yaml_methods - registry_methods)

    if missing_yaml:
        report.fail(
            "Registry methods missing method YAML files: " + ", ".join(missing_yaml)
        )
    else:
        report.ok("Every registered method has a method YAML file.")

    if extra_yaml:
        report.fail(
            "Method YAML files without registry entries: " + ", ".join(extra_yaml)
        )
    else:
        report.ok("Every method YAML file maps to a registry entry.")


def _check_experiment_plan_contracts(report: CheckReport) -> None:
    try:
        from config.experiment_plan import build_plan

        plan = build_plan(
            dataset_config_path=MNIST_SEED2_CONFIG,
            methods=["nnpu", "vpu"],
            methods_dir=METHODS_DIR,
            available_methods=["nnpu", "vpu"],
        )
    except Exception as exc:  # noqa: BLE001
        report.fail(
            "Failed to build MNIST seed2 ExperimentPlan.\n"
            f"Exception: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc().rstrip()}"
        )
        return

    expected_experiment = "MNIST_case-control_random_c0.1_seed2"
    failures = []
    if plan.dataset_name != "MNIST":
        failures.append(f"dataset_name={plan.dataset_name!r}")
    if plan.methods != ("nnpu", "vpu"):
        failures.append(f"methods={plan.methods!r}")
    if plan.total_runs != 2:
        failures.append(f"total_runs={plan.total_runs!r}")
    if len(plan.dataset_runs) != 1:
        failures.append(f"dataset_runs={len(plan.dataset_runs)!r}")

    run_methods = [run.method for run in plan.runs]
    if run_methods != ["nnpu", "vpu"]:
        failures.append(f"run method order={run_methods!r}")

    run_trainer_keys = [run.trainer_key for run in plan.runs]
    if run_trainer_keys != ["nnpu", "vpu"]:
        failures.append(f"run trainer_key order={run_trainer_keys!r}")

    run_names = [run.experiment_name for run in plan.runs]
    if run_names != [expected_experiment, expected_experiment]:
        failures.append(f"experiment names={run_names!r}")

    if plan.dataset_runs:
        data_run = plan.dataset_runs[0]
        expected_fields = {
            "scenario": "case-control",
            "selection_strategy": "random",
            "labeled_ratio": 0.1,
            "random_seed": 2,
        }
        for key, expected in expected_fields.items():
            if data_run.get(key) != expected:
                failures.append(f"dataset_run[{key}]={data_run.get(key)!r}")

    if failures:
        report.fail(
            "MNIST seed2 ExperimentPlan contract mismatch: " + "; ".join(failures)
        )
    else:
        report.ok(
            "MNIST seed2 ExperimentPlan preserves dataset-major run order and one dataset dry-run row."
        )


def _run_launcher(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUN_TRAIN), *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


def _combined_output(completed: subprocess.CompletedProcess[str]) -> str:
    return (completed.stdout or "") + (completed.stderr or "")


def _check_mnist_seed2_dry_run(report: CheckReport) -> None:
    args = [
        "--dataset-config",
        str(MNIST_SEED2_CONFIG),
        "--methods",
        "nnpu",
        "vpu",
        "--dry-run",
    ]
    try:
        completed = _run_launcher(args)
    except Exception as exc:  # noqa: BLE001
        report.fail(
            "Failed to execute MNIST seed2 dry-run command.\n"
            f"Exception: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc().rstrip()}"
        )
        return

    output = _combined_output(completed)
    expected_header = "Planned runs: 2 | datasets=MNIST | methods=nnpu, vpu"
    expected_dataset_line = (
        "dataset=MNIST | [1/1] scenario=case-control strategy=random c=0.1 seed=2"
    )
    dataset_lines = [
        line for line in output.splitlines() if line.startswith("dataset=")
    ]

    failures = []
    if completed.returncode != 0:
        failures.append(f"returncode={completed.returncode}")
    if expected_header not in output:
        failures.append(f"missing header {expected_header!r}")
    if dataset_lines != [expected_dataset_line]:
        failures.append(f"dataset lines={dataset_lines!r}")

    if failures:
        report.fail(
            "MNIST seed2 dry-run contract mismatch: "
            + "; ".join(failures)
            + "\nOutput:\n"
            + output.rstrip()
        )
    else:
        report.ok("MNIST seed2 dry-run reports the expected plan and dataset row.")


def _check_upu_fail_fast_dry_run(report: CheckReport) -> None:
    args = [
        "--dataset-config",
        str(MNIST_SEED2_CONFIG),
        "--methods",
        "upu",
        "--dry-run",
    ]
    try:
        completed = _run_launcher(args)
    except Exception as exc:  # noqa: BLE001
        report.fail(
            "Failed to execute UPU fail-fast dry-run command.\n"
            f"Exception: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc().rstrip()}"
        )
        return

    output = _combined_output(completed)
    failures = []
    if completed.returncode == 0:
        failures.append("returncode=0")
    if "Unknown or unregistered methods: upu" not in output:
        failures.append("missing unknown-method message")
    if "Available methods:" not in output:
        failures.append("missing available-methods message")

    if failures:
        report.fail(
            "UPU dry-run fail-fast contract mismatch: "
            + "; ".join(failures)
            + "\nOutput:\n"
            + output.rstrip()
        )
    else:
        report.ok("UPU dry-run fails fast as an unknown/unregistered method.")


def _check_removed_files(report: CheckReport) -> None:
    retained = [path.relative_to(PROJECT_ROOT) for path in REMOVED_FILES if path.exists()]
    if retained:
        report.fail(
            "Breaking refactor requires these old files to be removed: "
            + ", ".join(str(path) for path in retained)
        )
    else:
        report.ok("Removed legacy train_utils.py and data/vector_augment.py files.")


def _check_module_symbol_contracts(report: CheckReport) -> None:
    import importlib

    failures: list[str] = []
    for module_name, symbols in MOVED_MODULE_SYMBOLS.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{module_name}: import failed ({type(exc).__name__}: {exc})")
            continue
        missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
        if missing:
            failures.append(f"{module_name}: missing {', '.join(missing)}")

    try:
        import train.base_trainer as base_trainer_module
    except Exception as exc:  # noqa: BLE001
        failures.append(
            f"train.base_trainer: import failed ({type(exc).__name__}: {exc})"
        )
    else:
        base_cls = getattr(base_trainer_module, "BaseTrainer", None)
        missing = [
            name
            for name in BASE_TRAINER_LIFECYCLE_METHODS
            if base_cls is None or not hasattr(base_cls, name)
        ]
        if missing:
            failures.append("BaseTrainer missing lifecycle methods: " + ", ".join(missing))

    if failures:
        report.fail("Moved module contract mismatch:\n" + "\n".join(failures))
    else:
        report.ok("Moved modules expose Phase 2/3 target symbols.")


def _check_removed_legacy_imports(report: CheckReport) -> None:
    offenders: list[str] = []
    for py_path in sorted(_project_python_files()):
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            report.fail(
                f"Failed to parse Python file '{py_path.relative_to(PROJECT_ROOT)}'.\n"
                f"Exception: {type(exc).__name__}: {exc}"
            )
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in LEGACY_MODULE_FRAGMENTS or any(
                        fragment in alias.name for fragment in LEGACY_MODULE_FRAGMENTS
                    ):
                        offenders.append(
                            f"{py_path.relative_to(PROJECT_ROOT)}:{node.lineno} imports {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in LEGACY_MODULE_FRAGMENTS or any(
                    fragment in module for fragment in LEGACY_MODULE_FRAGMENTS
                ):
                    offenders.append(
                        f"{py_path.relative_to(PROJECT_ROOT)}:{node.lineno} imports from {module}"
                    )

    if offenders:
        report.fail(
            "Legacy train_utils/vector_augment imports remain:\n"
            + "\n".join(offenders)
        )
    else:
        report.ok("No project Python file imports removed legacy modules.")


def _check_metrics_extraction_contracts(report: CheckReport) -> None:
    try:
        import train.metrics as metrics_module
    except Exception as exc:  # noqa: BLE001
        report.fail(
            "Failed to import train.metrics for extraction check.\n"
            f"Exception: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc().rstrip()}"
        )
        return

    missing = [
        symbol
        for symbol in MOVED_METRIC_SYMBOLS
        if not hasattr(metrics_module, symbol)
    ]

    if missing:
        report.fail("train.metrics is missing moved symbols: " + ", ".join(missing))
    else:
        report.ok("train.metrics exposes the moved metric symbols.")


def _project_python_files() -> list[Path]:
    ignored_parts = {".venv", "__pycache__"}
    return [
        path
        for path in PROJECT_ROOT.rglob("*.py")
        if not ignored_parts.intersection(path.parts)
    ]


def _attribute_chain(node: ast.AST) -> list[str]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return list(reversed(parts))
    return []


def _assignment_targets(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.Assign):
        return list(node.targets)
    if isinstance(node, ast.AnnAssign):
        return [node.target]
    if isinstance(node, ast.AugAssign):
        return [node.target]
    return []


def _is_checkpoint_internal_chain(chain: list[str]) -> bool:
    return (
        len(chain) >= 3
        and chain[0] == "self"
        and chain[1] == "checkpoint_handler"
        and chain[2] in CHECKPOINT_INTERNAL_FIELDS
    )


def _check_no_staged_checkpoint_internal_writes(report: CheckReport) -> None:
    offenders: list[str] = []
    mutating_methods = {"update", "clear", "pop", "popitem", "setdefault"}

    for py_path in STAGED_TRAINER_FILES:
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            report.fail(
                f"Failed to parse staged trainer '{py_path.relative_to(PROJECT_ROOT)}'.\n"
                f"Exception: {type(exc).__name__}: {exc}"
            )
            continue

        for node in ast.walk(tree):
            for target in _assignment_targets(node):
                chain = _attribute_chain(target)
                if _is_checkpoint_internal_chain(chain):
                    offenders.append(
                        f"{py_path.relative_to(PROJECT_ROOT)}:{getattr(node, 'lineno', '?')} writes "
                        + ".".join(chain)
                    )

            if isinstance(node, ast.Call):
                chain = _attribute_chain(node.func)
                if (
                    len(chain) >= 4
                    and _is_checkpoint_internal_chain(chain[:-1])
                    and chain[-1] in mutating_methods
                ):
                    offenders.append(
                        f"{py_path.relative_to(PROJECT_ROOT)}:{node.lineno} mutates "
                        + ".".join(chain[:-1])
                    )

    if offenders:
        report.fail(
            "Staged trainers must use shared checkpoint helpers instead of writing internals:\n"
            + "\n".join(offenders)
        )
    else:
        report.ok("Staged trainers do not directly write checkpoint internals.")


def _check_model_factory_boundary(report: CheckReport) -> None:
    model_factory = PROJECT_ROOT / "train" / "model_factory.py"
    try:
        source = model_factory.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception as exc:  # noqa: BLE001
        report.fail(
            "Failed to parse train/model_factory.py for Phase 4 boundary check.\n"
            f"Exception: {type(exc).__name__}: {exc}"
        )
        return

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported = alias.name.lower()
                if any(fragment in imported for fragment in METHOD_PRIVATE_MODEL_FRAGMENTS):
                    offenders.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").lower()
            if any(fragment in module for fragment in METHOD_PRIVATE_MODEL_FRAGMENTS):
                offenders.append(f"line {node.lineno}: from {node.module} import ...")

    lowered = source.lower()
    for fragment in METHOD_PRIVATE_MODEL_FRAGMENTS:
        if fragment in lowered:
            offenders.append(f"contains method-private fragment '{fragment}'")

    if offenders:
        report.fail(
            "Shared model_factory must not know method-private model packages:\n"
            + "\n".join(offenders)
        )
    else:
        report.ok("Shared model_factory contains only public benchmark model selection.")


def _check_no_cross_method_nnpu_loss_imports(report: CheckReport) -> None:
    offenders: list[str] = []
    for py_path in sorted(_project_python_files()):
        rel = py_path.relative_to(PROJECT_ROOT)
        if rel.parts[:2] == ("train", "nnpu"):
            continue
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            report.fail(
                f"Failed to parse Python file '{rel}'.\n"
                f"Exception: {type(exc).__name__}: {exc}"
            )
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "nnpu.losses" in module:
                    offenders.append(f"{rel}:{node.lineno} imports from {module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "nnpu.losses" in alias.name:
                        offenders.append(f"{rel}:{node.lineno} imports {alias.name}")

    if offenders:
        report.fail(
            "Cross-method imports from train.nnpu.losses remain:\n"
            + "\n".join(offenders)
        )
    else:
        report.ok("Shared PU risk users no longer borrow from train.nnpu.losses.")


def _check_namespaced_config_dry_run(report: CheckReport) -> None:
    dataset_config = {
        "dataset": {
            "dataset_class": "MNIST",
            "data_dir": "./datasets",
            "label_scheme": {
                "positive_classes": [0, 2, 4, 6, 8],
                "negative_classes": [1, 3, 5, 7, 9],
            },
        },
        "sweep": {
            "random_seeds": [2],
            "c_values": [0.1],
            "scenarios": ["case-control"],
            "selection_strategies": ["random"],
        },
        "runtime": {
            "val_ratio": 0.01,
            "target_prevalence": None,
            "with_replacement": True,
            "also_print_dataset_stats": False,
        },
    }
    method_configs = {
        "method_a.yaml": {
            "metadata": {"method_key": "nnpu_alias", "trainer_key": "nnpu"},
            "method": {"optimizer": "adam", "batch_size": 128, "num_epochs": 1},
        },
        "method_b.yaml": {
            "metadata": {"method_key": "vpu", "trainer_key": "vpu"},
            "method": {"optimizer": "adam", "batch_size": 128, "num_epochs": 1},
        },
    }

    with tempfile.TemporaryDirectory(prefix="pu_bench_contract_") as tmp:
        tmp_path = Path(tmp)
        dataset_path = tmp_path / "mnist_namespaced.yaml"
        methods_dir = tmp_path / "methods"
        methods_dir.mkdir()
        dataset_path.write_text(yaml.safe_dump(dataset_config), encoding="utf-8")
        for filename, config in method_configs.items():
            (methods_dir / filename).write_text(
                yaml.safe_dump(config), encoding="utf-8"
            )

        completed = _run_launcher(
            [
                "--dataset-config",
                str(dataset_path),
                "--methods-dir",
                str(methods_dir),
                "--methods",
                "nnpu_alias",
                "vpu",
                "--dry-run",
            ]
        )

    output = _combined_output(completed)
    expected_header = "Planned runs: 2 | datasets=MNIST | methods=nnpu_alias, vpu"
    expected_dataset_line = (
        "dataset=MNIST | [1/1] scenario=case-control strategy=random c=0.1 seed=2"
    )
    dataset_lines = [
        line for line in output.splitlines() if line.startswith("dataset=")
    ]
    failures = []
    if completed.returncode != 0:
        failures.append(f"returncode={completed.returncode}")
    if expected_header not in output:
        failures.append(f"missing header {expected_header!r}")
    if dataset_lines != [expected_dataset_line]:
        failures.append(f"dataset lines={dataset_lines!r}")

    if failures:
        report.fail(
            "Namespaced temp config dry-run mismatch: "
            + "; ".join(failures)
            + "\nOutput:\n"
            + output.rstrip()
        )
    else:
        report.ok("Namespaced dataset/method config dry-run matches legacy MNIST seed2.")


def _check_plan_json_export(report: CheckReport) -> None:
    with tempfile.TemporaryDirectory(prefix="pu_bench_plan_json_") as tmp:
        tmp_path = Path(tmp)
        methods_dir = tmp_path / "methods"
        methods_dir.mkdir()
        method_config = {
            "metadata": {"method_key": "nnpu_alias", "trainer_key": "nnpu"},
            "method": {"optimizer": "adam", "batch_size": 128, "num_epochs": 1},
        }
        (methods_dir / "alias.yaml").write_text(
            yaml.safe_dump(method_config),
            encoding="utf-8",
        )
        plan_path = tmp_path / "plan.json"
        completed = _run_launcher(
            [
                "--dataset-config",
                str(MNIST_SEED2_CONFIG),
                "--methods-dir",
                str(methods_dir),
                "--methods",
                "nnpu_alias",
                "--plan-json",
                str(plan_path),
                "--dry-run",
            ]
        )
        output = _combined_output(completed)
        if completed.returncode != 0:
            report.fail(
                "Plan JSON dry-run command failed.\n"
                f"Output:\n{output.rstrip()}"
            )
            return
        try:
            plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            report.fail(
                "Failed to read exported plan JSON.\n"
                f"Exception: {type(exc).__name__}: {exc}"
            )
            return

    runs = plan_data.get("runs", [])
    failures = []
    if plan_data.get("total_runs") != 1:
        failures.append(f"total_runs={plan_data.get('total_runs')!r}")
    if [run.get("method") for run in runs] != ["nnpu_alias"]:
        failures.append(f"method order={[run.get('method') for run in runs]!r}")
    if [run.get("trainer_key") for run in runs] != ["nnpu"]:
        failures.append(
            f"trainer_key order={[run.get('trainer_key') for run in runs]!r}"
        )

    if failures:
        report.fail("Plan JSON export contract mismatch: " + "; ".join(failures))
    else:
        report.ok("Plan JSON export preserves run order, method, and trainer_key.")


def _check_no_unread_method_config_keys(report: CheckReport) -> None:
    shared_paths = [
        PROJECT_ROOT / "train" / "base",
        PROJECT_ROOT / "train" / "base_trainer.py",
        PROJECT_ROOT / "train" / "checkpointing.py",
        PROJECT_ROOT / "train" / "data_factory.py",
        PROJECT_ROOT / "train" / "metrics.py",
        PROJECT_ROOT / "train" / "model_factory.py",
        PROJECT_ROOT / "train" / "reproducibility.py",
    ]
    shared_text = _read_python_text(shared_paths)
    offenders: list[str] = []

    for yaml_path in sorted(METHODS_DIR.glob("*.yaml")):
        method = yaml_path.stem.lower()
        data, error = _load_yaml_file(yaml_path)
        if error is not None or not isinstance(data, dict) or method not in data:
            continue
        entry = data[method]
        if not isinstance(entry, dict):
            continue

        method_paths: list[Path] = []
        method_dir = PROJECT_ROOT / "train" / method
        if method_dir.exists():
            method_paths.append(method_dir)
        if method in {"p3mixc", "p3mixe"}:
            method_paths = [PROJECT_ROOT / "train" / "p3mix"]
        if method == "pn":
            method_paths = [PROJECT_ROOT / "train" / "pn_trainer.py"]

        search_text = shared_text + "\n" + _read_python_text(method_paths)
        for key, value in entry.items():
            if key in {"metadata", "extends"}:
                continue
            candidates = [(key, None)]
            if isinstance(value, dict):
                candidates.extend((key, subkey) for subkey in value)
            for top_key, subkey in candidates:
                token = str(subkey or top_key)
                pattern = re.compile(
                    r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])"
                )
                if not pattern.search(search_text):
                    rendered = f"{top_key}.{subkey}" if subkey is not None else top_key
                    offenders.append(f"{yaml_path.name}: {rendered}")

    if offenders:
        report.fail(
            "Method YAML keys not read by shared or method trainer code:\n"
            + "\n".join(offenders)
        )
    else:
        report.ok("Every method YAML key is read by shared or method trainer code.")


def _read_python_text(paths: list[Path]) -> str:
    chunks: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            chunks.extend(
                py_path.read_text(encoding="utf-8", errors="ignore")
                for py_path in path.rglob("*.py")
            )
        else:
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def _check_proxy_metric_label_contract(report: CheckReport) -> None:
    banned_patterns = {
        PROJECT_ROOT / "train" / "metrics.py": (
            "p_mask = t == 1",
            "u_mask = t != 1",
        ),
        PROJECT_ROOT / "train" / "vaepu" / "trainer.py": (
            "p_mask = t == 1",
            "u_mask = t != 1",
        ),
        PROJECT_ROOT / "train" / "puet" / "trainer.py": (
            "u_mask = pu != labeled_value",
        ),
    }
    offenders: list[str] = []
    for path, patterns in banned_patterns.items():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if pattern in text:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {pattern}")

    if offenders:
        report.fail(
            "Proxy metric evaluators must use explicit PU label metadata:\n"
            + "\n".join(offenders)
        )
        return

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset

        from train.metrics import evaluate_proxy_metrics

        class _ProxyDataset(Dataset):
            pu_metadata = {"pu_labeled_label": 7, "pu_unlabeled_label": -3}

            def __len__(self) -> int:
                return 2

            def __getitem__(self, idx: int):
                if idx == 0:
                    return (
                        torch.tensor([2.0]),
                        torch.tensor(7),
                        torch.tensor(1),
                        torch.tensor(0),
                        {},
                    )
                return (
                    torch.tensor([-2.0]),
                    torch.tensor(-3),
                    torch.tensor(0),
                    torch.tensor(1),
                    {},
                )

        class _ProxyModel(nn.Module):
            def forward(self, x):
                return x.view(-1)

        metrics = evaluate_proxy_metrics(
            _ProxyModel(),
            DataLoader(_ProxyDataset(), batch_size=2),
            torch.device("cpu"),
            prior=0.5,
            scenario="case-control",
        )
    except Exception as exc:  # noqa: BLE001
        report.fail(
            "Proxy metric metadata behavior check failed.\n"
            f"Exception: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc().rstrip()}"
        )
        return

    if metrics.get("proxy_acc") != 2.0 or metrics.get("proxy_auc") != 1.0:
        report.fail(f"Proxy metric metadata behavior mismatch: {metrics!r}")
    else:
        report.ok("Proxy metric evaluators use explicit PU label metadata.")


def _print_summary(report: CheckReport) -> None:
    for message in report.passed:
        print(f"[PASS] {message}")

    for message in report.failed:
        print(f"[FAIL] {message}")

    print(
        f"\nSummary: {len(report.passed)} passed, {len(report.failed)} failed."
    )


def main() -> int:
    report = CheckReport()

    registry_module, registry_error = _load_registry_module(report)
    yaml_methods = _check_method_yaml_contracts(report)

    if registry_module is not None:
        _check_registry_contracts(registry_module, report)
        registry_methods = set(registry_module.list_registered_methods())
        _check_registry_yaml_alignment(registry_methods, yaml_methods, report)
    elif registry_error is None:
        report.fail("Registry import failed for an unknown reason.")

    _check_experiment_plan_contracts(report)
    _check_mnist_seed2_dry_run(report)
    _check_upu_fail_fast_dry_run(report)
    _check_removed_files(report)
    _check_module_symbol_contracts(report)
    _check_removed_legacy_imports(report)
    _check_metrics_extraction_contracts(report)
    _check_no_staged_checkpoint_internal_writes(report)
    _check_model_factory_boundary(report)
    _check_no_cross_method_nnpu_loss_imports(report)
    _check_namespaced_config_dry_run(report)
    _check_plan_json_export(report)
    _check_no_unread_method_config_keys(report)
    _check_proxy_metric_label_contract(report)

    _print_summary(report)
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
