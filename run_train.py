"""run_train.py - New lightweight training launcher.

This script launches training runs by combining:
- A dataset config (which may define a grid over c/scenario/strategy/seeds)
- Per-method hyperparameters from config/methods/*.yaml

Key goals:
- No experiments concept; each combination is one run
- Minimize hyperparameter passing; dataset config is merged with method params

Usage:
  python -u run_train.py \
    --dataset-config config/datasets_typical/param_sweep_mnist.yaml \
    --methods nnpu upu  # optional; default: all available methods

Optional:
  --dry-run  Only list planned runs without executing training
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

# New method loader (relative import safe when running as script)
from config.method_loader import (
    list_available_methods as list_methods_new,
    load_method_params as load_method_new,
    DEFAULT_METHODS_DIR as NEW_METHODS_DIR,
)

# Dataset config loader (centralized)
from config.run_param_sweep import (
    load_dataset_config,
    expand_dataset_grid,
)


# Ensure project root on path for train.* imports
PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Mapping from method name to Trainer import path
TRAINER_IMPORT_PATHS = {
    "pn": "train.pn_trainer.PNTrainer",
    "nnpu": "train.nnpu_trainer.NNPUTrainer",
    "nnpusb": "train.nnpusb_trainer.NNPUSBTrainer",
    "vpu": "train.vpu_trainer.VPUTrainer",
    "distpu": "train.distpu_trainer.DistPUTrainer",
    "selfpu": "train.selfpu_trainer.SelfPUTrainer",
    "holisticpu": "train.holisticpu_trainer.HolisticPUTrainer",
    "robustpu": "train.robustpu_trainer.RobustPUTrainer",
    "p3mixc": "train.p3mixc_trainer.P3MIXCTrainer",
    "p3mixe": "train.p3mixe_trainer.P3MIXETrainer",
    "lagam": "train.lagam_trainer.LaGAMTrainer",
    "pulda": "train.pulda_trainer.PULDATrainer",
    "bbepu": "train.bbepu_trainer.BBEPUTrainer",
    "vaepu": "train.vaepu_trainer.VAEPUTrainer",
    "puet": "train.puet_trainer.PUETTrainer",
    "pan": "train.pan_trainer.PANTrainer",
    "lbe": "train.lbe_trainer.LBETrainer",
    "cgenpu": "train.cgenpu_trainer.CGenPUTrainer",
    "pulcpbf": "train.pulcpbf_trainer.PULCPBFTrainer",
}


def _lazy_import(path: str):
    mod_path, attr = path.rsplit(".", 1)
    mod = __import__(mod_path, fromlist=[attr])
    return getattr(mod, attr)


def _load_method_params(method_name: str, methods_dir: Path) -> Dict[str, Any]:
    # Delegate to new loader for consistent behavior
    return load_method_new(method_name, methods_dir)


def _expand_grid(dataset_cfg: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    # Delegate to centralized dataset config utilities
    return expand_dataset_grid(dataset_cfg)


def _build_experiment_name(
    dataset_class: str, data_cfg: Dict[str, Any], method: str
) -> str:
    c = data_cfg.get("labeled_ratio")
    scn = data_cfg.get("scenario")
    strat = data_cfg.get("selection_strategy")
    seed = data_cfg.get("random_seed")
    # New launcher has no 'experiment' concept; we use a deterministic per-run name
    # Method name is not embedded here to avoid duplication with results/{method}_ prefix
    return f"{dataset_class}_{scn}_{strat}_c{c:g}_seed{seed}"


def main():
    parser = argparse.ArgumentParser(description="Lightweight PU training launcher")
    parser.add_argument(
        "--dataset-config",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to dataset YAML(s) (supports multiple; each may define grids)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to run (default: all available in origin/configs/methods)",
    )
    parser.add_argument(
        "--methods-dir",
        type=str,
        default=str(NEW_METHODS_DIR),
        help="Directory containing per-method YAML configs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list planned runs without executing",
    )
    args = parser.parse_args()

    # Determine methods
    methods_dir = Path(args.methods_dir)
    if args.methods is None:
        method_names = list_methods_new(methods_dir)
    else:
        # Support comma-separated or whitespace-separated lists
        # argparse with nargs+ already splits on whitespace; we additionally split items on commas
        raw_methods = []
        for m in args.methods:
            raw_methods.extend([p for p in m.split(",") if p])
        method_names = [m.strip().lower() for m in raw_methods if m.strip()]
    # Filter by available trainers
    method_names = [m for m in method_names if m in TRAINER_IMPORT_PATHS]
    if not method_names:
        print("No valid methods found to run.")
        sys.exit(1)

    # Load one or more dataset configs and expand their grids
    dataset_cfg_paths: List[str] = list(args.dataset_config)
    datasets_expanded: List[Tuple[str, List[Dict[str, Any]]]] = []
    dataset_names: List[str] = []
    for cfg_path in dataset_cfg_paths:
        dataset_cfg = load_dataset_config(cfg_path)
        dataset_class, data_runs = _expand_grid(dataset_cfg)
        datasets_expanded.append((dataset_class, data_runs))
        dataset_names.append(dataset_class)

    # Prepare trainers (lazy import once)
    trainer_classes: Dict[str, Any] = {}
    for m in method_names:
        trainer_classes[m] = _lazy_import(TRAINER_IMPORT_PATHS[m])

    total = len(method_names) * sum(len(dr) for (_, dr) in datasets_expanded)
    unique_dataset_names = list(dict.fromkeys(dataset_names))
    print("=" * 80)
    print(
        f"Planned runs: {total} | datasets={', '.join(unique_dataset_names)} | methods={', '.join(method_names)}"
    )
    print("=" * 80)

    if args.dry_run:
        for dataset_class, data_runs in datasets_expanded:
            for i, d in enumerate(data_runs, 1):
                print(
                    f"dataset={dataset_class} | [{i}/{len(data_runs)}] scenario={d['scenario']} strategy={d['selection_strategy']} c={d['labeled_ratio']} seed={d['random_seed']}"
                )
        return

    # Execute runs
    import copy

    for dataset_class, data_runs in datasets_expanded:
        for i, data_cfg in enumerate(data_runs, 1):
            for method in method_names:
                try:
                    method_params = _load_method_params(method, methods_dir)
                    params = copy.deepcopy(method_params)
                    # Merge dataset settings into params
                    params.update(data_cfg)

                    # Determine experiment name per run
                    exp_name = _build_experiment_name(dataset_class, data_cfg, method)

                    # Instantiate and run trainer
                    trainer_cls = trainer_classes[method]
                    trainer = trainer_cls(
                        method=method, experiment=exp_name, params=params
                    )
                    trainer.run()
                    print(f"✔ Completed: {exp_name}")
                except Exception as exc:
                    import traceback

                    print(f"✗ Failed: method={method} data={data_cfg}")
                    print(f"Error: {exc}")
                    traceback.print_exc()
                    print("-" * 80)


if __name__ == "__main__":
    main()
