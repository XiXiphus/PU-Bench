"""run_train.py - New lightweight training launcher.

This script launches training runs by combining:
- A dataset config (which may define a grid over c/scenario/strategy/seeds)
- Per-method hyperparameters from config/methods/*.yaml

Key goals:
- No experiments concept; each combination is one run
- Minimize hyperparameter passing; dataset config is merged with method params

Usage:
  uv run python -u run_train.py \
    --dataset-config config/datasets_typical/param_sweep_mnist.yaml \
    --methods nnpu vpu  # optional; default: all available methods

Optional:
  --dry-run  Only list planned runs without executing training
"""

from __future__ import annotations

import sys
import argparse
import copy
import json
from pathlib import Path
from typing import List

# New method loader (relative import safe when running as script)
from config.method_loader import (
    list_available_methods as list_methods_new,
    DEFAULT_METHODS_DIR as NEW_METHODS_DIR,
)
from config.experiment_plan import (
    ExperimentPlan,
    build_plan,
    plans_to_dict,
    validate_requested_methods,
)


# Ensure project root on path for train.* imports
PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.registry import TRAINER_IMPORT_PATHS, import_trainer_class


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
    parser.add_argument(
        "--plan-json",
        type=str,
        default=None,
        help="Optional path to write the expanded execution plan as JSON",
    )
    args = parser.parse_args()

    # Determine methods
    methods_dir = Path(args.methods_dir)
    available_method_names = list_methods_new(methods_dir)
    if args.methods is None:
        method_names = available_method_names
    else:
        try:
            method_names = validate_requested_methods(args.methods)
        except ValueError as exc:
            print(str(exc))
            sys.exit(1)

    unknown_methods = sorted(set(method_names) - set(available_method_names))
    if unknown_methods:
        available = ", ".join(sorted(available_method_names))
        print(f"Unknown or unregistered methods: {', '.join(unknown_methods)}")
        print(f"Available methods: {available}")
        sys.exit(1)

    if not method_names:
        print("No valid methods found to run.")
        sys.exit(1)

    # Load one or more dataset configs and expand them into explicit run specs.
    dataset_cfg_paths: List[str] = list(args.dataset_config)
    plans: List[ExperimentPlan] = []
    for cfg_path in dataset_cfg_paths:
        try:
            plans.append(
                build_plan(
                    dataset_config_path=cfg_path,
                    methods=method_names,
                    methods_dir=methods_dir,
                    available_methods=available_method_names,
                )
            )
        except Exception as exc:
            print(f"Failed to build experiment plan for {cfg_path}: {exc}")
            sys.exit(1)

    unknown_trainer_keys = sorted(
        {
            run.trainer_key
            for plan in plans
            for run in plan.runs
            if run.trainer_key not in TRAINER_IMPORT_PATHS
        }
    )
    if unknown_trainer_keys:
        available = ", ".join(sorted(TRAINER_IMPORT_PATHS))
        print(f"Unknown or unregistered trainer keys: {', '.join(unknown_trainer_keys)}")
        print(f"Available trainer keys: {available}")
        sys.exit(1)

    total = sum(plan.total_runs for plan in plans)
    unique_dataset_names = list(dict.fromkeys(plan.dataset_name for plan in plans))
    plan_method_names = list(dict.fromkeys(method for plan in plans for method in plan.methods))
    print("=" * 80)
    print(
        f"Planned runs: {total} | datasets={', '.join(unique_dataset_names)} | methods={', '.join(plan_method_names)}"
    )
    print("=" * 80)

    if args.plan_json:
        plan_json_path = Path(args.plan_json)
        if plan_json_path.parent != Path("."):
            plan_json_path.parent.mkdir(parents=True, exist_ok=True)
        with plan_json_path.open("w", encoding="utf-8") as handle:
            json.dump(plans_to_dict(plans), handle, indent=2, sort_keys=True, default=str)

    if args.dry_run:
        for plan in plans:
            for i, d in enumerate(plan.dataset_runs, 1):
                print(
                    f"dataset={plan.dataset_name} | [{i}/{len(plan.dataset_runs)}] scenario={d['scenario']} strategy={d['selection_strategy']} c={d['labeled_ratio']} seed={d['random_seed']}"
                )
        return

    # Execute runs
    for plan in plans:
        for run in plan.runs:
            try:
                trainer_cls = import_trainer_class(run.trainer_key)
                trainer = trainer_cls(
                    method=run.method,
                    experiment=run.experiment_name,
                    params=copy.deepcopy(run.params),
                )
                trainer.run()
                print(f"✔ Completed: {run.experiment_name}")
            except Exception as exc:
                import traceback

                print(f"✗ Failed: method={run.method} data={run.dataset_params}")
                print(f"Error: {exc}")
                traceback.print_exc()
                print("-" * 80)


if __name__ == "__main__":
    main()
