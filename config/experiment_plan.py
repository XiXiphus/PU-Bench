"""Experiment plan helpers for PU-Bench.

This module adds a small planning layer on top of the existing dataset and
method YAML files without changing launcher behavior. The functions here are
pure config/data transforms so current callers can adopt them incrementally.
"""

from __future__ import annotations

import copy
import itertools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml

from .method_loader import DEFAULT_METHODS_DIR, list_available_methods, load_method_config
from .schema_adapter import normalize_dataset_config


SOURCE_HPARAMS_ENABLED_KEY = "use_source_hparams_by_dataset"
SOURCE_HPARAMS_BY_DATASET_KEY = "source_hparams_by_dataset"
# Partial source constants are documentation/bookkeeping only; they are not
# merged into run parameters as complete source recipes.
SOURCE_PARTIAL_HPARAMS_BY_DATASET_KEY = "source_partial_hparams_by_dataset"
SOURCE_HPARAMS_RESOLVED_KEY = "source_hparams_resolved_from"
RECOMMENDED_HPARAMS_ENABLED_KEY = "use_recommended_hparams_by_dataset"
RECOMMENDED_HPARAMS_BY_DATASET_KEY = "recommended_hparams_by_dataset"
RECOMMENDED_HPARAMS_RESOLVED_KEY = "recommended_hparams_resolved_from"


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One fully expanded dataset/method run."""

    dataset_name: str
    method: str
    trainer_key: str
    scenario: str
    selection_strategy: str
    labeled_ratio: float
    seed: int
    experiment_name: str
    dataset_params: Dict[str, Any]
    method_params: Dict[str, Any]
    method_metadata: Dict[str, Any]
    params: Dict[str, Any]
    case_control_mode: str | None = None


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """Expanded immutable view of one dataset config across one or more methods."""

    dataset_config_path: Path
    dataset_name: str
    methods: tuple[str, ...]
    methods_dir: Path
    raw_dataset_config: Dict[str, Any]
    dataset_runs: tuple[Dict[str, Any], ...]
    runs: tuple[RunSpec, ...]

    @property
    def total_runs(self) -> int:
        return len(self.runs)


def load_dataset_config(cfg_path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load a dataset sweep YAML into a plain dictionary."""

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError(
            f"dataset_config must deserialize to a mapping, got {type(cfg).__name__}"
        )
    return cfg


def expand_dataset_grid(
    dataset_cfg: Mapping[str, Any],
) -> tuple[str, List[Dict[str, Any]]]:
    """Expand a dataset config into run dictionaries via cartesian product.

    This intentionally mirrors the legacy behavior in ``config.run_param_sweep`` so
    callers can switch to the new module without changing semantics.
    """

    dataset_class = dataset_cfg.get("dataset_class")
    if not dataset_class:
        raise ValueError("dataset_config must include 'dataset_class'")

    seeds = dataset_cfg.get("random_seeds", [dataset_cfg.get("seed", 42)])
    c_vals = dataset_cfg.get("c_values")
    if c_vals is None:
        c_vals = [dataset_cfg.get("labeled_ratio", 0.2)]
    scenarios = dataset_cfg.get("scenarios", [dataset_cfg.get("scenario", "single")])
    strategies = dataset_cfg.get(
        "selection_strategies", [dataset_cfg.get("selection_strategy", "random")]
    )
    cc_modes = dataset_cfg.get(
        "case_control_modes",
        [dataset_cfg.get("case_control_mode", "naive_mode")],
    )

    base = dict(dataset_cfg)
    if "print_stats" not in base and "also_print_dataset_stats" in base:
        base["print_stats"] = bool(base.pop("also_print_dataset_stats"))

    base.setdefault("data_dir", "./")
    base.setdefault("val_ratio", 0.0)
    base.setdefault("target_prevalence", None)
    base.setdefault("with_replacement", True)
    base.setdefault("print_stats", False)

    runs: List[Dict[str, Any]] = []
    for seed, c, scenario, strategy, cc_mode in itertools.product(
        seeds, c_vals, scenarios, strategies, cc_modes
    ):
        run_cfg = dict(base)
        run_cfg["random_seed"] = int(seed)
        run_cfg["seed"] = int(seed)
        run_cfg["labeled_ratio"] = float(c)
        run_cfg["scenario"] = scenario
        run_cfg["selection_strategy"] = strategy
        run_cfg["case_control_mode"] = cc_mode
        for key in (
            "random_seeds",
            "c_values",
            "scenarios",
            "selection_strategies",
            "case_control_modes",
        ):
            run_cfg.pop(key, None)
        runs.append(run_cfg)

    return str(dataset_class), runs


def build_experiment_name(
    dataset_name: str,
    scenario: str,
    strategy: str,
    c: float,
    seed: int,
) -> str:
    """Build the legacy per-run experiment name."""

    return f"{dataset_name}_{scenario}_{strategy}_c{c:g}_seed{seed}"


def _normalize_lookup_key(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _deep_update(base: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _find_dataset_recipe(
    recipes: Any,
    data_run: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]] | None:
    if not isinstance(recipes, Mapping):
        return None

    dataset_key = _normalize_lookup_key(data_run.get("dataset_class", ""))
    if not dataset_key:
        return None

    recipe_by_key = {
        _normalize_lookup_key(key): (str(key), value)
        for key, value in recipes.items()
        if isinstance(value, Mapping)
    }
    return recipe_by_key.get(dataset_key)


def resolve_method_params_for_dataset(
    method_params: Mapping[str, Any],
    data_run: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply dataset-specific method recipes to one method/run pair.

    Method YAMLs still carry one benchmark fallback recipe, but source papers often
    provide dataset-specific optimizer constants.  This resolver applies source
    recipes first; for source-unsupported Bench datasets it may then apply a
    separate recommended Bench recipe.  The actual trainer receives one flat params
    dictionary and does not need method-private config plumbing.
    """

    resolved = copy.deepcopy(dict(method_params))
    resolved.pop(SOURCE_HPARAMS_RESOLVED_KEY, None)
    resolved.pop(RECOMMENDED_HPARAMS_RESOLVED_KEY, None)

    if not bool(resolved.get(SOURCE_HPARAMS_ENABLED_KEY, False)):
        source_matched = None
    else:
        source_matched = _find_dataset_recipe(
            resolved.get(SOURCE_HPARAMS_BY_DATASET_KEY),
            data_run,
        )
    if source_matched is not None:
        source_key, source_recipe = source_matched
        _deep_update(resolved, source_recipe)
        resolved[SOURCE_HPARAMS_RESOLVED_KEY] = (
            f"{SOURCE_HPARAMS_BY_DATASET_KEY}.{source_key}"
        )
        return resolved

    if bool(resolved.get(RECOMMENDED_HPARAMS_ENABLED_KEY, False)):
        recommended_matched = _find_dataset_recipe(
            resolved.get(RECOMMENDED_HPARAMS_BY_DATASET_KEY),
            data_run,
        )
        if recommended_matched is not None:
            recommended_key, recommended_recipe = recommended_matched
            _deep_update(resolved, recommended_recipe)
            resolved[RECOMMENDED_HPARAMS_RESOLVED_KEY] = (
                f"{RECOMMENDED_HPARAMS_BY_DATASET_KEY}.{recommended_key}"
            )
    return resolved


def validate_requested_methods(
    methods: Sequence[str] | None,
    available_methods: Iterable[str] | None = None,
) -> List[str]:
    """Normalize method names and ensure they exist in the available set."""

    normalized_available = None
    if available_methods is not None:
        normalized_available = sorted(
            {str(m).strip().lower() for m in available_methods if str(m).strip()}
        )

    if methods is None:
        if normalized_available is None:
            raise ValueError(
                "available_methods must be provided when methods is None"
            )
        return normalized_available

    normalized_methods: List[str] = []
    for method in methods:
        for part in str(method).split(","):
            method_name = part.strip().lower()
            if method_name:
                normalized_methods.append(method_name)

    if not normalized_methods:
        raise ValueError("No valid methods requested")

    if normalized_available is not None:
        unknown = sorted(set(normalized_methods) - set(normalized_available))
        if unknown:
            raise ValueError(
                "Unknown methods requested: "
                + ", ".join(unknown)
                + ". Available methods: "
                + ", ".join(normalized_available)
            )

    return normalized_methods


def build_plan(
    dataset_config_path: str | os.PathLike[str],
    methods: Sequence[str] | None,
    methods_dir: str | os.PathLike[str],
    available_methods: Iterable[str] | None = None,
) -> ExperimentPlan:
    """Build an ExperimentPlan from one dataset config and one method set."""

    dataset_config_path = Path(dataset_config_path)
    methods_dir = Path(methods_dir) if methods_dir else DEFAULT_METHODS_DIR
    if available_methods is None:
        available_methods = list_available_methods(methods_dir)

    method_names = validate_requested_methods(methods, available_methods)
    raw_dataset_config = load_dataset_config(dataset_config_path)
    dataset_cfg = normalize_dataset_config(raw_dataset_config)
    dataset_name, data_runs = expand_dataset_grid(dataset_cfg)

    method_config_map = {
        method_name: load_method_config(method_name, methods_dir)
        for method_name in method_names
    }
    resolved_method_names = [method_config_map[name].method_key for name in method_names]

    runs: List[RunSpec] = []
    for data_run in data_runs:
        for method_name in method_names:
            method_config = method_config_map[method_name]
            method_params = resolve_method_params_for_dataset(
                method_config.params,
                data_run,
            )
            method_metadata = copy.deepcopy(method_config.metadata)
            merged_params = copy.deepcopy(method_params)
            merged_params.update(data_run)
            merged_params["method_metadata"] = method_metadata
            runs.append(
                RunSpec(
                    dataset_name=dataset_name,
                    method=method_config.method_key,
                    trainer_key=method_config.trainer_key,
                    scenario=str(data_run["scenario"]),
                    selection_strategy=str(data_run["selection_strategy"]),
                    labeled_ratio=float(data_run["labeled_ratio"]),
                    seed=int(data_run["random_seed"]),
                    experiment_name=build_experiment_name(
                        dataset_name=dataset_name,
                        scenario=str(data_run["scenario"]),
                        strategy=str(data_run["selection_strategy"]),
                        c=float(data_run["labeled_ratio"]),
                        seed=int(data_run["random_seed"]),
                    ),
                    dataset_params=dict(data_run),
                    method_params=copy.deepcopy(method_params),
                    method_metadata=method_metadata,
                    params=merged_params,
                    case_control_mode=data_run.get("case_control_mode"),
                )
            )

    return ExperimentPlan(
        dataset_config_path=dataset_config_path,
        dataset_name=dataset_name,
        methods=tuple(resolved_method_names),
        methods_dir=methods_dir,
        raw_dataset_config=copy.deepcopy(raw_dataset_config),
        dataset_runs=tuple(copy.deepcopy(run) for run in data_runs),
        runs=tuple(runs),
    )


def run_spec_to_dict(run: RunSpec) -> Dict[str, Any]:
    return {
        "dataset_name": run.dataset_name,
        "method": run.method,
        "trainer_key": run.trainer_key,
        "scenario": run.scenario,
        "selection_strategy": run.selection_strategy,
        "labeled_ratio": run.labeled_ratio,
        "seed": run.seed,
        "experiment_name": run.experiment_name,
        "dataset_params": copy.deepcopy(run.dataset_params),
        "method_params": copy.deepcopy(run.method_params),
        "method_metadata": copy.deepcopy(run.method_metadata),
        "params": copy.deepcopy(run.params),
        "case_control_mode": run.case_control_mode,
    }


def plan_to_dict(plan: ExperimentPlan) -> Dict[str, Any]:
    return {
        "dataset_config_path": str(plan.dataset_config_path),
        "dataset_name": plan.dataset_name,
        "methods": list(plan.methods),
        "methods_dir": str(plan.methods_dir),
        "total_runs": plan.total_runs,
        "raw_dataset_config": copy.deepcopy(plan.raw_dataset_config),
        "dataset_runs": copy.deepcopy(list(plan.dataset_runs)),
        "runs": [run_spec_to_dict(run) for run in plan.runs],
    }


def plans_to_dict(plans: Sequence[ExperimentPlan]) -> Dict[str, Any]:
    plan_list = [plan_to_dict(plan) for plan in plans]
    return {
        "total_runs": sum(plan["total_runs"] for plan in plan_list),
        "datasets": [plan["dataset_name"] for plan in plan_list],
        "methods": list(dict.fromkeys(method for plan in plans for method in plan.methods)),
        "plans": plan_list,
        "runs": [
            run for plan in plan_list for run in plan["runs"]
        ],
    }


__all__ = [
    "ExperimentPlan",
    "RunSpec",
    "build_experiment_name",
    "build_plan",
    "expand_dataset_grid",
    "load_dataset_config",
    "plan_to_dict",
    "plans_to_dict",
    "resolve_method_params_for_dataset",
    "run_spec_to_dict",
    "validate_requested_methods",
]
