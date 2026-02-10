from typing import Any, Dict, List, Tuple

import itertools
import yaml


def load_dataset_config(cfg_path: str) -> Dict[str, Any]:
    """Load dataset YAML configuration.

    The configuration may define grid lists like:
      - random_seeds, c_values, scenarios, selection_strategies
    and fixed knobs like:
      - data_dir, val_ratio, target_prevalence, with_replacement, print_stats
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def expand_dataset_grid(
    dataset_cfg: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Expand dataset configuration into per-run dictionaries via cartesian product.

    Returns (dataset_class, runs_list)
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

    # Backward-compatible aliases / defaults
    # - allow using 'also_print_dataset_stats' in YAML; map to 'print_stats'
    if "print_stats" not in base and "also_print_dataset_stats" in base:
        base["print_stats"] = bool(base.pop("also_print_dataset_stats"))

    base.setdefault("data_dir", "./")
    base.setdefault("val_ratio", 0.0)
    base.setdefault("target_prevalence", None)
    base.setdefault("with_replacement", True)
    base.setdefault("print_stats", False)

    runs: List[Dict[str, Any]] = []
    for seed, c, scn, strat, cc_mode in itertools.product(
        seeds, c_vals, scenarios, strategies, cc_modes
    ):
        d = dict(base)
        d["random_seed"] = seed
        # Alias for trainers expecting 'seed'
        d["seed"] = int(seed)
        d["labeled_ratio"] = float(c)
        d["scenario"] = scn
        d["selection_strategy"] = strat
        d["case_control_mode"] = cc_mode
        for k in [
            "random_seeds",
            "c_values",
            "scenarios",
            "selection_strategies",
            "case_control_modes",
        ]:
            d.pop(k, None)
        runs.append(d)

    return str(dataset_class), runs


__all__ = ["load_dataset_config", "expand_dataset_grid"]
