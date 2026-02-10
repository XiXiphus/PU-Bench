"""method_loader.py - Load per-method hyperparameters from the new config directory.

This utility provides a stable API for:
- Listing available methods (by YAML filename stems)
- Loading a specific method's flattened hyperparameter dictionary

All method YAMLs are stored under Reissue/config/methods/ and should be
flattened (no YAML anchors/merges), with a simple top-level mapping:

  method_name:
    <param>: <value>
    ...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List

import yaml


DEFAULT_METHODS_DIR = Path(__file__).resolve().parent / "methods"


def list_available_methods(methods_dir: str | os.PathLike | None = None) -> List[str]:
    """Return all method names available in the methods directory.

    Method names are derived from YAML filenames without extension.
    """
    dir_path = Path(methods_dir) if methods_dir else DEFAULT_METHODS_DIR
    return sorted([p.stem.lower() for p in dir_path.glob("*.yaml")])


def load_method_params(
    method_name: str, methods_dir: str | os.PathLike | None = None
) -> Dict[str, Any]:
    """Load flattened hyperparameters for a given method.

    Expects a YAML file at <methods_dir>/<method_name>.yaml containing a top-level
    key equal to the method name.
    """
    dir_path = Path(methods_dir) if methods_dir else DEFAULT_METHODS_DIR
    yaml_path = dir_path / f"{method_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Method config not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}
    if method_name not in data:
        raise KeyError(
            f"Top-level key '{method_name}' not found in {yaml_path}. "
            f"Ensure the YAML is flattened and keyed by the method name."
        )
    params = data[method_name] or {}
    if not isinstance(params, dict):
        raise TypeError(
            f"Method params for '{method_name}' must be a mapping, got {type(params)}"
        )
    return params
