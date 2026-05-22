"""Load per-method hyperparameters and metadata."""

from __future__ import annotations

import os
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .schema_adapter import normalize_method_config


DEFAULT_METHODS_DIR = Path(__file__).resolve().parent / "methods"


@dataclass(frozen=True, slots=True)
class MethodConfig:
    method_key: str
    trainer_key: str
    params: Dict[str, Any]
    metadata: Dict[str, Any]
    source_path: Path


def list_method_configs(
    methods_dir: str | os.PathLike | None = None,
) -> List[MethodConfig]:
    """Return normalized method configs discovered from a methods directory."""

    dir_path = Path(methods_dir) if methods_dir else DEFAULT_METHODS_DIR
    configs = [_load_method_config_from_path(path) for path in sorted(dir_path.glob("*.yaml"))]

    seen: dict[str, Path] = {}
    for config in configs:
        previous = seen.get(config.method_key)
        if previous is not None:
            raise ValueError(
                f"Duplicate method_key '{config.method_key}' in {previous} and {config.source_path}"
            )
        seen[config.method_key] = config.source_path
    return configs


def list_available_methods(methods_dir: str | os.PathLike | None = None) -> List[str]:
    """Return all method keys declared in the methods directory."""

    return sorted(config.method_key for config in list_method_configs(methods_dir))


def load_method_config(
    method_name: str,
    methods_dir: str | os.PathLike | None = None,
) -> MethodConfig:
    """Load one normalized method config by method key."""

    requested = str(method_name).strip().lower()
    dir_path = Path(methods_dir) if methods_dir else DEFAULT_METHODS_DIR

    exact_path = dir_path / f"{requested}.yaml"
    if exact_path.exists():
        exact_config = _load_method_config_from_path(exact_path)
        if exact_config.method_key == requested:
            return exact_config

    matches = [
        config for config in list_method_configs(dir_path) if config.method_key == requested
    ]
    if not matches:
        raise FileNotFoundError(f"Method config not found for method_key: {requested}")
    if len(matches) > 1:
        paths = ", ".join(str(config.source_path) for config in matches)
        raise ValueError(f"Duplicate method config for method_key '{requested}': {paths}")
    return matches[0]


def load_method_params(
    method_name: str, methods_dir: str | os.PathLike | None = None
) -> Dict[str, Any]:
    """Load flattened hyperparameters for a given method key."""

    return load_method_config(method_name, methods_dir).params


def _load_method_config_from_path(path: Path) -> MethodConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(
            f"Method config '{path}' must deserialize to a mapping, got {type(data).__name__}"
        )

    default_method_key, entry = _extract_method_entry(data, fallback_method_key=path.stem.lower())
    extends = entry.pop("extends", None)
    if extends:
        base_config = load_method_config(str(extends), path.parent)
        method_key, trainer_key, params, metadata = normalize_method_config(
            {default_method_key: entry},
            fallback_method_key=default_method_key,
        )
        merged_params = copy.deepcopy(base_config.params)
        merged_params.update(params)
        merged_metadata = copy.deepcopy(base_config.metadata)
        merged_metadata.update(metadata)
        return MethodConfig(
            method_key=method_key,
            trainer_key=trainer_key,
            params=merged_params,
            metadata=merged_metadata,
            source_path=path,
        )

    method_key, trainer_key, params, metadata = normalize_method_config(
        data,
        fallback_method_key=path.stem.lower(),
    )
    return MethodConfig(
        method_key=method_key,
        trainer_key=trainer_key,
        params=params,
        metadata=metadata,
        source_path=path,
    )


def _extract_method_entry(
    data: Dict[str, Any],
    *,
    fallback_method_key: str,
) -> tuple[str, Dict[str, Any]]:
    if fallback_method_key in data and isinstance(data[fallback_method_key], dict):
        return fallback_method_key, dict(data[fallback_method_key])
    if len(data) == 1:
        only_key = next(iter(data))
        only_value = data[only_key]
        if isinstance(only_value, dict):
            return str(only_key).lower(), dict(only_value)
    return fallback_method_key, dict(data)


__all__ = [
    "DEFAULT_METHODS_DIR",
    "MethodConfig",
    "list_available_methods",
    "list_method_configs",
    "load_method_config",
    "load_method_params",
]
