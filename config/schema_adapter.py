"""Compatibility adapters for PU-Bench config schemas."""

from __future__ import annotations

from typing import Any, Mapping


DATASET_SECTIONS = ("dataset", "sweep", "runtime")
METHOD_SECTIONS = ("metadata", "method", "runtime")


def _as_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping, got {type(value).__name__}")
    return dict(value)


def normalize_dataset_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten legacy or namespaced dataset config into trainer-ready params."""

    raw = dict(raw_config)
    if not any(section in raw for section in DATASET_SECTIONS):
        return raw

    normalized = {
        key: value for key, value in raw.items() if key not in DATASET_SECTIONS
    }
    for section in DATASET_SECTIONS:
        normalized.update(
            _as_mapping(raw.get(section), context=f"dataset config section '{section}'")
        )
    return normalized


def normalize_method_config(
    raw_config: Mapping[str, Any],
    *,
    fallback_method_key: str,
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    """Flatten legacy or namespaced method config.

    Returns ``(method_key, trainer_key, params, metadata)``.
    """

    raw = dict(raw_config)
    method_key = fallback_method_key.lower()

    if method_key in raw and isinstance(raw[method_key], Mapping):
        entry = dict(raw[method_key])
        return _normalize_method_entry(entry, default_method_key=method_key)

    if len(raw) == 1:
        only_key = next(iter(raw))
        only_value = raw[only_key]
        if isinstance(only_value, Mapping) and only_key not in METHOD_SECTIONS:
            return _normalize_method_entry(
                dict(only_value), default_method_key=str(only_key).lower()
            )

    return _normalize_method_entry(raw, default_method_key=method_key)


def _normalize_method_entry(
    entry: Mapping[str, Any],
    *,
    default_method_key: str,
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    entry_dict = dict(entry)
    metadata = _as_mapping(
        entry_dict.get("metadata"), context="method config section 'metadata'"
    )
    method_key = str(metadata.get("method_key", default_method_key)).lower()
    trainer_key = str(metadata.get("trainer_key", method_key)).lower()

    if "method" in entry_dict or "runtime" in entry_dict:
        params: dict[str, Any] = {
            key: value for key, value in entry_dict.items() if key not in METHOD_SECTIONS
        }
        params.update(
            _as_mapping(entry_dict.get("method"), context="method config section 'method'")
        )
        params.update(
            _as_mapping(
                entry_dict.get("runtime"), context="method config section 'runtime'"
            )
        )
    else:
        params = {key: value for key, value in entry_dict.items() if key != "metadata"}

    return method_key, trainer_key, params, metadata


__all__ = ["normalize_dataset_config", "normalize_method_config"]
