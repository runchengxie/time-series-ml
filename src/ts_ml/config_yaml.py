"""YAML configuration file loader.

Loads experiment parameters from a YAML file.
CLI arguments always take precedence over YAML values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return a dict of overrides.

    Only keys present in the YAML are returned — missing keys
    fall through to Settings defaults.
    """
    try:
        import yaml
    except ImportError:
        print("[config] PyYAML not installed; install with: uv pip install pyyaml")
        return {}

    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        print(f"[config] {path}: expected a mapping, got {type(data).__name__}")
        return {}

    return {k: v for k, v in data.items() if v is not None}
