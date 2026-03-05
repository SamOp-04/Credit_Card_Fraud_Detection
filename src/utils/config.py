"""Configuration loader for the MLOps pipeline."""

import yaml
from pathlib import Path
from typing import Any


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent.parent
