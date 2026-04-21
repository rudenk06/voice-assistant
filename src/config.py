import os
import yaml
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = get_project_root() / "config" / "assistant.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Resolve relative paths against project root
    root = get_project_root()
    _resolve_paths(config, root)
    return config


def _resolve_paths(obj, root: Path):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and ("path" in key or key in ("file",)):
                if not os.path.isabs(value):
                    obj[key] = str(root / value)
            elif isinstance(value, (dict, list)):
                _resolve_paths(value, root)
    elif isinstance(obj, list):
        for item in obj:
            _resolve_paths(item, root)
