from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a small platform override without mutating its base."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(config_path: Path, seen: set[Path]) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    if config_path in seen:
        chain = " -> ".join(str(path) for path in (*seen, config_path))
        raise ValueError(f"Circular config inheritance: {chain}")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = Path(parent).expanduser()
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    return _merge(_load_yaml(parent_path, seen | {config_path}), config)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "config" / "default.yaml"
    config_path = config_path.expanduser().resolve()
    config = _load_yaml(config_path, set())
    config["_config_dir"] = str(config_path.parent)
    return config


def resolve_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    # Paths in default.yaml are relative to the project root.
    return (PROJECT_ROOT / path).resolve()


def robot_kinematics_config(config: dict[str, Any]) -> tuple[Path, str, str]:
    """Return URDF, base frame and TCP frame for both supported config layouts."""
    robot = config["robot"]
    rars01 = robot.get("rars01")
    if isinstance(rars01, dict):
        return (
            resolve_path(config, rars01["urdf_path"]),
            "base_link",
            rars01.get("end_effector_frame", "End_link"),
        )
    return (
        resolve_path(config, robot["urdf"]),
        robot["base_frame"],
        robot["tcp_frame"],
    )
