"""Transport-free RARS01 joint limits for simulation and offline planning.

The real adapter owns the same SDK/URDF intersection and command epsilon.
This module only reads ArmConfiguration; it never constructs RarsArm or opens serial.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

import numpy as np

from rars01_graspnet.config import resolve_path


def _sdk_configuration(config):
    hardware = config["robot"]["rars01"]
    candidates = []
    override = os.environ.get("RARS01_SDK_PYTHON_PATH")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(resolve_path(config, hardware["sdk_python_path"]))
    # Existing standalone grasp checkout on the development machine.
    candidates.append(Path.home() / "RARS_sdk_grasp_net/rars_arm_sdk/build-python")
    for path in candidates:
        if path.is_dir() and any(path.glob("rars_arm_py*.so")):
            sys.path.insert(0, str(path))
            return importlib.import_module("rars_arm_py").ArmConfiguration(), str(path)
    raise RuntimeError(
        "RARS_SDK_CONFIG_UNAVAILABLE: set RARS01_SDK_PYTHON_PATH to the "
        "built rars_arm_sdk/build-python directory; no serial was opened"
    )


def arm_limit_contract(config, model):
    """Intersect unchanged RARS SDK and URDF bounds using real-driver signs."""
    sdk_config, sdk_path = _sdk_configuration(config)
    hardware = config["robot"]["rars01"]
    directions = np.asarray(hardware["joint_directions"], dtype=float)
    if len(directions) != 7 or not np.all(np.isin(directions, (-1.0, 1.0))):
        raise ValueError("RARS01 joint_directions must contain seven +/-1 signs")
    if model.nq < 6:
        raise ValueError("RARS01 control URDF must contain six arm joints")
    sdk_lower = np.empty(6)
    sdk_upper = np.empty(6)
    for index in range(6):
        motor = sdk_config.motor(index)
        sign = directions[index] * float(motor.direction)
        transformed = (sign * float(motor.joint_position_min),
                       sign * float(motor.joint_position_max))
        sdk_lower[index], sdk_upper[index] = min(transformed), max(transformed)
    urdf_lower = np.asarray(model.lowerPositionLimit[:6], dtype=float).copy()
    urdf_upper = np.asarray(model.upperPositionLimit[:6], dtype=float).copy()
    lower = np.maximum(urdf_lower, sdk_lower)
    upper = np.minimum(urdf_upper, sdk_upper)
    if np.any(lower >= upper):
        raise ValueError("RARS01 URDF and SDK joint limits do not overlap")
    epsilon = float(hardware["command_limit_epsilon_rad"])
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("RARS01 command_limit_epsilon_rad must be finite and non-negative")
    return lower, upper, epsilon, {
        "sdk_python_path": sdk_path,
        "urdf_lower_rad": urdf_lower.tolist(), "urdf_upper_rad": urdf_upper.tolist(),
        "sdk_lower_rad": sdk_lower.tolist(), "sdk_upper_rad": sdk_upper.tolist(),
        "joint_directions": directions[:6].tolist(),
        "command_limit_epsilon_rad": epsilon,
    }


def command_positions_within_limits(values, lower, upper, epsilon):
    """Apply original GraspDriver command-side epsilon and clamp to hard bounds."""
    values = np.asarray(values, dtype=float)
    violation = np.maximum(lower - values, values - upper)
    if not np.all(np.isfinite(values)) or float(np.max(violation)) > epsilon:
        index = int(np.argmax(violation))
        raise RuntimeError(
            f"RARS01 joint{index + 1} start {values[index]:.6f} rad outside "
            f"[{lower[index]:.6f}, {upper[index]:.6f}] by more than "
            f"command_limit_epsilon_rad={epsilon:.6f}"
        )
    return np.clip(values, lower, upper)
