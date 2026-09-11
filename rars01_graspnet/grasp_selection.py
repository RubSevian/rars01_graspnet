"""Pure, testable costs for selecting an executable grasp candidate."""

from __future__ import annotations

import numpy as np


def normalized_grasp_costs(scores: np.ndarray) -> np.ndarray:
    """Return ``1 - minmax(score)``; equal scores deliberately tie."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values
    span = float(values.max() - values.min())
    if span <= np.finfo(np.float64).eps:
        return np.full(values.shape, 0.5, dtype=np.float64)
    return 1.0 - (values - values.min()) / span


def joint_limit_cost(
    configurations: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Maximum squared normalized distance to the centre of joint limits."""
    q = np.asarray(configurations, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64).reshape(1, -1)
    hi = np.asarray(upper, dtype=np.float64).reshape(1, -1)
    midpoint = 0.5 * (lo + hi)
    normalized = 2.0 * np.abs(q - midpoint) / (hi - lo)
    return float(np.clip(np.max(normalized * normalized), 0.0, 1.0))


def motion_cost(
    configurations: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    normalization: float,
) -> float:
    """Normalized accumulated joint motion for the executed configuration chain."""
    q = np.asarray(configurations, dtype=np.float64)
    if len(q) < 2:
        return 0.0
    ranges = np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64)
    normalized_steps = np.diff(q, axis=0) / ranges.reshape(1, -1)
    distance = float(np.linalg.norm(normalized_steps, axis=1).sum())
    if normalization <= 0.0:
        raise ValueError("motion normalization must be positive")
    return float(np.clip(distance / normalization, 0.0, 1.0))
