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


def best_validated_candidate(candidates, path_validator=None, *, cancel_check=None, metrics=None):
    """Stable cost order preserves exhaustive filter-then-min selection.

    Costs must be finite and validation must not change candidate costs. A
    timeout/cancellation is propagated, never treated as geometric rejection.
    """
    candidates = list(candidates)
    if any(not np.isfinite(item.cost) for item in candidates):
        raise ValueError("nonfinite grasp candidate cost")
    ranked = sorted(enumerate(candidates), key=lambda pair: pair[1].cost)
    if metrics is not None:
        metrics.update(candidate_count=len(candidates), checked_count=0,
                       selected_original_index=None, selected_cost=None)
    for original_index, candidate in ranked:
        if cancel_check is not None:
            cancel_check()
        if path_validator is not None:
            if metrics is not None:
                metrics['checked_count'] += 1
            if not path_validator(candidate):
                continue
        if metrics is not None:
            metrics.update(selected_original_index=original_index, selected_cost=float(candidate.cost))
        return candidate
    return None
