"""Pure 100 Hz RARS trajectory generator shared by real driver and simulation.

The interpolation and duration formulas are extracted unchanged from GraspDriver.
No transport, SDK or ROS imports are allowed here.
"""
from __future__ import annotations

import numpy as np


def joint_motion_samples(q_start, q_end, minimum_duration_s, dt, velocity_limits):
    q_start = np.asarray(q_start, dtype=np.float64).reshape(-1)
    q_end = np.asarray(q_end, dtype=np.float64).reshape(q_start.shape)
    velocity_limits = np.asarray(velocity_limits, dtype=np.float64).reshape(q_start.shape)
    if minimum_duration_s <= 0.0 or dt <= 0.0 or np.any(velocity_limits <= 0.0):
        raise ValueError("RARS01 duration, controller dt and velocity limits must be positive")
    required_duration = float(np.max(1.875 * np.abs(q_end - q_start) / (0.8 * velocity_limits)))
    duration = max(float(minimum_duration_s), required_duration)
    sample_count = max(2, int(np.ceil(duration / dt)) + 1)
    phase = np.linspace(0.0, 1.0, sample_count)
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    points = [q_start + factor * (q_end - q_start) for factor in blend]
    return points, duration


def waypoint_motion_samples(q_start, waypoints, minimum_duration_s, dt, velocity_limits):
    q_start = np.asarray(q_start, dtype=np.float64).reshape(-1)
    q_targets = [np.asarray(point, dtype=np.float64).reshape(q_start.shape) for point in waypoints]
    velocity_limits = np.asarray(velocity_limits, dtype=np.float64).reshape(q_start.shape)
    if not q_targets or minimum_duration_s <= 0.0 or dt <= 0.0 or np.any(velocity_limits <= 0.0):
        raise ValueError("RARS01 waypoints, duration, controller dt and limits must be valid")
    q_all = [q_start, *q_targets]
    segment_minimums = np.array(
        [float(np.max(1.875 * np.abs(q_next - q_prev) / (0.8 * velocity_limits)))
         for q_prev, q_next in zip(q_all[:-1], q_all[1:])], dtype=np.float64)
    required_duration = float(segment_minimums.sum())
    duration = max(float(minimum_duration_s), required_duration)
    delta_norms = np.array(
        [float(np.linalg.norm(q_next - q_prev)) for q_prev, q_next in zip(q_all[:-1], q_all[1:])])
    if float(delta_norms.sum()) > 1e-12:
        extra_weights = delta_norms / delta_norms.sum()
    else:
        extra_weights = np.full(len(segment_minimums), 1.0 / len(segment_minimums))
    segment_durations = segment_minimums + (duration - required_duration) * extra_weights
    points: list[np.ndarray] = []
    for q_prev, q_next, segment_duration in zip(q_all[:-1], q_all[1:], segment_durations):
        sample_count = max(1, int(np.ceil(segment_duration / dt)))
        phase = np.arange(1, sample_count + 1, dtype=np.float64) / sample_count
        blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
        points.extend(q_prev + factor * (q_next - q_prev) for factor in blend)
    return points, duration
