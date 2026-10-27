"""Configured SDK box approximation, without changing its occupancy criterion."""
import numpy as np


def prepare_candidates(grasps, options):
    """Force the experimental opening, preserving raw network output separately."""
    result = np.asarray(grasps).copy()
    if result.ndim != 2 or result.shape[1] != 17 or not np.isfinite(result).all():
        raise ValueError('Expected finite Nx17 grasps')
    changed = int(np.count_nonzero(result[:, 1] != options['opening_width_m']))
    result[:, 1] = options['opening_width_m']
    return result, changed


def detect_boxes(detector, group_type, grasps, options, threshold):
    """Use the existing SDK detector in small batches; height is filter-only.

    The SDK couples palm thickness to finger thickness. Its normalized volume
    score and strict comparisons are intentionally preserved.
    """
    detector.finger_length = options['finger_length_m']
    detector.finger_width = options['finger_thickness_m']
    model = np.asarray(grasps).copy()
    model[:, 2] = options['finger_height_m']
    rejected = []
    for start in range(0, len(model), 8):
        rejected.extend(detector.detect(
            group_type(model[start:start + 8]),
            approach_dist=options['approach_distance_m'],
            collision_thresh=threshold).tolist())
    return np.asarray(rejected, dtype=bool)
