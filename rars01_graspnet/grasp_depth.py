"""Depth adjustment before any grasp collision or target-membership filtering."""
import numpy as np


def apply_depth_floor(grasps, minimum_m=0., maximum_m=None):
    minimum=float(minimum_m)
    if not np.isfinite(minimum) or minimum<0:
        raise ValueError('minimum_grasp_depth_m must be finite and non-negative')
    if maximum_m is not None:
        maximum=float(maximum_m)
        if not np.isfinite(maximum) or maximum<=0 or minimum>maximum:
            raise ValueError('minimum_grasp_depth_m exceeds valid max_grasp_depth_m')
    original=np.asarray(grasps)
    if original.ndim!=2 or original.shape[1]!=17 or not np.isfinite(original).all() or np.any(original[:,3]<0):
        raise ValueError('invalid decoded grasp array')
    result=original.astype(np.float64,copy=True)
    changed=result[:,3]<minimum
    result[:,3]=np.maximum(result[:,3],minimum)
    return result,int(changed.sum())
