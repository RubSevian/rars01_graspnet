"""Local inference region in camera metres; never crops the collision cloud."""
import math
import numpy as np


def validate_roi(options):
    if not isinstance(options, dict):
        raise ValueError('graspnet.target_roi must be a mapping')
    cfg = dict(enabled=False, margin_m=.12, lower_quantile=.01, upper_quantile=.99)
    cfg.update(options)
    if type(cfg['enabled']) is not bool:
        raise ValueError('graspnet.target_roi.enabled must be boolean')
    for key in ('margin_m', 'lower_quantile', 'upper_quantile'):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]):
            raise ValueError('invalid graspnet.target_roi.' + key)
    if cfg['margin_m'] <= 0 or not 0 <= cfg['lower_quantile'] < cfg['upper_quantile'] <= 1:
        raise ValueError('invalid graspnet.target_roi bounds')
    return cfg


def inference_region(points, valid_depth, target_mask, mask_source, options):
    cfg = validate_roi(options)
    if not cfg['enabled']:
        return np.arange(len(points)), dict(enabled=False, full_cloud_points=len(points), pool_points=len(points))
    if mask_source != 'segmentation' or target_mask is None:
        raise ValueError('TARGET_ROI_REQUIRES_SEGMENTATION')
    mask = np.asarray(target_mask)
    if mask.shape != valid_depth.shape or not np.isfinite(mask).all():
        raise ValueError('TARGET_ROI_INVALID_MASK')
    target = mask.astype(bool)[valid_depth]
    target_points = points[target]
    if not len(target_points):
        raise ValueError('TARGET_ROI_NO_VALID_DEPTH')
    lo, hi = np.quantile(target_points, [cfg['lower_quantile'], cfg['upper_quantile']], axis=0)
    lo -= cfg['margin_m']
    hi += cfg['margin_m']
    ids = np.flatnonzero(((points >= lo) & (points <= hi)).all(axis=1))
    if not len(ids):
        raise ValueError('TARGET_ROI_EMPTY')
    return ids, dict(cfg, full_cloud_points=len(points), pool_points=len(ids),
                     target_points=int(target.sum()), bounds_camera_m=[lo.tolist(), hi.tolist()])
