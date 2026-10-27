"""Offline-safe geometry and timing features for soft grasp ranking."""
import numpy as np
from .real_grasp_trajectory import waypoint_motion_samples


def target_shape(depth_mm, K, mask, config, *, mask_source='segmentation', min_depth=.05, max_depth=1.):
    """Visible target PCA, gated by coverage, elongation and subsample agreement.

    Confidence is a heuristic gate, not a calibrated probability or contact test.
    All vectors stay in the capture camera frame, like GraspNet rotations.
    """
    inactive=lambda reason,**kw:dict(active=False,confidence=0.,reason=reason,**kw)
    if mask_source!='segmentation' or mask is None:return inactive('NO_SEGMENTATION')
    d=np.asarray(depth_mm,dtype=float)/1000.;m=np.asarray(mask,dtype=bool);K=np.asarray(K,dtype=float)
    if d.ndim!=2 or m.shape!=d.shape or K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0:
        raise ValueError('INVALID_SHAPE_INPUT')
    if not m.any():return inactive('EMPTY_MASK')
    if m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any():return inactive('CLIPPED_TARGET')
    valid=m&np.isfinite(d)&(d>=min_depth)&(d<=max_depth)&(d>0)
    coverage=float(valid.sum()/m.sum())
    if coverage<float(config.get('shape_min_depth_fraction',.7)):return inactive('LOW_DEPTH_COVERAGE',coverage=coverage)
    v,u=np.nonzero(valid);z=d[valid]
    xyz=np.column_stack(((u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z))
    # Deterministic 3 mm voxel representation reduces pixel-density bias.
    voxel=float(config.get('shape_voxel_m',.003))
    if not np.isfinite(voxel) or voxel<=0:raise ValueError('INVALID_SHAPE_VOXEL')
    _,indices=np.unique(np.floor(xyz/voxel).astype(np.int64),axis=0,return_index=True)
    xyz=xyz[np.sort(indices)]
    lo,hi=np.quantile(xyz,[.02,.98],axis=0);xyz=xyz[np.all((xyz>=lo)&(xyz<=hi),axis=1)]
    if len(xyz)<int(config.get('shape_min_points',100)):return inactive('TOO_FEW_POINTS',points=len(xyz))
    def pca(points):
        values,vectors=np.linalg.eigh(np.cov(points,rowvar=False))
        return values,vectors[:,-1]
    values,axis=pca(xyz);ratio=float(values[-1]/max(values[-2],1e-12))
    low=float(config.get('shape_ratio_min',2.5));high=float(config.get('shape_ratio_full',5.))
    if not 1<low<high:raise ValueError('INVALID_SHAPE_RATIO')
    angles=[float(np.rad2deg(np.arccos(np.clip(abs(axis@pca(xyz[i::2])[1]),0,1)))) for i in (0,1)]
    details=dict(axis_camera=axis.tolist(),center_camera=np.median(xyz,axis=0).tolist(),
                 eigenvalue_ratio=ratio,subsample_angle_deg=max(angles),coverage=coverage,points=len(xyz))
    if max(angles)>float(config.get('shape_max_axis_disagreement_deg',15.)):return inactive('UNSTABLE_AXIS',**details)
    confidence=float(np.clip((ratio-low)/(high-low),0,1))
    return dict(active=confidence>0,confidence=confidence,reason='ELONGATED' if confidence>0 else 'AMBIGUOUS_SHAPE',**details)


def route_time(current, pre, approach, retreat, robot, retreat_speed_scale=1.):
    """Quantized stream time from the same waypoint generator; excludes settling/HOME.

    Includes pregrasp, approach, retreat and return to scan; closure/confirmation
    is excluded because its duration depends on future contact feedback.
    """
    motion=robot['motion'];minimum=float(motion.get('minimum_duration_s',4.5))
    ready=max(minimum,float(robot['ready_pose']['duration']));back=max(minimum,float(motion['retreat_duration_s']))
    rate=float(robot['rars01']['command_rate_hz']);vel=np.asarray(robot['rars01']['position_velocity_limits_rad_s'][:6])
    if not np.isfinite(retreat_speed_scale) or not 0<retreat_speed_scale<=1:raise ValueError('INVALID_RETREAT_SPEED_SCALE')
    phases=((current,[pre],ready,1.),(pre,approach,ready,1.),
            (approach[-1],retreat,ready,retreat_speed_scale),(retreat[-1],[current],back,retreat_speed_scale))
    return sum(len(waypoint_motion_samples(start,targets,duration,1/rate,vel)[0])/rate/scale
               for start,targets,duration,scale in phases)


def route_ik_errors(ik, pre_pose, grasp_pose, retreat_pose, pre, approach, retreat):
    """Independent FK residuals at all commanded Cartesian waypoints; no IK solve."""
    errors=[ik.fk_errors(pre_pose,pre)]
    for begin,end,points in ((pre_pose,grasp_pose,approach),(grasp_pose,retreat_pose,retreat)):
        for i,q in enumerate(points,1):
            pose=np.asarray(begin,dtype=float).copy()
            pose[:3]+=(np.asarray(end[:3])-pose[:3])*i/len(points)
            errors.append(ik.fk_errors(tuple(pose),q))
    return np.asarray(errors)
