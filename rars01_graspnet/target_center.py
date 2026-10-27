"""Visible-surface centre for soft ranking; no IK, inference or robot IO."""
import numpy as np


def transform_points(points, transform):
    t=np.asarray(transform,dtype=float)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]):
        raise ValueError('INVALID_CENTER_TRANSFORM')
    if not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(t[:3,:3]),1.,atol=1e-5):
        raise ValueError('INVALID_CENTER_ROTATION')
    return np.asarray(points,dtype=float)@t[:3,:3].T+t[:3,3]


def target_center(depth_mm,K,mask,camera_to_base,config,*,mask_source='segmentation',min_depth=.05,max_depth=1.):
    """Voxel-balanced trimmed centroid of measured visible target in arm base.

    Not a centre of mass. Ambiguous PCA shape does not disable this feature.
    """
    inactive=lambda reason,**kw:dict(active=False,reason=reason,frame='base_link',**kw)
    if mask is None or mask_source!='segmentation':return inactive('NO_SEGMENTATION')
    d=np.asarray(depth_mm,dtype=float)/1000.;m=np.asarray(mask,dtype=bool);k=np.asarray(K,dtype=float)
    if d.ndim!=2 or m.shape!=d.shape or k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:raise ValueError('INVALID_CENTER_INPUT')
    voxel=float(config.get('center_voxel_m',.003));fraction=float(config.get('center_min_depth_fraction',.7));minimum=int(config.get('center_min_points',100));trim=float(config.get('center_trim_quantile',.95));floor=float(config.get('center_min_radius_m',.02))
    if not np.isfinite([voxel,fraction,trim,floor]).all() or voxel<=0 or floor<=0 or not 0<fraction<=1 or not .5<trim<=1 or minimum<3:raise ValueError('INVALID_CENTER_CONFIG')
    if not m.any():return inactive('EMPTY_MASK')
    if m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any():return inactive('CLIPPED_TARGET')
    valid=m&np.isfinite(d)&(d>=min_depth)&(d<=max_depth)&(d>0)
    coverage=float(valid.sum()/m.sum())
    if coverage<fraction:return inactive('LOW_DEPTH_COVERAGE',coverage=coverage)
    v,u=np.nonzero(valid);z=d[valid]
    points=transform_points(np.column_stack(((u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z)),camera_to_base)
    # Equal voxel weight, with mean position inside each voxel (no artificial depth).
    _,inverse=np.unique(np.floor(points/voxel).astype(np.int64),axis=0,return_inverse=True)
    counts=np.bincount(inverse);points=np.column_stack([np.bincount(inverse,weights=points[:,i])/counts for i in range(3)])
    if len(points)<minimum:return inactive('TOO_FEW_POINTS',points=len(points),coverage=coverage)
    distances=np.linalg.norm(points-np.median(points,axis=0),axis=1)
    points=points[distances<=np.quantile(distances,trim)]
    if len(points)<minimum:return inactive('TOO_FEW_POINTS_AFTER_TRIM',points=len(points),coverage=coverage)
    center=points.mean(axis=0);radius=max(floor,float(np.quantile(np.linalg.norm(points-center,axis=1),trim)))
    return dict(active=True,reason='VISIBLE_SURFACE',frame='base_link',center_base_m=center.tolist(),radius_m=radius,points=len(points),coverage=coverage)


def center_feature(center,translation_camera,rotation_camera,depth_m,camera_to_base,*,finger_length_m=.08,offset_base=(0.,0.,0.)):
    """Middle of the usable jaw length, between inner faces; not End_link TCP."""
    if not np.isfinite(finger_length_m) or finger_length_m<=0:raise ValueError('INVALID_CENTER_FINGER_LENGTH')
    p=np.asarray(translation_camera)+np.asarray(rotation_camera)[:,0]*(depth_m-finger_length_m/2)
    p=transform_points(p,camera_to_base)+np.asarray(offset_base)
    result=dict(active=False,cost=0.,grasp_center_base_m=p.tolist())
    if center and center.get('active'):
        if center.get('frame')!='base_link':raise ValueError('CENTER_FRAME_MISMATCH')
        radius=float(center['radius_m']);c=np.asarray(center['center_base_m'])
        if c.shape!=(3,) or not np.isfinite(c).all() or not np.isfinite(radius) or radius<=0 or not np.isfinite(p).all():raise ValueError('INVALID_TARGET_CENTER')
        distance=float(np.linalg.norm(p-c))
        result.update(active=True,cost=min(distance/radius,1.),distance_m=distance,target_center_base_m=c.tolist(),radius_m=radius)
    return result
