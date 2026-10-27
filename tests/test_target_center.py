import numpy as np
import pytest
from rars01_graspnet.target_center import target_center,center_feature,transform_points


def scene():
    depth=np.full((30,40),700.);mask=np.zeros(depth.shape,bool);mask[5:25,7:33]=True
    return depth,np.array([[200.,0,20],[0,200,15],[0,0,1]]),mask


def test_cloud_and_center_follow_camera_transform():
    d,k,m=scene();cfg={'center_min_points':10,'center_trim_quantile':1.}
    base=target_center(d,k,m,np.eye(4),cfg)
    t=np.eye(4);t[:3,:3]=[[0,0,1],[1,0,0],[0,1,0]];t[:3,3]=[.1,.2,.3]
    rotated=target_center(d,k,m,t,cfg)
    np.testing.assert_allclose(rotated['center_base_m'],transform_points(base['center_base_m'],t),atol=1e-9)
    assert np.isclose(rotated['radius_m'],base['radius_m'])


def test_mask_depth_gates_and_ambiguous_shape_not_required():
    d,k,m=scene();assert target_center(d,k,m,np.eye(4),{})['active']
    d[m]=0;assert target_center(d,k,m,np.eye(4),{})['reason']=='LOW_DEPTH_COVERAGE'
    d[:]=700;m[0,10]=True;assert target_center(d,k,m,np.eye(4),{})['reason']=='CLIPPED_TARGET'
    assert not target_center(d,k,m,np.eye(4),{},mask_source='bbox')['active']


def test_working_jaw_center_not_tcp_and_flip_invariance():
    center=dict(active=True,frame='base_link',center_base_m=[0,0,0],radius_m=.1)
    a=center_feature(center,[0,0,0],np.eye(3),.04,np.eye(4));assert a['cost']==0
    b=center_feature(center,[0,0,0],np.diag([1,-1,-1]),.05,np.eye(4))
    assert np.isclose(b['distance_m'],.01)
    c=center_feature(center,[0,0,0],np.eye(3),.05,np.eye(4));assert b==c
    assert center_feature(None,[0,0,0],np.eye(3),.04,np.eye(4))['cost']==0


def test_reject_invalid_transform_and_config():
    d,k,m=scene();t=np.eye(4);t[0,0]=2
    with pytest.raises(ValueError):target_center(d,k,m,t,{})
    with pytest.raises(ValueError):target_center(d,k,m,np.eye(4),{'center_voxel_m':0})


def test_center_adds_bounded_cost_only_when_enabled():
    from rars01_graspnet.grasp_selection import candidate_cost_terms
    args=(np.zeros((2,6)),-np.ones(6),np.ones(6),.5,.08,.1,[0,0,-1])
    old=candidate_cost_terms(*args,{'weight_center':0},center={'active':True,'cost':1.})
    new=candidate_cost_terms(*args,{'weight_center':.05},center={'active':True,'cost':.4})
    assert np.isclose(new['total']-old['total'],.02)
    missing=candidate_cost_terms(*args,{'weight_center':.05},center={'active':False})
    assert missing['total']==old['total']
