import numpy as np
import pytest
from rars01_graspnet.grasp_depth import apply_depth_floor


def test_floor_preserves_origins_rotations_scores_and_deeper_candidates():
    raw=np.zeros((4,17),dtype=np.float32);raw[:,3]=[.01,.02,.03,.06]
    raw[:,4:13]=np.eye(3).reshape(1,9);raw[:,13:16]=[.1,.2,.7]
    before=raw.copy();out,count=apply_depth_floor(raw,.04,.08)
    assert count==3
    np.testing.assert_allclose(out[:,3],[.04,.04,.04,.06])
    np.testing.assert_array_equal(out[:,np.arange(17)!=3],raw[:,np.arange(17)!=3])
    np.testing.assert_array_equal(raw,before)
    disabled,n=apply_depth_floor(raw,0.,.08)
    np.testing.assert_array_equal(disabled,raw);assert n==0
    for v in (-.01,np.nan,np.inf,.081):
        with pytest.raises(ValueError):apply_depth_floor(raw,v,.08)


def test_adjustment_happens_before_cloud_collision_and_audit_keeps_raw(monkeypatch):
    import torch
    from utils import graspnet_utils as gu
    raw=np.zeros((1,17));raw[0,0]=.5;raw[0,1]=.08;raw[0,2]=.02;raw[0,3]=.03;raw[0,4:13]=np.eye(3).ravel();raw[0,15]=.7
    monkeypatch.setattr(gu,'pred_decode',lambda _: [torch.from_numpy(raw)])
    seen=[]
    class Detector:
        def __init__(self,*a,**kw):pass
        def detect(self,grasps,**kw):
            seen.append(grasps.depths.copy())
            return np.array([True]) # Changed candidate really is rejected, no fallback to old depth.
    monkeypatch.setattr(gu,'ModelFreeCollisionDetector',Detector)
    diagnostics={};out,counts=gu.infer_grasps(lambda x:x,{},np.zeros((1,3)),.01,diagnostics=diagnostics,minimum_grasp_depth_m=.04,max_grasp_depth_m=.08)
    assert len(out)==0 and counts['collision_removed']==1 and counts['depth_adjusted_count']==1
    np.testing.assert_allclose(seen[0],[.04]);assert diagnostics['decoded_grasps'][0,3]==.03
    assert diagnostics['depth_adjusted_grasps'][0,3]==.04
