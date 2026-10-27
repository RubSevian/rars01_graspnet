import itertools
from pathlib import Path
import numpy as np
import pytest
from rars01_graspnet.cloud_gripper_collision import RarsCloudCollision,mesh_vertices,hull,inside,validate_options
from rars01_graspnet.config import load_config
ROOT=Path(__file__).resolve().parents[1]

def options():return load_config(ROOT/'config/offline_rars_convex_experiment.yaml')['graspnet']['cloud_collision']
def grasp():
    g=np.zeros(17);g[1:4]=[.06,.02,.04];g[4:13]=np.eye(3).ravel();g[13:16]=[.1,.2,.7];return g

def test_hull_contains_surfaces_but_not_free_space():
    cube=np.array(list(itertools.product((-1.,1.),repeat=3)))
    np.testing.assert_array_equal(inside(np.array([[0,0,0],[1,0,0],[1.01,0,0]]),hull(cube)),[True,True,False])

def test_urdf_origins_dimensions_and_enclosure():
    c=options();vs=mesh_vertices(c['urdf_path'],c['mesh_root'])
    np.testing.assert_allclose(vs['gripper_left'].min(0),[-.105,0,-.023],atol=1e-7)
    np.testing.assert_allclose(vs['gripper_left'].max(0),[0,.03,.023],atol=1e-7)
    detector=RarsCloudCollision(np.empty((0,3)),.01,c)
    assert inside(vs['gripper_left'],detector.jaw).all()
    assert inside(vs['gripper_right']*[1,-1,-1],detector.jaw).all()
    assert inside(vs['End_link'],detector.body).all()

def test_depth_translation_flip_and_rigid_frame_invariance():
    c=options();g=grasp();rng=np.random.default_rng(7);points=rng.uniform([-.15,-.12,-.08],[.06,.12,.1],(5000,3))+g[13:16]
    detector=RarsCloudCollision(points,.0001,c);a,_=detector.evaluate(g[None]);m=detector.point_masks(g);hit=np.logical_or.reduce(list(m.values()))
    flip=g.copy();flip[4:13]=(np.eye(3)@np.diag([1,-1,-1])).ravel()
    np.testing.assert_array_equal(hit,np.logical_or.reduce(list(detector.point_masks(flip).values())))
    shifted=g.copy();shifted[3]+=.01;shifted[13]-=.01
    np.testing.assert_array_equal(hit,np.logical_or.reduce(list(detector.point_masks(shifted).values())))
    # Same points in a rotated camera frame, without re-voxelizing them.
    R=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    detector.scene_points=detector.scene_points@R.T
    rotated=g.copy();rotated[4:13]=R.ravel();rotated[13:16]=g[13:16]@R.T
    np.testing.assert_array_equal(hit,np.logical_or.reduce(list(detector.point_masks(rotated).values())))

def test_open_approach_can_collide_when_candidate_width_does_not():
    c=options();g=grasp()
    # At 100mm opening, mid-jaw material sits at Y~65mm, outside a 60mm opening jaw.
    local=np.array([[-.06,.065,0]])
    points=local+np.array([g[3],0,0])+g[13:16]
    detector=RarsCloudCollision(points,.001,c);m=detector.point_masks(g)
    assert not m['candidate_jaw_1'][0] and m['open_approach_jaw_1'][0]
    assert detector.evaluate(g[None])[0][0]

def test_empty_and_invalid_inputs():
    c=options();d=RarsCloudCollision(np.empty((0,3)),.01,c)
    assert not d.evaluate(grasp()[None])[0][0]
    for key,value in [('model','typo'),('max_collision_points',-1),('max_collision_points',True),('opening_width_m',0),('approach_distance_m',float('nan'))]:
        with pytest.raises(ValueError):validate_options(dict(c,**{key:value}))
    with pytest.raises(ValueError):RarsCloudCollision(np.array([[np.nan,0,0]]),.01,c)

def test_actual_infer_routes_to_new_filter_and_preserves_diagnostics(monkeypatch):
    import torch
    from utils import graspnet_utils as gu
    import rars01_graspnet.cloud_gripper_collision as mod
    raw=grasp()[None];seen=[]
    monkeypatch.setattr(gu,'pred_decode',lambda _: [torch.from_numpy(raw)])
    class Detector:
        def __init__(self,cloud,voxel,opts):seen.append(opts)
        def evaluate(self,g):
            assert g[0,3]==.05
            return np.array([True]),[dict(occupied_points=2)]
    monkeypatch.setattr(mod,'RarsCloudCollision',Detector)
    diagnostics={};out,counts=gu.infer_grasps(lambda x:x,{},np.empty((0,3)),.01,minimum_grasp_depth_m=.05,max_grasp_depth_m=.08,cloud_collision=options(),diagnostics=diagnostics)
    assert len(out)==0 and counts['collision_removed']==1
    assert seen[0]['model']=='rars01_convex' and diagnostics['cloud_collision_details'][0]['occupied_points']==2


def test_working_profile_keeps_legacy_collision_threshold_with_configured_boxes():
    cfg = load_config(ROOT/'config/jetson_orin_nano.yaml')
    assert cfg['graspnet']['cloud_collision']['model'] == 'graspnet_boxes'
    assert cfg['graspnet']['cloud_collision']['finger_thickness_m'] == .03
    assert cfg['graspnet']['collision_thresh'] == .01
