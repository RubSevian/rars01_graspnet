"""Configured boxes preserve SDK decisions and propagate opening through inference."""
from pathlib import Path
import numpy as np
import pytest
from rars01_graspnet.config import load_config
from rars01_graspnet.cloud_gripper_collision import validate_options
from rars01_graspnet.graspnet_box_filter import prepare_candidates, detect_boxes
ROOT=Path(__file__).resolve().parents[1]

def options():return load_config(ROOT/'config/jetson_orin_nano.yaml')['graspnet']['cloud_collision']
def grasps(n=19):
 g=np.zeros((n,17));g[:,1:4]=[.065,.02,.04];g[:,4:13]=np.eye(3).ravel();g[:,13]=np.linspace(-.1,.1,n);return g

def test_batched_boxes_match_original_sdk_exactly():
 from utils import graspnet_utils as gu
 raw=grasps();backup=raw.copy();g,count=prepare_candidates(raw,options());assert count==len(raw)
 np.testing.assert_array_equal(raw,backup)
 points=np.random.default_rng(12).uniform([-.2,-.1,-.04],[.15,.1,.04],size=(2400,3))
 det=gu.ModelFreeCollisionDetector(points,voxel_size=.01)
 actual=detect_boxes(det,gu.GraspGroup,g,options(),.01)
 expected_g=g.copy();expected_g[:,2]=.03
 expected=det.detect(gu.GraspGroup(expected_g),approach_dist=.05,collision_thresh=.01)
 np.testing.assert_array_equal(actual,expected)
 np.testing.assert_array_equal(g[:,2],raw[:,2])
 assert not detect_boxes(det,gu.GraspGroup,g[:0],options(),.01).any()

@pytest.mark.parametrize('threshold',[.01,0.])
def test_live_infer_preserves_raw_and_forces_opening_even_when_filter_disabled(monkeypatch,threshold):
 import torch
 from utils import graspnet_utils as gu
 raw=grasps(2);monkeypatch.setattr(gu,'pred_decode',lambda _:[torch.from_numpy(raw)])
 diagnostics={}
 out,counts=gu.infer_grasps(lambda x:x,{},np.empty((0,3)),threshold,minimum_grasp_depth_m=.045,max_grasp_depth_m=.08,cloud_collision=options(),diagnostics=diagnostics)
 assert len(out)==2 and counts['width_adjusted_count']==2
 np.testing.assert_array_equal(diagnostics['decoded_grasps'],raw)
 np.testing.assert_array_equal(diagnostics['depth_adjusted_grasps'][:,1],raw[:,1])
 assert np.all(out.widths==.1) and np.all(out.depths==.045)
 assert np.all(out.heights==.02)

@pytest.mark.parametrize('key,value',[('finger_thickness_m',0),('finger_height_m',True),('finger_length_m',float('nan')),('opening_width_m',-.1),('approach_distance_m',None)])
def test_bad_dimensions_rejected(key,value):
 with pytest.raises(ValueError):validate_options(dict(options(),**{key:value}))
