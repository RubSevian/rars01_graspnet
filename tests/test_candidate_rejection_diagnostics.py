"""Candidate rejection reasons; synthetic candidates, no camera or motor IO."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from rars01_graspnet.real_grasp_selection import _select_executable_grasp


def select(ranked, diagnostics):
    with patch('rars01_graspnet.real_grasp_selection._rank_grasps', return_value=ranked):
        return _select_executable_grasp(
            SimpleNamespace(current_joints=lambda:np.zeros(6),joint_limits=lambda:(np.full(6,-3.),np.full(6,3.))),
            SimpleNamespace(current_joints=lambda:np.zeros(6)), None, np.eye(4),
            .1,.1,0.,.1,0.,0.,'rars01',False,False,np.zeros(3),16,{},
            dict(weight_grasp=1.,weight_joint=1.,weight_motion=1.,motion_normalization=1.),
            rejection_diagnostics=diagnostics)


def test_empty_candidates_are_not_reported_as_unreachable():
    diagnostics=[]
    assert select([],diagnostics) is None
    assert diagnostics==[dict(reason='NO_CANDIDATES')]


def test_excessive_depth_is_reported_before_ik():
    class Ranked(list):
        scores=np.array([.9])
    diagnostics=[]
    assert select(Ranked([SimpleNamespace(depth=.2)]),diagnostics) is None
    assert diagnostics==[dict(reason='DEPTH_LIMIT',rank=1,measured_m=.2,limit_m=.1)]
