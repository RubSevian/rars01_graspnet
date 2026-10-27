import numpy as np
import pytest
from rars01_graspnet.real_grasp_ik import reverse_approach_retreat

CFG = dict(waypoint_count=2, max_joint_step_deg=15.,
           ik_position_tolerance_m=.004, ik_orientation_tolerance_deg=7.5)

class Checker:
    cancel_check = None
    def joint_limits(self): return np.full(6, -1.), np.ones(6)
    def fk_errors(self, pose, q): return abs(pose[0]-q[0]), 0.

def inputs():
    pre=np.zeros(6);grasp=pre.copy();grasp[0]=.2
    return pre,grasp,np.array([grasp*.5,grasp])

def test_reverse_preserves_branch_and_returns_to_pregrasp_without_mutation():
    pre,grasp,app=inputs();before=app.copy()
    path=reverse_approach_retreat(Checker(),pre,grasp,pre,pre,app,CFG)
    np.testing.assert_allclose(path,[app[0],pre])
    np.testing.assert_array_equal(app,before)
    path[0][0]=99
    np.testing.assert_array_equal(app,before)

def test_different_retreat_target_requires_independent_ik():
    pre,grasp,app=inputs();other=pre.copy();other[2]=.01
    assert reverse_approach_retreat(Checker(),pre,grasp,other,pre,app,CFG) is None

@pytest.mark.parametrize('kind',['jump','limit','nan','fk','length','rotation'])
def test_invalid_reuse_is_rejected(kind):
    pre,grasp,app=inputs();check=Checker();cfg=dict(CFG)
    if kind=='jump': cfg['max_joint_step_deg']=1.
    if kind=='limit':app[0,1]=2.
    if kind=='nan':app[0,0]=np.nan
    if kind=='fk':app[0,0]=.12
    if kind=='length':app=app[:1]
    if kind=='rotation':check.fk_errors=lambda p,q:(0.,.2)
    assert reverse_approach_retreat(check,pre,grasp,pre,pre,app,cfg) is None

def test_cancel_is_propagated():
    pre,grasp,app=inputs();check=Checker()
    def cancel():raise RuntimeError('cancelled')
    check.cancel_check=cancel
    with pytest.raises(RuntimeError,match='cancelled'):
        reverse_approach_retreat(check,pre,grasp,pre,pre,app,CFG)
