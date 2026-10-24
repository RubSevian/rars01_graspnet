from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
import pytest
from rars01_graspnet.pinocchio_math import solve_ik, IKParams, compute_fk, get_end_effector_frame_id
from rars01_graspnet.real_grasp_ik import IkChecker

URDF = Path(__file__).resolve().parents[2] / 'rars01_description/urdf/rars01_control.urdf'

def checker():
    model = pin.buildModelFromUrdf(str(URDF))
    arm = SimpleNamespace(groups={'arm': SimpleNamespace(num_joints=6)},
                          load_kinematic_model=lambda: model,
                          get_state=lambda **kw: (np.zeros(6), np.zeros(6)))
    return IkChecker(arm)

def test_cache_uses_exact_target_and_seed_and_resets():
    c=checker(); logs=[];c.begin_operation(logs)
    q=np.array([.2,1.,.8,-.3,.1,-.4]);p,r,_=compute_fk(c._model,np.r_[q,0.,0.])
    pose=(*p,*pin.rpy.matrixToRpy(r))
    a=c.solve(*pose,reference_joints=q);b=c.solve(*pose,reference_joints=q)
    assert a.success and b.success and logs[-1]['cache_hit']
    a.joints[:]=99
    np.testing.assert_array_equal(c.solve(*pose,reference_joints=q).joints,q)
    c.solve(*pose,reference_joints=q+.001)
    assert not logs[-1]['cache_hit']
    assert logs[-1]['attempts'][0]['iterations'] > 0
    record=logs[-1]
    np.testing.assert_allclose(np.array(record['joints'])+record['upper_margin_rad'],record['joint_upper_rad'])
    np.testing.assert_allclose(np.array(record['joints'])-record['lower_margin_rad'],record['joint_lower_rad'])
    assert len(record['attempts'][0]['joints'])==6
    assert record['position_tolerance_m']==c._position_tolerance_m
    c.begin_operation(logs);c.solve(*pose,reference_joints=q)
    assert not logs[-1]['cache_hit']

def test_cancellation_interrupts_inner_solver_and_records_reason():
    c=checker();logs=[];count=0
    def cancel():
        nonlocal count
        count+=1
        if count>=3:raise RuntimeError('MISSION_CANCELLED')
    c.begin_operation(logs,cancel)
    with pytest.raises(RuntimeError,match='MISSION_CANCELLED'):c.solve(2,2,2,0,0,0)
    assert count==3 and logs[-1]['reason']=='MISSION_CANCELLED'

def test_stagnation_preserves_result_without_exhausting_budget():
    c=checker();m=c._model;frame=c._end_frame_id;t=pin.SE3(np.eye(3),np.array([2.,2.,2.]))
    a=solve_ik(m,m.createData(),frame,t,np.zeros(6),IKParams(max_iter=200))
    b=solve_ik(m,m.createData(),frame,t,np.zeros(6),IKParams(max_iter=400))
    assert not a.success and a.iterations < 200
    assert a.termination=='no_improving_step'
    np.testing.assert_array_equal(a.q,b.q)

def test_rebase_preserves_world_tcp_with_rotations():
    from utils.transforms import pose6d_to_mat4,mat4_to_pose6d
    capture=pose6d_to_mat4(.1,.2,.3,.2,-.4,.6)
    live=pose6d_to_mat4(-.3,.1,.4,-.3,.2,-.1)
    camera=pose6d_to_mat4(.1,0,.05,.1,.2,.3)
    tcp=pose6d_to_mat4(.2,.1,.6,.4,-.2,.7)
    rebased=np.linalg.inv(live)@capture@camera@tcp
    np.testing.assert_allclose(live@pose6d_to_mat4(*mat4_to_pose6d(rebased)),capture@camera@tcp,atol=1e-12)
