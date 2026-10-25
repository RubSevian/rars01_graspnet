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

# Recorded pregrasp: feasible at 10 degrees despite strict solver stagnation.
RECORDED_POSE = (.325752, -.280068, -.103402, .615557, .338793, -.362985)
RECORDED_Q = np.array([-1.0262440730073026, 2.601655993756175,
                       .6275959268525444, 1.4, -.656429372507589,
                       .2699073327414773])


def test_recorded_pose_acceptance_uses_separate_fk_budgets():
    c = checker()
    c._position_tolerance_m = .004
    c._orientation_tolerance_rad = np.deg2rad(10.)
    logs = []
    c.begin_operation(logs)
    solution = c.solve(*RECORDED_POSE, reference_joints=RECORDED_Q)
    assert solution.success
    assert solution.position_error_m <= .004
    assert solution.orientation_error_rad <= np.deg2rad(10.)
    assert logs[0]['attempts'][0]['accepted']
    assert not logs[0]['attempts'][0]['solver_success']
    np.testing.assert_allclose(c.fk_errors(RECORDED_POSE, solution.joints),
                               [solution.position_error_m, solution.orientation_error_rad])
    c._orientation_tolerance_rad = np.deg2rad(2.)
    c.begin_operation([])
    assert not c.solve(*RECORDED_POSE, reference_joints=RECORDED_Q).success


def test_feasible_intermediate_pose_survives_later_invalid_iterate():
    from rars01_graspnet.pinocchio_math import IKResult
    c = checker()
    q = np.array([.2, 1., .8, -.3, .1, -.4])
    p, r, _ = compute_fk(c._model, np.r_[q, 0., 0.])
    pose = (*p, *pin.rpy.matrixToRpy(r))
    bad_q = q.copy(); bad_q[0] += .8
    bad_p, bad_r, _ = compute_fk(c._model, np.r_[bad_q, 0., 0.])

    def solver(*args, iterate_callback, **kwargs):
        iterate_callback(q, .1, p, r)
        iterate_callback(bad_q, .09, bad_p, bad_r)
        return IKResult(bad_q, False, .09, 2, 'no_improving_step')

    c._solve_ik = solver
    result = c.solve(*pose, reference_joints=q)
    assert result.success
    np.testing.assert_array_equal(result.joints, q)


@pytest.mark.parametrize('failure', ['position', 'orientation', 'limit', 'nan'])
def test_solver_success_never_overrides_fk_or_joint_checks(failure):
    from rars01_graspnet.pinocchio_math import IKResult
    c = checker()
    q = np.array([.2, 1., .8, -.3, .1, -.4])
    p, r, _ = compute_fk(c._model, np.r_[q, 0., 0.])
    target = pin.SE3(r, p)
    if failure == 'position':
        target.translation = p + [.1, 0., 0.]
    elif failure == 'orientation':
        target.rotation = r @ pin.rpy.rpyToMatrix(.3, 0., 0.)
    elif failure == 'limit':
        q[0] = c._upper[0] + .01
        p, r, _ = compute_fk(c._model, np.r_[q, 0., 0.])
        target = pin.SE3(r, p)
    else:
        q[0] = np.nan

    def solver(*args, iterate_callback, **kwargs):
        iterate_callback(q, 0., p, r)
        return IKResult(q, True, 0., 0, 'initial_converged')

    c._solve_ik = solver
    result = c.solve(*target.translation, *pin.rpy.matrixToRpy(target.rotation),
                     reference_joints=np.zeros(6))
    assert not result.success


def test_bounded_fallback_completes_recorded_approach_and_retreat():
    from rars01_graspnet.real_grasp_ik import _plan_cartesian_segment
    from utils.transforms import pose6d_to_mat4, mat4_to_pose6d
    c = checker(); c._bounded_fallback = True
    c._position_tolerance_m = .004
    c._orientation_tolerance_rad = np.deg2rad(10)
    logs = []; c.begin_operation(logs)
    pre = c.solve(*RECORDED_POSE, reference_joints=RECORDED_Q)
    assert pre.success
    t = pose6d_to_mat4(*RECORDED_POSE); t[:3, 3] += .08*t[:3, 0]
    end = mat4_to_pose6d(t)
    cfg = dict(waypoint_count=20, joint_step_warn_deg=10, max_joint_step_deg=15,
               ik_position_tolerance_m=.004, ik_orientation_tolerance_deg=10)
    approach = _plan_cartesian_segment(c, RECORDED_POSE, end, pre.joints, cfg, 'approach', False)
    assert approach is not None
    back = _plan_cartesian_segment(c, end, RECORDED_POSE, approach[-1], cfg, 'retreat', False)
    assert back is not None
    assert any(a.get('method') == 'bounded_least_squares' and a['accepted']
               for log in logs for a in log['attempts'])
    assert all(log['position_error_m'] <= .004 and
               log['orientation_error_rad'] <= np.deg2rad(10) for log in logs)


def test_bounded_fallback_cancellation_and_strict_rejection():
    c = checker(); c._bounded_fallback = True
    target = c._pos_rot_to_se3(np.array(RECORDED_POSE[:3]),
                              roll=RECORDED_POSE[3], pitch=RECORDED_POSE[4], yaw=RECORDED_POSE[5])
    c._position_tolerance_m = .004; c._orientation_tolerance_rad = np.deg2rad(2)
    assert not c._solve_bounded(target, [RECORDED_Q], {'attempts': []}).success
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 3: raise RuntimeError('MISSION_CANCELLED')
    c.cancel_check = cancel
    record = {'attempts': []}
    with pytest.raises(RuntimeError, match='MISSION_CANCELLED'):
        c._solve_bounded(target, [RECORDED_Q], record)
    assert calls == 3 and record['attempts'][-1]['termination'] == 'interrupted'


def test_primary_success_does_not_run_fallback(monkeypatch):
    c = checker(); c._bounded_fallback = True
    def unexpected(*args): raise AssertionError('fallback called after primary success')
    monkeypatch.setattr(c, '_solve_bounded', unexpected)
    q = np.array([.2, 1., .8, -.3, .1, -.4])
    p, r, _ = compute_fk(c._model, np.r_[q, 0., 0.])
    assert c.solve(*p, *pin.rpy.matrixToRpy(r), reference_joints=q).success


def test_bounded_solver_preparation_is_ready_before_first_solve(monkeypatch):
    import builtins
    base=checker()
    c=IkChecker(base._arm,position_tolerance_m=.004,
                orientation_tolerance_rad=np.deg2rad(7.5),bounded_fallback=True)
    assert c.solver_preparation['ready']
    prepared=dict(c.solver_preparation)
    assert c.prepare_bounded_solver()==prepared
    original_import=builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if name=='scipy.optimize':raise AssertionError('Late SciPy import in mission')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded_import)
    target=c._pos_rot_to_se3(np.array(RECORDED_POSE[:3]),
                            roll=RECORDED_POSE[3],pitch=RECORDED_POSE[4],yaw=RECORDED_POSE[5])
    result=c._solve_bounded(target,[RECORDED_Q],{'attempts':[]})
    assert result.success
    assert result.position_error_m<=.004
    assert result.orientation_error_rad<=np.deg2rad(7.5)
