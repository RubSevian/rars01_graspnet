import numpy as np
import pytest
from rars01_graspnet.grasp_ranking import target_shape,route_time
from rars01_graspnet.grasp_selection import normalized_grasp_costs,candidate_cost_terms
from rars01_graspnet.real_grasp_trajectory import waypoint_motion_samples


def test_fixed_score_does_not_depend_on_competing_candidates():
    c=dict(score_normalization='fixed',score_min=0.,score_max=1.)
    assert normalized_grasp_costs(np.array([.4,.35]),c)[0]==normalized_grasp_costs(np.array([.4,.01,.99]),c)[0]
    assert normalized_grasp_costs(np.array([.4,.4]),c).tolist()==pytest.approx([.6,.6])


def test_shape_gates_and_rotation_invariant_sign():
    d=np.full((160,220),700.);K=np.array([[400.,0,110],[0,400.,80],[0,0,1]])
    m=np.zeros(d.shape,bool);m[70:90,40:180]=True
    shape=target_shape(d,K,m,{})
    assert shape['active'] and abs(shape['axis_camera'][0])>.99
    m[:]=False;m[40:120,70:150]=True
    assert not target_shape(d,K,m,{})['active']
    m[:]=False;m[0:20,40:180]=True
    assert target_shape(d,K,m,{})['reason']=='CLIPPED_TARGET'
    assert not target_shape(d,K,m,{},mask_source='bbox')['active']
    m[:]=False;m[70:90,40:180]=True;d[70:90,40:150]=0
    assert target_shape(d,K,m,{})['reason']=='LOW_DEPTH_COVERAGE'


def test_soft_shape_ik_and_turn_costs_do_not_change_admission():
    base=dict(weight_grasp=0.,weight_motion=0.,weight_joint=0.,weight_width=0.,weight_shape=1.)
    args=(np.array([[0.],[.2],[0.]]),np.array([-3.]),np.array([3.]),0.,.05,.1,[0,0,-1])
    shape=dict(active=True,confidence=.6,axis_camera=[1,0,0])
    cost=lambda axis:candidate_cost_terms(*args,base,shape=shape,closing_axis=axis)['total']
    assert cost([1,0,0])==cost([-1,0,0])==pytest.approx(.6)
    assert cost([0,1,0])==0
    c=dict(base,weight_shape=0.,weight_ik=1.,ik_position_scale_m=.004,ik_orientation_scale_deg=7.5)
    assert candidate_cost_terms(*args,c,ik_errors=[[0.,0.]])['total']==0
    assert candidate_cost_terms(*args,c,ik_errors=[[.004,np.deg2rad(7.5)]])['total']==pytest.approx(1.)
    with pytest.raises(ValueError):candidate_cost_terms(*args,c,ik_errors=[[np.nan,0.]])


def test_route_time_matches_stream_generator_and_trial_scaling():
    robot=dict(motion=dict(minimum_duration_s=4.5,retreat_duration_s=4.5),ready_pose=dict(duration=4.5),
               rars01=dict(command_rate_hz=100,position_velocity_limits_rad_s=[.5]*6))
    q=np.zeros(6);pre=np.ones(6)*.2;approach=[pre+.01,pre+.02];retreat=[pre+.01,pre]
    durations=[len(waypoint_motion_samples(a,b,4.5,.01,np.ones(6)*.5)[0])/100
               for a,b in [(q,[pre]),(pre,approach),(approach[-1],retreat),(retreat[-1],[q])]]
    assert route_time(q,pre,approach,retreat,robot)==pytest.approx(sum(durations))
    assert route_time(q,pre,approach,retreat,robot,.25)==pytest.approx(sum(durations[:2])+4*sum(durations[2:]))


def test_residual_features_follow_every_cartesian_waypoint_without_solving():
    from rars01_graspnet.grasp_ranking import route_ik_errors
    class FK:
        def __init__(self):self.targets=[]
        def fk_errors(self,pose,q):
            self.targets.append(np.array(pose))
            return abs(pose[0]-q[0]),0.
    fk=FK();pre=[0.,0,0,0,0,0];grasp=[.08,0,0,0,0,0]
    errors=route_ik_errors(fk,pre,grasp,pre,[0.],[[.04],[.08]],[[.04],[0.]])
    assert len(errors)==5
    np.testing.assert_allclose(errors,0.)
    np.testing.assert_allclose(np.array(fk.targets)[:,0],[0.,.04,.08,.04,0.])
