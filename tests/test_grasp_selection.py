import numpy as np

from rars01_graspnet.grasp_selection import (
    joint_limit_cost,
    motion_cost,
    normalized_grasp_costs,
)


def test_grasp_scores_are_normalized_before_costing():
    assert np.allclose(normalized_grasp_costs(np.array([4.0, 2.0, 3.0])), [0.0, 1.0, 0.5])
    assert np.allclose(normalized_grasp_costs(np.array([4.0, 4.0])), [0.5, 0.5])


def test_joint_and_motion_costs_use_joint_ranges():
    lower = np.array([-1.0, -2.0])
    upper = np.array([1.0, 2.0])
    chain = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]])

    assert joint_limit_cost(chain, lower, upper) == 1.0
    assert np.isclose(motion_cost(chain, lower, upper, normalization=2.0), np.sqrt(0.125))


def test_cost_order_matches_exhaustive_validation_with_ties_and_rejections():
    from types import SimpleNamespace
    from rars01_graspnet.grasp_selection import best_validated_candidate
    rng = np.random.default_rng(20261024)
    for _ in range(500):
        candidates = [SimpleNamespace(cost=float(rng.integers(0, 6)), valid=bool(rng.integers(0, 2)))
                      for _ in range(int(rng.integers(0, 30)))]
        expected = min((c for c in candidates if c.valid), key=lambda c: c.cost, default=None)
        seen = []; metrics = {}
        def validate(candidate):
            seen.append(candidate)
            return candidate.valid
        actual = best_validated_candidate(candidates, validate, metrics=metrics)
        assert actual is expected
        assert metrics['checked_count'] == len(seen)
        assert [c.cost for c in seen] == sorted(c.cost for c in seen)
        if actual is not None:
            assert seen[-1] is actual
            assert all(not c.valid for c in seen[:-1])


def test_validation_errors_and_cancel_are_not_candidate_rejection():
    import pytest
    from types import SimpleNamespace
    from rars01_graspnet.grasp_selection import best_validated_candidate
    candidates = [SimpleNamespace(cost=0.), SimpleNamespace(cost=1.)]
    def fail(*args):
        raise RuntimeError('COLLISION_CANCELLED:SHUTDOWN')
    with pytest.raises(RuntimeError, match='CANCELLED'):
        best_validated_candidate(candidates, fail)
    with pytest.raises(RuntimeError, match='CANCELLED'):
        best_validated_candidate(candidates, lambda c: True, cancel_check=fail)
    for value in (float('nan'), float('inf')):
        with pytest.raises(ValueError, match='nonfinite'):
            best_validated_candidate([SimpleNamespace(cost=value)], lambda c: True)


def test_candidate_audit_preserves_selection_and_records_both_orientations(capsys):
    # Exercise the production selector without importing CUDA or the camera SDK.
    import ast
    import copy
    import __future__
    from pathlib import Path
    from types import SimpleNamespace as NS
    from rars01_graspnet.grasp_selection import best_validated_candidate
    source=ast.parse((Path(__file__).resolve().parents[1]/'rars01_graspnet/real_grasp_selection.py').read_text())
    names={'_select_executable_grasp','_translate_pose','_pose_z_ok'}
    functions=[node for node in source.body if isinstance(node,ast.FunctionDef) and node.name in names]
    class Group:
        def __init__(self,n):
            self.items=[NS(score=3.-i,width=.05,height=.02,depth=.01,
                           translation=np.array([.1,.2,.3]),rotation_matrix=np.eye(3)) for i in range(n)]
            self.scores=np.array([g.score for g in self.items]);self.grasp_group_array=np.arange(n*17).reshape(n,17)
        def __len__(self):return len(self.items)
        def __getitem__(self,i):return self.items[i]
    def poses(grasp,*a,**kw):
        pose=(.1,.2,.3,0.,0.,0.)
        return pose,pose,pose
    scope=dict(np=np,normalized_grasp_costs=normalized_grasp_costs,joint_limit_cost=joint_limit_cost,
        motion_cost=motion_cost,best_validated_candidate=best_validated_candidate,
        _rank_grasps=lambda *a,**k:Group(2),_parallel_flip_grasp=copy.deepcopy,
        graspnet_utils=NS(grasp_to_base_poses=poses),ExecutableGrasp=lambda **kw:NS(**kw),
        IkSolution=lambda success,error,joints:NS(success=success,error=error,joints=joints))
    exec(compile(ast.Module(body=functions,type_ignores=[]),'production_selector','exec',flags=__future__.annotations.compiler_flag),scope)
    def run(audit,verbose=True):
        calls=[]
        def solve(*a,**kw):
            calls.append(1)
            # First branch executable, second rejected at pregrasp.
            return NS(success=len(calls)<=3,error=0.,joints=np.zeros(6))
        ik=NS(current_joints=lambda:np.zeros(6),joint_limits=lambda:(-np.ones(6),np.ones(6)),solve=solve)
        rejection=[];metrics={}
        candidate=scope['_select_executable_grasp'](ik,NS(grasp_tcp_transform=lambda w:np.eye(4)),Group(3),np.eye(4),
            .08,.08,0.,.1,0.,0.,'synthetic',True,True,np.zeros(3),1,{},
            dict(weight_grasp=.5,weight_joint=.25,weight_motion=.25,motion_normalization=2.),
            rejection_diagnostics=rejection,selection_metrics=metrics,candidate_diagnostics=audit,verbose=verbose)
        return candidate,calls,rejection,metrics
    plain,plain_calls,plain_rejection,plain_metrics=run(None)
    audit={};logged,calls,rejection,metrics=run(audit)
    assert logged.cost==plain.cost and calls==plain_calls
    assert rejection==plain_rejection and metrics==plain_metrics
    assert audit['input_count']==3 and audit['after_nms']==2 and audit['after_candidate_limit']==1
    assert audit['ik_feasible']==1 and audit['rejected_ik']==1
    assert audit['selected']=='rank=1 flip=0'
    assert [r['status'] for r in audit['orientations']]==['ik_feasible','rejected']
    assert audit['orientations'][1]['failed_stage']=='pregrasp'
    assert audit['orientations'][0]['translation_camera_m']==[.1,.2,.3]
    assert audit['orientations'][0]['pregrasp_End_link_pose6d']==[.1,.2,.3,0.,0.,0.]
    capsys.readouterr()
    quiet,quiet_calls,quiet_rejection,quiet_metrics=run(None,verbose=False)
    assert capsys.readouterr().out==''
    assert quiet.cost==plain.cost and quiet_calls==plain_calls and quiet_rejection==plain_rejection
    assert quiet_metrics==plain_metrics
    import json
    json.dumps(audit,allow_nan=False)
