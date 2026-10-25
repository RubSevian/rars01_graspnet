#!/usr/bin/env python3
"""Replay recorded pregrasp targets with production IK; never imports robot IO.

This is a kinematic diagnostic, not execution approval. Historical joints, legs
and scene are absent from short journals; reference seeds are explicitly synthetic.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rars01_graspnet.config import load_config
from rars01_graspnet.real_grasp_ik import IkChecker, _plan_cartesian_segment
from utils.transforms import pose6d_to_mat4, mat4_to_pose6d


def replay(workspace, journal, prior_report):
    config_path = workspace / 'repos/rars01_graspnet/config/jetson_orin_nano.yaml'
    before = hashlib.sha256(config_path.read_bytes()).hexdigest()
    cfg = load_config(config_path)
    grasp = cfg['grasp_pipeline']['grasp']
    model = pin.buildModelFromUrdf(str(workspace / 'repos/rars01_description/urdf/rars01_control.urdf'))
    arm = SimpleNamespace(groups={'arm': SimpleNamespace(num_joints=6)},
                          load_kinematic_model=lambda: model,
                          get_state=lambda **kw: (np.zeros(6), np.zeros(6)))
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    source = next(row for row in reversed(records)
                  if row.get('operation') == 'grasp_plan' and row.get('event') == 'compute_result'
                  and row.get('summary', {}).get('failed_ik'))
    lower, upper = model.lowerPositionLimit[:6], model.upperPositionLimit[:6]
    rng = np.random.default_rng(7)
    seeds = [('zero', np.zeros(6)), ('midpoint', (lower + upper) / 2)]
    seeds += [(f'random_{i}', rng.uniform(lower, upper)) for i in range(11)]
    prior = json.loads(prior_report.read_text())['rows'] if prior_report else []
    rows = []
    for failed in source['summary']['failed_ik']:
        pre = tuple(failed['target_pose6d'])
        tcp = pose6d_to_mat4(*pre)
        tcp[:3, 3] += float(grasp['pregrasp_offset_m']) * tcp[:3, 0]
        end = mat4_to_pose6d(tcp)
        retreat = tcp.copy()
        retreat[:3, 3] -= float(grasp.get('retreat_offset_m', grasp['pregrasp_offset_m'])) * tcp[:3, 0]
        retreat = mat4_to_pose6d(retreat)
        refs = seeds + [(f'independent_{r["orientation_tolerance_deg"]}_deg', np.array(r['joints']))
                        for r in prior if r['candidate'] == failed['candidate']]
        for deg in (2, 10):
            cart = dict(grasp['cartesian_ik'], ik_orientation_tolerance_deg=deg)
            checker = IkChecker(arm, 3, float(cart['ik_position_tolerance_m']), np.deg2rad(deg),
                                bounded_fallback=grasp.get('ik_bounded_fallback_enabled', False))
            for name, reference in refs:
                logs = []
                checker.begin_operation(logs)
                checker.context = {'stage': 'pregrasp'}
                pre_solution = checker.solve(*pre, reference_joints=reference)
                row = dict(candidate=failed['candidate'], tolerance_deg=deg, reference=name,
                           reference_joints=reference.tolist(), pregrasp_pose=list(pre),
                           grasp_pose=list(end), retreat_pose=list(retreat),
                           pregrasp_ok=bool(pre_solution.success), approach_ok=False,
                           retreat_ok=False, collision_status='NOT_RUN_KINEMATIC_PATH_INCOMPLETE')
                if pre_solution.success:
                    approach = _plan_cartesian_segment(checker, pre, end, pre_solution.joints,
                                                       cart, 'approach', verbose=False)
                    row['approach_ok'] = approach is not None
                    if approach is not None:
                        back = _plan_cartesian_segment(checker, end, retreat, approach[-1],
                                                       cart, 'retreat', verbose=False)
                        row['retreat_ok'] = back is not None
                        if back is not None:
                            row['collision_status'] = 'UNVERIFIED_MISSING_RECORDED_SCENE'
                            row['joint_path'] = [q.tolist() for q in (pre_solution.joints, *approach, *back)]
                row['ik'] = logs
                rows.append(row)
                print(f"{failed['candidate']} {deg}deg {name}: pre={row['pregrasp_ok']} "
                      f"approach={row['approach_ok']} retreat={row['retreat_ok']}", flush=True)
    assert before == hashlib.sha256(config_path.read_bytes()).hexdigest()
    return dict(source=str(journal), request_id=source.get('request_id'),
                exact_recorded_seed_available=False, recorded_collision_scene_available=False,
                config_unchanged=True, config_sha256=before,
                note='No hardware. No claim of global infeasibility. No collision approval.', rows=rows)


def independent_path_diagnostic(workspace, report):
    """Diagnostic alternative only; does not replace production IK or collision checks."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from rars01_graspnet.pinocchio_math import get_end_effector_frame_id
    from rars01_graspnet.real_grasp_ik import _cartesian_waypoint_pose
    m = pin.buildModelFromUrdf(str(workspace / 'repos/rars01_description/urdf/rars01_control.urdf'))
    d = m.createData()
    f = get_end_effector_frame_id(m)
    lo, hi = m.lowerPositionLimit[:6], m.upperPositionLimit[:6]
    results=[]
    for row in report['rows']:
     if not row['pregrasp_ok'] or row['reference']!='independent_10_deg':continue
     q=np.array(row['ik'][0]['joints']);out={'candidate':row['candidate'],'tolerance_deg':10,'method':'scipy least_squares normalized FK residual, diagnostic only','waypoints':[]};tol=np.deg2rad(10)
     for stage,start,end in [('approach',row['pregrasp_pose'],row['grasp_pose']),('retreat',row['grasp_pose'],row['retreat_pose'])]:
      for i in range(1,21):
       T=pose6d_to_mat4(*_cartesian_waypoint_pose(start,end,i/20))
       def err(q):
        full=pin.neutral(m);full[:6]=q;pin.forwardKinematics(m,d,full);pin.updateFramePlacements(m,d);pose=d.oMf[f]
        return np.r_[pose.translation-T[:3,3],Rotation.from_matrix(T[:3,:3].T@pose.rotation).as_rotvec()]
       def fun(q):return err(q)/np.r_[np.full(3,.004),np.full(3,tol)]
       fit=least_squares(fun,np.clip(q,lo+1e-12,hi-1e-12),bounds=(lo,hi),max_nfev=500,ftol=1e-12,xtol=1e-12,gtol=1e-12)
       e=err(fit.x);pe=np.linalg.norm(e[:3]);oe=np.linalg.norm(e[3:]);step=np.max(np.abs(fit.x-q));ok=bool(pe<=.004 and oe<=tol and step<=np.deg2rad(15))
       point=dict(stage=stage,waypoint=i,position_mm=pe*1000,orientation_deg=float(np.rad2deg(oe)),max_joint_step_deg=float(np.rad2deg(step)),accepted=ok,joints=fit.x.tolist());out['waypoints'].append(point);print(point,flush=True)
       if not ok:break
       q=fit.x.copy()
      if not ok:break
     out['full_path_accepted']=len(out['waypoints'])==40 and all(x['accepted'] for x in out['waypoints']);results.append(out)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--prior-report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--independent-search', action='store_true',
                        help='Also diagnose the 10-degree path with SciPy; never approves execution')
    args = parser.parse_args()
    report = replay(args.workspace, args.journal, args.prior_report)
    if args.independent_search:
        report['independent_path_diagnostic'] = independent_path_diagnostic(args.workspace, report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
