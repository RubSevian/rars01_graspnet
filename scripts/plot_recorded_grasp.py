#!/usr/bin/env python3
"""Matplotlib reconstruction of a recorded offline IK path; no robot IO."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/grasp_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.transforms import pose6d_to_mat4, transform_graspnet_grasp_to_end_link_base_with_retreat
from rars01_graspnet.config import load_config


def main(root, output):
    report=json.loads((root/'runtime/grasp_debug/offline_path_bounded_fallback.json').read_text())
    row=next(r for r in report['rows'] if r['retreat_ok'] and r['candidate']=='rank=1 flip=1')
    cfg=load_config(root/'repos/rars01_graspnet/config/jetson_orin_nano.yaml')
    gripper=cfg['robot']['gripper']['rars01']; grasp=cfg['grasp_pipeline']['grasp']
    with np.load(root/'runtime/grasp_debug/latest.npz',allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
        if meta['request_id'] != report['request_id']:raise ValueError('Snapshot/request mismatch')
        # Historical snapshot used depth for the filtered candidate array.
        candidates=data['depth'].copy()
    candidate=candidates[np.argsort(-candidates[:,0])][0]
    R=candidate[4:13].reshape(3,3) @ np.diag([1.,-1.,-1.])
    tool=np.eye(4);tool[:3,:3]=gripper['R_grasp_End_link'];tool[:3,3]=gripper['jaw_center_End_link_m']
    camera_poses=transform_graspnet_grasp_to_end_link_base_with_retreat(
        candidate[13:16],R,candidate[3],np.eye(4),tool,
        grasp['pregrasp_offset_m'],grasp.get('retreat_offset_m',grasp['pregrasp_offset_m']),False)
    pre=pose6d_to_mat4(*row['pregrasp_pose']);target=pose6d_to_mat4(*row['grasp_pose'])
    camera_to_base=pre @ np.linalg.inv(pose6d_to_mat4(*camera_poses[1]))
    object_point=(camera_to_base @ np.r_[candidate[13:16],1])[:3]
    model=pin.buildModelFromUrdf(str(root/'repos/rars01_description/urdf/rars01_control.urdf'))
    frame=model.getFrameId('End_link'); data=model.createData()
    def fk(q):
        full=pin.neutral(model);full[:6]=q
        pin.forwardKinematics(model,data,full);pin.updateFramePlacements(model,data)
        pts=np.array([np.zeros(3)]+[data.oMi[model.getJointId(f'joint{i}')].translation.copy() for i in range(1,7)]+[data.oMf[frame].translation.copy()])
        return pts,data.oMf[frame].homogeneous.copy()
    start_pts,start_T=fk(row['joint_path'][0]);end_pts,end_T=fk(row['joint_path'][20])
    tcp=np.array([fk(q)[1][:3,3] for q in row['joint_path'][:21]])
    fig=plt.figure(figsize=(17,9),layout='constrained')
    gs=fig.add_gridspec(2,2,width_ratios=[1.25,1]);ax=fig.add_subplot(gs[:,0],projection='3d');zoom=fig.add_subplot(gs[0,1],projection='3d');side=fig.add_subplot(gs[1,1])
    blue='#377ac2';green='#128474';orange='#d87714'
    def draw_gripper(axis,T,color):
        # Symbol only: use configured jaw-center offset; not a collision mesh.
        center=np.array(gripper['jaw_center_End_link_m'],dtype=float)
        width=float(candidate[1]);points=np.array([center+[.04,-width/2,0],center+[0,-width/2,0],center+[0,width/2,0],center+[.04,width/2,0]])
        world=(T[:3,:3]@points.T).T+T[:3,3]
        axis.plot(*world.T,color=color,lw=2)
    for axis in (ax,zoom):
        for pts,T,color,label in ((start_pts,start_T,blue,'Рука в предпозиции (IK)'),(end_pts,end_T,green,'Рука у захвата (IK)')):
            shown=pts if axis is ax else pts[-2:]
            axis.plot(*shown.T,'o-',color=color,lw=2.7,ms=4,label=label)
            draw_gripper(axis,T,color)
        axis.scatter(0,0,0,c='black',s=70,label='Основание руки / робот условно')
        axis.scatter(*object_point,c=orange,s=100,marker='o',label='Опорная точка GraspNet на объекте')
        axis.plot(*np.array([pre[:3,3],target[:3,3]]).T,'--',color='black',lw=1.8,label='Заданный подход 8 см')
        axis.plot(*tcp.T,color='#a753ba',lw=1.5,label='TCP по найденным точкам IK')
        axis.scatter(*target[:3,3],c='red',marker='x',s=65,label='Цель TCP (не центр кружки)')
        axis.quiver(*target[:3,3],*(target[:3,0]*.06),color='red',arrow_length_ratio=.2)
        axis.quiver(*end_T[:3,3],*(end_T[:3,0]*.06),color=green,arrow_length_ratio=.2)
        axis.set(xlabel='X основания руки, м',ylabel='Y, м',zlabel='Z, м');axis.view_init(elev=23,azim=-65)
    ax.set_title('Реконструкция руки по URDF\nОфлайн-кандидат №1, переворот схвата, допуск 10°',fontsize=12)
    allpoints=np.vstack([start_pts,end_pts,object_point]);center=(allpoints.min(0)+allpoints.max(0))/2;radius=np.ptp(allpoints,axis=0).max()*.62
    for setter,c in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):setter(c-radius,c+radius)
    ax.set_box_aspect((1,1,1));ax.legend(loc='upper left',fontsize=8)
    center=(pre[:3,3]+target[:3,3])/2
    for setter,c in zip((zoom.set_xlim,zoom.set_ylim,zoom.set_zlim),center):setter(c-.10,c+.10)
    zoom.set_box_aspect((1,1,1));zoom.set_title('Крупно: схват и точка объекта\nКрасная стрелка — заданная +X, зелёная — найденная',fontsize=10)
    side.plot([pre[0,3],target[0,3]],[pre[2,3],target[2,3]],'ko--',label='Проекция подхода на X–Z')
    side.plot(tcp[:,0],tcp[:,2],color='#a753ba',label='TCP по IK')
    side.scatter(object_point[0],object_point[2],color=orange,s=80)
    side.annotate('Объект: точка GraspNet',object_point[[0,2]],xytext=(-145,-25),textcoords='offset points',arrowprops={'arrowstyle':'->'})
    side.annotate('Предпозиция',pre[[0,2],3],xytext=(5,15),textcoords='offset points')
    side.annotate('Цель TCP',target[[0,2],3],xytext=(-55,-20),textcoords='offset points')
    delta=(target[:3,3]-pre[:3,3])*100
    side.set_title(f'8 см по локальной +X схвата:\nΔX={delta[0]:+.2f}, ΔY={delta[1]:+.2f}, ΔZ={delta[2]:+.2f} см',fontsize=10)
    side.set(xlabel='X основания руки, м',ylabel='Z, м');side.axis('equal');side.grid(alpha=.3);side.legend(fontsize=8)
    fig.suptitle('Расчётная схема, не измеренная поза робота. Форма и ориентация кружки неизвестны.\nСхват нарисован условно; коллизии не подтверждены. Длина 8 см — полный 3D-отрезок.',fontsize=12)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=160);fig.savefig(output.with_suffix('.svg'))
    output.with_suffix('.json').write_text(json.dumps(dict(request_id=report['request_id'],candidate=row['candidate'],reference=row['reference'],object_graspnet_point_base_m=object_point.tolist(),pregrasp_base_m=pre[:3,3].tolist(),tcp_target_base_m=target[:3,3].tolist(),approach_delta_cm=delta.tolist(),approach_angle_below_xy_deg=float(np.rad2deg(np.arcsin(-target[2,0]))),note='Offline IK at 10 degrees; object shape unknown; collisions unverified'),indent=2)+'\n')
    print(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.workspace,a.output)
