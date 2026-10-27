"""Cloud occupancy against conservative RARS01 mesh hulls in End_link.
No robot I/O. Non-watertight STL is enclosed by convex hulls, not filled as-is.
The symmetric enclosure preserves the parallel-gripper 180-degree equivalence.
"""
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def validate_options(options):
    c=dict(options or {})
    mode=c.get('model','graspnet')
    if mode not in ('graspnet','graspnet_boxes','rars01_convex'):raise ValueError('Invalid cloud collision model')
    if mode=='graspnet':return c
    if mode=='graspnet_boxes':
        for key in ('finger_length_m','finger_thickness_m','finger_height_m','opening_width_m','approach_distance_m'):
            v=c.get(key)
            if type(v) not in (int,float) or not np.isfinite(v) or not 0<v<=.2:
                raise ValueError('Invalid cloud box '+key)
        return c
    for key in ('urdf_path','mesh_root'):
        if not isinstance(c.get(key),str) or not c[key]:raise ValueError('Missing cloud collision '+key)
    for key in ('approach_distance_m','opening_width_m'):
        v=c.get(key)
        if type(v) not in (int,float) or not np.isfinite(v) or not 0<v<=.2:raise ValueError('Invalid cloud collision '+key)
    v=c.get('max_collision_points')
    if type(v) is not int or v<0:raise ValueError('Invalid max_collision_points')
    return c


def _transform(origin):
    t=np.eye(4)
    if origin is not None:
        t[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
        t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return t


@lru_cache(maxsize=4)
def mesh_vertices(urdf_path,mesh_root):
    import trimesh
    tree=ET.parse(urdf_path).getroot();vertices={}
    for name in ('gripper_left','gripper_right','End_link'):
        link=tree.find(f"link[@name='{name}']");parts=[]
        for collision in link.findall('collision'):
            mesh=collision.find('geometry/mesh')
            filename=mesh.get('filename')
            if not filename.startswith('package://rars01_description/'):raise ValueError('Unsupported mesh URI')
            path=Path(mesh_root)/filename.split('package://rars01_description/')[1]
            points=np.asarray(trimesh.load(path,force='mesh').vertices)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            t=_transform(collision.find('origin'))
            if name!='End_link':
                joint=tree.find(f"joint[@name='{name}_joint']")
                expected=np.array([0,1 if name=='gripper_left' else -1,0])
                if joint.get('type')!='prismatic' or not np.allclose(np.fromstring(joint.find('axis').get('xyz'),sep=' '),expected):raise ValueError('Unsupported jaw joint')
                t=_transform(joint.find('origin'))@t
            parts.append(points@t[:3,:3].T+t[:3,3])
        vertices[name]=np.concatenate(parts)
    return vertices


def hull(vertices):
    v=np.asarray(vertices,float);h=ConvexHull(v)
    return (v.min(0),v.max(0),h.equations)


def inside(points,shape):
    lo,hi,eq=shape;mask=np.zeros(len(points),bool)
    idx=np.flatnonzero(np.all((points>=lo-1e-9)&(points<=hi+1e-9),axis=1))
    if len(idx):mask[idx]=np.all(points[idx]@eq[:,:3].T+eq[:,3]<=1e-9,axis=1)
    return mask


class RarsCloudCollision:
    def __init__(self,scene_points,voxel_size,options):
        self.options=validate_options(options);self.voxel_size=float(voxel_size)
        if not np.isfinite(voxel_size) or voxel_size<=0:raise ValueError('Invalid cloud voxel')
        points=np.asarray(scene_points,np.float32)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Invalid cloud')
        _,idx=np.unique(np.floor(points/voxel_size).astype(np.int64),axis=0,return_index=True)
        self.scene_points=points[np.sort(idx)]
        v=mesh_vertices(self.options['urdf_path'],self.options['mesh_root'])
        flip=np.array([1,-1,-1])
        # Enclose both equivalent orientations, so post-filter IK flips stay valid.
        self.jaw_vertices=np.concatenate([v['gripper_left'],v['gripper_right']*flip])
        self.body_vertices=np.concatenate([v['End_link'],v['End_link']*flip])
        self.jaw=hull(self.jaw_vertices);self.body=hull(self.body_vertices)
        shift=np.array([-self.options['approach_distance_m'],0,0])
        self.jaw_sweep=hull(np.concatenate([self.jaw_vertices,self.jaw_vertices+shift]))
        self.body_sweep=hull(np.concatenate([self.body_vertices,self.body_vertices+shift]))

    def point_masks(self,g):
        g=np.asarray(g,float);w=g[1];depth=g[3];R=g[4:13].reshape(3,3)
        if not np.isfinite(g).all() or w<0 or w>self.options['opening_width_m']+1e-6 or depth<0 or not np.allclose(R.T@R,np.eye(3),atol=1e-5):raise ValueError('Invalid gripper candidate')
        local=(self.scene_points-g[13:16])@R-np.array([depth,0,0])
        result={}
        for side in (-1,1):
            q=local.copy();q[:,1:]*=side
            result['candidate_jaw_'+str(side)]=inside(q-np.array([0,w/2,0]),self.jaw)
            result['open_approach_jaw_'+str(side)]=inside(q-np.array([0,self.options['opening_width_m']/2,0]),self.jaw_sweep)
        result['body_and_approach']=inside(local,self.body_sweep)
        return result

    def evaluate(self,grasps):
        grasps=np.asarray(grasps,float)
        if grasps.ndim!=2 or grasps.shape[1]!=17:raise ValueError('Expected Nx17 grasps')
        rows=[];rejected=[]
        for i,g in enumerate(grasps):
            masks=self.point_masks(g);occupied=np.logical_or.reduce(list(masks.values()));count=int(occupied.sum())
            bad=count>self.options['max_collision_points'];rejected.append(bad)
            rows.append(dict(input_index=i,collision=bad,occupied_points=count,parts={k:int(v.sum()) for k,v in masks.items()}))
        return np.array(rejected,bool),rows
