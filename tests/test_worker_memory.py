"""No camera, model or robot: worker payload/lifecycle and preprocessing parity."""
import ast
import importlib.util
import json
import pickle
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('memory_worker',ROOT/'utils/graspnet_worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

class Group:
    def __init__(self,array):self.grasp_group_array=array

class Connection:
    def __init__(self,request):self.requests=iter([request,{'type':'stop'}]);self.sent=[];self.closed=False
    def recv(self):return next(self.requests)
    def send(self,value):self.sent.append(value)
    def close(self):self.closed=True

def exercise(diagnostics,n=10000,crash=False):
    array=np.arange(n*17,dtype=np.float64).reshape(n,17)
    def points(*args,**kwargs):
        assert kwargs['visualize'] is False
        if crash:raise ValueError('synthetic failure')
        return {},None,np.zeros((8,3))
    def infer(*args,diagnostics=None):
        if diagnostics is not None:diagnostics['decoded_grasps']=array.copy()
        return Group(array),{'decoded':n}
    fake=types.SimpleNamespace(build_net=lambda *a:object(),build_end_points=points,
        infer_grasps=infer,
        filter_grasps_by_bbox=lambda g,*a,**kw:Group(g.grasp_group_array[::2]),
        filter_grasps_by_width=lambda g,*a:Group(g.grasp_group_array[::2]),
        filter_grasps_by_depth=lambda g,*a:g)
    package=types.ModuleType('utils');package.graspnet_utils=fake
    request=dict(type='infer',color_bgr=np.zeros((4,4,3),np.uint8),depth_mm=np.ones((4,4),np.uint16),K=np.eye(3),num_point=10000,min_depth=.1,max_depth=1.,collision_thresh=.01,voxel_size=.01,bbox_xyxy=(0,0,4,4),target_margin_px=0,target_expand_ratio=1.,max_grasp_width_m=.1,max_grasp_depth_m=.1,include_diagnostics=diagnostics)
    connection=Connection(request)
    with patch.dict(sys.modules,{'utils':package}):worker._worker_main(connection,'unused',300)
    assert connection.closed
    return connection.sent[-1]

class MemoryTests(unittest.TestCase):
    def test_selected_grasps_and_counts_identical_without_unused_diagnostics(self):
        for n in (0,100,10000):
            full=exercise(True,n);lean=exercise(False,n)
            np.testing.assert_array_equal(full['grasps'],lean['grasps'])
            self.assertEqual(full['counts'],lean['counts'])
            self.assertEqual(lean['filter_counts'],dict(decoded=n,after_cloud_collision=n,
                after_bbox=n//2,after_width=n//4,after_depth=n//4))
            self.assertEqual(lean['filter_parameters']['bbox_xyxy'],(0,0,4,4))
            self.assertNotIn('pre_bbox_grasps',lean);self.assertNotIn('bbox_grasps',lean)
            self.assertIn('pre_bbox_grasps',full);self.assertIn('bbox_grasps',full)
            self.assertNotIn('decoded_grasps',lean)
            np.testing.assert_array_equal(full['decoded_grasps'],np.arange(n*17).reshape(n,17))
    def test_decoded_snapshot_is_taken_before_cloud_collision(self):
        from contextlib import nullcontext
        source=ast.parse((ROOT/'utils/graspnet_utils.py').read_text())
        fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='infer_grasps')
        array=np.arange(3*17,dtype=float).reshape(3,17)
        class G(Group):
            def __len__(self):return len(self.grasp_group_array)
            def __getitem__(self,mask):return G(self.grasp_group_array[mask])
        class Tensor:
            def detach(self):return self
            def cpu(self):return self
            def numpy(self):return array.copy()
        detector=types.SimpleNamespace(detect=lambda *a,**kw:np.array([True,False,True]))
        scope=dict(np=np,torch=types.SimpleNamespace(no_grad=nullcontext),pred_decode=lambda _: [Tensor()],
            GraspNet=object,GraspGroup=G,DEFAULT_VOXEL_SIZE=.01,Optional=__import__('typing').Optional,
            ModelFreeCollisionDetector=lambda *a,**kw:detector)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'infer_grasps','exec'),scope)
        audit={}
        full,counts=scope['infer_grasps'](lambda x:x,{},np.zeros((4,3)),.01,diagnostics=audit)
        lean,lean_counts=scope['infer_grasps'](lambda x:x,{},np.zeros((4,3)),.01)
        np.testing.assert_array_equal(audit['decoded_grasps'],array)
        np.testing.assert_array_equal(full.grasp_group_array,array[1:2])
        np.testing.assert_array_equal(lean.grasp_group_array,full.grasp_group_array)
        self.assertEqual(counts,lean_counts);self.assertEqual(counts['collision_removed'],2)

    def test_worker_error_closes_pipe(self):
        self.assertEqual(exercise(False,crash=True)['type'],'error')
    def test_preprocessing_numbers_unchanged_when_visualizer_skipped(self):
        # Execute the actual production function without importing CUDA/SDK.
        source=ast.parse((ROOT/'utils/graspnet_utils.py').read_text())
        fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='build_end_points')
        fn.returns=None
        for arg in [*fn.args.args,*fn.args.kwonlyargs]:arg.annotation=None
        calls=[]
        class Tensor:
            def __init__(self,a):self.array=a
            def cuda(self,**kwargs):return self.array
        def cloud(depth,camera,organized):
            y,x=np.indices(depth.shape);z=depth/1000.
            return np.stack([x*z/200,y*z/200,z],axis=-1)
        o3d=types.SimpleNamespace(geometry=types.SimpleNamespace(PointCloud=lambda:calls.append(1) or types.SimpleNamespace()),utility=types.SimpleNamespace(Vector3dVector=lambda a:a.copy()))
        import cv2
        scope=dict(np=np,cv2=cv2,torch=types.SimpleNamespace(from_numpy=Tensor),CameraInfo=lambda *a:object(),create_point_cloud_from_depth_image=cloud,o3d=o3d)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'production_build_end_points','exec'),scope)
        rng=np.random.default_rng(17);color=rng.integers(0,255,(80,120,3),dtype=np.uint8);depth=rng.integers(0,1000,(80,120),dtype=np.uint16)
        args=(color,depth,np.eye(3),10000,.1,1.)
        np.random.seed(17);full=scope['build_end_points'](*args)
        np.random.seed(17);lean=scope['build_end_points'](*args,visualize=False)
        self.assertEqual(len(calls),1);self.assertIsNone(lean[1]);self.assertIsNotNone(full[1])
        for key in full[0]:np.testing.assert_array_equal(full[0][key],lean[0][key])
        np.testing.assert_array_equal(full[2],lean[2])

if __name__=='__main__':unittest.main()
