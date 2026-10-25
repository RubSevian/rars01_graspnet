"""CPU tests of local inference / full collision separation."""
import ast
from pathlib import Path
import types
import numpy as np
import pytest
from rars01_graspnet.graspnet_roi import inference_region, validate_roi


def test_invalid_configuration():
    for opts in ({'enabled':1}, {'margin_m':0}, {'margin_m':float('nan')},
                 {'lower_quantile':.99,'upper_quantile':.01}):
        with pytest.raises(ValueError): validate_roi(opts)


def test_missing_or_wrong_mask_is_explicit_error():
    points=np.array([[0,0,.7],[.4,0,.7]])
    valid=np.ones((1,2),bool)
    for mask,source in ((None,'unknown'),(valid,'bbox'),(np.zeros_like(valid),'segmentation'),
                        (np.ones((2,2)),'segmentation')):
        with pytest.raises(ValueError):
            inference_region(points,valid,mask,source,{'enabled':True})


def test_production_preprocessing_crops_only_network_input():
    import cv2
    path=Path(__file__).resolve().parents[1]/'utils/graspnet_utils.py'
    fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='build_end_points')
    fn.returns=None
    for arg in [*fn.args.args,*fn.args.kwonlyargs]:arg.annotation=None
    class Tensor:
        def __init__(self,a):self.a=a
        def cuda(self,**kwargs):return self.a
    points=np.array([[[0,0,.7],[.05,0,.7],[.4,0,.7]]],dtype=np.float32)
    scope=dict(np=np,cv2=cv2,torch=types.SimpleNamespace(from_numpy=Tensor),
               CameraInfo=lambda *a:None,create_point_cloud_from_depth_image=lambda *a,**kw:points,o3d=None)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope)
    color=np.zeros((1,3,3),np.uint8);depth=np.full((1,3),700,np.uint16)
    mask=np.array([[True,False,False]])
    audit={}
    ep,_,raw=scope['build_end_points'](color,depth,np.eye(3),20,.05,1.,visualize=False,
        target_mask=mask,mask_source='segmentation',target_roi={'enabled':True,'margin_m':.12},sampling_diagnostics=audit)
    np.testing.assert_array_equal(raw,points.reshape(-1,3))
    assert ep['point_clouds'].shape==(1,20,3)
    assert np.max(ep['point_clouds'][0,:,0])==points[0,1,0]
    assert audit['pool_points']==2 and audit['full_cloud_points']==3
    # Explicitly disabled mode keeps all points even without a mask.
    ep,_,raw=scope['build_end_points'](color,depth,np.eye(3),3,.05,1.,visualize=False,target_roi={'enabled':False})
    assert np.max(ep['point_clouds'][0,:,0])==points[0,2,0]


def test_saved_banana_region_matches_offline_experiment():
    root=Path(__file__).resolve().parents[3]
    frame=root/'docs/figures/last_three_runs_20261025/frame.npz'
    if not frame.exists():pytest.skip('Saved local experiment is not part of portable tests')
    z=np.load(frame);depth=z['rgbd_depth_mm'];K=z['K'];valid=(depth>50)&(depth<1000)
    y,x=np.indices(depth.shape);d=depth/1000.
    points=np.stack(((x-K[0,2])*d/K[0,0],(y-K[1,2])*d/K[1,1],d),axis=-1)[valid]
    ids,audit=inference_region(points,valid,z['target_mask'],'segmentation',{'enabled':True,'margin_m':.12})
    assert len(ids)==102150
    assert audit['full_cloud_points']==552000
