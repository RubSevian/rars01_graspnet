"""Exercise planner geometry in a fresh interpreter that forbids inference imports."""
import os
from pathlib import Path
import subprocess
import sys


def test_cold_geometry_and_selection_do_not_import_inference():
    root = Path(__file__).resolve().parents[1]
    code = r'''
import importlib.abc
import sys
class NoInference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'ultralytics', 'graspnet', 'pointnet2_ops') or fullname == 'utils.graspnet_utils':
            raise AssertionError('Inference dependency leaked: '+fullname)
sys.meta_path.insert(0, NoInference())
import numpy as np
from utils import grasp_geometry as g
from rars01_graspnet.real_grasp_selection import _rank_grasps, _select_executable_grasp
from graspnetAPI import GraspGroup
from unittest.mock import patch
from types import SimpleNamespace as NS
raw=np.zeros((4,17));raw[:,0]=[.7,.9,.8,.6];raw[:,1:4]=[.06,.02,.03]
raw[:,4:13]=np.eye(3).ravel();raw[:,13:16]=[[0,0,.5],[.001,0,.5],[.1,0,.5],[.101,0,.5]]
before=raw.copy();group=GraspGroup(raw)
# Force the deployed optional-native fallback and verify order/deduplication.
with patch.object(GraspGroup,'nms',side_effect=ImportError('native unavailable')):
    ranked=_rank_grasps(group)
np.testing.assert_array_equal(ranked.scores,[.9,.8])
np.testing.assert_array_equal(group.grasp_group_array,before)
# Preserve the native-success dispatch too.
with patch.object(GraspGroup,'nms',return_value=group[:1]) as native:
    assert len(g.nms_grasp_group(group))==1
    native.assert_called_once_with(.03,np.deg2rad(30.))
assert len(g.nms_grasp_group(GraspGroup(np.empty((0,17)))))==0
pose,pre,retreat=map(np.asarray,g.grasp_to_base_poses(group[0],np.eye(4),.08,.05,0.,'rars01',np.eye(4),False))
np.testing.assert_allclose(pose[:3],[.03,0,.5])
np.testing.assert_allclose(pre[:3],[-.05,0,.5])
np.testing.assert_allclose(retreat[:3],[-.02,0,.5])
try:g.grasp_to_base_poses(group[0],np.eye(4),.08,.05,.01,'rars01',np.eye(4),False)
except ValueError:pass
else:raise AssertionError('Invalid insertion depth accepted')
rejections=[]
assert _select_executable_grasp(NS(),NS(),GraspGroup(np.empty((0,17))),np.eye(4),
    .1,.1,0.,.1,0.,0.,'rars01',False,False,np.zeros(3),16,{},
    dict(weight_grasp=1.,weight_joint=1.,weight_motion=1.,motion_normalization=1.),
    rejection_diagnostics=rejections) is None
assert rejections==[dict(reason='NO_CANDIDATES')]
assert 'torch' not in sys.modules and 'utils.graspnet_utils' not in sys.modules
'''
    env = dict(os.environ, PYTHONPATH=str(root))
    subprocess.run([sys.executable, '-c', code], env=env, cwd=root, check=True, timeout=60)
