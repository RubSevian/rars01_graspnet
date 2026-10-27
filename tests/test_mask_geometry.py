import numpy as np
import pytest
from types import SimpleNamespace as NS
from rars01_graspnet.mask_geometry import restore_mask
from utils.yolo_utils import detection_mask

@pytest.mark.parametrize('shape,padded,offset', [((720,1280),(384,640),(12,0)), ((720,1280),(640,640),(140,0)), ((1280,720),(640,384),(0,12))])
def test_known_pixel_rectangle_survives_letterbox(shape,padded,offset):
    # A rectangle in half-resolution content; padding is independent of restoration.
    mask=np.zeros(padded,np.float32);top,left=offset
    mask[top+100:top+120,left+140:left+170]=1
    expected=np.zeros(shape,np.uint8);expected[200:240,280:340]=1
    np.testing.assert_array_equal(restore_mask(mask,shape),expected)

def test_native_mask_is_not_shifted():
    mask=np.zeros((720,1280),np.float32);mask[550:600,700:750]=1
    np.testing.assert_array_equal(restore_mask(mask,mask.shape),mask)

def test_odd_padding_matches_ultralytics():
    import torch
    from ultralytics.utils.ops import scale_masks
    mask=np.random.default_rng(4).integers(0,2,(384,640)).astype(np.float32)
    expected=scale_masks(torch.tensor(mask)[None,None],(719,1280),mode='nearest')[0,0].numpy()
    np.testing.assert_array_equal(restore_mask(mask,(719,1280)),expected)

def test_production_parser_removes_padding_and_keeps_source():
    import torch
    mask=torch.zeros((1,384,640));mask[0,112:132,140:170]=1
    actual,source=detection_mask(NS(masks=NS(data=mask)),0,(720,1280),(280,200,340,240),with_source=True)
    assert source=='segmentation'
    assert np.count_nonzero(actual)==40*60
    assert actual[200:240,280:340].all()

def test_missing_mask_fallback_is_explicit_bbox():
    mask,source=detection_mask(NS(masks=None),0,(50,80),(10,20,15,25),with_source=True)
    assert source=='bbox' and mask.sum()==36
