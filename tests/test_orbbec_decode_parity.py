"""Known RGB/Y16 conversions and original acquisition stamps, no SDK context."""
import sys,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from drivers.camera.orbbec_gemini2 import OrbbecGemini2
class Frame:
 def __init__(self,data,depth=False):self.data=data;self.depth=depth
 def get_data(self):return self.data
 def get_width(self):return 3
 def get_height(self):return 2
 def get_format(self):return 2
 def get_depth_scale(self):return .5
 def get_system_timestamp_us(self):return 123456789
 def get_timestamp_us(self):return 987654
class Frames:
 def __init__(self):self.color=Frame(np.arange(18,dtype=np.uint8));self.depth=Frame(np.array([0,1,2,3,65534,65535],dtype=np.uint16),True)
 def get_color_frame(self):return self.color
 def get_depth_frame(self):return self.depth
class Tests(unittest.TestCase):
 def test_request_driven_decoder_preserves_pixels_units_and_timestamps(self):
  c=OrbbecGemini2();f=Frames();c._pipeline=object()
  with patch.dict(sys.modules,{'pyorbbecsdk':types.SimpleNamespace(OBFormat=types.SimpleNamespace(MJPG=1,RGB=2))}):
   color,depth=c.decode_frames(f)
  np.testing.assert_array_equal(color,np.arange(18,dtype=np.uint8).reshape(2,3,3)[:,:,::-1])
  np.testing.assert_array_equal(depth,np.rint(f.depth.data.reshape(2,3)*.5).astype(np.uint16))
  self.assertEqual(c.last_frame_timing,c.frame_timing(f))
 def test_acquisition_returns_owned_original_frameset_without_decode(self):
  c=OrbbecGemini2();f=Frames();c._pending_frames=f;c._pipeline=object()
  self.assertIs(c.acquire_frames(),f);self.assertIsNone(c._pending_frames);self.assertIsNone(getattr(c,'last_frame_timing',None))
if __name__=='__main__':unittest.main()
