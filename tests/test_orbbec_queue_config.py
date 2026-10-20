"""SDK XML and mocked lifecycle only; never creates a real SDK Context."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drivers.camera.orbbec_queue import SDKQueueConfig
from drivers.camera.orbbec_gemini2 import OrbbecGemini2

class QueueTests(unittest.TestCase):
    def test_only_two_queue_values_change_and_cleanup_is_repeatable(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source.xml';source.write_text('<Config><Memory><PipelineFrameQueueSize>10</PipelineFrameQueueSize><FrameProcessingBlockQueueSize>10</FrameProcessingBlockQueueSize><MaxFrameBufferSize>2048</MaxFrameBufferSize></Memory><Other>unchanged</Other></Config>')
            original=source.read_bytes()
            for size in (10,2,1):
                config=SDKQueueConfig(source,size);path=config.path;tree=ET.parse(path)
                self.assertEqual(tree.findtext('Memory/PipelineFrameQueueSize'),str(size))
                self.assertEqual(tree.findtext('Memory/FrameProcessingBlockQueueSize'),str(size))
                self.assertEqual(tree.findtext('Memory/MaxFrameBufferSize'),'2048')
                self.assertEqual(tree.findtext('Other'),'unchanged')
                self.assertEqual(source.read_bytes(),original)
                config.close();config.close();self.assertFalse(path.exists())
    def test_invalid_queue_rejected_before_sdk(self):
        for size in (True,0,3,-1,'2'):
            with self.assertRaises(ValueError):OrbbecGemini2(sdk_queue_size=size)
    def test_mock_context_device_pipeline_reopen_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source.xml';source.write_text('<Config><Memory><PipelineFrameQueueSize>10</PipelineFrameQueueSize><FrameProcessingBlockQueueSize>10</FrameProcessingBlockQueueSize></Memory></Config>')
            paths=[];devices=[];device=object()
            class Context:
                def __init__(self,path=None):
                    if path:paths.append(Path(path));assert Path(path).is_file()
                def query_devices(self):return types.SimpleNamespace(get_device_by_index=lambda i:device)
                def set_logger_severity(self,*a):pass
            class Pipeline:
                def __init__(self,d=None):devices.append(d)
                def get_stream_profile_list(self,*a):return object()
                def start(self,*a):pass
                def stop(self):pass
                def get_camera_param(self):return types.SimpleNamespace(rgb_intrinsic=types.SimpleNamespace(fx=500,fy=500,cx=320,cy=240))
            class Config:
                def enable_stream(self,*a):pass
                def set_frame_aggregate_output_mode(self,*a):pass
            sdk=types.ModuleType('pyorbbecsdk');sdk.__file__=str(Path(d)/'__init__.py')
            for k,v in dict(Context=Context,Pipeline=Pipeline,Config=Config,AlignFilter=lambda **kw:object(),OBSensorType=types.SimpleNamespace(COLOR_SENSOR=1,DEPTH_SENSOR=2),OBFormat=types.SimpleNamespace(MJPG=1,RGB=2,Y16=3),OBAlignMode=object(),OBFrameAggregateOutputMode=types.SimpleNamespace(FULL_FRAME_REQUIRE=1),OBStreamType=types.SimpleNamespace(COLOR_STREAM=1),OBLogSeverity=types.SimpleNamespace(FATAL=1)).items():setattr(sdk,k,v)
            camera=OrbbecGemini2(alignment='software',sdk_queue_size=2,sdk_config_path=str(source));camera._profile=lambda *a:object();camera._load_distortion=lambda:np.zeros(5)
            with patch.dict(sys.modules,{'pyorbbecsdk':sdk}):
                for _ in range(2):
                    camera.open();self.assertIs(camera._sdk_device,device);camera.close()
                    self.assertIsNone(camera._sdk_context);self.assertIsNone(camera._queue_config)
                    self.assertFalse(paths[-1].exists())
                class BrokenContext:
                    def __init__(self,path):paths.append(Path(path));raise RuntimeError('synthetic SDK failure')
                sdk.Context=BrokenContext
                with self.assertRaisesRegex(RuntimeError,'synthetic SDK'):camera.open()
                self.assertFalse(paths[-1].exists());self.assertIsNone(camera._queue_config)
            self.assertEqual(devices,[device,device])

if __name__=='__main__':unittest.main()
