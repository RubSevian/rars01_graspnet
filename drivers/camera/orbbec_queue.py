"""Private SDK XML for controlled queue A/B; never writes SDK/global config."""
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

class SDKQueueConfig:
    def __init__(self,source,size):
        if type(size) is not int or size not in (1,2,10):
            raise ValueError('SDK queue experiment requires1,2 or10')
        tree=ET.parse(source)
        for name in ('PipelineFrameQueueSize','FrameProcessingBlockQueueSize'):
            node=tree.getroot().find('Memory/'+name)
            if node is None:raise ValueError('SDK config missing '+name)
            node.text=str(size)
        self._directory=TemporaryDirectory(prefix='go2_orbbec_queue_')
        self.path=Path(self._directory.name)/'OrbbecSDKConfig.xml'
        try:tree.write(self.path,encoding='utf-8',xml_declaration=True)
        except BaseException:self.close();raise
    def close(self):
        self._directory.cleanup()
