"""Real spawn/IPC lifecycle tests; no model, camera, ROS or actuator."""
import sys
import time
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.yolo_worker import YoloWorker


def fake_worker(pipe, config, root, classes, threads):
    pipe.send({'type':'ready'})
    if config.get('never_read'):time.sleep(10);return
    while True:
        r=pipe.recv()
        if r['type']=='stop':return
        if config.get('crash'):return
        time.sleep(config.get('delay',0))
        pipe.send(dict(type='result',id=r['id']+config.get('wrong_id',0),
            capture_ns=r['capture_ns'],started_ns=1,completed_ns=2,
            detections=[int(r['image'].sum())]))


def make(**cfg):
    return YoloWorker(cfg,'.',[],1,5.,_target=fake_worker)


def test_round_trip_keeps_request_identity_and_frame():
    worker=make()
    try:
        for i in (1,2):
            result=worker.detect(np.ones((2,2,3),np.uint8)*i,123+i,time.monotonic()+1.,lambda:None)
            assert result['id']==i and result['capture_ns']==123+i
            assert result['detections']==[12*i]
    finally:worker.close()
    assert not worker.is_alive()


def test_timeout_discards_child_and_late_result():
    worker=make(delay=1.)
    with pytest.raises(TimeoutError):worker.detect(np.zeros((1,1,3)),1,time.monotonic()+.1,lambda:None)
    assert not worker.is_alive()
    with pytest.raises(RuntimeError,match='EXITED'):
        worker.detect(np.zeros((1,1,3)),2,time.monotonic()+1.,lambda:None)


def test_cancel_checked_while_inference_running():
    worker=make(delay=1.);start=time.monotonic();checks=[]
    def check():
        checks.append(time.monotonic())
        if checks[-1]-start>.1:raise ValueError('cancelled')
    with pytest.raises(ValueError,match='cancelled'):
        worker.detect(np.zeros((1,1,3)),1,start+2.,check)
    assert len(checks)>=3 and not worker.is_alive()


def test_timeout_also_bounds_blocked_image_send():
    worker=make(never_read=True)
    start=time.monotonic()
    with pytest.raises(TimeoutError):
        worker.detect(np.zeros((720,1280,3),np.uint8),1,start+.1,lambda:None)
    assert time.monotonic()-start<3. and not worker.is_alive()


@pytest.mark.parametrize('config,match',[({'crash':True},'EXITED'),({'wrong_id':1},'MISMATCH')])
def test_invalid_or_missing_response_revokes_worker(config,match):
    worker=make(**config)
    with pytest.raises(RuntimeError,match=match):
        worker.detect(np.zeros((1,1,3)),1,time.monotonic()+1.,lambda:None)
    assert not worker.is_alive()
