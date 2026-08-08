import numpy as np

from rars01_graspnet.contracts import CameraIntrinsics, Header, RgbdFrame


def test_rgbd_contract_carries_ros_ready_metadata():
    header = Header(stamp_ns=123, frame_id="camera_color_optical_frame", sequence=4)
    frame = RgbdFrame(
        header=header,
        color_bgr=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_mm=np.zeros((2, 3), dtype=np.uint16),
        intrinsics=CameraIntrinsics(np.eye(3), np.zeros((1, 5)), 3, 2),
    )
    assert frame.header.frame_id == "camera_color_optical_frame"
    assert frame.header.stamp_ns == 123
    assert frame.color_bgr.shape[:2] == frame.depth_mm.shape

