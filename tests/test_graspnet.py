import numpy as np

from rars01_graspnet.contracts import CameraIntrinsics, Detection2D, Header, RgbdFrame
from rars01_graspnet.graspnet import _frame_cloud, select_target


def test_select_target_prefers_requested_class_and_confidence():
    header = Header(1, "camera")
    detections = [
        Detection2D(header, "bottle", 0.99, (0, 0, 1, 1)),
        Detection2D(header, "cup", 0.71, (0, 0, 1, 1)),
        Detection2D(header, "cup", 0.92, (0, 0, 1, 1)),
    ]
    assert select_target(detections, "cup").confidence == 0.92
    assert select_target(detections, "tool") is None


def test_frame_cloud_uses_sdk_intrinsics_and_depth_scale():
    header = Header(1, "camera")
    intrinsics = CameraIntrinsics(
        np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]),
        np.zeros(5), 3, 3,
    )
    frame = RgbdFrame(header, np.zeros((3, 3, 3), np.uint8),
                      np.full((3, 3), 500, np.uint16), intrinsics)
    points, colors, sampled, sampled_colors = _frame_cloud(frame, 20, 0.1, 1.0)
    assert points.shape == (9, 3)
    assert colors.shape == (9, 3)
    assert sampled.shape == (20, 3)
    assert sampled_colors.shape == (20, 3)
    assert np.any(np.all(np.isclose(points, [0.0, 0.0, 0.5]), axis=1))
