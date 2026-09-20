"""Offline regressions for the local controller: no SDK, camera or motors."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from rars01_graspnet.pinocchio_math import (  # noqa: E402
    IKParams, cartesian_geodesic, compute_fk, solve_ik,
)
from rars01_graspnet.pose_controller import RarsPoseController  # noqa: E402


URDF = Path(__file__).resolve().parents[2] / "rars01_description/urdf/rars01_control.urdf"


class FakeArm:
    def __init__(self):
        self.q = np.array([0.2, 0.1, 0.1, -0.1, 0.0, 0.0])
        self._home = np.zeros(6)
        self.groups = {"arm": SimpleNamespace(num_joints=6, _pv_vlim=np.full(6, 2.0))}
        self.disconnected = False

    def load_kinematic_model(self):
        return pin.buildModelFromUrdf(str(URDF))

    def get_state(self):
        return self.q.copy(), np.zeros(6), np.zeros(6)

    def disconnect(self):
        self.disconnected = True


def test_ik_preserves_baseline_ready_solution():
    model = FakeArm().load_kinematic_model()
    target = pin.SE3(pin.rpy.rpyToMatrix(0.0, 0.7, 0.0), np.array([0.3, 0.0, 0.3]))
    result = solve_ik(model, model.createData(), model.getFrameId("End_link"),
                      target, np.zeros(model.nq), IKParams(), 6)
    assert result.success
    # Captured from the stable controller before moving its math into the project.
    expected = [-0.0011288214512597985, 0.9720482366198061, 1.3737185500480948,
                -1.1016706273527133, -0.0008633701917392083, -0.0007272069253235443]
    np.testing.assert_allclose(result.q, expected, atol=1e-8)
    full = np.r_[result.q, 0.0, 0.0]
    np.testing.assert_allclose(compute_fk(model, full)[2], target.homogeneous, atol=1e-8)


def test_cartesian_geodesic_preserves_endpoints():
    start = pin.SE3.Identity()
    end = pin.SE3(pin.rpy.rpyToMatrix(0.2, -0.4, 0.1), np.array([0.3, 0.1, 0.2]))
    poses = cartesian_geodesic(start, end, 1.0, 0.01)
    assert len(poses) == 101
    np.testing.assert_allclose(poses[0].homogeneous, start.homogeneous, atol=1e-12)
    np.testing.assert_allclose(poses[-1].homogeneous, end.homogeneous, atol=1e-12)


def test_end_waits_for_measured_home_before_disconnect(monkeypatch):
    arm = FakeArm()
    controller = RarsPoseController(arm)
    controller._running = True
    clock = {"now": 0.0, "final_sent": None}

    def sleep(dt):
        clock["now"] += dt
        if np.array_equal(controller._q_target, arm._home):
            if clock["final_sent"] is None:
                clock["final_sent"] = clock["now"]
            # The physical motor lags behind the final commanded position.
            if clock["now"] - clock["final_sent"] >= 0.2:
                arm.q[:] = arm._home
            else:
                arm.q[:] = arm._home + 0.03
        else:
            arm.q[:] = controller._q_target

    fake_time = SimpleNamespace(monotonic=lambda: clock["now"], sleep=sleep)
    monkeypatch.setattr("rars01_graspnet.pose_controller.time", fake_time)
    controller.end()
    assert arm.disconnected
    assert not controller._running
    assert clock["now"] - clock["final_sent"] >= 0.2
    np.testing.assert_array_equal(arm.q, arm._home)


def test_home_stops_previous_sender_and_uses_configured_target(monkeypatch):
    arm = FakeArm()
    arm._home = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
    controller = RarsPoseController(arm)
    controller._running = True
    joined = []

    def join(timeout):
        assert controller._stop_send.is_set()
        joined.append(timeout)

    controller._send_thread = SimpleNamespace(join=join, is_alive=lambda: False)
    monkeypatch.setattr("rars01_graspnet.pose_controller.time", SimpleNamespace(
        monotonic=lambda: 0.0,
        sleep=lambda dt: arm.q.__setitem__(slice(None), controller._q_target),
    ))
    controller.safe_home()
    assert joined
    assert not arm.disconnected
    np.testing.assert_array_equal(controller._q_target, arm._home)


def test_sender_keeps_last_target_for_position_hold(monkeypatch):
    controller = RarsPoseController(FakeArm())
    controller._traj = [np.full(6, 0.1), np.full(6, 0.2)]
    controller._moving = True
    monkeypatch.setattr("rars01_graspnet.pose_controller.time", SimpleNamespace(sleep=lambda dt: None))
    controller._send_loop(0.1)
    assert not controller._moving
    np.testing.assert_array_equal(controller._q_target, np.full(6, 0.2))
