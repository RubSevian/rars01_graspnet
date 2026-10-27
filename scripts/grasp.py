"""
Visual grasping demo with selectable GraspNet or central-mask estimation.

Keys:
  G/Space: run GraspNet on the current RGB-D frame and execute a grasp.
  R: resume live preview.
  Q/Esc: release, home, and exit.

Usage:
    python scripts/grasp.py --dry-run
    python scripts/grasp.py --target-class cup
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"
_SHARED_GRASPNET_ROOT = PROJECT_ROOT.parent / "rars01_graspnet" / "sdk" / "graspnet-baseline"
GRASPNET_ROOT = Path(os.environ.get(
    "GRASPNET_ROOT",
    str(_LOCAL_GRASPNET_ROOT if _LOCAL_GRASPNET_ROOT.is_dir() else _SHARED_GRASPNET_ROOT),
)).expanduser().resolve()


def _prepare_imports() -> None:
    project_root = str(PROJECT_ROOT)
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)

    graspnet_paths = [
        GRASPNET_ROOT,
        *(GRASPNET_ROOT / subdir for subdir in ("models", "dataset", "utils", "pointnet2", "graspnetAPI")),
    ]
    for path in reversed(graspnet_paths):
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(1, path_str)


_prepare_imports()

from drivers.camera import make_camera  # noqa: E402
from drivers.robot.grasp_driver import (  # noqa: E402
    GRIPPER_MAX_DISTANCE_M,
    GraspDriver,
    RarsArmAdapter,
)
import utils.graspnet_utils as graspnet_utils  # noqa: E402
from utils.graspnet_worker import GraspNetWorker  # noqa: E402
from utils.camera_utils import compose_cam_to_base_transform, configure_camera, load_config, load_hand_eye  # noqa: E402
from utils.ordinary_grasp import (  # noqa: E402
    GraspPose,
    draw_grasp as draw_central_mask_grasp,
    estimate_grasps as estimate_central_mask_grasps,
)
from utils.transforms import (  # noqa: E402
    canonicalize_parallel_gripper_tcp_rotation,
    graspnet_rotation_to_rars_tcp_rotation,
    graspnet_rotation_to_tcp_rotation,
    pose6d_to_mat4,
    rotation_matrix_to_euler_zyx,
)
from rars01_graspnet.grasp_selection import (  # noqa: E402
    joint_limit_cost,
    motion_cost,
    normalized_grasp_costs,
)
from utils.yolo_utils import (  # noqa: E402
    YoloDetection,
    detect_objects,
    load_yolo as load_yolo_from_config,
)
from graspnetAPI import Grasp, GraspGroup  # noqa: E402
from rars01_graspnet.real_grasp_selection import ExecutableGrasp, _rank_grasps, _parallel_flip_grasp, _pose_z_ok, _translate_pose, _select_executable_grasp
from rars01_graspnet.real_grasp_ik import IkChecker, IkSolution, _plan_cartesian_segment

_PARALLEL_FLIP_X = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def _wait_motion(controller: Any, duration: float, extra: float = 0.6) -> None:
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + extra + 2.0)
    else:
        time.sleep(duration + extra)
    # RarsArmAdapter deliberately executes its 100 Hz callback in a background
    # thread.  Surface its exception immediately: otherwise the main UI keeps
    # printing the remaining grasp steps after the SDK has stopped commands,
    # which lets the STM watchdog disable the motors without the actual cause
    # appearing in the terminal log.
    transport = getattr(controller, "arm", None)
    failure = getattr(transport, "_failure", None)
    if failure is not None:
        raise RuntimeError(f"RARS01 control loop failed: {failure}") from failure
    if transport is not None and not getattr(transport, "_running", True):
        raise RuntimeError("RARS01 control loop stopped unexpectedly")


def _rotation_error_rad(target: np.ndarray, actual: np.ndarray) -> float:
    relative = np.asarray(target, dtype=np.float64).T @ np.asarray(actual, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


def _wait_rars_end_link(
    controller: Any,
    grasp_driver: GraspDriver,
    target6d: tuple[float, ...],
    target_joints: np.ndarray,
    planned_duration_s: float,
    started_at: float,
    feedback_cfg: dict[str, Any],
    label: str,
) -> bool:
    """Require fresh, stable measured End_link feedback before the next stage."""
    position_tolerance = float(feedback_cfg["grasp_position_tolerance_m"])
    orientation_tolerance = np.deg2rad(float(feedback_cfg["grasp_orientation_tolerance_deg"]))
    velocity_tolerance = np.deg2rad(float(feedback_cfg["velocity_tolerance_deg_s"]))
    settle_time = float(feedback_cfg["settle_time_s"])
    max_age = float(feedback_cfg["max_feedback_age_s"])
    deadline = started_at + float(planned_duration_s) + float(feedback_cfg["timeout_margin_s"])
    target = pose6d_to_mat4(*target6d)
    previous_sequence = -1
    settle_started: Optional[float] = None
    last_reason = "waiting"
    last_log: Optional[tuple[Any, ...]] = None

    while time.monotonic() < deadline:
        transport = getattr(controller, "arm", None)
        failure = getattr(transport, "_failure", None)
        if failure is not None:
            print(f"[Feedback/{label}] ABORT transport error: {failure}")
            return False
        thread = getattr(controller, "_send_thread", None)
        sending_done = thread is None or not thread.is_alive()
        try:
            feedback = grasp_driver.get_end_link_feedback(refresh=sending_done)
        except Exception as exc:
            print(f"[Feedback/{label}] ABORT feedback error: {exc}")
            return False

        if not feedback.valid:
            print(f"[Feedback/{label}] ABORT invalid feedback")
            return False
        if feedback.age_s > max_age:
            last_reason = "stale feedback"
            settle_started = None
        elif feedback.sequence == previous_sequence:
            # Do not count a cached sample as additional evidence.  Its age is
            # still checked above; the next fresh sample decides whether the
            # existing settle interval remains valid.
            last_reason = "waiting for next feedback sequence"
        else:
            previous_sequence = feedback.sequence
            position_error = float(np.linalg.norm(feedback.pose[:3, 3] - target[:3, 3]))
            orientation_error = _rotation_error_rad(target[:3, :3], feedback.pose[:3, :3])
            max_velocity = float(np.max(np.abs(feedback.velocity)))
            ready = (
                sending_done
                and position_error <= position_tolerance
                and orientation_error <= orientation_tolerance
                and max_velocity <= velocity_tolerance
            )
            last_log = (feedback, position_error, orientation_error, max_velocity)
            if ready:
                settle_started = time.monotonic() if settle_started is None else settle_started
                if time.monotonic() - settle_started >= settle_time:
                    print(
                        f"[Feedback/{label}] READY planned={planned_duration_s:.3f}s "
                        f"seq={feedback.sequence} age={feedback.age_s:.3f}s "
                        f"pos={position_error * 1000.0:.2f}mm "
                        f"rot={np.rad2deg(orientation_error):.2f}deg "
                        f"max_vel={np.rad2deg(max_velocity):.2f}deg/s"
                    )
                    return True
                last_reason = "settling"
            else:
                settle_started = None
                last_reason = "trajectory/pose/velocity outside tolerance"
        time.sleep(0.01)

    if last_log is not None:
        feedback, position_error, orientation_error, max_velocity = last_log
        thread = getattr(controller, "_send_thread", None)
        sending_done = thread is None or not thread.is_alive()
        command = np.asarray(controller._q_target, dtype=np.float64).copy()
        command_error = float(np.max(np.abs(command - target_joints)))
        joint_errors = feedback.joints - target_joints
        worst_joint = int(np.argmax(np.abs(joint_errors)))
        print(
            f"[Feedback/{label}] ABORT timeout planned={planned_duration_s:.3f}s "
            f"seq={feedback.sequence} age={feedback.age_s:.3f}s "
            f"q={np.round(feedback.joints, 5).tolist()} "
            f"q_target={np.round(target_joints, 5).tolist()} "
            f"pos={position_error * 1000.0:.2f}mm "
            f"rot={np.rad2deg(orientation_error):.2f}deg "
            f"max_vel={np.rad2deg(max_velocity):.2f}deg/s reason={last_reason} "
            f"elapsed={time.monotonic() - started_at:.3f}s "
            f"sending_done={sending_done} "
            f"q_command={np.round(command, 5).tolist()} "
            f"command_target_error={command_error:.6f}rad "
            f"worst_joint={worst_joint + 1} "
            f"joint_error={np.rad2deg(joint_errors[worst_joint]):+.2f}deg"
        )
    else:
        print(f"[Feedback/{label}] ABORT timeout: {last_reason}")
    return False


def _move_ready(controller: Any, ready_cfg: dict[str, Any]) -> bool:
    """Move to ready and report an IK/path creation failure to the caller."""
    duration = float(ready_cfg.get("duration", 3.0))
    started = controller.move_to_traj(
        x=float(ready_cfg.get("x", 0.25)),
        y=float(ready_cfg.get("y", 0.0)),
        z=float(ready_cfg.get("z", 0.35)),
        roll=float(ready_cfg.get("roll", 0.0)),
        pitch=float(ready_cfg.get("pitch", 1.2)),
        yaw=float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    if not started:
        print("[Robot] Ready trajectory was not started")
        return False
    _wait_motion(controller, duration)
    return True





def _execute_grasp(
    controller: Any,
    grasp_driver: GraspDriver,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    retreat6d: tuple[float, ...],
    ready_cfg: dict[str, Any],
    motion_cfg: dict[str, Any],
    dry_run: bool,
    joint_targets: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    cartesian_joint_paths: Optional[tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]] = None,
) -> bool:
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d
    xr, yr, zr, rxr, ryr, rzr = retreat6d

    print(f"[Grasp] pregrasp xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f}) rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f})")
    print(f"[Grasp] grasp    xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f}) rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})")
    print(f"[Grasp] retreat  xyz=({xr:+.3f},{yr:+.3f},{zr:+.3f}) rpy=({rxr:+.3f},{ryr:+.3f},{rzr:+.3f})")

    if dry_run:
        print("[Grasp] dry run; skip motion")
        return False

    print("[Grasp] Open gripper")
    grasp_driver.open_gripper()

    pregrasp_duration = float(motion_cfg.get("pregrasp_duration_s", 2.0))
    grasp_duration = float(motion_cfg.get("grasp_duration_s", 1.5))
    retreat_duration = float(motion_cfg.get("retreat_duration_s", 1.5))
    feedback_cfg = {
        "grasp_position_tolerance_m": 0.005,
        "grasp_orientation_tolerance_deg": 3.0,
        "velocity_tolerance_deg_s": 2.0,
        "settle_time_s": 0.2,
        "timeout_margin_s": 1.5,
        "max_feedback_age_s": 0.10,
        **dict(motion_cfg.get("motion_feedback", {})),
    }

    print("[Grasp] Move to pregrasp")
    stage_started = time.monotonic()
    if joint_targets is not None:
        duration = grasp_driver.move_rars_joint_target(joint_targets[0], pregrasp_duration)
    else:
        if not controller.move_to_traj(xp, yp, zp, rxp, ryp, rzp, duration=pregrasp_duration):
            print("[Grasp] Pregrasp IK failed")
            return False
        duration = pregrasp_duration
    if joint_targets is not None:
        if not _wait_rars_end_link(
            controller, grasp_driver, pre6d, joint_targets[0], duration, stage_started,
            feedback_cfg, "pregrasp",
        ):
            print("[Grasp] ABORT before Cartesian approach")
            return False
    else:
        _wait_motion(controller, duration)

    print("[Grasp] Move to grasp")
    stage_started = time.monotonic()
    if cartesian_joint_paths is not None:
        duration = grasp_driver.move_rars_joint_waypoints(
            cartesian_joint_paths[0], grasp_duration
        )
    elif joint_targets is not None:
        duration = grasp_driver.move_rars_joint_target(joint_targets[1], grasp_duration)
    else:
        if not controller.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=grasp_duration):
            print("[Grasp] Grasp IK failed")
            return False
        duration = grasp_duration
    if joint_targets is not None:
        grasp_target_joints = (
            cartesian_joint_paths[0][-1]
            if cartesian_joint_paths is not None else joint_targets[1]
        )
        if not _wait_rars_end_link(
            controller, grasp_driver, grasp6d, grasp_target_joints, duration, stage_started,
            feedback_cfg, "grasp",
        ):
            print("[Grasp] ABORT: gripper remains open")
            return False
    else:
        _wait_motion(controller, duration)

    print("[Grasp] Closing")
    ok = grasp_driver.grasp()
    print("[Grasp] Holding object" if ok else "[Grasp] Empty grasp")
    # Let the jaw controller transition from contact torque to position
    # holding before the arm starts lifting.
    grip_settle_s = float(motion_cfg.get("grip_settle_s", 0.35))
    if ok and grip_settle_s > 0.0:
        print(f"[Grasp] Stabilize grip ({grip_settle_s:.2f}s)")
        time.sleep(grip_settle_s)

    print("[Grasp] Retreat")
    if cartesian_joint_paths is not None:
        duration = grasp_driver.move_rars_joint_waypoints(
            cartesian_joint_paths[1], retreat_duration
        )
        _wait_motion(controller, duration)
    elif joint_targets is not None:
        duration = grasp_driver.move_rars_joint_target(joint_targets[2], retreat_duration)
        _wait_motion(controller, duration)
    elif controller.move_to_traj(xr, yr, zr, rxr, ryr, rzr, duration=retreat_duration):
        _wait_motion(controller, retreat_duration)

    print("[Grasp] Return ready")
    return _move_ready(controller, ready_cfg) and ok


def _print_grasp(grasp: Grasp, robot_backend: str) -> None:
    rotation_fn = (
        graspnet_rotation_to_rars_tcp_rotation
        if robot_backend == "rars01"
        else graspnet_rotation_to_tcp_rotation
    )
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(rotation_fn(grasp.rotation_matrix))
    print("\n[G] Best GraspNet grasp:")
    print(f"  score={grasp.score:.4f} width={grasp.width:.4f} height={grasp.height:.4f} depth={grasp.depth:.4f}")
    print(f"  position_xyz={grasp.translation.tolist()}")
    print(f"  graspnet_rpy={rotation_matrix_to_euler_zyx(grasp.rotation_matrix).tolist()}")
    print(f"  tcp_rpy={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")




def _select_central_mask_grasp(
    grasps: list[GraspPose], target_class: Optional[str]
) -> Optional[GraspPose]:
    valid = [grasp for grasp in grasps if grasp.is_valid]
    if target_class:
        target = target_class.casefold()
        exact = [grasp for grasp in valid if grasp.class_name.casefold() == target]
        contains = [grasp for grasp in valid if target in grasp.class_name.casefold()]
        valid = exact or contains
    return max(valid, key=lambda grasp: grasp.conf) if valid else None


def _central_mask_to_graspnet(grasp: GraspPose) -> Grasp:
    """Adapt the upstream central-mask grasp axes to the shared executor."""
    if grasp.position is None or grasp.rotation is None:
        raise ValueError("central-mask grasp is incomplete")
    approach_into_object = -np.asarray(grasp.rotation[:, 2], dtype=np.float64)
    opening_axis = np.asarray(grasp.rotation[:, 1], dtype=np.float64)
    third_axis = np.cross(approach_into_object, opening_axis)
    rotation = np.column_stack((approach_into_object, opening_axis, third_axis))
    return Grasp(
        float(grasp.conf),
        float(grasp.jaw_width_m),
        0.02,
        0.02,
        rotation,
        np.asarray(grasp.position, dtype=np.float64),
        -1,
    )




def _grasp_with_base_approach(
    grasp: Grasp,
    T_cam2base: np.ndarray,
    approach_base: np.ndarray,
) -> Grasp:
    """Keep the mask opening axis while imposing an approach in base_link."""
    R_cam2base = np.asarray(T_cam2base, dtype=np.float64)[:3, :3]
    approach = R_cam2base.T @ np.asarray(approach_base, dtype=np.float64)
    approach /= max(float(np.linalg.norm(approach)), 1e-8)

    opening = np.asarray(grasp.rotation_matrix[:, 1], dtype=np.float64)
    opening -= float(np.dot(opening, approach)) * approach
    opening_norm = float(np.linalg.norm(opening))
    if opening_norm < 1e-8:
        opening = np.asarray(grasp.rotation_matrix[:, 2], dtype=np.float64)
        opening -= float(np.dot(opening, approach)) * approach
        opening_norm = float(np.linalg.norm(opening))
    if opening_norm < 1e-8:
        raise ValueError("central-mask opening axis is parallel to the requested approach")
    opening /= opening_norm
    third_axis = np.cross(approach, opening)
    third_axis /= max(float(np.linalg.norm(third_axis)), 1e-8)
    rotation = np.column_stack((approach, opening, third_axis))
    return Grasp(
        float(grasp.score),
        float(grasp.width),
        float(grasp.height),
        float(grasp.depth),
        rotation,
        np.asarray(grasp.translation, dtype=np.float64),
        int(grasp.object_id),
    )


def _print_central_mask_grasp(grasp: GraspPose) -> None:
    print("\n[G] Central-mask grasp:")
    print(f"  class={grasp.class_name} conf={grasp.conf:.4f}")
    print(f"  center_px={grasp.center_px} angle_deg={grasp.angle_deg:.2f}")
    print(f"  width={grasp.jaw_width_m:.4f} m position_xyz={grasp.position.tolist()}")








def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraspNet/central-mask robot grasp demo")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--robot-backend", choices=("rars01",), default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="GraspNet checkpoint; default is graspnet.checkpoint from YAML",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="skip grasp execution; Home -> ready is still performed",
    )
    parser.add_argument("--camera-type", choices=("realsense_d435i", "realsense_d405", "orbbec_gemini2"), default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--num-view", type=int, default=300)
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--min-depth", type=float, default=0.05, help="meters")
    parser.add_argument("--max-depth", type=float, default=2.0, help="meters")
    parser.add_argument("--target-class", default=None)
    parser.add_argument("--extra-yolo-class", action="append", default=[], help="add open-vocabulary YOLO class")
    parser.add_argument("--target-margin-px", type=int, default=None)
    parser.add_argument("--target-expand-ratio", type=float, default=None, help="YOLO bbox expansion ratio")
    parser.add_argument("--no-yolo", action="store_true", help="disable YOLO and use full-scene GraspNet")
    parser.add_argument("--yolo-model", default=None)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--yolo-iou", type=float, default=None)
    parser.add_argument("--infer-every-live", type=int, default=None)
    parser.add_argument("--pregrasp-offset", type=float, default=None, help="meters")
    parser.add_argument("--retreat-offset", type=float, default=None, help="meters")
    parser.add_argument("--min-base-z", type=float, default=None, help="minimum executable TCP z in base frame, meters")
    parser.add_argument("--no-open3d", action="store_true", help="do not open Open3D after inference")
    parser.add_argument(
        "--open3d-grasps",
        choices=("final", "bbox", "pre-bbox"),
        default="final",
        help="Open3D grasp set: final, bbox, or pre-bbox",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = configure_camera(load_config(Path(args.config)), args)

    robot_cfg = cfg.get("robot", {})
    robot_backend = str(args.robot_backend or robot_cfg.get("backend", "rars01")).lower()
    max_grasp_width_m = (
        float(robot_cfg.get("rars01", {}).get("max_grasp_width_m", 0.100))
        if robot_backend == "rars01" else GRIPPER_MAX_DISTANCE_M
    )
    ready_cfg = robot_cfg.get(
        "ready_pose",
        {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
    )
    motion_cfg = robot_cfg.get("motion", {})
    pipeline_cfg = cfg.get("grasp_pipeline", {})
    grasp_mode = str(pipeline_cfg.get("mode", "graspnet")).lower()
    if grasp_mode not in ("graspnet", "central_mask"):
        raise ValueError("grasp_pipeline.mode must be 'graspnet' or 'central_mask'")
    if grasp_mode == "central_mask" and args.no_yolo:
        raise ValueError("central_mask mode requires YOLO; remove --no-yolo")
    grasp_cfg = pipeline_cfg.get("grasp", {})
    pregrasp_offset_m = float(args.pregrasp_offset if args.pregrasp_offset is not None else grasp_cfg.get("pregrasp_offset_m", 0.08))
    retreat_offset_m = float(args.retreat_offset if args.retreat_offset is not None else pregrasp_offset_m)
    insertion_depth_m = float(grasp_cfg.get("insertion_depth_m", 0.0))
    max_grasp_depth_m = float(robot_cfg.get("rars01", {}).get("max_grasp_depth_m", 0.080))
    if not 0.0 <= insertion_depth_m <= max_grasp_depth_m:
        raise ValueError("insertion_depth_m must be between 0 and robot.rars01.max_grasp_depth_m")
    min_base_z_m = float(args.min_base_z if args.min_base_z is not None else grasp_cfg.get("min_base_z_m", 0.03))
    min_jaw_z_m = float(grasp_cfg.get("min_jaw_z_m", 0.01))
    compensation_cfg = grasp_cfg.get("position_compensation_base_m", {})
    position_compensation_base_m = np.array(
        [float(compensation_cfg.get(axis, 0.0)) for axis in ("x", "y", "z")],
        dtype=np.float64,
    )
    ik_retry_count = int(grasp_cfg.get("ik_retry_count", 3))
    ik_candidate_limit = int(grasp_cfg.get("ik_candidate_limit", 20))
    if ik_candidate_limit <= 0:
        raise ValueError("grasp_pipeline.grasp.ik_candidate_limit must be positive")
    cartesian_ik_cfg = dict(grasp_cfg.get("cartesian_ik", {}))
    cartesian_ik_cfg.setdefault("waypoint_count", 20)
    cartesian_ik_cfg.setdefault("joint_step_warn_deg", 10.0)
    cartesian_ik_cfg.setdefault("max_joint_step_deg", 15.0)
    cartesian_ik_cfg.setdefault("ik_position_tolerance_m", 0.010)
    cartesian_ik_cfg.setdefault("ik_orientation_tolerance_deg", 5.0)
    if not 2 <= int(cartesian_ik_cfg["waypoint_count"]) <= 100:
        raise ValueError("cartesian_ik.waypoint_count must be between 2 and 100")
    if not (
        0.0 < float(cartesian_ik_cfg["joint_step_warn_deg"])
        <= float(cartesian_ik_cfg["max_joint_step_deg"])
    ):
        raise ValueError("cartesian_ik joint step thresholds must satisfy 0 < warn <= reject")
    candidate_selection_cfg = dict(grasp_cfg.get("candidate_selection", {}))
    candidate_selection_cfg.setdefault("weight_grasp", 0.35)
    candidate_selection_cfg.setdefault("weight_joint", 0.25)
    candidate_selection_cfg.setdefault("weight_motion", 0.30)
    candidate_selection_cfg.setdefault("motion_normalization", 2.0)
    weights = np.asarray(
        [
            candidate_selection_cfg["weight_grasp"],
            candidate_selection_cfg["weight_joint"],
            candidate_selection_cfg["weight_motion"],
        ],
        dtype=np.float64,
    )
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("candidate_selection weights must be non-negative and non-zero")
    if float(candidate_selection_cfg["motion_normalization"]) <= 0.0:
        raise ValueError("candidate_selection.motion_normalization must be positive")
    depth_quantile = float(grasp_cfg.get("depth_quantile", 0.5))
    central_mask_approach = str(
        grasp_cfg.get("central_mask_approach", "camera_ray")
    ).lower()
    if central_mask_approach not in ("camera_ray", "vertical"):
        raise ValueError(
            "grasp_pipeline.grasp.central_mask_approach must be "
            "'camera_ray' or 'vertical'"
        )
    graspnet_cfg = cfg.get("graspnet", {})
    checkpoint_path = graspnet_utils.resolve_checkpoint_path(
        str(args.checkpoint or graspnet_cfg.get("checkpoint", "models/checkpoint-rs.tar")),
        project_root=PROJECT_ROOT,
        graspnet_root=GRASPNET_ROOT,
    )
    target_class = args.target_class or graspnet_cfg.get("target_class")
    target_expand_ratio = float(
        args.target_expand_ratio
        if args.target_expand_ratio is not None
        else graspnet_cfg.get("target_expand_ratio", 1.0)
    )

    cam_cfg = cfg["camera"]
    print(f"=== Init camera: {cam_cfg['type']} {cam_cfg.get('color_width')}x{cam_cfg.get('color_height')}@{cam_cfg.get('fps')} ===")
    cam = make_camera(cfg)

    last_detections: list[YoloDetection] = []
    selected_target: Optional[Any] = None
    last_target_status = "target detector warming up..."
    status = "warming up camera..."
    frozen = False
    last_display: Optional[np.ndarray] = None
    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0
    window_name = f"Main - {grasp_mode} Grasp"
    top_k = int(cfg.get("graspnet", {}).get("top_k", 50))
    vis: Optional[graspnet_utils.Open3DGraspWindow] = None
    graspnet_worker: Optional[GraspNetWorker] = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(cam_cfg.get("color_width", 1280)), int(cam_cfg.get("color_height", 720)))
    print("\n[Keys] G/SPACE=grasp  R=resume  Q/ESC=quit\n")

    arm: Optional[Any] = None
    controller: Optional[Any] = None
    grasp_driver: Optional[GraspDriver] = None
    ik_checker: Optional[IkChecker] = None
    T_hand_eye: Optional[np.ndarray] = None
    robot_ready = False

    try:
        cam.open()
        cam.warm_up(args.warmup)
        K = cam.K.astype(np.float64)
        print("Camera intrinsics:")
        print(K)

        cam_type = str(cam_cfg.get("type", "")).lower()
        T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
        if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
            print("[WARN] Hand-eye calibration unavailable; grasp execution disabled")
            T_hand_eye = None
        print("=== Load models ===")
        yolo_model, yolo_opts = load_yolo_from_config(
            cfg,
            project_root=PROJECT_ROOT,
            no_yolo=args.no_yolo,
            model_override=args.yolo_model,
            device_override=args.yolo_device,
            conf_override=args.yolo_conf,
            iou_override=args.yolo_iou,
            infer_every_override=args.infer_every_live,
            extra_classes=args.extra_yolo_class,
        )
        last_target_status = "YOLO disabled: full-scene GraspNet" if yolo_model is None else "target detector warming up..."
        if grasp_mode == "graspnet":
            print("[Pipeline] starting isolated GraspNet CUDA worker")
            graspnet_worker = GraspNetWorker(checkpoint_path, args.num_view)
        print(f"[Pipeline] grasp mode: {grasp_mode}")

        print("=== Init robot ===")
        from rars01_graspnet.pose_controller import RarsPoseController

        if robot_backend == "rars01":
            answer = input(
                "RARS01: place the arm in zero/home, clear the path and type START: "
            ).strip()
            if answer != "START":
                print("[RARS01] Cancelled before serial or motors were opened")
                return 0
            arm = RarsArmAdapter(robot_cfg, PROJECT_ROOT)
            controller = RarsPoseController(
                arm,
                # Joints 1..6 use the STM POS/VEL profile.  The gripper is
                # independently kept in MIT by GraspDriver.
                dt=1.0 / arm.rate,
                arm_control_mode="posvel",
            )
            mode_name = "posvel arm + mit gripper (RARS transport)"
        else:
            raise ValueError("Only robot.backend: rars01 is supported")

        grasp_driver = GraspDriver(
            arm,
            controller,
            gripper_config=robot_cfg.get("gripper"),
        )
        grasp_driver.start()
        robot_ready = True
        ik_checker = IkChecker(
            arm,
            retry_count=ik_retry_count,
            bounded_fallback=grasp_cfg.get("ik_bounded_fallback_enabled", False),
            position_tolerance_m=float(cartesian_ik_cfg["ik_position_tolerance_m"]),
            orientation_tolerance_rad=np.deg2rad(
                float(cartesian_ik_cfg["ik_orientation_tolerance_deg"])
            ),
        )
        print(f"[Robot] mode: {mode_name}")
        print("[Robot] Move ready")
        if not _move_ready(controller, ready_cfg):
            raise RuntimeError("Could not start the ready trajectory")

        while True:
            grasp_driver.check_health()
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                continue

            frame_index += 1
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                fps_value = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            if not frozen and yolo_model is not None and (frame_index == 1 or frame_index % int(yolo_opts["infer_every"]) == 0):
                try:
                    _, last_detections = detect_objects(yolo_model, color_bgr, yolo_opts)
                    selected_target = graspnet_utils.select_target(last_detections, target_class)
                    last_target_status = graspnet_utils.target_status_text(selected_target, last_detections, target_class)
                except Exception as exc:
                    last_detections = []
                    selected_target = None
                    last_target_status = f"YOLO failed: {exc}"

            if frozen and last_display is not None:
                display = last_display.copy()
            else:
                display_base = color_bgr
                if yolo_model is not None:
                    display_base = graspnet_utils.draw_detections_overlay(color_bgr, last_detections, selected_target, target_class)
                display = graspnet_utils.draw_status(
                    display_base,
                    f"LIVE {fps_value:.1f}fps | {status}",
                    last_target_status,
                    title=f"Main - {grasp_mode} Grasp",
                )
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                status = "live preview"
                continue

            if key in (ord("g"), ord("G"), ord(" ")):
                print(f"\n[G] Capture and run {grasp_mode}")
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[G] Frame capture failed")
                    continue

                central_best: Optional[GraspPose] = None
                try:
                    if grasp_mode == "graspnet":
                        if graspnet_worker is None:
                            raise RuntimeError("GraspNet worker is unavailable")
                        # Reuse the latest live YOLO target. Only GraspNet CUDA
                        # runs in the child process; this process remains free
                        # to keep the RARS01 command loop alive at 100 Hz.
                        if yolo_model is not None and selected_target is None:
                            print("[G] No current YOLO target")
                            continue
                        worker_result = graspnet_worker.infer(
                            snap_color, snap_depth, K,
                            num_point=args.num_point,
                            min_depth=args.min_depth,
                            max_depth=args.max_depth,
                            collision_thresh=args.collision_thresh,
                            voxel_size=args.voxel_size,
                            bbox_xyxy=(selected_target.bbox_xyxy if selected_target else None),
                            target_margin_px=int(
                                args.target_margin_px
                                if args.target_margin_px is not None
                                else graspnet_cfg.get("target_margin_px", 12)
                            ),
                            target_expand_ratio=target_expand_ratio,
                            max_grasp_width_m=max_grasp_width_m,
                            max_grasp_depth_m=max_grasp_depth_m,
                            minimum_grasp_depth_m=graspnet_cfg.get("minimum_grasp_depth_m",0.),
                            timeout_s=float(graspnet_cfg.get("worker_timeout_s", 30.0)),
                        )
                        candidate_grasps = GraspGroup(worker_result["grasps"])
                        pre_bbox_grasps = GraspGroup(worker_result["pre_bbox_grasps"])
                        bbox_grasps = GraspGroup(worker_result["bbox_grasps"])
                        result = graspnet_utils.GraspNetFrameResult(
                            grasps=candidate_grasps,
                            pre_bbox_grasps=pre_bbox_grasps,
                            bbox_grasps=bbox_grasps,
                            best=graspnet_utils.select_best_grasp(candidate_grasps),
                            status="",
                            target_status=last_target_status,
                            detections=last_detections,
                            selected_target=selected_target,
                            o3d_cloud=None,
                            raw_cloud=np.empty((0, 3), dtype=np.float32),
                        )
                        counts = worker_result["counts"]
                        label = (
                            f"{selected_target.class_name} {selected_target.conf:.2f}"
                            if selected_target else "full scene"
                        )
                        result.status = (
                            f"{label} grasps={len(candidate_grasps)}/{len(bbox_grasps)}/"
                            f"{len(pre_bbox_grasps)} decoded={counts['decoded']} "
                            f"collide={counts['collision_removed']}/{counts['pre_collision']} "
                            f"inference={worker_result['elapsed_s']:.2f}s"
                        )
                        status = result.status
                        last_target_status = result.target_status
                        last_detections = result.detections
                        selected_target = result.selected_target
                        candidate_grasps = result.grasps
                        if result.best is None:
                            if counts["decoded"] == 0:
                                reason = "network decoded 0 candidates"
                            elif len(pre_bbox_grasps) == 0:
                                reason = (
                                    f"collision filter removed all "
                                    f"({counts['collision_removed']}/{counts['pre_collision']})"
                                )
                            elif len(bbox_grasps) == 0:
                                reason = "no grasp center inside target YOLO region"
                            else:
                                reason = "all candidates exceed gripper width/depth limits"
                            print(f"[G] No valid GraspNet grasp: {reason}")
                            continue

                        vis_grasps = graspnet_utils.visualization_grasps(result, args.open3d_grasps)
                        if not args.no_open3d:
                            try:
                                if vis is None:
                                    vis = graspnet_utils.Open3DGraspWindow("GraspNet Grasps", top_k)
                                vis.update(result.o3d_cloud, vis_grasps)
                                print(f"[G] Open3D {args.open3d_grasps} candidates={len(vis_grasps)}")
                            except Exception as exc:
                                print(f"[G] Open3D failed: {exc}")
                                if vis is not None:
                                    vis.close()
                                    vis = None

                        display_base = graspnet_utils.draw_detections_overlay(
                            snap_color, last_detections, selected_target, target_class
                        )
                    else:
                        snap_results, last_detections = detect_objects(yolo_model, snap_color, yolo_opts)
                        selected_target = graspnet_utils.select_target(last_detections, target_class)
                        last_target_status = graspnet_utils.target_status_text(
                            selected_target, last_detections, target_class
                        )
                        central_grasps = estimate_central_mask_grasps(
                            snap_results, snap_depth, K, depth_quantile=depth_quantile
                        )
                        central_best = _select_central_mask_grasp(central_grasps, target_class)
                        if central_best is None:
                            print("[G] No valid central-mask grasp")
                            continue
                        if central_best.jaw_width_m > max_grasp_width_m:
                            print(
                                f"[G] Central grasp width {central_best.jaw_width_m:.4f} m "
                                f"exceeds gripper limit {max_grasp_width_m:.4f} m"
                            )
                            continue
                        candidate = _central_mask_to_graspnet(central_best)
                        flipped_candidate = _parallel_flip_grasp(candidate)
                        candidate_grasps = GraspGroup(
                            np.stack((candidate.grasp_array, flipped_candidate.grasp_array))
                        )
                        status = (
                            f"central_mask target={central_best.class_name} "
                            f"conf={central_best.conf:.2f}"
                        )
                        display_base = graspnet_utils.draw_detections_overlay(
                            snap_color, last_detections, selected_target, target_class
                        )
                        draw_central_mask_grasp(display_base, central_best)
                except Exception as exc:
                    status = f"inference failed: {exc}"
                    print(f"[G] {status}")
                    continue

                print(f"[G] {status}")
                frozen = True
                snap_display = graspnet_utils.draw_status(
                    display_base,
                    f"SNAPSHOT | {status}",
                    last_target_status,
                    frozen=True,
                    title=f"Main - {grasp_mode} Grasp",
                )
                last_display = snap_display

                if T_hand_eye is None:
                    if grasp_mode == "graspnet":
                        graspnet_utils.draw_best_grasp_projection(snap_display, result.best, K)
                    last_display = snap_display
                    print("[G] Hand-eye calibration unavailable")
                    continue

                T_cam2base = compose_cam_to_base_transform(grasp_driver.get_tcp_pose(), T_hand_eye, cfg)
                if central_best is not None and central_mask_approach == "vertical":
                    candidate = _grasp_with_base_approach(
                        candidate,
                        T_cam2base,
                        np.array([0.0, 0.0, -1.0], dtype=np.float64),
                    )
                    flipped_candidate = _parallel_flip_grasp(candidate)
                    candidate_grasps = GraspGroup(
                        np.stack((candidate.grasp_array, flipped_candidate.grasp_array))
                    )
                if central_best is not None:
                    print(f"[G] central-mask approach={central_mask_approach}")
                from rars01_graspnet.grasp_ranking import target_shape
                shape = target_shape(snap_depth,K,getattr(selected_target,"mask",None),candidate_selection_cfg,
                    mask_source=getattr(selected_target,"mask_source","unknown"),min_depth=args.min_depth,max_depth=args.max_depth)
                selected = _select_executable_grasp(
                    ik_checker,
                    grasp_driver,
                    candidate_grasps,
                    T_cam2base,
                    pregrasp_offset_m,
                    retreat_offset_m,
                    insertion_depth_m if grasp_mode == "graspnet" else 0.0,
                    max_grasp_depth_m,
                    min_base_z_m,
                    min_jaw_z_m,
                    robot_backend,
                    grasp_mode == "graspnet",
                    grasp_mode == "graspnet",
                    position_compensation_base_m,
                    ik_candidate_limit,
                    cartesian_ik_cfg,
                    candidate_selection_cfg,
                    max_grasp_width_m=max_grasp_width_m,
                    ranking_robot=cfg["robot"],
                    target_shape_camera=shape,
                )
                if selected is None:
                    print(f"[G] No IK-reachable grasp above min_base_z={min_base_z_m:.3f}m ")
                    continue
                best = selected.grasp
                grasp6d = selected.grasp6d
                pre6d = selected.pregrasp6d
                retreat6d = selected.retreat6d

                if central_best is None:
                    _print_grasp(best, robot_backend)
                    graspnet_utils.draw_best_grasp_projection(snap_display, best, K)
                else:
                    _print_central_mask_grasp(central_best)
                last_display = snap_display

                _execute_grasp(
                    controller,
                    grasp_driver,
                    grasp6d,
                    pre6d,
                    retreat6d,
                    ready_cfg,
                    motion_cfg,
                    dry_run=args.dry_run,
                    joint_targets=(
                        selected.pregrasp_joints,
                        selected.grasp_joints,
                        selected.retreat_joints,
                    ) if robot_backend == "rars01" else None,
                    cartesian_joint_paths=(
                        selected.approach_joint_waypoints,
                        selected.retreat_joint_waypoints,
                    ) if robot_backend == "rars01" else None,
                )

            if vis is not None and not vis.poll():
                vis.close()
                vis = None

    finally:
        print("\n[Exit] Release gripper and home")
        try:
            if (robot_ready and grasp_driver is not None
                    and controller is not None and getattr(controller, "_running", False)):
                grasp_driver.release_gripper()
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            if controller is not None and getattr(controller, "_running", False):
                controller.end()
            elif arm is not None:
                arm.disconnect()
        except Exception as exc:
            print(f"[Exit] {exc}")
            try:
                if arm is not None:
                    arm.disconnect()
            except Exception as disconnect_exc:
                print(f"[Exit] disconnect: {disconnect_exc}")
        try:
            cam.close()
        except Exception:
            pass
        if graspnet_worker is not None:
            graspnet_worker.close()
        if vis is not None:
            vis.close()
        cv2.destroyAllWindows()
        print("Done.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
