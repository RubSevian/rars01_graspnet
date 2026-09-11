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
    RarsRebotArm,
    selected_arm_config,
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
    graspnet_rotation_to_rebot_tcp_rotation,
    pose6d_to_mat4,
    rotation_matrix_to_euler_zyx,
)
from utils.yolo_utils import (  # noqa: E402
    YoloDetection,
    detect_objects,
    load_yolo as load_yolo_from_config,
)
from graspnetAPI import Grasp, GraspGroup  # noqa: E402

_PARALLEL_FLIP_X = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def _wait_motion(controller: Any, duration: float, extra: float = 0.6) -> None:
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + extra + 2.0)
    else:
        time.sleep(duration + extra)
    # RarsRebotArm deliberately executes its 100 Hz callback in a background
    # thread.  Surface its exception immediately: otherwise the main UI keeps
    # printing the remaining grasp steps after the SDK has stopped commands,
    # which lets the STM watchdog disable the motors without the actual cause
    # appearing in the terminal log.
    transport = getattr(controller, "rebotarm", None)
    failure = getattr(transport, "_failure", None)
    if failure is not None:
        raise RuntimeError(f"RARS01 control loop failed: {failure}") from failure
    if transport is not None and not getattr(transport, "_running", True):
        raise RuntimeError("RARS01 control loop stopped unexpectedly")


def _move_ready(controller: Any, ready_cfg: dict[str, Any]) -> None:
    duration = float(ready_cfg.get("duration", 3.0))
    controller.move_to_traj(
        x=float(ready_cfg.get("x", 0.25)),
        y=float(ready_cfg.get("y", 0.0)),
        z=float(ready_cfg.get("z", 0.35)),
        roll=float(ready_cfg.get("roll", 0.0)),
        pitch=float(ready_cfg.get("pitch", 1.2)),
        yaw=float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    _wait_motion(controller, duration)


@dataclass(frozen=True)
class IkSolution:
    success: bool
    error: float
    joints: np.ndarray


@dataclass(frozen=True)
class ExecutableGrasp:
    grasp: Grasp
    grasp6d: tuple[float, ...]
    pregrasp6d: tuple[float, ...]
    retreat6d: tuple[float, ...]
    pregrasp_joints: np.ndarray
    grasp_joints: np.ndarray
    retreat_joints: np.ndarray


class IkChecker:
    def __init__(self, arm: Any, retry_count: int = 3) -> None:
        from reBotArm_control_py.kinematics import (
            get_end_effector_frame_id,
            load_robot_model,
            pad_q_for_model,
            pos_rot_to_se3,
            solve_ik,
        )
        from reBotArm_control_py.kinematics.inverse_kinematics import IKParams

        self._arm = arm
        self._arm_group = arm.groups.get("arm")
        if self._arm_group is None:
            raise ValueError("Hardware config missing groups.arm")
        self._n = self._arm_group.num_joints
        self._pad_q_for_model = pad_q_for_model
        self._pos_rot_to_se3 = pos_rot_to_se3
        self._solve_ik = solve_ik
        load_arm_model = getattr(arm, "load_kinematic_model", None)
        self._model = load_arm_model() if load_arm_model is not None else load_robot_model()
        self._end_frame_id = get_end_effector_frame_id(self._model)
        self._params = IKParams(max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6)
        self._retry_count = max(0, int(retry_count))
        self._lower = np.asarray(self._model.lowerPositionLimit[:self._n], dtype=np.float64)
        self._upper = np.asarray(self._model.upperPositionLimit[:self._n], dtype=np.float64)

    def current_joints(self) -> np.ndarray:
        return np.asarray(
            self._arm.get_state(request_feedback=False)[0][:self._n], dtype=np.float64
        ).copy()

    def solve(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        reference_joints: Optional[np.ndarray] = None,
    ) -> IkSolution:
        reference = (
            self.current_joints()
            if reference_joints is None
            else np.asarray(reference_joints, dtype=np.float64).reshape(self._n)
        )
        target = self._pos_rot_to_se3(
            np.array([x, y, z], dtype=np.float64), roll=roll, pitch=pitch, yaw=yaw
        )
        midpoint = 0.5 * (self._lower + self._upper)
        seeds = [reference]
        for fraction in (0.25, 0.50, 0.75)[:self._retry_count]:
            seeds.append((1.0 - fraction) * reference + fraction * midpoint)

        best_result = None
        for seed in seeds:
            result = self._solve_ik(
                self._model,
                self._model.createData(),
                self._end_frame_id,
                target,
                self._pad_q_for_model(self._model, seed, self._n),
                self._params,
                controlled_joints=self._n,
            )
            if best_result is None or float(result.error) < float(best_result.error):
                best_result = result
            if result.success:
                q = np.asarray(result.q[:self._n], dtype=np.float64)
                # Seeds are ordered from the previous chain point toward the
                # joint-range midpoint, so the first success is the most local.
                return IkSolution(True, float(result.error), q.copy())
        assert best_result is not None
        return IkSolution(
            False,
            float(best_result.error),
            np.asarray(best_result.q[:self._n], dtype=np.float64).copy(),
        )

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

    print("[Grasp] Move to pregrasp")
    if joint_targets is not None:
        duration = grasp_driver.move_rars_joint_target(joint_targets[0], pregrasp_duration)
    else:
        if not controller.move_to_traj(xp, yp, zp, rxp, ryp, rzp, duration=pregrasp_duration):
            print("[Grasp] Pregrasp IK failed")
            return False
        duration = pregrasp_duration
    _wait_motion(controller, duration)

    print("[Grasp] Move to grasp")
    if joint_targets is not None:
        duration = grasp_driver.move_rars_joint_target(joint_targets[1], grasp_duration)
    else:
        if not controller.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=grasp_duration):
            print("[Grasp] Grasp IK failed")
            return False
        duration = grasp_duration
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
    if joint_targets is not None:
        duration = grasp_driver.move_rars_joint_target(joint_targets[2], retreat_duration)
        _wait_motion(controller, duration)
    elif controller.move_to_traj(xr, yr, zr, rxr, ryr, rzr, duration=retreat_duration):
        _wait_motion(controller, retreat_duration)

    print("[Grasp] Return ready")
    _move_ready(controller, ready_cfg)
    return ok


def _print_grasp(grasp: Grasp, robot_backend: str) -> None:
    rotation_fn = (
        graspnet_rotation_to_rars_tcp_rotation
        if robot_backend == "rars01"
        else graspnet_rotation_to_rebot_tcp_rotation
    )
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(rotation_fn(grasp.rotation_matrix))
    print("\n[G] Best GraspNet grasp:")
    print(f"  score={grasp.score:.4f} width={grasp.width:.4f} height={grasp.height:.4f} depth={grasp.depth:.4f}")
    print(f"  position_xyz={grasp.translation.tolist()}")
    print(f"  graspnet_rpy={rotation_matrix_to_euler_zyx(grasp.rotation_matrix).tolist()}")
    print(f"  tcp_rpy={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")


def _rank_grasps(grasps: GraspGroup, apply_nms: bool = True) -> GraspGroup:
    ranked = GraspGroup(grasps.grasp_group_array.copy())
    if apply_nms and len(ranked) > 1:
        try:
            ranked = ranked.nms()
        except Exception as exc:
            print(f"[WARN] GraspNet NMS skipped: {exc}")
    ranked.sort_by_score()
    return ranked


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


def _parallel_flip_grasp(grasp: Grasp) -> Grasp:
    """Return the same parallel-jaw grasp with the transverse axes reversed."""
    rotation = np.asarray(grasp.rotation_matrix, dtype=np.float64) @ _PARALLEL_FLIP_X
    return Grasp(
        float(grasp.score),
        float(grasp.width),
        float(grasp.height),
        float(grasp.depth),
        rotation,
        np.asarray(grasp.translation, dtype=np.float64),
        int(grasp.object_id),
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


def _pose_z_ok(pose6d: tuple[float, ...], min_z: float) -> bool:
    return float(pose6d[2]) >= float(min_z)


def _translate_pose(
    pose6d: tuple[float, ...], offset_base_m: np.ndarray
) -> tuple[float, ...]:
    pose = np.asarray(pose6d, dtype=np.float64).copy()
    pose[:3] += np.asarray(offset_base_m, dtype=np.float64).reshape(3)
    return tuple(float(value) for value in pose)


def _select_executable_grasp(
    ik_checker: IkChecker,
    grasp_driver: GraspDriver,
    grasps: GraspGroup,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    retreat_offset_m: float,
    insertion_depth_m: float,
    max_grasp_depth_m: float,
    min_base_z_m: float,
    min_jaw_z_m: float,
    robot_backend: str,
    allow_parallel_flip: bool,
    apply_nms: bool,
    prefer_current_orientation: bool,
    position_compensation_base_m: np.ndarray,
    candidate_limit: int,
) -> Optional[ExecutableGrasp]:
    ranked = _rank_grasps(grasps, apply_nms=apply_nms)
    skipped_low = 0
    skipped_jaw = 0
    skipped_depth = 0
    skipped_ik = 0
    worst_err = 0.0

    pose_candidates = []
    current_rotation = (
        grasp_driver.get_tcp_pose()[:3, :3]
        if prefer_current_orientation else None
    )
    for idx in range(min(len(ranked), candidate_limit)):
        grasp = ranked[idx]
        if float(grasp.depth) > max_grasp_depth_m:
            skipped_depth += 1
            continue
        orientation_candidates = (
            (grasp, _parallel_flip_grasp(grasp)) if allow_parallel_flip else (grasp,)
        )
        for branch_index, oriented_grasp in enumerate(orientation_candidates):
            T_grasp_tcp = grasp_driver.grasp_tcp_transform(float(oriented_grasp.width))
            grasp6d, pre6d, retreat6d = graspnet_utils.grasp_to_base_poses(
                oriented_grasp,
                T_cam2base,
                pregrasp_offset_m,
                retreat_offset_m,
                insertion_depth_m,
                tcp_convention=robot_backend,
                T_grasp_tcp=T_grasp_tcp,
                # Both equivalent parallel-gripper orientations are checked here.
                allow_parallel_flip=False,
            )
            grasp6d = _translate_pose(grasp6d, position_compensation_base_m)
            pre6d = _translate_pose(pre6d, position_compensation_base_m)
            retreat6d = _translate_pose(retreat6d, position_compensation_base_m)
            rotation_cost = 0.0
            if current_rotation is not None:
                target_rotation = pose6d_to_mat4(*pre6d)[:3, :3]
                relative = current_rotation.T @ target_rotation
                cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
                rotation_cost = float(np.arccos(cosine))
            pose_candidates.append(
                (rotation_cost, idx, branch_index, grasp, T_grasp_tcp, grasp6d, pre6d, retreat6d)
            )

    if prefer_current_orientation:
        pose_candidates.sort(key=lambda item: item[0])

    current_joints = ik_checker.current_joints()
    for exec_idx, candidate in enumerate(pose_candidates):
        _, original_idx, branch_index, grasp, T_grasp_tcp, grasp6d, pre6d, retreat6d = candidate
        if not all(
            _pose_z_ok(pose, min_base_z_m) for pose in (pre6d, grasp6d, retreat6d)
        ):
            skipped_low += 1
            continue
        if (
            robot_backend == "rars01"
            and min(
                grasp_driver.minimum_jaw_height(pose, float(grasp.width))
                for pose in (pre6d, grasp6d, retreat6d)
            ) < min_jaw_z_m
        ):
            skipped_jaw += 1
            continue

        pre = ik_checker.solve(*pre6d, reference_joints=current_joints)
        grasp_solution = (
            ik_checker.solve(*grasp6d, reference_joints=pre.joints)
            if pre.success else IkSolution(False, pre.error, pre.joints)
        )
        retreat = (
            ik_checker.solve(*retreat6d, reference_joints=grasp_solution.joints)
            if grasp_solution.success
            else IkSolution(False, grasp_solution.error, grasp_solution.joints)
        )
        worst_err = max(worst_err, pre.error, grasp_solution.error, retreat.error)
        if pre.success and grasp_solution.success and retreat.success:
            print(f"[G] Executable rank={exec_idx + 1}/{len(ranked)} score={grasp.score:.4f}")
            if prefer_current_orientation:
                print(
                    f"[G] Central-mask orientation branch={original_idx + 1}/2 "
                    f"(minimum wrist rotation first)"
                )
            elif allow_parallel_flip:
                print(f"[G] parallel-gripper orientation branch={branch_index + 1}/2")
            if robot_backend == "rars01":
                print(
                    "[G] jaw center in End_link [m]: "
                    f"{np.round(T_grasp_tcp[:3, 3], 5).tolist()}"
                )
            if skipped_depth or skipped_low or skipped_jaw or skipped_ik:
                print(
                    f"[G] Skipped depth={skipped_depth} low_z={skipped_low} jaw_z={skipped_jaw} "
                    f"ik_fail={skipped_ik}"
                )
            return ExecutableGrasp(
                grasp=grasp,
                grasp6d=grasp6d,
                pregrasp6d=pre6d,
                retreat6d=retreat6d,
                pregrasp_joints=pre.joints,
                grasp_joints=grasp_solution.joints,
                retreat_joints=retreat.joints,
            )
        skipped_ik += 1

    print(
        f"[G] No executable grasp: depth={skipped_depth} low_z={skipped_low} jaw_z={skipped_jaw} "
        f"ik_fail={skipped_ik} max_err={worst_err:.4f}"
    )
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraspNet/central-mask robot grasp demo")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--robot-backend", choices=("rebot", "rars01"), default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="GraspNet checkpoint; default is graspnet.checkpoint from YAML",
    )
    parser.add_argument("--dry-run", action="store_true", help="estimate only; do not move the arm")
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
    robot_backend = str(args.robot_backend or robot_cfg.get("backend", "rebot")).lower()
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

    rebotarm: Optional[Any] = None
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
        from reBotArm_control_py.controllers import RebotArmEndPose

        if robot_backend == "rars01":
            answer = input(
                "RARS01: place the arm in zero/home, clear the path and type START: "
            ).strip()
            if answer != "START":
                print("[RARS01] Cancelled before serial or motors were opened")
                return 0
            rebotarm = RarsRebotArm(robot_cfg, PROJECT_ROOT)
            controller = RebotArmEndPose(
                rebotarm,
                # Joints 1..6 use the STM POS/VEL profile.  The gripper is
                # independently kept in MIT by GraspDriver.
                dt=1.0 / rebotarm.rate,
                arm_control_mode="posvel",
                use_gravity_ff=False,
            )
            mode_name = "posvel arm + mit gripper (RARS transport)"
        else:
            from reBotArm_control_py.actuator import RebotArm

            selected = selected_arm_config(robot_cfg.get("repo_root"))
            rebotarm = RebotArm()
            controller = RebotArmEndPose(rebotarm, arm_control_mode=selected.controller_mode)
            mode_name = selected.controller_mode

        grasp_driver = GraspDriver(
            rebotarm,
            controller,
            gripper_config=robot_cfg.get("gripper"),
            repo_root=robot_cfg.get("repo_root"),
        )
        grasp_driver.start()
        robot_ready = True
        ik_checker = IkChecker(rebotarm, retry_count=ik_retry_count)
        print(f"[Robot] mode: {mode_name}")
        print("[Robot] Move ready")
        _move_ready(controller, ready_cfg)

        while True:
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
                            print("[G] No valid GraspNet grasp")
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
                    grasp_mode == "central_mask",
                    position_compensation_base_m,
                    ik_candidate_limit,
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
            elif rebotarm is not None:
                rebotarm.disconnect()
        except Exception as exc:
            print(f"[Exit] {exc}")
            try:
                if rebotarm is not None:
                    rebotarm.disconnect()
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
