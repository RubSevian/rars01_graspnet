#!/usr/bin/env python3
"""RARS01 port of reBot-DevArm-Grasp/scripts/grasp.py's one-shot pipeline."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from rars01_graspnet.camera import camera_from_config
from rars01_graspnet.config import load_config, resolve_path
from rars01_graspnet.detection import detector_from_config, draw_detections
from rars01_graspnet.graspnet import estimator_from_config, select_target
from rars01_graspnet.gripper_geometry import RarsGripperGeometry
from rars01_graspnet.hand_eye import camera_to_base, load, pose_transform
from rars01_graspnet.ik import pregrasp_transform, solve_pose_ik, tcp_target_from_grasp
from rars01_graspnet.kinematics import RarsKinematics
from rars01_graspnet.robot import robot_from_config
from rars01_graspnet.trajectory import track_cartesian_trajectory


@dataclass(frozen=True)
class _ExecutableGrasp:
    candidate: object
    opening: object
    pre: object
    grasp: object
    T_grasp_camera: np.ndarray
    rank: int
    T_pregrasp_tcp: np.ndarray
    T_grasp_tcp: np.ndarray
    T_retreat_tcp: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--target", default="banana")
    parser.add_argument("--execute", action="store_true",
                        help="authorize execution after G/Space selects a reachable grasp")
    parser.add_argument("--dry-run", action="store_true",
                        help="authorize home -> seed and selection, but no grasp execution")
    parser.add_argument("--no-open3d", action="store_true")
    args = parser.parse_args()
    if args.execute and args.dry_run:
        raise RuntimeError("Choose only one of --dry-run or --execute")
    config = load_config(args.config)
    rc, ac = config["robot"], config["calibration"]["auto"]
    safety, planning = config["safety"], config["planning"]
    reference = planning["reference_grasp"]
    expected_home = np.asarray(ac["expected_start_joints_rad"], dtype=np.float64)
    seed = np.asarray(ac["seed_joints_rad"], dtype=np.float64)
    rate_hz = float(safety.get("trajectory_rate_hz", 50.0))
    max_step = float(safety.get("max_joint_step_rad", 0.02))

    if not args.execute and not args.dry_run:
        print("Configuration check only. Use --dry-run or --execute to open hardware.")
        return
    mode = "EXECUTION" if args.execute else "DRY-RUN"
    answer = input(
        f"{mode}: place RARS01 in zero/home, clear the full path, and type GRASP: "
    ).strip()
    if answer != "GRASP":
        print("Cancelled before models, camera, serial or motors were opened.")
        return

    detector = detector_from_config(config, resolve_path)
    estimator = estimator_from_config(config, resolve_path)
    kinematics = RarsKinematics(
        resolve_path(config, rc["urdf"]), rc["base_frame"], rc["tcp_frame"]
    )
    T_camera_tcp = load(resolve_path(config, config["calibration"]["output"]))
    geometry = RarsGripperGeometry.from_config(planning["gripper_geometry"])
    preview = None

    with camera_from_config(config) as camera, robot_from_config(config) as robot:
        robot.enable()
        hold = robot.current_joints()
        error = float(np.max(np.abs(hold[:6] - expected_home)))
        tolerance = float(ac.get("start_tolerance_rad", 0.15))
        if error > tolerance:
            raise RuntimeError(f"Current pose is not zero/home; max error={error:.3f} rad")
        robot.set_cleanup_return(
            home=expected_home, via=seed,
            via_duration_s=float(ac.get("return_duration_s", 2.0)),
            home_duration_s=float(ac.get("home_return_duration_s", 4.0)),
            rate_hz=rate_hz, max_joint_step_rad=max_step,
        )
        print("[Robot] Move home -> ready/seed")
        hold = robot.move_joints(
            seed, duration_s=float(ac.get("seed_move_duration_s", 4.0)),
            rate_hz=rate_hz, max_joint_step_rad=max_step,
        )
        camera.warm_up()
        detections = []
        frame_index = 0
        selected = None
        quit_requested = False
        window = "RARS01 - reBot-style grasp"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("[Keys] G/Space=infer and select executable grasp | R=resume | Q/Esc=home")

        while not quit_requested:
            selected = None
            with robot.continuous_hold(hold, rate_hz=rate_hz) as check_hold:
                while True:
                    check_hold()
                    if preview is not None and not preview.poll():
                        preview.close()
                        preview = None
                    frame = camera.read()
                    if frame is None:
                        continue
                    frame_index += 1
                    if frame_index == 1 or frame_index % max(
                        1, int(config["yolo"].get("inference_every_n_frames", 3))
                    ) == 0:
                        detections = detector.detect(frame)
                    output = draw_detections(frame.color_bgr, detections)
                    _status(output, "G/Space: infer+select | R: live | Q: home")
                    cv2.imshow(window, output)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        quit_requested = True
                        break
                    if key in (ord("r"), ord("R")):
                        continue
                    if key not in (ord("g"), ord("G"), ord(" ")):
                        continue
                    if select_target(detections, args.target) is None:
                        print(f"[G] Target '{args.target}' is not visible")
                        continue

                    print("[G] Capture one RGB-D frame and run GraspNet")
                    result = estimator.infer(frame, detections, args.target)
                    check_hold()
                    print(f"[G] grasps={len(result.candidates)} decoded={result.decoded_count} "
                          f"collision_removed={result.collision_removed}")
                    T_camera_base = camera_to_base(
                        kinematics.forward(hold[:6]), T_camera_tcp
                    )
                    selected = _select_executable(
                        result.candidates, T_camera_base, kinematics, hold[:6],
                        geometry, planning, reference,
                    )
                    if selected is None:
                        print("[G] No candidate has reachable pregrasp AND grasp; press G for a new frame")
                        continue
                    candidate, opening, pre, grasp = (
                        selected.candidate, selected.opening, selected.pre, selected.grasp
                    )
                    T_grasp_camera, rank = selected.T_grasp_camera, selected.rank
                    print(f"[G] Executable rank={rank}/{len(result.candidates)} "
                          f"score={candidate.score:.4f} width={candidate.width_m:.4f} m")
                    print("[G] pregrasp q [deg]:", np.round(np.rad2deg(pre.joints), 2).tolist())
                    print("[G] grasp q [deg]:", np.round(np.rad2deg(grasp.joints), 2).tolist())
                    _draw_grasp(output, T_grasp_camera, frame.intrinsics.K)
                    cv2.imshow(window, output)
                    cv2.waitKey(1)
                    if not args.no_open3d:
                        if preview is None:
                            preview = _Open3DWindow(estimator)
                        preview.update(result, candidate, T_grasp_camera)
                    break

            if quit_requested:
                break
            if selected is None:
                continue
            if not args.execute:
                print("[DRY-RUN] Reachable grasp selected; skip motion.")
                quit_requested = True
                continue

            opening = selected.opening
            cart_common = dict(
                rate_hz=float(reference.get("cartesian_rate_hz", 50.0)),
                max_joint_step_rad=max_step,
                joint_margin_rad=float(planning.get("ik_joint_margin_rad", 0.05)),
                position_tolerance_m=float(planning.get("ik_position_tolerance_m", 0.002)),
                rotation_tolerance_deg=float(planning.get("ik_rotation_tolerance_deg", 2.0)),
                null_gain=float(reference.get("cartesian_null_gain", 0.1)),
            )
            pre_duration = float(reference.get("pregrasp_move_duration_s", 2.0))
            grasp_duration = float(reference.get("grasp_move_duration_s", 1.5))
            retreat_duration = float(reference.get("retreat_move_duration_s", 1.5))
            ready_duration = float(reference.get("ready_move_duration_s", 3.0))

            print(f"[Grasp] Open gripper to {opening.motor_angle_rad:.4f} rad")
            hold = robot.move_gripper(
                opening.motor_angle_rad,
                duration_s=float(planning["execution"].get("gripper_open_duration_s", 2.0)),
                rate_hz=rate_hz,
            )
            print("[Grasp] Move seed -> pregrasp")
            hold = _move_to_traj(
                robot, kinematics, hold, selected.T_pregrasp_tcp,
                duration_s=pre_duration, common=cart_common,
            )
            print("[Grasp] Move pregrasp -> grasp")
            robot.set_cleanup_route(
                waypoints=(to_pre[-1], seed, expected_home),
                durations_s=(float(reference.get("retreat_move_duration_s", 3.0)),
                             float(ac.get("return_duration_s", 2.0)),
                             float(ac.get("home_return_duration_s", 4.0))),
                rate_hz=rate_hz, max_joint_step_rad=max_step,
            )
            hold = _move_to_traj(
                robot, kinematics, hold, selected.T_grasp_tcp,
                duration_s=grasp_duration, common=cart_common,
            )
            close_cfg = rc["control"]["gripper_close"]
            print(f"[Grasp] Close until {float(close_cfg['torque_limit_nm']):.2f} Nm")
            close = robot.close_gripper_until_torque(
                torque_limit_nm=float(close_cfg["torque_limit_nm"]),
                kp=float(close_cfg["kp"]), kd=float(close_cfg["kd"]),
                close_rate_rad_s=float(close_cfg["close_rate_rad_s"]),
                minimum_angle_rad=float(close_cfg["minimum_angle_rad"]),
                timeout_s=float(close_cfg["timeout_s"]),
                stable_samples=int(close_cfg.get("stable_samples", 2)), rate_hz=rate_hz,
            )
            print(f"[Grasp] Contact torque={close.measured_torque_nm:.3f} Nm; retreat")
            hold = _move_to_traj(
                robot, kinematics, hold, selected.T_retreat_tcp,
                duration_s=retreat_duration, common=cart_common,
            )
            print("[Grasp] Return ready")
            robot.set_cleanup_route(
                waypoints=(seed, expected_home),
                durations_s=(float(ac.get("return_duration_s", 2.0)),
                             float(ac.get("home_return_duration_s", 4.0))),
                rate_hz=rate_hz, max_joint_step_rad=max_step,
            )
            hold = _move_to_traj(
                robot, kinematics, hold, kinematics.forward(seed),
                duration_s=ready_duration, common=cart_common,
            )
            robot.set_cleanup_route(
                waypoints=(expected_home,),
                durations_s=(float(ac.get("home_return_duration_s", 4.0)),),
                rate_hz=rate_hz, max_joint_step_rad=max_step,
            )
            input("[Exit] Support the object and press Enter to release and home: ")
            print("[Exit] Release gripper and home")
            hold = robot.move_gripper(
                opening.motor_angle_rad,
                duration_s=float(planning["execution"].get("gripper_open_duration_s", 2.0)),
                rate_hz=rate_hz,
            )
            hold = robot.move_gripper(0.0, duration_s=1.5, rate_hz=rate_hz)
            print("[Robot] Move seed -> home")
            hold = robot.move_joints(
                expected_home, duration_s=float(ac.get("home_return_duration_s", 4.0)),
                rate_hz=rate_hz, max_joint_step_rad=max_step,
            )
            robot.clear_cleanup_route()
            quit_requested = True

    if preview is not None:
        preview.close()
    cv2.destroyAllWindows()
    print("Done: cleanup returned home and disabled motors.")


def _move_to_traj(robot, kinematics, state, target, *, duration_s, common):
    """RARS SDK equivalent of RebotArmEndPose.move_to_traj()."""
    points = track_cartesian_trajectory(
        kinematics, np.asarray(state, dtype=np.float64)[:6], target,
        duration_s=duration_s, **common,
    )
    return robot.follow_joint_trajectory(
        points, duration_s=duration_s,
        max_joint_step_rad=float(common["max_joint_step_rad"]),
    )


def _select_executable(candidates, T_camera_base, kinematics, reference_joints,
                       geometry, planning, config):
    pregrasp_distance = float(config.get("pregrasp_distance_m", 0.08))
    retreat_distance = float(config.get("retreat_distance_m", pregrasp_distance))
    insertion_depth = float(config.get("insertion_depth_m", 0.015))
    limit = min(len(candidates), int(config.get("candidate_limit", 30)))
    skipped_width = skipped_height = skipped_ik = 0
    for rank, candidate in enumerate(candidates[:limit], start=1):
        if candidate.width_m > geometry.maximum_width_m:
            skipped_width += 1
            continue
        if bool(config.get("open_to_maximum", True)):
            # The reference driver calls open_gripper() without a requested
            # width, which means its configured maximum opening.
            opening = geometry.opening_at_angle(geometry.maximum_angle_rad)
        else:
            clearance = float(planning.get("gripper_opening_clearance_m", 0.010))
            try:
                opening = geometry.solve_opening(candidate.width_m + clearance)
            except ValueError:
                skipped_width += 1
                continue
        T_raw = T_camera_base @ pose_transform(
            candidate.pose.position_m, candidate.pose.rotation
        )
        T_grasp_base = _canonical_rars_grasp(T_raw, opening.T_grasp_End_link)
        # Same sign and order as reBot transform_grasp_pose_to_base_with_retreat:
        # insert along tool +X first, then compute pregrasp and retreat from it.
        T_grasp_base[:3, 3] += insertion_depth * T_grasp_base[:3, 0]
        T_pregrasp_base = pregrasp_transform(T_grasp_base, pregrasp_distance)
        T_retreat_base = pregrasp_transform(T_grasp_base, retreat_distance)
        T_grasp_tcp = tcp_target_from_grasp(T_grasp_base, opening.T_grasp_End_link)
        T_pregrasp_tcp = tcp_target_from_grasp(T_pregrasp_base, opening.T_grasp_End_link)
        T_retreat_tcp = tcp_target_from_grasp(T_retreat_base, opening.T_grasp_End_link)
        if (T_grasp_base[2, 3] < float(config.get("minimum_grasp_center_z_m", 0.025))
                or min(T_grasp_tcp[2, 3], T_pregrasp_tcp[2, 3])
                < float(config.get("minimum_tcp_z_m", 0.035))):
            skipped_height += 1
            continue
        common = dict(
            joint_margin_rad=float(planning.get("ik_joint_margin_rad", 0.05)),
            random_starts=int(config.get("ik_screen_random_starts", 0)),
            position_tolerance_m=float(planning.get("ik_position_tolerance_m", 0.002)),
            rotation_tolerance_deg=float(planning.get("ik_rotation_tolerance_deg", 2.0)),
        )
        pre = solve_pose_ik(kinematics, T_pregrasp_tcp, reference_joints, **common)
        grasp = solve_pose_ik(
            kinematics, T_grasp_tcp, pre.joints if pre.success else reference_joints, **common
        )
        if not pre.success or not grasp.success:
            skipped_ik += 1
            print(f"[G] Skip rank={rank}: pre={pre.success} grasp={grasp.success} "
                  f"errors={pre.position_error_m * 1000:.1f}/{pre.rotation_error_deg:.1f}, "
                  f"{grasp.position_error_m * 1000:.1f}/{grasp.rotation_error_deg:.1f}")
            continue
        T_grasp_camera = np.linalg.inv(T_camera_base) @ T_grasp_base
        print(f"[G] Skipped width={skipped_width} height={skipped_height} ik={skipped_ik}")
        return _ExecutableGrasp(
            candidate, opening, pre, grasp, T_grasp_camera, rank,
            T_pregrasp_tcp, T_grasp_tcp, T_retreat_tcp,
        )
    print(f"[G] Rejected: width={skipped_width} height={skipped_height} ik={skipped_ik}")
    return None


def _canonical_rars_grasp(T_grasp_base, T_grasp_End_link):
    """Same Rx(pi) canonicalization as reBot, after the RARS axis mapping."""
    flip = np.diag([1.0, -1.0, -1.0])
    alternatives = []
    for branch in (np.eye(3), flip):
        grasp = np.asarray(T_grasp_base, dtype=np.float64).copy()
        grasp[:3, :3] = grasp[:3, :3] @ branch
        end_rotation = grasp[:3, :3] @ T_grasp_End_link[:3, :3].T
        roll = float(Rotation.from_matrix(end_rotation).as_euler("xyz")[0])
        alternatives.append((abs(roll), grasp))
    return min(alternatives, key=lambda item: item[0])[1]


def _status(image, text):
    cv2.putText(image, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_grasp(image, transform, K):
    origin = transform[:3, 3]
    points = [origin] + [origin + 0.045 * transform[:3, i] for i in range(3)]
    if any(point[2] <= 0 for point in points):
        return
    pixels = [(int(K[0, 0] * p[0] / p[2] + K[0, 2]),
               int(K[1, 1] * p[1] / p[2] + K[1, 2])) for p in points]
    for endpoint, color in zip(pixels[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0)), strict=True):
        cv2.arrowedLine(image, pixels[0], endpoint, color, 3, cv2.LINE_AA)


class _Open3DWindow:
    def __init__(self, estimator):
        import open3d as o3d
        self.o3d, self.estimator = o3d, estimator
        self.vis = o3d.visualization.Visualizer()
        if not self.vis.create_window("GraspNet Grasps", width=1280, height=720):
            raise RuntimeError("Open3D visualizer window could not be created")
        self.geometries = []

    def update(self, result, selected, T_grasp_camera):
        for geometry in self.geometries:
            self.vis.remove_geometry(geometry, reset_bounding_box=False)
        cloud = self.o3d.geometry.PointCloud()
        cloud.points = self.o3d.utility.Vector3dVector(result.raw_cloud_xyz_m)
        cloud.colors = self.o3d.utility.Vector3dVector(result.raw_cloud_rgb)
        array = np.concatenate((
            [selected.score, selected.width_m, 0.02, 0.02],
            T_grasp_camera[:3, :3].reshape(-1), T_grasp_camera[:3, 3], [-1.0],
        )).astype(np.float64)[None]
        gripper = self.estimator.GraspGroup(array).to_open3d_geometry_list()[0]
        gripper.paint_uniform_color([1.0, 0.1, 0.1])
        self.geometries = [cloud, gripper]
        for geometry in self.geometries:
            self.vis.add_geometry(geometry, reset_bounding_box=True)
        self.poll()

    def poll(self):
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        return alive

    def close(self):
        self.vis.destroy_window()


if __name__ == "__main__":
    main()
