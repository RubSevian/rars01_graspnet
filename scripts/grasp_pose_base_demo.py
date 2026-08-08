#!/usr/bin/env python3
"""YOLO + GraspNet pose in base_link without executing a grasp trajectory."""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from rars01_graspnet.camera import camera_from_config
from rars01_graspnet.config import load_config, resolve_path
from rars01_graspnet.detection import detector_from_config, draw_detections
from rars01_graspnet.graspnet import estimator_from_config, select_target
from rars01_graspnet.gripper_geometry import RarsGripperGeometry
from rars01_graspnet.hand_eye import grasp_to_base, load
from rars01_graspnet.ik import pregrasp_transform, solve_pose_ik, tcp_target_from_grasp
from rars01_graspnet.kinematics import RarsKinematics
from rars01_graspnet.robot import robot_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--target", default=None,
                        help="object class, for example banana; defaults to config")
    parser.add_argument("--enable-feedback", action="store_true",
                        help="enable and hold motors so real joint feedback is available")
    parser.add_argument("--move-to-seed", action="store_true",
                        help="move from configured zero/home to the verified calibration seed")
    parser.add_argument("--execute", action="store_true",
                        help="explicitly authorize --move-to-seed")
    args = parser.parse_args()
    if not args.enable_feedback:
        raise RuntimeError(
            "This preview needs real joint feedback. Secure the already-positioned arm and "
            "pass --enable-feedback. The script holds the current pose and never moves it."
        )
    if args.move_to_seed and not args.execute:
        raise RuntimeError("--move-to-seed requires explicit --execute")
    if args.move_to_seed:
        answer = input(
            "Place RARS01 in zero/home; preview will move to the verified seed. Type MOVE: "
        ).strip()
        if answer != "MOVE":
            print("Cancelled before camera, serial or motors were opened.")
            return

    config = load_config(args.config)
    rc, gc = config["robot"], config["graspnet"]
    ac, safety = config["calibration"]["auto"], config["safety"]
    planning = config.get("planning", {})
    target_class = args.target if args.target is not None else gc.get("target_class")
    detector = detector_from_config(config, resolve_path)
    estimator = estimator_from_config(config, resolve_path)
    kinematics = RarsKinematics(
        resolve_path(config, rc["urdf"]), rc["base_frame"], rc["tcp_frame"]
    )
    T_camera_tcp = load(resolve_path(config, config["calibration"]["output"]))
    infer_every = max(1, int(config["yolo"].get("inference_every_n_frames", 3)))
    hold_rate = float(safety.get("trajectory_rate_hz", 50.0))
    detections, frame_count = [], 0
    frozen_frame = None
    best_camera_xyz = None
    status = "G/Space: calculate grasp | R: resume | Q/Esc: quit"

    if args.move_to_seed:
        print("SEED preview: the only motion is zero/home -> verified seed -> zero/home.")
    else:
        print("NO-MOTION preview: motors only hold their measured starting position.")
    print(f"Target: {target_class or 'highest-confidence detection'}")
    with camera_from_config(config) as camera, robot_from_config(config) as robot:
        robot.enable()
        hold = robot.current_joints()
        if args.move_to_seed:
            expected_home = np.asarray(ac["expected_start_joints_rad"], dtype=np.float64)
            error = float(np.max(np.abs(hold[:6] - expected_home)))
            tolerance = float(ac.get("start_tolerance_rad", 0.15))
            if error > tolerance:
                raise RuntimeError(
                    f"Current pose is not zero/home; max error={error:.3f} rad "
                    f"(allowed {tolerance:.3f})"
                )
            seed_values = ac.get("seed_joints_rad")
            if seed_values is None:
                raise RuntimeError("calibration.auto.seed_joints_rad is not configured")
            seed = np.asarray(seed_values, dtype=np.float64)
            robot.set_cleanup_return(
                home=expected_home, via=seed,
                via_duration_s=float(ac.get("return_duration_s", 2.0)),
                home_duration_s=float(ac.get("home_return_duration_s", 4.0)),
                rate_hz=hold_rate,
                max_joint_step_rad=float(safety.get("max_joint_step_rad", 0.02)),
            )
            print("Moving zero/home -> verified seed...")
            hold = robot.move_joints(
                seed, duration_s=float(ac.get("seed_move_duration_s", 4.0)),
                rate_hz=hold_rate,
                max_joint_step_rad=float(safety.get("max_joint_step_rad", 0.02)),
            )
        print("Holding q [rad]:", np.round(hold[:6], 5).tolist())
        with robot.continuous_hold(hold, rate_hz=hold_rate) as check_hold:
            camera.warm_up()
            while True:
                check_hold()
                frame = camera.read()
                if frame is None:
                    continue
                frame_count += 1
                if frozen_frame is None and (frame_count == 1 or frame_count % infer_every == 0):
                    detections = detector.detect(frame)
                shown_frame = frozen_frame or frame
                output = draw_detections(shown_frame.color_bgr, detections)
                target = select_target(detections, target_class)
                if target is not None:
                    x1, y1, x2, y2 = target.bbox_xyxy
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 80, 255), 3)
                if best_camera_xyz is not None:
                    _draw_candidate(output, best_camera_xyz, shown_frame.intrinsics.K)
                _draw_text(output, status)
                cv2.imshow("RARS01 grasp pose in base_link (NO MOTION)", output)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("r"), ord("R")):
                    frozen_frame = None
                    best_camera_xyz = None
                    status = "live preview"
                    continue
                if key not in (ord("g"), ord("G"), ord(" ")):
                    continue
                if target_class and target is None:
                    status = f"target '{target_class}' not found"
                    print(status)
                    continue

                # The arm is stationary, but use feedback adjacent to the frozen RGB-D frame.
                joints = robot.current_joints()
                T_tcp_base = kinematics.forward(joints)
                frozen_frame = frame
                started = time.perf_counter()
                try:
                    result = estimator.infer(frame, detections, target_class)
                except Exception as exc:
                    frozen_frame = None
                    status = f"GraspNet failed: {exc}"
                    print(status)
                    continue
                check_hold()
                elapsed = time.perf_counter() - started
                status = (f"grasps={len(result.candidates)} decoded={result.decoded_count} "
                          f"collision={result.collision_removed} time={elapsed:.2f}s")
                print(status)
                if not result.candidates:
                    best_camera_xyz = None
                    continue

                best = result.candidates[0]
                best_camera_xyz = best.pose.position_m
                T_grasp_base = grasp_to_base(
                    T_tcp_base, T_camera_tcp, best.pose.position_m, best.pose.rotation
                )
                print(f"best score={best.score:.4f}, required width={best.width_m:.4f} m")
                print("T_grasp_camera =")
                print(np.array2string(_pose_matrix(best.pose.position_m, best.pose.rotation),
                                      precision=6, suppress_small=True))
                print("T_grasp_base =")
                print(np.array2string(T_grasp_base, precision=6, suppress_small=True))
                xyz = T_grasp_base[:3, 3]
                print(f"grasp in base_link: x={xyz[0]:+.4f}, y={xyz[1]:+.4f}, z={xyz[2]:+.4f} m")
                _print_dry_ik(
                    kinematics, T_grasp_base, joints[:6], best.score, best.width_m,
                    gc.get("max_grasp_width_m"), planning, safety,
                )
    cv2.destroyAllWindows()
    if args.move_to_seed:
        print("Cleanup requested return through seed to zero/home; robot was disabled.")
    else:
        print("Robot disabled. No changed position target was sent.")


def _pose_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def _draw_text(image: np.ndarray, value: str) -> None:
    cv2.putText(image, value, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, value, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_candidate(image: np.ndarray, xyz: np.ndarray, K: np.ndarray) -> None:
    x, y, z = xyz
    if z <= 0:
        return
    pixel = (int(round(K[0, 0] * x / z + K[0, 2])),
             int(round(K[1, 1] * y / z + K[1, 2])))
    cv2.drawMarker(image, pixel, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)


def _print_dry_ik(kinematics, T_grasp_base: np.ndarray, current_joints: np.ndarray,
                  score: float, width_m: float, max_width_m, planning: dict,
                  safety: dict) -> None:
    geometry = RarsGripperGeometry.from_config(planning["gripper_geometry"])
    clearance = float(planning.get("gripper_opening_clearance_m", 0.010))
    try:
        grasp_opening = geometry.solve_opening(width_m)
        pregrasp_opening = geometry.solve_opening(width_m + clearance)
    except ValueError as exc:
        print("\nOFFLINE IK DRY-RUN (never sent to motors)")
        print("EXECUTION BLOCKED:")
        print(" -", exc)
        return
    pregrasp_distance = float(planning.get("pregrasp_distance_m", 0.08))
    common = {
        "joint_margin_rad": float(planning.get("ik_joint_margin_rad", 0.05)),
        "random_starts": int(planning.get("ik_random_starts", 32)),
        "position_tolerance_m": float(planning.get("ik_position_tolerance_m", 0.002)),
        "rotation_tolerance_deg": float(planning.get("ik_rotation_tolerance_deg", 2.0)),
    }
    T_pregrasp_base = pregrasp_transform(T_grasp_base, pregrasp_distance)
    T_pregrasp_tcp_base = tcp_target_from_grasp(
        T_pregrasp_base, pregrasp_opening.T_grasp_End_link
    )
    T_grasp_tcp_base = tcp_target_from_grasp(
        T_grasp_base, grasp_opening.T_grasp_End_link
    )
    pre = solve_pose_ik(kinematics, T_pregrasp_tcp_base, current_joints, **common)
    grasp_reference = pre.joints if pre.success else current_joints
    grasp = solve_pose_ik(kinematics, T_grasp_tcp_base, grasp_reference, **common)

    print("\nOFFLINE IK DRY-RUN (never sent to motors)")
    print(f"gripper grasp: width={grasp_opening.actual_width_m:.4f} m, "
          f"motor={grasp_opening.motor_angle_rad:.4f} rad / "
          f"{np.rad2deg(grasp_opening.motor_angle_rad):.2f} deg")
    print(f"gripper pre-open: width={pregrasp_opening.actual_width_m:.4f} m, "
          f"motor={pregrasp_opening.motor_angle_rad:.4f} rad / "
          f"{np.rad2deg(pregrasp_opening.motor_angle_rad):.2f} deg")
    print("grasp center in End_link [m]:",
          np.round(grasp_opening.center_End_link_m, 5).tolist())
    print("pre-grasp xyz:", np.round(T_pregrasp_tcp_base[:3, 3], 5).tolist())
    _print_ik_solution("pre-grasp", pre)
    _print_ik_solution("grasp", grasp)
    if pre.success and grasp.success:
        delta_deg = np.rad2deg(grasp.joints - pre.joints)
        print("pre-grasp -> grasp delta [deg]:", np.round(delta_deg, 2).tolist())

    blockers = []
    minimum_score = float(planning.get("minimum_grasp_score", 0.50))
    if score < minimum_score:
        blockers.append(f"score {score:.3f} < {minimum_score:.3f}")
    if max_width_m is None:
        blockers.append("max_grasp_width_m is not measured")
    elif width_m > float(max_width_m):
        blockers.append(f"required width {width_m:.4f} m > maximum {float(max_width_m):.4f} m")
    if not bool(planning.get("tool_offset_verified", False)):
        blockers.append("URDF jaw-center geometry is not yet physically verified")
    if not pre.success or not grasp.success:
        blockers.append("6D IK did not meet tolerance")
    if not bool(safety.get("allow_motion", False)):
        blockers.append("safety.allow_motion=false")
    print("EXECUTION BLOCKED:")
    for reason in blockers or ["no blocker reported (execution code is still absent)"]:
        print(" -", reason)


def _print_ik_solution(name: str, solution) -> None:
    state = "OK" if solution.success else "FAILED"
    print(f"{name} IK: {state}, solutions={solution.successful_starts}, "
          f"position_error={solution.position_error_m * 1000:.3f} mm, "
          f"rotation_error={solution.rotation_error_deg:.3f} deg")
    print(f"{name} q [rad]:", np.round(solution.joints, 5).tolist())
    print(f"{name} q [deg]:", np.round(np.rad2deg(solution.joints), 2).tolist())


if __name__ == "__main__":
    main()
