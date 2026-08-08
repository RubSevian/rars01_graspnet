#!/usr/bin/env python3
"""Interactive eye-in-hand calibration without modifying the URDF."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from rars01_graspnet.aruco import ArucoPoseEstimator
from rars01_graspnet.camera import camera_from_config
from rars01_graspnet.config import load_config, resolve_path
from rars01_graspnet.hand_eye import append_sample, load_samples, save, solve
from rars01_graspnet.kinematics import RarsKinematics
from rars01_graspnet.robot import robot_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="RARS01 eye-in-hand calibration by ArUco")
    parser.add_argument("--config")
    parser.add_argument("--joints", nargs=6, type=float, metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                        help="offline/debug joint angles in radians instead of hardware feedback")
    parser.add_argument(
        "--enable-feedback", action="store_true",
        help="enable all motors so hardware joint feedback is available",
    )
    parser.add_argument(
        "--capture-once", action="store_true",
        help="append one hardware sample and exit, disabling motors in cleanup",
    )
    parser.add_argument(
        "--solve-only", action="store_true",
        help="solve saved samples without opening the camera or robot",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    robot_config = config["robot"]
    calibration_config = config["calibration"]
    kinematics = RarsKinematics(
        resolve_path(config, robot_config["urdf"]),
        robot_config["base_frame"], robot_config["tcp_frame"],
    )
    detector = ArucoPoseEstimator(
        calibration_config["marker_dictionary"], calibration_config["marker_id"],
        calibration_config["marker_size_m"],
    )
    sample_path = resolve_path(config, calibration_config["samples"])
    if args.solve_only:
        _solve_saved(sample_path, config, calibration_config)
        return
    if args.capture_once and args.joints:
        raise RuntimeError("--capture-once requires real hardware feedback, not --joints")
    if args.capture_once and not args.enable_feedback:
        raise RuntimeError("--capture-once requires explicit --enable-feedback")
    tcp_base_samples: list[np.ndarray] = []
    marker_camera_samples: list[np.ndarray] = []

    robot_context = _FixedJoints(args.joints) if args.joints else robot_from_config(config)
    with camera_from_config(config) as camera:
        camera.warm_up()
        with robot_context as robot:
            if not args.joints:
                if not args.enable_feedback:
                    raise RuntimeError(
                        "RARS01 provides no feedback while disabled. Hardware calibration requires "
                        "--enable-feedback and controlled robot motion; do not move enabled joints by hand."
                    )
                robot.enable()
            saved_count = len(load_samples(sample_path)) if args.capture_once else 0
            print("SPACE: capture pair; S: solve/save; Q/Esc: exit")
            if args.capture_once:
                print(f"One-shot mode: {saved_count} samples already saved. SPACE appends one and exits.")
            print("Use only a tested controller to change pose; never move enabled joints by hand.")
            while True:
                frame = camera.read()
                if frame is None:
                    continue
                observation = detector.detect(frame.color_bgr, camera.K, camera.D)
                display = detector.draw(frame.color_bgr, observation, camera.K, camera.D)
                count = saved_count if args.capture_once else len(tcp_base_samples)
                cv2.putText(display, f"samples: {count}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
                cv2.imshow("RARS01 hand-eye", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key == ord(" "):
                    if observation is None:
                        print("[skip] target ArUco is not visible")
                        continue
                    joints = robot.current_joints()
                    T_tcp_base = kinematics.forward(joints)
                    if args.capture_once:
                        saved_count = append_sample(
                            sample_path, T_tcp_base, observation.T_marker_camera, joints
                        )
                    else:
                        tcp_base_samples.append(T_tcp_base)
                        marker_camera_samples.append(observation.T_marker_camera.copy())
                    print(f"[sample {saved_count if args.capture_once else len(tcp_base_samples)}] "
                          f"q={np.round(joints[:6], 4).tolist()} "
                          f"marker_z={observation.T_marker_camera[2, 3]:.3f} m")
                    if args.capture_once:
                        print("Sample saved; exiting now so robot cleanup disables all motors.")
                        break
                if key in (ord("s"), ord("S")):
                    try:
                        _solve_arrays(tcp_base_samples, marker_camera_samples, config, calibration_config)
                    except RuntimeError as exc:
                        print(f"[wait] {exc}")
        if not args.joints:
            print("Robot disconnected; cleanup requested motor disable.")
    cv2.destroyAllWindows()


def _solve_saved(sample_path: Path, config: dict, calibration_config: dict) -> None:
    samples = load_samples(sample_path)
    print(f"Loaded {len(samples)} samples from {sample_path}")
    _solve_arrays(list(samples.T_tcp_base), list(samples.T_marker_camera), config, calibration_config)


def _solve_arrays(tcp_samples: list[np.ndarray], marker_samples: list[np.ndarray],
                  config: dict, calibration_config: dict) -> None:
    minimum = int(calibration_config.get("minimum_samples", 10))
    if len(tcp_samples) < minimum:
        raise RuntimeError(f"Need at least {minimum} samples, got {len(tcp_samples)}")
    result = solve(tcp_samples, marker_samples, calibration_config["method"])
    output = resolve_path(config, calibration_config["output"])
    save(result, output, len(tcp_samples), calibration_config["method"])
    print("Saved:", output)
    print("T_camera_End_link (camera -> End_link):\n", result.T_camera_tcp)
    print(f"method: {result.method}")
    print(f"residual: {result.translation_rms_m * 1000:.1f} mm, "
          f"{result.rotation_rms_deg:.2f} deg")


class _FixedJoints:
    """Only for checking FK/UI without connecting hardware."""
    def __init__(self, joints):
        self.joints = np.asarray(joints, dtype=np.float64)

    def current_joints(self):
        return self.joints

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None


if __name__ == "__main__":
    main()
