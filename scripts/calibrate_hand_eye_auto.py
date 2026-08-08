#!/usr/bin/env python3
"""Automatic 50-pose eye-in-hand calibration using the direct RARS SDK."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np

from rars01_graspnet.aruco import ArucoPoseEstimator
from rars01_graspnet.camera import camera_from_config
from rars01_graspnet.config import load_config, resolve_path
from rars01_graspnet.hand_eye import append_sample, load_samples, save, solve
from rars01_graspnet.kinematics import RarsKinematics
from rars01_graspnet.robot import robot_from_config
from rars01_graspnet.trajectory import calibration_joint_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--execute", action="store_true",
                        help="explicitly authorize automatic robot motion")
    parser.add_argument("--yes", action="store_true", help="skip the MOVE confirmation prompt")
    parser.add_argument("--resume", action="store_true", help="append to an existing sample file")
    parser.add_argument("--count", type=int, help="override calibration.auto.count")
    parser.add_argument(
        "--seed-joints", nargs=6, type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="verified calibration-center joint pose in radians",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    rc, cc, safety = config["robot"], config["calibration"], config["safety"]
    ac = cc["auto"]
    seed_values = args.seed_joints if args.seed_joints is not None else ac.get("seed_joints_rad")
    if not args.execute:
        print("Dry safety stop: no hardware was opened and no motion was sent.")
        print("Run with --execute from zero/home and provide a verified --seed-joints pose.")
        return
    if seed_values is None:
        raise RuntimeError(
            "Calibration seed is not configured. Find a real joint pose where ArUco is visible, "
            "then pass --seed-joints J1 J2 J3 J4 J5 J6 or set calibration.auto.seed_joints_rad."
        )
    if not args.yes:
        answer = input(
            "Place RARS01 in zero/home; it will move to the verified seed. Type MOVE: "
        ).strip()
        if answer != "MOVE":
            print("Cancelled before camera, serial or motors were opened.")
            return

    count = int(args.count or ac.get("count", 50))
    sample_path = resolve_path(config, cc["samples"])
    output_path = resolve_path(config, cc["output"])
    if sample_path.exists() and not args.resume:
        backup = _backup_path(sample_path)
        sample_path.replace(backup)
        print(f"Previous samples archived: {backup}")

    kinematics = RarsKinematics(
        resolve_path(config, rc["urdf"]), rc["base_frame"], rc["tcp_frame"]
    )
    detector = ArucoPoseEstimator(
        cc["marker_dictionary"], cc["marker_id"], cc["marker_size_m"]
    )
    captured, skipped = 0, 0

    with camera_from_config(config) as camera:
        camera.warm_up()
        print("Camera ready. Robot will now connect and enable for automatic calibration.")
        with robot_from_config(config) as robot:
            kp, kd = robot.position_gains()
            print("MIT position KP:", kp.tolist())
            print("MIT position KD:", kd.tolist())
            robot.enable()
            start_all = robot.current_joints()
            robot.hold_positions(start_all)
            expected_start = np.asarray(ac["expected_start_joints_rad"], dtype=np.float64)
            start_error = float(np.max(np.abs(start_all[:6] - expected_start)))
            if start_error > float(ac.get("start_tolerance_rad", 0.15)):
                raise RuntimeError(
                    f"Current pose is not the configured zero/home pose; "
                    f"max error={start_error:.3f} rad"
                )
            seed = np.asarray(seed_values, dtype=np.float64)
            sdk_lower, sdk_upper = robot.arm_joint_limits()
            targets = calibration_joint_targets(
                seed, np.maximum(kinematics.lower_limits, sdk_lower),
                np.minimum(kinematics.upper_limits, sdk_upper),
                ac["joint_amplitude_rad"], count, kinematics,
                margin_rad=float(ac.get("joint_limit_margin_rad", 0.05)),
                min_tcp_z_m=float(ac.get("min_tcp_z_m", 0.10)),
                max_tcp_translation_m=float(ac.get("max_tcp_translation_from_seed_m", 0.12)),
            )
            robot.set_cleanup_return(
                home=expected_start, via=seed,
                via_duration_s=float(ac.get("return_duration_s", 2.0)),
                home_duration_s=float(ac.get("home_return_duration_s", 4.0)),
                rate_hz=float(safety.get("trajectory_rate_hz", 50)),
                max_joint_step_rad=float(safety.get("max_joint_step_rad", 0.02)),
            )
            print("Initial home joints:", np.round(start_all[:6], 4).tolist())
            print("Verified calibration seed joints:", np.round(seed, 4).tolist())
            print(f"Generated {len(targets)} local targets; each target returns to seed.")
            print("Moving home -> seed...")
            seed_hold = robot.move_joints(
                seed, duration_s=float(ac.get("seed_move_duration_s", 4.0)),
                rate_hz=float(safety.get("trajectory_rate_hz", 50)),
                max_joint_step_rad=float(safety.get("max_joint_step_rad", 0.02)),
            )
            seed_marker, abort = _wait_stable_marker(
                camera, detector, robot, seed_hold, float(ac.get("settle_s", 0.6)),
                float(ac.get("seed_marker_timeout_s", 5.0)),
                int(ac.get("marker_stable_frames", 4)), 0, len(targets),
            )
            if abort:
                print("Operator requested stop at seed pose")
                return
            if seed_marker is None:
                raise RuntimeError(
                    "ArUco is not stable in the configured seed pose; sweep was not started"
                )
            print("Seed pose and ArUco visibility confirmed; starting automatic sweep.")

            try:
                for index, target in enumerate(targets, 1):
                    print(f"\n[Auto {index}/{len(targets)}] target={np.round(target, 3).tolist()}")
                    hold = robot.move_joints(
                        target, duration_s=float(ac.get("move_duration_s", 2.5)),
                        rate_hz=float(safety.get("trajectory_rate_hz", 50)),
                        max_joint_step_rad=float(safety.get("max_joint_step_rad", 0.02)),
                    )
                    observation, abort = _wait_stable_marker(
                        camera, detector, robot, hold, float(ac.get("settle_s", 0.6)),
                        float(ac.get("marker_timeout_s", 2.5)),
                        int(ac.get("marker_stable_frames", 4)), index, len(targets),
                    )
                    if abort:
                        print("Operator requested stop")
                        break
                    if observation is None:
                        skipped += 1
                        print("[skip] stable ArUco was not visible")
                    else:
                        joints = robot.current_joints()
                        total = append_sample(
                            sample_path, kinematics.forward(joints),
                            observation.T_marker_camera, joints,
                        )
                        captured += 1
                        print(f"[sample {total}] marker_z={observation.T_marker_camera[2, 3]:.3f} m")
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        print("Operator requested stop")
                        break
                    robot.move_joints(
                        seed, duration_s=float(ac.get("return_duration_s", 2.0)),
                        rate_hz=float(safety.get("trajectory_rate_hz", 50)),
                        max_joint_step_rad=float(safety.get("max_joint_step_rad", 0.02)),
                    )
            finally:
                print("Sweep ending; cleanup will return via seed to home.")
        print("Robot disconnected; cleanup requested motor disable.")
    cv2.destroyAllWindows()

    samples = load_samples(sample_path)
    minimum = int(cc.get("minimum_samples", 10))
    print(f"Automatic collection finished: captured={captured}, skipped={skipped}, total={len(samples)}")
    if len(samples) < minimum:
        print(f"Test sweep completed; calibration was not solved ({len(samples)} < {minimum}).")
        return
    result = solve(list(samples.T_tcp_base), list(samples.T_marker_camera), cc["method"])
    save(result, output_path, len(samples), cc["method"])
    print("Saved:", output_path)
    print("T_camera_End_link (camera -> End_link):\n", result.T_camera_tcp)
    print(f"method={result.method} residual={result.translation_rms_m * 1000:.1f} mm, "
          f"{result.rotation_rms_deg:.2f} deg")


def _wait_stable_marker(camera, detector, robot, hold, settle_s: float, timeout_s: float,
                        stable_needed: int, index: int, total: int):
    settle_until = time.monotonic() + settle_s
    deadline = settle_until + timeout_s
    stable, latest, latest_frame = 0, None, None
    while time.monotonic() < deadline:
        frame = camera.read()
        if frame is None:
            continue
        latest_frame = frame
        current = detector.detect(frame.color_bgr, camera.K, camera.D)
        display = detector.draw(frame.color_bgr, current, camera.K, camera.D)
        phase = "SEED" if index == 0 else f"AUTO {index}/{total}"
        text = f"{phase} ArUco stable {stable}/{stable_needed}"
        cv2.putText(display, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 220, 0), 2)
        cv2.imshow("RARS01 automatic hand-eye", display)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            return None, True
        robot.hold_positions(hold)
        if time.monotonic() < settle_until:
            stable = 0
            continue
        if current is None:
            stable, latest = 0, None
        else:
            stable += 1
            latest = current
            if stable >= stable_needed:
                return latest, False
    return None, False


def _backup_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.stem}.{stamp}.bak{path.suffix}")


if __name__ == "__main__":
    main()
