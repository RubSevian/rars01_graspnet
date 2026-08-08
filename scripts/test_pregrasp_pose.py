#!/usr/bin/env python3
"""Move RARS01 only to a previously computed and MoveIt-checked pre-grasp pose."""
from __future__ import annotations

import argparse

import numpy as np

from rars01_graspnet.config import load_config, resolve_path
from rars01_graspnet.kinematics import RarsKinematics
from rars01_graspnet.robot import robot_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--pregrasp-joints", nargs=6, type=float, required=True,
                        metavar=("J1", "J2", "J3", "J4", "J5", "J6"))
    parser.add_argument("--gripper-angle", type=float, required=True,
                        help="pre-open motor-7 angle in radians")
    parser.add_argument("--confirmed-in-moveit", action="store_true",
                        help="confirm this exact joint pose was collision-checked in MoveIt")
    parser.add_argument("--execute", action="store_true",
                        help="explicitly authorize home -> seed -> pre-grasp motion")
    args = parser.parse_args()

    config = load_config(args.config)
    rc, ac = config["robot"], config["calibration"]["auto"]
    safety, execution = config["safety"], config["planning"]["execution"]
    kinematics = RarsKinematics(
        resolve_path(config, rc["urdf"]), rc["base_frame"], rc["tcp_frame"]
    )
    target = np.asarray(args.pregrasp_joints, dtype=np.float64)
    margin = float(config["planning"].get("ik_joint_margin_rad", 0.05))
    if np.any(target < kinematics.lower_limits + margin) or np.any(
        target > kinematics.upper_limits - margin
    ):
        raise RuntimeError("Pre-grasp target is outside URDF limits with safety margin")
    T_target = kinematics.forward(target)
    minimum_z = float(execution.get("minimum_pregrasp_tcp_z_m", 0.060))
    if T_target[2, 3] < minimum_z:
        raise RuntimeError(
            f"Pre-grasp End_link z={T_target[2, 3]:.4f} m is below configured "
            f"minimum {minimum_z:.4f} m"
        )
    print("Validated pre-grasp q [deg]:", np.round(np.rad2deg(target), 2).tolist())
    print("Pre-grasp End_link xyz [m]:", np.round(T_target[:3, 3], 5).tolist())
    print(f"Pre-open gripper angle: {args.gripper_angle:.4f} rad / "
          f"{np.rad2deg(args.gripper_angle):.2f} deg")
    if not args.execute:
        print("Dry-run only: camera, serial and motors were not opened.")
        return
    if not args.confirmed_in_moveit:
        raise RuntimeError(
            "Execution requires --confirmed-in-moveit after checking this exact pose and path"
        )
    answer = input(
        "Place RARS01 in zero/home, clear the path, and type PREGRASP: "
    ).strip()
    if answer != "PREGRASP":
        print("Cancelled before serial or motors were opened.")
        return

    expected_home = np.asarray(ac["expected_start_joints_rad"], dtype=np.float64)
    seed_values = ac.get("seed_joints_rad")
    if seed_values is None:
        raise RuntimeError("calibration.auto.seed_joints_rad is not configured")
    seed = np.asarray(seed_values, dtype=np.float64)
    rate_hz = float(safety.get("trajectory_rate_hz", 50.0))
    max_step = float(safety.get("max_joint_step_rad", 0.02))

    with robot_from_config(config) as robot:
        robot.enable()
        start = robot.current_joints()
        start_error = float(np.max(np.abs(start[:6] - expected_home)))
        tolerance = float(ac.get("start_tolerance_rad", 0.15))
        if start_error > tolerance:
            raise RuntimeError(
                f"Current pose is not zero/home; max error={start_error:.3f} rad"
            )
        robot.set_cleanup_return(
            home=expected_home, via=seed,
            via_duration_s=float(ac.get("return_duration_s", 2.0)),
            home_duration_s=float(ac.get("home_return_duration_s", 4.0)),
            rate_hz=rate_hz, max_joint_step_rad=max_step,
        )
        print("Moving home -> seed...")
        robot.move_joints(
            seed, duration_s=float(ac.get("seed_move_duration_s", 4.0)),
            rate_hz=rate_hz, max_joint_step_rad=max_step,
        )
        print("Opening gripper to the planned pre-open angle...")
        robot.move_gripper(
            args.gripper_angle,
            duration_s=float(execution.get("gripper_open_duration_s", 2.0)),
            rate_hz=rate_hz,
        )
        print("Moving seed -> pre-grasp. Keep the E-stop ready...")
        reached = robot.move_joints(
            target,
            duration_s=float(execution.get("pregrasp_move_duration_s", 6.0)),
            rate_hz=rate_hz, max_joint_step_rad=max_step,
        )
        print("PRE-GRASP REACHED. Inspect clearance; no approach or closing will occur.")
        with robot.continuous_hold(reached, rate_hz=rate_hz) as check_hold:
            response = input("Type RETURN to go back through seed to home: ").strip()
            check_hold()
            if response != "RETURN":
                print("Unrecognized input; safe cleanup return will still run.")
    print("Cleanup requested pre-grasp -> seed -> home -> disable.")


if __name__ == "__main__":
    main()
