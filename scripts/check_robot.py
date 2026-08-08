#!/usr/bin/env python3
"""RARS01 connection test and explicitly armed feedback/FK diagnostic."""
from __future__ import annotations

import argparse
import time

import numpy as np

from rars01_graspnet.config import load_config, resolve_path
from rars01_graspnet.kinematics import RarsKinematics
from rars01_graspnet.robot import robot_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="RARS01 connection or enabled-feedback monitor")
    parser.add_argument("--config")
    parser.add_argument("--rate", type=float, default=2.0, help="print rate in Hz")
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--enable-feedback", action="store_true",
        help="ENABLE ALL MOTORS to receive feedback; robot must be secured and E-stop ready",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    rc = config["robot"]
    fk = RarsKinematics(resolve_path(config, rc["urdf"]), rc["base_frame"], rc["tcp_frame"])
    print("URDF joints:", ", ".join(fk.joint_names))
    print("URDF lower:", np.round(fk.lower_limits, 4))
    print("URDF upper:", np.round(fk.upper_limits, 4))

    with robot_from_config(config) as robot:
        if not args.enable_feedback:
            status = robot.arm.communication_status()
            print("Serial receiver opened:", status.connected)
            print("No motor feedback is expected while motors are disabled.")
            print("For an armed feedback test use --enable-feedback after securing the robot.")
            return
        print("WARNING: enabling all seven motors. Keep the workspace clear and E-stop ready.")
        robot.enable()
        print("Motors enabled; reading feedback. Ctrl+C/Quit will disable them in finally cleanup.")
        while True:
            state = robot.read_state()
            q = state.position[:6]
            inside = (q >= fk.lower_limits) & (q <= fk.upper_limits)
            T = fk.forward(q)
            status = robot.arm.communication_status()
            print("\nq [rad]:", np.round(q, 5))
            print("valid:  ", state.valid.astype(int), "error:", state.error)
            print("limits: ", inside.astype(int))
            print("TCP xyz [m]:", np.round(T[:3, 3], 5))
            print("temperature MOS/rotor:", np.round(state.mos_temperature, 1),
                  np.round(state.rotor_temperature, 1))
            print("serial frames valid/invalid/timeouts:", status.valid_frames,
                  status.invalid_frames, status.read_timeouts)
            if not np.all(inside):
                bad = [fk.joint_names[i] for i in np.flatnonzero(~inside)]
                print("WARNING: joints outside URDF limits:", bad)
            if np.any(state.error):
                print("WARNING: motor fault/status is non-zero")
            if args.once:
                break
            time.sleep(max(0.01, 1.0 / args.rate))


if __name__ == "__main__":
    main()
