#!/usr/bin/env python3
"""Read current joints and print base/End_link/camera transforms."""
from __future__ import annotations

import argparse
import numpy as np

from rars01_graspnet.config import load_config, resolve_path
from rars01_graspnet.hand_eye import camera_to_base, load
from rars01_graspnet.kinematics import RarsKinematics
from rars01_graspnet.robot import robot_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--enable-feedback", action="store_true",
                        help="enable all motors; required by current RARS01 feedback protocol")
    args = parser.parse_args()
    config = load_config(args.config)
    if not args.enable_feedback:
        raise RuntimeError("Current joint feedback requires explicit --enable-feedback")
    rc, cc = config["robot"], config["calibration"]
    fk = RarsKinematics(resolve_path(config, rc["urdf"]), rc["base_frame"], rc["tcp_frame"])
    T_camera_tcp = load(resolve_path(config, cc["output"]))
    with robot_from_config(config) as robot:
        robot.enable()
        q = robot.current_joints()
    T_tcp_base = fk.forward(q)
    print("q arm [rad]:", np.round(q[:6], 6))
    print("T_End_link_base:\n", T_tcp_base)
    print("T_camera_base:\n", camera_to_base(T_tcp_base, T_camera_tcp))


if __name__ == "__main__":
    main()
