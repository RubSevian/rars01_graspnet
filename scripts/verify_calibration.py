#!/usr/bin/env python3
"""Show the ArUco pose in base coordinates using the saved hand-eye matrix."""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from rars01_graspnet.aruco import ArucoPoseEstimator
from rars01_graspnet.camera import camera_from_config
from rars01_graspnet.config import load_config, resolve_path
from rars01_graspnet.hand_eye import camera_to_base, load
from rars01_graspnet.kinematics import RarsKinematics
from rars01_graspnet.robot import robot_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify RARS01 hand-eye calibration")
    parser.add_argument("--config")
    parser.add_argument("--enable-feedback", action="store_true",
                        help="enable all motors; required by current RARS01 feedback protocol")
    args = parser.parse_args()
    config = load_config(args.config)
    rc, cc = config["robot"], config["calibration"]
    fk = RarsKinematics(resolve_path(config, rc["urdf"]), rc["base_frame"], rc["tcp_frame"])
    hand_eye = load(resolve_path(config, cc["output"]))
    detector = ArucoPoseEstimator(cc["marker_dictionary"], cc["marker_id"], cc["marker_size_m"])

    if not args.enable_feedback:
        raise RuntimeError("Verification needs joint feedback; pass --enable-feedback after securing the robot")
    with camera_from_config(config) as camera, robot_from_config(config) as robot:
        robot.enable()
        hold = robot.current_joints()
        robot.hold_positions(hold)
        camera.warm_up()
        print("Motors are enabled. Do not move joints by hand. Q/Esc disables and exits.")
        while True:
            frame = camera.read()
            if frame is None:
                continue
            observation = detector.detect(frame.color_bgr, camera.K, camera.D)
            display = detector.draw(frame.color_bgr, observation, camera.K, camera.D)
            if observation is not None:
                T_tcp_base = fk.forward(robot.current_joints())
                T_marker_base = camera_to_base(T_tcp_base, hand_eye) @ observation.T_marker_camera
                xyz = T_marker_base[:3, 3]
                text = f"marker in base: {xyz[0]:+.3f} {xyz[1]:+.3f} {xyz[2]:+.3f} m"
                cv2.putText(display, text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 220, 0), 2)
                print("\r" + text, end="", flush=True)
            cv2.imshow("Calibration verification", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
