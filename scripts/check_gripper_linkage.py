#!/usr/bin/env python3
"""Compare commanded linkage widths with measurements on the real gripper."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drivers.robot.grasp_driver import GraspDriver, RarsArmAdapter  # noqa: E402
from rars01_graspnet.config import load_config  # noqa: E402
from rars01_graspnet.gripper_geometry import RarsGripperGeometry  # noqa: E402


def geometry_from_config(config: dict) -> RarsGripperGeometry:
    robot = config["robot"]
    hardware = robot["rars01"]
    gripper = robot["gripper"]["rars01"]
    return RarsGripperGeometry(
        linkage_radius_m=gripper["linkage_radius_m"],
        connecting_rod_length_m=gripper["connecting_rod_length_m"],
        carriage_width_m=gripper["carriage_width_m"],
        maximum_width_m=hardware["max_grasp_width_m"],
        jaw_center_End_link_m=gripper["jaw_center_End_link_m"],
        jaw_depth_m=gripper["jaw_depth_m"],
        jaw_height_m=gripper["jaw_height_m"],
        R_grasp_End_link=gripper["R_grasp_End_link"],
        motor_angle_limit_rad=gripper["angle_open"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RARS01 gripper linkage calibration")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "jetson_orin_nano.yaml"))
    parser.add_argument("--widths-mm", type=float, nargs="+", default=[20, 40, 60, 80, 100])
    parser.add_argument(
        "--execute", action="store_true",
        help="enable the real robot and command the listed widths",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    geometry = geometry_from_config(config)
    widths_m = [value / 1000.0 for value in args.widths_mm]

    for width in widths_m:
        opening = geometry.solve_opening(width)
        print(f"{width * 1000:6.1f} mm -> motor {opening.motor_angle_rad:.6f} rad")
    if not args.execute:
        return 0

    if input(
        "Вручную полностью закройте губки, поставьте руку в home и введите START: "
    ).strip() != "START":
        return 1

    robot_cfg = config["robot"]
    arm = RarsArmAdapter(robot_cfg, PROJECT_ROOT)
    try:
        from rars01_graspnet.pose_controller import RarsPoseController

        controller = RarsPoseController(
            arm, dt=1.0 / arm.rate, arm_control_mode="posvel"
        )
        driver = GraspDriver(
            arm, controller, gripper_config=robot_cfg.get("gripper"),
        )
        # Keep motor 7 passive while startup feedback settles.  Never infer
        # the closed zero from transient feedback: it is an explicit config.
        driver.start(passive_gripper=True)
        time.sleep(0.5)
        configured_closed = float(robot_cfg["gripper"]["rars01"]["closed_position_rad"])
        print(f"Закрытая позиция из YAML: {configured_closed:+.6f} рад")
        for index, width in enumerate(widths_m):
            if index > 0:
                input(f"Enter: открыть на {width * 1000:.1f} мм...")
            before, _, _ = driver.get_gripper_state()
            target = driver.motor_position_for_width(width)
            print(f"Команда мотору 7: {before:+.6f} -> {target:+.6f} рад")
            driver.open_gripper(width, timeout=4.0)
            motor_pos, _, _ = driver.get_gripper_state()
            moved = motor_pos - before
            print(f"Feedback мотора 7: {motor_pos:+.6f} рад, движение {moved:+.6f} рад")
            if abs(moved) < 0.005:
                raise RuntimeError(
                    "Мотор 7 не сдвинулся. Если он упёрся в закрытие, измените "
                    "robot.gripper.rars01.counterclockwise и повторите тест."
                )
            measured = input("Фактическое расстояние между внутренними краями, мм: ").strip()
            if measured:
                error = float(measured.replace(",", ".")) - width * 1000.0
                print(f"ошибка {error:+.2f} мм; feedback мотора {motor_pos:+.6f} рад")
    finally:
        arm.disconnect()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nПроверка остановлена, моторы отключены.")
        raise SystemExit(130)
