"""Automatic eye-in-hand calibration: RARS01, RGB-D camera and ArUco.

Traverse the validated baseline set of 50 Cartesian poses using POS/VEL.
Save samples and the camera-to-End_link transform under config/calibration.
Manual gravity-guided motion requires a separate RARS01 implementation.
"""

import os
import sys
import threading
import argparse
import queue
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

from drivers.camera import make_camera
from drivers.robot.grasp_driver import GraspDriver, RarsArmAdapter
from rars01_graspnet.pose_controller import RarsPoseController
from calibration.hand_eye import CalibMode, HandEyeCalibrator
from utils.camera_utils import load_config
from utils.transforms import rotation_matrix_to_euler_zyx


# ==========================================
# Preset calibration poses in Cartesian space, using meters and radians.
# (x, y, z, roll, pitch, yaw)
# pitch > 0 points the end effector downward toward the ArUco marker.
# ==========================================
CALIB_POSES_XYZ = [
    # Center region, large pitch, yaw sweep.
    (0.28, -0.16, 0.26, -0.30, 0.80, -0.90),
    (0.28, -0.08, 0.26,  0.30, 0.80, -0.45),
    (0.28,  0.00, 0.26, -0.30, 0.80,  0.00),
    (0.28,  0.08, 0.26,  0.30, 0.80,  0.45),
    (0.28,  0.16, 0.26, -0.30, 0.80,  0.90),
    # Center region, medium pitch.
    (0.27, -0.16, 0.31,  0.30, 0.55, -0.90),
    (0.27, -0.08, 0.31, -0.30, 0.55, -0.45),
    (0.27,  0.00, 0.31,  0.30, 0.55,  0.00),
    (0.27,  0.08, 0.31, -0.30, 0.55,  0.45),
    (0.27,  0.16, 0.31,  0.30, 0.55,  0.90),
    # Center region, small pitch.
    (0.26, -0.14, 0.34, -0.40, 0.35, -0.80),
    (0.26,  0.00, 0.34,  0.40, 0.35,  0.00),
    (0.26,  0.14, 0.34, -0.40, 0.35,  0.80),
    # Forward region, yaw +/- 1 rad.
    (0.37,  0.00, 0.27,  0.00, 0.65,  0.00),
    (0.37,  0.00, 0.27,  0.00, 0.65,  1.00),
    (0.37,  0.00, 0.27,  0.00, 0.65, -1.00),
    (0.37,  0.08, 0.27,  0.50, 0.65,  0.50),
    (0.37, -0.08, 0.27, -0.50, 0.65, -0.50),
    # Lateral poses with larger x.
    (0.33,  0.18, 0.27,  0.50, 0.50,  0.55),
    (0.33, -0.18, 0.27, -0.50, 0.50, -0.55),
    # Lateral poses with larger y.
    (0.20,  0.22, 0.28,  0.60, 0.40,  0.70),
    (0.20, -0.22, 0.28, -0.60, 0.40, -0.70),
    # Diagonal forward-y poses with large roll.
    (0.24, -0.20, 0.31,  0.70, 0.45, -1.00),
    (0.24,  0.20, 0.31, -0.70, 0.45,  1.00),
    (0.25, -0.15, 0.29, -0.60, 0.62, -0.50),
    (0.25,  0.15, 0.29,  0.60, 0.62,  0.50),
    # High poses.
    (0.21, -0.09, 0.40, -0.40, 0.25, -0.60),
    (0.21,  0.00, 0.40,  0.40, 0.25,  0.00),
    (0.21,  0.09, 0.40, -0.40, 0.25,  0.60),
    (0.20, -0.09, 0.40,  0.40, 0.28, -0.60),
    (0.20,  0.09, 0.40, -0.40, 0.28,  0.60),
    # Low poses.
    (0.30, -0.10, 0.24,  0.40, 0.70, -0.70),
    (0.30,  0.00, 0.24,  0.00, 0.75,  0.00),
    (0.30,  0.10, 0.24, -0.40, 0.70,  0.70),
    # Roll extremes.
    (0.26,  0.12, 0.30,  0.80, 0.50,  0.30),
    (0.26, -0.12, 0.30, -0.80, 0.50, -0.30),
    # Large roll and yaw combinations for rotation diversity.
    (0.29, -0.10, 0.28,  0.90, 0.60, -0.40),
    (0.29,  0.10, 0.28, -0.90, 0.60,  0.40),
    (0.28, -0.18, 0.30,  0.85, 0.55, -0.80),
    (0.28,  0.18, 0.30, -0.85, 0.55,  0.80),
    # Forward reach with different roll/yaw combinations.
    (0.35, -0.12, 0.30,  0.60, 0.58, -0.70),
    (0.35,  0.12, 0.30, -0.60, 0.58,  0.70),
    (0.34, -0.06, 0.28,  0.40, 0.72,  0.80),
    (0.34,  0.06, 0.28, -0.40, 0.72, -0.80),
    # High poses with large roll.
    (0.22, -0.14, 0.38,  0.75, 0.32, -0.70),
    (0.22,  0.14, 0.38, -0.75, 0.32,  0.70),
    # Lateral poses with large pitch and opposite yaw.
    (0.31,  0.20, 0.27,  0.30, 0.68,  0.95),
    (0.31, -0.20, 0.27, -0.30, 0.68, -0.95),
    # Mid-range poses covering roll, pitch, and yaw.
    (0.30,  0.05, 0.32,  0.75, 0.42,  0.65),
    (0.30, -0.05, 0.32, -0.75, 0.42, -0.65),
]

DEFAULT_AUTO_MOVE_DURATION_S = 3.0
AUTO_SETTLE_EXTRA_S = 0.6
AUTO_MARKER_TIMEOUT_S = 2.5
AUTO_MARKER_STABLE_FRAMES = 4
MIN_CALIB_SAMPLES = 5


def make_input_thread(line_queue: queue.Queue) -> threading.Thread:
    def _loop():
        while True:
            try:
                line_queue.put(input())
            except EOFError:
                line_queue.put(None)
                break
            except KeyboardInterrupt:
                line_queue.put(None)
                break
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


# ==========================================
# Main flow.
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Eye-in-hand calibration data collection")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--robot-backend", choices=("rars01",), default=None)
    parser.add_argument("--manual", action="store_true",
                        help="unsupported: hand-guided gravity mode requires RARS01-specific implementation")
    args = parser.parse_args()
    if args.manual:
        parser.error("Manual gravity mode is not supported for RARS01; use automatic calibration")

    root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_config(config_path)
    robot_cfg = cfg.get("robot", {})
    robot_backend = str(args.robot_backend or robot_cfg.get("backend", "rars01")).lower()

    cam_type   = cfg["camera"]["type"]
    calib_dir  = root / "config" / "calibration" / cam_type
    aruco_cfg  = cfg["calibration"]["aruco"]
    he_method  = cfg["calibration"].get("hand_eye_method", "TSAI")
    save_path  = calib_dir / "hand_eye.npz"
    samples_path = calib_dir / "hand_eye_samples.npz"
    auto_move_duration_s = float(
        cfg["calibration"].get("auto", {}).get(
            "move_duration_s", DEFAULT_AUTO_MOVE_DURATION_S
        )
    )
    if auto_move_duration_s <= 0.0:
        raise ValueError("calibration.auto.move_duration_s must be positive")
    auto_home_duration_s = float(cfg["calibration"].get("auto", {}).get("home_duration_s", 4.0))
    auto_home_timeout_s = float(cfg["calibration"].get("auto", {}).get("home_timeout_s", 8.0))
    if auto_home_duration_s <= 0.0 or auto_home_timeout_s <= 0.0:
        raise ValueError("calibration.auto.home_duration_s and home_timeout_s must be positive")

    # Camera.
    cam = make_camera(cfg)
    cam.setup_aruco(
        marker_length_m=aruco_cfg["marker_length_m"],
        dict_id=aruco_cfg.get("dict_id", 0),
        target_marker_id=aruco_cfg.get("target_marker_id"),
    )

    # Calibrator.
    calibrator = HandEyeCalibrator(CalibMode.EYE_IN_HAND, method=he_method)

    # Robot.
    mode_str = f"auto ({len(CALIB_POSES_XYZ)} preset poses)"
    arm = None
    controller: RarsPoseController | None = None
    grasp_driver: GraspDriver | None = None
    auto_controller_mode: str | None = None
    auto = {
        "enabled": True,
        "idx": 0,
        "pose_idx": None,
        "phase": "idle",
        "settle_until": 0.0,
        "timeout_at": 0.0,
        "stable_frames": 0,
        "status": "waiting to start",
        "finished": False,
    }
    result_saved = False

    print(f"\n=== Eye-in-Hand Calibration ===")
    print(f"Camera: {cam_type}  |  Mode: {mode_str}  |  Solver: {he_method}")
    print(f"ArUco size: {aruco_cfg['marker_length_m']*100:.0f}cm  |  Output: {save_path}")
    print()

    # Open the camera first so the arm is not enabled if camera setup fails.
    try:
        cam.open()
        print("Warming up camera...", end="", flush=True)
        cam.warm_up(20)
        print(" ready\n")
    except Exception as e:
        try:
            cam.close()
        except Exception:
            pass
        print(f"[Camera] Initialization failed: {e}")
        sys.exit(1)

    try:
        if robot_backend != "rars01":
            raise ValueError("Only robot.backend: rars01 is supported")
        answer = input(
            "RARS01: place the arm in zero/home, clear all 50-pose workspace "
            "and type START: "
        ).strip()
        if answer != "START":
            print("[RARS01] Cancelled before serial or motors were opened")
            cam.close()
            return
        arm = RarsArmAdapter(robot_cfg, root)
        auto_controller_mode = "posvel"
        controller = RarsPoseController(
            arm, dt=1.0 / arm.rate, arm_control_mode=auto_controller_mode,
        )
        grasp_driver = GraspDriver(
            arm, controller, gripper_config=robot_cfg.get("gripper"),
        )
        grasp_driver.start()
        print(
            f"[Robot] Auto mode ready, control mode: POS/VEL. "
            f"{len(CALIB_POSES_XYZ)} preset poses will be traversed."
        )
    except Exception as e:
        try:
            if arm is not None:
                arm.disconnect()
        except Exception:
            pass
        try:
            cam.close()
        except Exception:
            pass
        print(f"[Robot] Connection failed: {e}")
        sys.exit(1)

    print("[Controls] Auto traversal and capture  c/q=stop and solve  pos=print current TCP pose")
    print()

    latest_pose = None
    line_queue: queue.Queue | None = None
    if sys.stdin.isatty():
        line_queue = queue.Queue()
        make_input_thread(line_queue)
    else:
        print("[Hint] Non-interactive terminal detected; terminal commands are disabled")

    def _print_fk() -> None:
        try:
            T = grasp_driver.get_tcp_pose()
            t = T[:3, 3]
            R = T[:3, :3]
            _r, _p, _y = rotation_matrix_to_euler_zyx(R)
            print(f"  FK: x={t[0]:+.3f} y={t[1]:+.3f} z={t[2]:+.3f} m"
                  f"  rpy=[{_r:+.2f} {_p:+.2f} {_y:+.2f}] rad")
        except Exception as e:
            print(f"  [Error] {e}")

    def capture_sample(cur, source: str) -> bool:
        if cur is None:
            print("  [Skip] Marker is not visible; adjust the pose and try again")
            return False

        print(f"\n[Sample {calibrator.n_samples + 1}] {source}")
        print(f"  ArUco: x={cur.T_marker2cam[0,3]:.3f} "
              f"y={cur.T_marker2cam[1,3]:.3f} "
              f"z={cur.T_marker2cam[2,3]:.3f} m")
        try:
            T_g2b = grasp_driver.get_tcp_pose()
            t = T_g2b[:3, 3]
            print(f"  End effector (FK): x={t[0]:.4f} y={t[1]:.4f} z={t[2]:.4f} m")
            calibrator.add_sample(T_g2b, cur.T_marker2cam)
            print(f"  [OK] Recorded, total samples: {calibrator.n_samples}"
                  + ("  will solve automatically on finish" if calibrator.n_samples >= 15 else ""))
            return True
        except Exception as e:
            print(f"  [Error] Failed to read TCP pose: {e}")
            return False

    def compute_and_save(reason: str) -> bool:
        nonlocal result_saved
        print(f"\n[Finish] {reason}")
        if calibrator.n_samples < MIN_CALIB_SAMPLES:
            print(f"[Result] Not enough samples ({calibrator.n_samples} < {MIN_CALIB_SAMPLES}); calibration was not solved")
            if save_path.exists():
                print("[Result] Existing hand_eye.npz was not updated")
            return False

        # Preserve the collected measurements even if a solver or installation
        # issue occurs.  The result file is intentionally left untouched until
        # a complete new calibration has been computed.
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            samples_path,
            T_gripper2base=np.asarray([s.T_gripper2base for s in calibrator._samples]),
            T_marker2cam=np.asarray([s.T_marker2cam for s in calibrator._samples]),
        )
        print(f"[Result] Solving with {calibrator.n_samples} samples...")
        try:
            result = calibrator.calibrate(min_samples=MIN_CALIB_SAMPLES)
            HandEyeCalibrator.save(result, save_path)
            t = result.T_result[:3, 3]
            R = result.T_result[:3, :3]
            print(f"[Result] T_cam2gripper translation: x={t[0]:.4f} y={t[1]:.4f} z={t[2]:.4f} m")
            print(f"[Result] Rotation matrix:\n{R}")
            print(f"[Result] [OK] Saved to {save_path}")
            if calibrator.n_samples < 15:
                print("[Result] Tip: fewer than 15 samples; collect more samples for better accuracy")
            result_saved = True
            return True
        except Exception as e:
            print(f"[Result] [Error] Solve failed: {e}")
            return False

    def start_next_auto_pose() -> bool:
        if not auto["enabled"] or controller is None:
            return False

        total = len(CALIB_POSES_XYZ)
        while auto["idx"] < total:
            idx = auto["idx"]
            x, y, z, roll, pitch, yaw = CALIB_POSES_XYZ[idx]
            print(f"\n[Auto] Pose {idx+1}/{total}: "
                  f"pos=({x:.2f},{y:.2f},{z:.2f}) rpy=({roll:.2f},{pitch:.2f},{yaw:.2f})")
            if robot_backend == "rars01":
                planned_duration_s = grasp_driver.move_rars_calibration_pose(
                    x, y, z, roll, pitch, yaw, auto_move_duration_s,
                )
            else:
                planned_duration_s = (
                    auto_move_duration_s
                    if controller.move_to_traj(
                        x, y, z, roll=roll, pitch=pitch, yaw=yaw,
                        duration=auto_move_duration_s,
                    )
                    else None
                )
            if planned_duration_s is not None:
                now = time.monotonic()
                auto["pose_idx"] = idx
                auto["phase"] = "settling"
                auto["settle_until"] = now + planned_duration_s + AUTO_SETTLE_EXTRA_S
                auto["timeout_at"] = auto["settle_until"] + AUTO_MARKER_TIMEOUT_S
                auto["stable_frames"] = 0
                auto["status"] = f"pose {idx+1}/{total} moving"
                return False

            print(f"[Auto] Pose {idx+1}/{total} has no safe IK solution, skipping")
            auto["idx"] += 1

        auto["phase"] = "done"
        auto["finished"] = True
        auto["status"] = "all poses completed"
        print("\n[Auto] All preset poses completed")
        return True

    def tick_auto(cur) -> bool:
        if not auto["enabled"] or auto["finished"]:
            return auto["finished"]

        if auto["phase"] == "idle":
            return start_next_auto_pose()

        pose_idx = auto["pose_idx"]
        total = len(CALIB_POSES_XYZ)
        now = time.monotonic()

        if auto["phase"] == "settling":
            remain = auto["settle_until"] - now
            if remain > 0.0:
                auto["stable_frames"] = 0
                auto["status"] = f"pose {pose_idx+1}/{total} moving/settling {remain:.1f}s"
                return False
            auto["phase"] = "searching"

        if cur is not None:
            auto["stable_frames"] += 1
            remain = max(0.0, auto["timeout_at"] - now)
            auto["status"] = (
                f"pose {pose_idx+1}/{total} marker stable "
                f"{auto['stable_frames']}/{AUTO_MARKER_STABLE_FRAMES}  remaining {remain:.1f}s"
            )
            if auto["stable_frames"] >= AUTO_MARKER_STABLE_FRAMES:
                capture_sample(cur, f"auto pose {pose_idx+1}/{total}")
                auto["idx"] += 1
                auto["phase"] = "idle"
                auto["stable_frames"] = 0
                return start_next_auto_pose()
        else:
            auto["stable_frames"] = 0
            remain = max(0.0, auto["timeout_at"] - now)
            auto["status"] = f"pose {pose_idx+1}/{total} waiting for ArUco {remain:.1f}s"

        if now >= auto["timeout_at"]:
            print(f"[Auto] Pose {pose_idx+1}/{total} timed out without ArUco, skipping")
            auto["idx"] += 1
            auto["phase"] = "idle"
            auto["stable_frames"] = 0
            return start_next_auto_pose()

        return False

    def safe_home_and_disconnect() -> None:
        """Return home first, then stop control and disconnect."""
        if arm is None or auto_controller_mode is None:
            return
        try:
            print("[Robot] Homing and disconnecting...")
            if grasp_driver is not None:
                if not grasp_driver.home_rars(auto_home_duration_s, auto_home_timeout_s):
                    raise RuntimeError("RARS01 did not reach home before timeout")
            arm.stop_control_loop()
        except Exception as e:
            print(f"[Robot] Homing failed: {e}")
        try:
            arm.disconnect()
        except Exception:
            pass

    def handle_line(raw: str) -> bool:
        if raw is None:
            print("\n[Interrupt] Terminal input closed; stopping and trying to solve")
            return True

        line = raw.strip().lower()

        if line in {"q", "c"}:
            return True

        if line == "pos":
            _print_fk()
            return False

        if line:
            print("  Auto commands: c/q=finish and solve  pos=print current TCP pose")
        return False

    # Main loop.
    WIN = "Eye-in-Hand Calibration  (operate in terminal)"
    finish_reason = "normal finish"
    try:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

        while True:
            bgr, _ = cam.get_frame()
            if bgr is not None:
                pose = cam.detect_aruco(bgr)
                latest_pose = pose
                if tick_auto(pose):
                    finish_reason = "auto traversal completed"
                vis  = cam.draw_aruco(bgr)
                n    = calibrator.n_samples

                def osd(text, y, color=(220, 220, 220)):
                    cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, color, 1, cv2.LINE_AA)

                if pose:
                    osd(f"[ID={pose.id}] z={pose.T_marker2cam[2,3]:.3f}m  samples:{n}",
                        28, (80, 220, 80))
                else:
                    osd(f"No marker  samples:{n}", 28, (80, 80, 220))
                osd(f"AUTO: {auto['status']}", 50, (180, 180, 60))

                filled = min(n, 15) * (400 // 15)
                cv2.rectangle(vis, (10, 70), (10 + filled, 82), (0, 200, 100), -1)
                cv2.rectangle(vis, (10, 70), (410, 82), (160, 160, 160), 1)
                cv2.putText(vis, f"{n}/15", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                mode_label = "AUTO"
                cv2.putText(vis, mode_label, (vis.shape[1] - 200, vis.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 60), 1)

                cv2.imshow(WIN, vis)

            if cv2.waitKey(30) & 0xFF in [ord('q'), ord('Q'), 27]:
                finish_reason = "window exit"
                break
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                finish_reason = "window closed"
                break

            try:
                if line_queue is not None and handle_line(line_queue.get_nowait()):
                    finish_reason = "user interrupted"
                    break
            except queue.Empty:
                pass

            if auto["finished"]:
                break

    except KeyboardInterrupt:
        finish_reason = "Ctrl+C interrupt"
        print("\n[Ctrl+C] Stopping and trying to solve")

    finally:
        cv2.destroyAllWindows()
        cam.close()
        if controller is not None:
            safe_home_and_disconnect()
        compute_and_save(finish_reason)

    print(f"\nDone, total samples: {calibrator.n_samples}.")
    if calibrator.n_samples > 0 and not result_saved:
        print("Tip: hand_eye.npz was not generated; collect more samples and try again.")


if __name__ == "__main__":
    main()
    os._exit(0)
