from pathlib import Path

from rars01_graspnet.camera import camera_from_config
from rars01_graspnet.config import load_config, robot_kinematics_config
from rars01_graspnet.robot import robot_from_config
from utils.camera_utils import load_config as load_script_config


def test_jetson_config_inherits_robot_and_overrides_memory_profile():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "jetson_orin_nano.yaml")

    assert config["robot"]["rars01"]["command_rate_hz"] == 100.0
    assert config["robot"]["rars01"]["max_grasp_width_m"] == 0.09
    assert config["yolo"]["model_name"] == "yoloe-26s-seg.pt"
    assert config["graspnet"]["num_point"] == 10000
    assert config["graspnet"]["top_k"] == 20


def test_grasp_script_config_loader_follows_jetson_extends():
    root = Path(__file__).resolve().parents[1]
    config = load_script_config(root / "config" / "jetson_orin_nano.yaml")
    assert config["camera"]["type"] == "orbbec_gemini2"


def test_camera_adapter_accepts_project_color_dimension_names():
    root = Path(__file__).resolve().parents[1]
    camera = camera_from_config(load_config(root / "config" / "jetson_orin_nano.yaml"))

    assert (camera.width, camera.height, camera.fps) == (1280, 720, 30)


def test_jetson_robot_kinematics_uses_nested_rars01_settings():
    root = Path(__file__).resolve().parents[1]
    urdf, base_frame, tcp_frame = robot_kinematics_config(
        load_config(root / "config" / "jetson_orin_nano.yaml")
    )

    assert urdf.name == "rars01.urdf"
    assert base_frame == "base_link"
    assert tcp_frame == "End_link"


def test_jetson_robot_config_is_accepted_by_hardware_adapter(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "jetson_orin_nano.yaml")
    captured = {}

    class FakeRobot:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr("rars01_graspnet.robot.RarsRobot", FakeRobot)
    robot_from_config(config)

    assert captured["args"][:2] == ("/dev/ttyACM0", 921600)
    assert captured["kwargs"]["position_kp"] == [70.0, 120.0, 120.0, 50.0, 20.0, 20.0, 20.0]


def test_robot_adapter_adds_configured_sdk_path(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "jetson_orin_nano.yaml")
    added = []

    class FakePath(list):
        def insert(self, index, value):
            added.append((index, value))

    class FakeRobot:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr("rars01_graspnet.robot.sys.path", FakePath())
    monkeypatch.setattr("rars01_graspnet.robot.RarsRobot", FakeRobot)
    robot_from_config(config)

    assert added == [(0, str(root.parent / "rars_arm_sdk" / "build-python"))]
