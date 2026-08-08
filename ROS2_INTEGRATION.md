# ROS 2 integration contract

The Python algorithms are transport-independent. A future ROS 2 adapter should
only convert messages to/from the dataclasses in `rars01_graspnet.contracts`.

| Python contract | ROS 2 representation |
|---|---|
| `Header` | `std_msgs/Header` (`stamp_ns` is split into sec/nanosec) |
| `RgbdFrame.color_bgr` | `sensor_msgs/Image`, encoding `bgr8` |
| `RgbdFrame.depth_mm` | `sensor_msgs/Image`, encoding `16UC1` |
| `CameraIntrinsics` | `sensor_msgs/CameraInfo` |
| `RobotState` | `sensor_msgs/JointState` plus diagnostics |
| `Detection2D` | `vision_msgs/Detection2D` plus mask/depth result |
| `GraspCandidate` | `geometry_msgs/PoseStamped` plus score/width |

Frame names are part of the public contract:

```text
base_link
End_link
camera_color_optical_frame
```

The SDK camera timestamps frames with the system clock. A ROS camera adapter
must preserve the incoming ROS message timestamp instead. Algorithms must never
read the wall clock themselves after a frame has been constructed.

Future adapters:

```text
RosCameraSource: synchronized Image + Image + CameraInfo -> RgbdFrame
RosRobotStateSource: JointState -> RobotState
RosDetectionPublisher: Detection2D -> vision_msgs/Detection2D
RosGraspPublisher: GraspCandidate -> PoseStamped/custom metadata
MoveItRobotDriver: GraspCandidate -> planning/execution action
```

Neither YOLO nor GraspNet may import `rclpy`; this keeps inference testable
outside ROS and allows it to run in the CUDA `uv` environment.

