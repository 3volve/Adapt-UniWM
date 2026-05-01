from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from PIL import Image


@dataclass(frozen=True)
class UniWMAction:
    dx: float
    dy: float
    dyaw: float
    raw_text: str
    is_stop: bool = False


@dataclass(frozen=True)
class HabitatDiscreteAction:
    action: str
    amount: Optional[float] = None
    axis: Optional[str] = None
    reason: str = ""


@dataclass(frozen=True)
class HabitatActionSpaceSpec:
    move_forward_action: str = "move_forward"
    turn_left_action: str = "turn_left"
    turn_right_action: str = "turn_right"
    stop_action: str = "stop"
    strafe_left_action: Optional[str] = None
    strafe_right_action: Optional[str] = None
    move_backward_action: Optional[str] = None
    forward_step_m: float = 0.25
    turn_angle_deg: float = 10.0
    strafe_step_m: Optional[float] = None
    backward_step_m: Optional[float] = None
    compose_turn_then_move: bool = True
    rotation_first_when_mixed: bool = True
    linear_tolerance_ratio: float = 0.5
    angular_tolerance_ratio: float = 0.5
    linear_deadband_m: float = 0.02
    angular_deadband_rad: float = 0.02
    max_consecutive_forward_steps: int = 1
    max_consecutive_turn_steps: int = 1


@dataclass(frozen=True)
class ObservationSchema:
    rgb_sensor_key: str = "rgb"
    goal_sensor_keys: Tuple[str, ...] = (
        "instance_imagegoal",
        "imagegoal",
        "goal_image",
        "goal_observation",
    )
    image_size: Tuple[int, int] = (256, 256)
    image_mode: str = "RGB"
    start_position_indices: Tuple[int, int] = (0, 2)
    start_rotation_format: str = "xyzw"
    start_pose_template: str = "Starting Point Coordinate: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}\n"


@dataclass(frozen=True)
class UniWMInputBundle:
    start_observation: Any
    goal_observation: Any
    current_observation: Any
    start_pose_str: str
    action_text: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def unpack(self) -> Dict[str, Any]:
        return {
            "start_observation": self.start_observation,
            "goal_observation": self.goal_observation,
            "current_observation": self.current_observation,
            "start_pose_str": self.start_pose_str,
        }


@dataclass(frozen=True)
class HabitatActionConversion:
    raw_action: UniWMAction
    snapped_action: UniWMAction
    habitat_actions: List[HabitatDiscreteAction]
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class HabitatUniWMConverterConfig:
    action_space: HabitatActionSpaceSpec
    observation: ObservationSchema
    bin_step: float = 0.01

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HabitatUniWMConverterConfig":
        observation_mapping = data.get("observation_mapping", {})
        if "image_size" in observation_mapping:
            observation_mapping = {
                **observation_mapping,
                "image_size": tuple(observation_mapping["image_size"]),
            }
        if "goal_sensor_keys" in observation_mapping:
            observation_mapping = {
                **observation_mapping,
                "goal_sensor_keys": tuple(observation_mapping["goal_sensor_keys"]),
            }
        if "start_position_indices" in observation_mapping:
            observation_mapping = {
                **observation_mapping,
                "start_position_indices": tuple(observation_mapping["start_position_indices"]),
            }
        return cls(
            action_space=HabitatActionSpaceSpec(**data.get("action_space", {})),
            observation=ObservationSchema(**observation_mapping),
            bin_step=float(data.get("bin_step", 0.01)),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "HabitatUniWMConverterConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)


def pose_to_str(pose: Sequence[float], template: str) -> str:
    if len(pose) < 3:
        raise AssertionError(f"Expected pose with [x, y, yaw], got {pose}")
    return template.format(x=pose[0], y=pose[1], yaw=pose[2])
