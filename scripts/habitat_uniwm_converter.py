from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from scripts.action_utils import extract_bin_values
from scripts.habitat_uniwm_schemas import (
    HabitatActionConversion,
    HabitatDiscreteAction,
    HabitatUniWMConverterConfig,
    ObservationSchema,
    UniWMAction,
    UniWMInputBundle,
    pose_to_str,
)


ObservationLike = Union[Image.Image, np.ndarray, torch.Tensor, Mapping[str, Any]]


class HabitatUniWMConverter:
    """Schema-driven converter between Habitat step data and UniWM-facing inputs."""

    def __init__(self, config: HabitatUniWMConverterConfig):
        self.config = config

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "HabitatUniWMConverter":
        return cls(HabitatUniWMConverterConfig.from_yaml(str(path)))

    def habitat_action_ranges(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        spec = self.config.action_space

        max_dx = spec.forward_step_m * max(1, spec.max_consecutive_forward_steps)
        min_dx = -spec.backward_step_m if spec.move_backward_action and spec.backward_step_m else 0.0

        if spec.strafe_left_action and spec.strafe_right_action and spec.strafe_step_m:
            max_dy = spec.strafe_step_m
            min_dy = -spec.strafe_step_m
        else:
            max_dy = 0.0
            min_dy = 0.0

        max_dxy = max(abs(min_dx), abs(max_dx), abs(min_dy), abs(max_dy))
        max_dyaw = np.deg2rad(spec.turn_angle_deg) * max(1, spec.max_consecutive_turn_steps)
        return (-max_dxy, max_dxy), (-max_dyaw, max_dyaw)

    def parse_uniwm_action(self, action_text: str) -> UniWMAction:
        normalized = action_text.strip()
        if normalized.lower() == "stop":
            return UniWMAction(dx=0.0, dy=0.0, dyaw=0.0, raw_text=normalized, is_stop=True)

        step = self.config.bin_step
        return UniWMAction(
            dx=extract_bin_values(normalized, "dx", step),
            dy=extract_bin_values(normalized, "dy", step),
            dyaw=extract_bin_values(normalized, "dyaw", step),
            raw_text=normalized,
            is_stop=False,
        )

    def snap_uniwm_action_to_habitat(self, action: Union[str, UniWMAction]) -> UniWMAction:
        parsed = self.parse_uniwm_action(action) if isinstance(action, str) else action
        spec = self.config.action_space

        if parsed.is_stop:
            return parsed

        linear_deadband = float(spec.linear_deadband_m)
        angular_step = float(np.deg2rad(spec.turn_angle_deg))
        angular_deadband = float(spec.angular_deadband_rad)

        snapped_dx = 0.0
        snapped_dy = 0.0
        snapped_dyaw = 0.0

        if parsed.dx >= linear_deadband:
            snapped_dx = spec.forward_step_m
        elif parsed.dx <= -linear_deadband and spec.move_backward_action and spec.backward_step_m:
            snapped_dx = -spec.backward_step_m

        if spec.strafe_left_action and spec.strafe_right_action and spec.strafe_step_m:
            if parsed.dy >= linear_deadband:
                snapped_dy = spec.strafe_step_m
            elif parsed.dy <= -linear_deadband:
                snapped_dy = -spec.strafe_step_m

        if abs(parsed.dyaw) >= angular_deadband:
            snapped_dyaw = angular_step if parsed.dyaw > 0 else -angular_step

        return UniWMAction(
            dx=float(snapped_dx),
            dy=float(snapped_dy),
            dyaw=float(snapped_dyaw),
            raw_text=parsed.raw_text,
            is_stop=False,
        )

    def uniwm_action_to_habitat(self, action: Union[str, UniWMAction]) -> HabitatActionConversion:
        parsed = self.parse_uniwm_action(action) if isinstance(action, str) else action
        snapped = self.snap_uniwm_action_to_habitat(parsed)
        spec = self.config.action_space
        warnings = []

        if parsed.is_stop:
            return HabitatActionConversion(
                raw_action=parsed,
                snapped_action=parsed,
                habitat_actions=[HabitatDiscreteAction(action=spec.stop_action, reason="UniWM predicted stop")],
                warnings=[],
            )

        habitat_actions = []
        linear_threshold = min(float(spec.forward_step_m) * spec.linear_tolerance_ratio, float(spec.linear_deadband_m))
        angular_step = float(np.deg2rad(spec.turn_angle_deg))
        angular_threshold = min(angular_step * spec.angular_tolerance_ratio, float(spec.angular_deadband_rad))

        turn_actions = []
        if abs(snapped.dyaw) >= angular_threshold:
            turn_name = spec.turn_left_action if snapped.dyaw > 0 else spec.turn_right_action
            turn_actions = [
                HabitatDiscreteAction(action=turn_name, amount=angular_step, axis="dyaw", reason="mapped from snapped UniWM dyaw")
            ]

        move_actions = []
        if snapped.dx >= linear_threshold:
            move_actions = [
                HabitatDiscreteAction(
                    action=spec.move_forward_action,
                    amount=spec.forward_step_m,
                    axis="dx",
                    reason="mapped from snapped UniWM dx",
                )
            ]
        elif snapped.dx <= -linear_threshold:
            if spec.move_backward_action and spec.backward_step_m:
                move_actions = [
                    HabitatDiscreteAction(
                        action=spec.move_backward_action,
                        amount=spec.backward_step_m,
                        axis="dx",
                        reason="mapped from snapped UniWM negative dx",
                    )
                ]
            else:
                warnings.append("Negative dx cannot be mapped because backward movement is not configured.")

        strafe_actions = []
        if abs(snapped.dy) >= linear_threshold:
            if spec.strafe_left_action and spec.strafe_right_action and spec.strafe_step_m:
                strafe_name = spec.strafe_left_action if snapped.dy > 0 else spec.strafe_right_action
                strafe_actions = [
                    HabitatDiscreteAction(
                        action=strafe_name,
                        amount=spec.strafe_step_m,
                        axis="dy",
                        reason="mapped from snapped UniWM dy",
                    )
                ]
            else:
                warnings.append("Non-zero dy cannot be mapped because strafing is not configured.")
        elif abs(parsed.dy) >= float(spec.linear_deadband_m):
            warnings.append("Non-zero dy cannot be mapped because strafing is not configured.")

        linear_actions = move_actions + strafe_actions

        if spec.compose_turn_then_move:
            habitat_actions.extend(turn_actions if spec.rotation_first_when_mixed else linear_actions)
            habitat_actions.extend(linear_actions if spec.rotation_first_when_mixed else turn_actions)
        else:
            habitat_actions.extend(linear_actions)
            habitat_actions.extend(turn_actions)

        if not habitat_actions:
            warnings.append("UniWM action did not produce any Habitat action above the configured deadband.")
        if (parsed.dx, parsed.dy, parsed.dyaw) != (snapped.dx, snapped.dy, snapped.dyaw):
            warnings.append("UniWM action was snapped to Habitat-configured move/turn distances before conversion.")

        return HabitatActionConversion(
            raw_action=parsed,
            snapped_action=snapped,
            habitat_actions=habitat_actions,
            warnings=warnings,
        )

    def habitat_step_to_uniwm_input(
        self,
        *,
        mode: str,
        start_observation: ObservationLike,
        goal_observation: Optional[ObservationLike] = None,
        habitat_step_result: Optional[Mapping[str, Any]] = None,
        current_observation: Optional[ObservationLike] = None,
        episode: Optional[Any] = None,
        start_pose: Optional[Sequence[float]] = None,
        decoded_action: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UniWMInputBundle:
        schema = self.config.observation
        if mode == "single_step_visualization" and not decoded_action:
            raise AssertionError("decoded_action is required when formatting a visualization input")
        if mode not in {"action_reasoning", "single_step_visualization", "task_level_evaluation"}:
            raise AssertionError(f"Unsupported UniWM mode: {mode}")

        step_observation = self._extract_step_observation(habitat_step_result, current_observation)
        goal_observation = self._extract_goal_observation(
            goal_observation=goal_observation,
            habitat_step_result=habitat_step_result,
            current_observation=current_observation,
            start_observation=start_observation,
            episode=episode,
        )
        if goal_observation is None:
            raise AssertionError(
                "Goal observation is required. Pass goal_observation explicitly or include one of "
                f"{schema.goal_sensor_keys} in the Habitat observation."
            )
        if start_pose is None:
            start_pose = self.extract_start_pose(episode)

        start_img = self._to_pil_image(start_observation, schema)
        goal_img = self._to_pil_image(goal_observation, schema)
        current_img = self._to_pil_image(step_observation, schema) if step_observation is not None else start_img
        start_pose_str = pose_to_str(start_pose, schema.start_pose_template)
        bundle_metadata = {
            **self.extract_episode_metadata(episode),
            **(metadata or {}),
        }

        return UniWMInputBundle(
            start_observation=start_img,
            goal_observation=goal_img,
            current_observation=current_img,
            start_pose_str=start_pose_str,
            action_text=decoded_action,
            metadata=bundle_metadata,
        )

    def _extract_step_observation(
        self,
        habitat_step_result: Optional[Mapping[str, Any]],
        current_observation: Optional[ObservationLike],
    ) -> Optional[ObservationLike]:
        if current_observation is not None:
            return current_observation
        if habitat_step_result is None:
            return None
        if "observations" in habitat_step_result:
            return habitat_step_result["observations"]
        return habitat_step_result

    def _extract_goal_observation(
        self,
        *,
        goal_observation: Optional[ObservationLike],
        habitat_step_result: Optional[Mapping[str, Any]],
        current_observation: Optional[ObservationLike],
        start_observation: Optional[ObservationLike],
        episode: Optional[Any],
    ) -> Optional[ObservationLike]:
        if goal_observation is not None:
            return goal_observation

        candidates = [
            current_observation,
            self._extract_step_observation(habitat_step_result, None),
            start_observation,
        ]
        for candidate in candidates:
            goal = self._extract_goal_from_mapping(candidate)
            if goal is not None:
                return goal

        return self._extract_goal_from_episode(episode)

    def _extract_goal_from_mapping(self, observation: Optional[ObservationLike]) -> Optional[ObservationLike]:
        if not isinstance(observation, MappingABC):
            return None

        for key in self.config.observation.goal_sensor_keys:
            if key in observation and observation[key] is not None:
                return observation[key]

        desired_goal = observation.get("desired_goal")
        if isinstance(desired_goal, MappingABC):
            for key in self.config.observation.goal_sensor_keys:
                if key in desired_goal and desired_goal[key] is not None:
                    return desired_goal[key]

        return None

    def _extract_goal_from_episode(self, episode: Optional[Any]) -> Optional[ObservationLike]:
        if episode is None:
            return None
        for key in self.config.observation.goal_sensor_keys:
            value = self._read_field(episode, key)
            if value is not None:
                return value
        info = self._read_field(episode, "info")
        if isinstance(info, MappingABC):
            for key in self.config.observation.goal_sensor_keys:
                if key in info and info[key] is not None:
                    return info[key]
        return None

    def extract_start_pose(self, episode: Optional[Any]) -> Tuple[float, float, float]:
        if episode is None:
            raise AssertionError("start_pose is required when no Habitat episode is provided.")

        position = self._read_field(episode, "start_position")
        if position is None:
            raise AssertionError("Habitat episode is missing start_position; pass start_pose explicitly.")
        if len(position) <= max(self.config.observation.start_position_indices):
            raise AssertionError(
                "Habitat episode start_position is shorter than configured start_position_indices "
                f"{self.config.observation.start_position_indices}: {position}"
            )

        first_idx, second_idx = self.config.observation.start_position_indices
        yaw = self._yaw_from_rotation(self._read_field(episode, "start_rotation"))
        return (float(position[first_idx]), float(position[second_idx]), yaw)

    def extract_episode_metadata(self, episode: Optional[Any]) -> Dict[str, Any]:
        if episode is None:
            return {}

        fields = (
            "episode_id",
            "scene_id",
            "start_position",
            "start_rotation",
            "goal_object_id",
            "goal_image_id",
            "object_category",
            "goal_key",
            "info",
        )
        metadata = {
            field: self._metadata_value(self._read_field(episode, field))
            for field in fields
            if self._read_field(episode, field) is not None
        }

        goals = self._read_field(episode, "goals")
        if goals is not None:
            metadata["goals"] = self._goal_metadata(goals)

        return metadata

    def _goal_metadata(self, goals: Any) -> Dict[str, Any]:
        if not isinstance(goals, SequenceABC) or isinstance(goals, (str, bytes)):
            return {"raw": self._metadata_value(goals)}
        result: Dict[str, Any] = {"count": len(goals)}
        if goals:
            first_goal = goals[0]
            goal_fields = (
                "object_id",
                "object_name",
                "object_name_id",
                "object_category",
                "position",
                "radius",
                "object_surface_area",
            )
            result["first"] = {
                field: self._metadata_value(self._read_field(first_goal, field))
                for field in goal_fields
                if self._read_field(first_goal, field) is not None
            }
            image_goals = self._read_field(first_goal, "image_goals")
            if image_goals is not None:
                result["first"]["image_goal_count"] = len(image_goals)
        return result

    def _yaw_from_rotation(self, rotation: Optional[Sequence[float]]) -> float:
        if rotation is None:
            return 0.0
        if len(rotation) == 1:
            return float(rotation[0])
        if len(rotation) != 4:
            raise AssertionError(f"Expected start_rotation as yaw or quaternion with 4 values, got {rotation}")

        values = [float(value) for value in rotation]
        fmt = self.config.observation.start_rotation_format.lower()
        if fmt == "xyzw":
            x, y, z, w = values
        elif fmt == "wxyz":
            w, x, y, z = values
        else:
            raise AssertionError(f"Unsupported start_rotation_format '{self.config.observation.start_rotation_format}'")

        return float(math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))

    def _read_field(self, value: Any, field: str) -> Any:
        if value is None:
            return None
        if isinstance(value, MappingABC):
            return value.get(field)
        return getattr(value, field, None)

    def _metadata_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.ndarray):
            return value.tolist()
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, MappingABC):
            return {str(key): self._metadata_value(item) for key, item in value.items()}
        if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
            return [self._metadata_value(item) for item in value]
        return str(value)

    def _to_pil_image(self, observation: ObservationLike, schema: ObservationSchema) -> Image.Image:
        if isinstance(observation, Image.Image):
            return observation.convert(schema.image_mode).resize(schema.image_size)

        if torch.is_tensor(observation):
            tensor = observation.detach().cpu()
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.dim() != 3:
                raise AssertionError(f"Expected image tensor with 3 dims, got shape {tuple(tensor.shape)}")
            if tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            observation = tensor.numpy()

        if isinstance(observation, np.ndarray):
            array = observation
            if array.ndim != 3:
                raise AssertionError(f"Expected image array with 3 dims, got shape {array.shape}")
            if array.shape[0] in (1, 3):
                array = np.transpose(array, (1, 2, 0))
            if array.dtype != np.uint8:
                max_value = float(array.max()) if array.size else 0.0
                if max_value <= 1.0:
                    array = array * 255.0
                array = np.clip(array, 0, 255).astype(np.uint8)
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            return Image.fromarray(array, mode=schema.image_mode).resize(schema.image_size)

        if isinstance(observation, MappingABC):
            if schema.rgb_sensor_key not in observation:
                raise AssertionError(
                    f"Habitat observation is missing configured rgb key '{schema.rgb_sensor_key}'. "
                    f"Available keys: {sorted(observation.keys())}"
                )
            return self._to_pil_image(observation[schema.rgb_sensor_key], schema)

        raise AssertionError(
            f"Unsupported Habitat observation type {type(observation)}. "
            "Expected PIL image, ndarray, tensor, or observation dict."
        )


def load_converter(config_path: Union[str, Path]) -> HabitatUniWMConverter:
    return HabitatUniWMConverter.from_yaml(config_path)
