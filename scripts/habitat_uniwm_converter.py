from __future__ import annotations

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
from scripts.prompt_builder import build_action_prompt, build_viz_prompt


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

        linear_threshold = spec.forward_step_m * spec.linear_tolerance_ratio
        angular_step = float(np.deg2rad(spec.turn_angle_deg))
        angular_threshold = angular_step * spec.angular_tolerance_ratio

        snapped_dx = 0.0
        snapped_dy = 0.0
        snapped_dyaw = 0.0

        if parsed.dx >= linear_threshold:
            snapped_dx = spec.forward_step_m
        elif parsed.dx <= -linear_threshold and spec.move_backward_action and spec.backward_step_m:
            snapped_dx = -spec.backward_step_m

        if abs(parsed.dyaw) >= angular_threshold:
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
        linear_threshold = spec.forward_step_m * spec.linear_tolerance_ratio
        angular_step = float(np.deg2rad(spec.turn_angle_deg))
        angular_threshold = angular_step * spec.angular_tolerance_ratio

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

        if abs(parsed.dy) >= linear_threshold:
            warnings.append("Non-zero dy is ignored by the Habitat converter; only turn, move, or turn+move are emitted.")

        if spec.compose_turn_then_move:
            habitat_actions.extend(turn_actions if spec.rotation_first_when_mixed else move_actions)
            habitat_actions.extend(move_actions if spec.rotation_first_when_mixed else turn_actions)
        else:
            habitat_actions.extend(move_actions)
            habitat_actions.extend(turn_actions)

        if not habitat_actions:
            warnings.append("UniWM action did not produce any Habitat action above the configured thresholds.")
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
        goal_observation: ObservationLike,
        habitat_step_result: Optional[Mapping[str, Any]] = None,
        current_observation: Optional[ObservationLike] = None,
        start_pose: Sequence[float],
        decoded_action: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UniWMInputBundle:
        schema = self.config.observation
        step_observation = self._extract_step_observation(habitat_step_result, current_observation)
        start_img = self._to_pil_image(start_observation, schema)
        goal_img = self._to_pil_image(goal_observation, schema)
        current_img = self._to_pil_image(step_observation, schema) if step_observation is not None else None
        start_pose_str = pose_to_str(start_pose, schema.start_pose_template)

        if mode == "action_reasoning":
            dxy_range, dyaw_range = self.habitat_action_ranges()
            input_text = build_action_prompt(
                start_pose_str=start_pose_str,
                dxy_range=dxy_range,
                dyaw_range=dyaw_range,
            )
            images = [start_img, goal_img, current_img]
        elif mode == "single_step_visualization":
            if not decoded_action:
                raise AssertionError("decoded_action is required when formatting a visualization input")
            input_text = build_viz_prompt(decoded_action=decoded_action, start_pose_str=start_pose_str)
            images = [start_img, goal_img, current_img]
        elif mode == "task_level_evaluation":
            input_text = (
                "Task: Full Navigation Task\n"
                "Description: Navigate from the start observation to the goal observation. "
                "Start Observation: <image>\nGoal Observation: <image>\n"
                f"{start_pose_str}"
            )
            images = [start_img, goal_img]
        else:
            raise AssertionError(f"Unsupported UniWM mode: {mode}")

        return UniWMInputBundle(
            mode=mode,
            input_text=input_text,
            input_imgs=[image for image in images if image is not None],
            start_pose_str=start_pose_str,
            metadata=metadata or {},
        )

    def to_processor_payload(self, bundle: UniWMInputBundle) -> Dict[str, Any]:
        return {
            "text": [bundle.input_text],
            "images": bundle.input_imgs,
            "return_tensors": "pt",
        }

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

        if isinstance(observation, Mapping):
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
