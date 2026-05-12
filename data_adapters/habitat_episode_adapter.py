from __future__ import annotations

import math, torch
from typing import Any

import numpy as np
from PIL import Image
from collections.abc import Mapping

import habitat
from habitat.config.default import get_config
from habitat.core.simulator import Observations
from habitat.core.dataset import Episode
from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalNavEpisode

from gym.spaces import Dict as GymDictSpace
from scripts.action_utils import extract_bin_values
from scripts.habitat_uniwm_schemas import (
    UniWMInputBundle,
    SourceFormatter,
    HabitatOutputBundle,
)

EXPECTED_HABITAT_ACTIONS: Mapping[str, str] = {
    "stop": "stop",
    "forward": "move_forward",
    "left": "turn_left",
    "right": "turn_right"
}


class HabitatEpisodeAdapter:
    """Thin Habitat environment adapter.

    Responsibilities:
    - build/close Habitat env
    - reset Habitat env
    - step Habitat env with already-mapped Habitat actions
    - return HabitatOutputBundle
    """

    source_mode = "habitat"

    def __init__(
        self,
        config_path: str = "benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml",
        split: str = "val_mini",
        data_path: str = (
            "data/datasets/instance_imagenav/hm3d/"
            "instance_imagenav_hm3d_v3/val_mini/val_mini.json.gz"
        ),
        scenes_dir: str = "data/scene_datasets",
        max_episode_steps: int = 500,
        extra_overrides: list[str] | None | None = None,
    ):
        self.current_episode: InstanceImageGoalNavEpisode | Episode | None = None
        self.step_index: int = 0
        self.last_step: HabitatOutputBundle | None = None
        self.start_obs: Observations = Observations({})
        self.goal_image: np.ndarray

        self.config = get_config(
            config_path=config_path,
            overrides=[
                f"habitat.dataset.split={split}",
                f"habitat.dataset.data_path={data_path}",
                f"habitat.dataset.scenes_dir={scenes_dir}",
                f"habitat.environment.max_episode_steps={max_episode_steps}",
                *extra_overrides,
            ],
        )

        self.env = habitat.Env(config=self.config)
        assert self.env is not None

    def reset(self) -> HabitatOutputBundle:
        obs: Observations = self.env.reset()
        self.current_episode = self.env.current_episode
        self.step_index = 0
        self.start_obs = obs

        return self._pack_step(
            obs=obs,
            done=bool(self.env.episode_over),
            action_taken=None,
        )

    def step(self, habitat_action: str) -> HabitatOutputBundle:
        obs = self.env.step(habitat_action)
        self.current_episode = self.env.current_episode
        self.step_index += 1

        return self._pack_step(
            obs=obs,
            done=bool(self.env.episode_over),
            action_taken=habitat_action,
        )

    @property
    def action_space(self) -> object:
        return self.env.action_space

    @property
    def metrics(self) -> dict[str, object]:
        return dict(self.env.get_metrics())

    @property
    def habitat_config(self) -> Any:
        return self.config

    def close(self) -> None:
        self.env.close()

        self.current_episode = None
        self.last_step = None
        self.start_obs =  Observations({})
        self.step_index = 0

    def _pack_step(
        self,
        *,
        obs: Observations,
        done: bool,
        action_taken: str | None,
    ) -> HabitatOutputBundle:
        episode = self.current_episode
        if episode is None:
            raise AssertionError("Expected habitat.Env.current_episode to be set after reset/step.")

        step = HabitatOutputBundle(
            start_obs=self.start_obs,
            current_obs=obs,
            done=done,
            metrics=self.metrics,
            episode=episode,
            step_index=self.step_index,
            action_taken=action_taken,
        )

        self.last_step = step
        return step


class HabitatUniWMFormatter(SourceFormatter):
    """Strict converter between Habitat output bundles and UniWM input bundles."""

    source_mode = "habitat"

    GOAL_KEY: str = "instance_imagegoal"
    IMAGE_MODE = "RGB"
    RGB_KEY: str = "rgb"
    START_POS_IDX: tuple[int, int] = (0, 2)
    START_POSE_TEMPLATE: str = "Starting Point Coordinate: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}\n"

    def __init__(self, bin_step: float = 0.01, linear_deadband: float = 0.02, angular_deadband: float = 0.02, image_size: tuple[int, int] = (256, 256), cfg: Any | None = None):
        self.bin_step: float = bin_step
        self.linear_deadband: float = linear_deadband
        self.angular_deadband: float = angular_deadband
        self.image_size: tuple[int, int] = image_size

        if cfg is not None:
            turn_step_size: float = cfg.habitat.simulator.turn_angle
            self.right_angle_turn_repeats = round(90 / turn_step_size)
            self.forward_step_size: float = cfg.habitat.simulator.forward_step_size

    def convert_action(self, action_text: str) -> list[str]:
        dx, dy, dyaw, action_text = 0.0, 0.0, 0.0, action_text.strip()
        is_stop = action_text.lower() == "stop"

        dx: float = extract_bin_values(action_text, "dx", self.bin_step),
        dy: float = extract_bin_values(action_text, "dy", self.bin_step),
        dyaw: float = extract_bin_values(action_text, "dyaw", self.bin_step),

        if is_stop:
            return [EXPECTED_HABITAT_ACTIONS["stop"]]

        if abs(dy) >= self.linear_deadband:
            raise AssertionError(
                "UniWM action requests lateral movement, but the current Habitat InstanceImageNav "
                f"action space does not support strafing: {action_text!r}"
            )
        if dx <= -self.linear_deadband:
            raise AssertionError(
                "UniWM action requests backward movement, but the current Habitat InstanceImageNav "
                f"action space does not support it: {action_text!r}"
            )

        all_actions: list[str] = []
        # Add a forward movement
        if dx >= self.linear_deadband:
            forward_action = EXPECTED_HABITAT_ACTIONS["forward"]
            all_actions.append(forward_action)

        # TODO: Replace these hacky approaches for backward and strafe movements with updating Habitat movement options directly, or better restrict UniWM to not allow these options.
        # Add a backward movement
        elif dx <= -self.linear_deadband:
            # Turn 180 degrees, move forward once, then turn back around.
            turn_around = [EXPECTED_HABITAT_ACTIONS["right"]] * (self.right_angle_turn_repeats * 2)

            all_actions.extend(turn_around)
            all_actions.append(EXPECTED_HABITAT_ACTIONS["forward"])
            all_actions.extend(turn_around)

        # Add a strafe movement
        if abs(dy) >= self.linear_deadband:
            # Turn 90 degrees in direction of strafe, move forward once, then turn 90 degrees in opposite direction of strafe
            turns_order = (EXPECTED_HABITAT_ACTIONS["left"], EXPECTED_HABITAT_ACTIONS["right"]) if dy > 0 else (EXPECTED_HABITAT_ACTIONS["right"], EXPECTED_HABITAT_ACTIONS["left"])

            turn_first = [turns_order[0]] * self.right_angle_turn_repeats
            turn_second = [turns_order[1]] * self.right_angle_turn_repeats

            all_actions.extend(turn_first)
            all_actions.append(EXPECTED_HABITAT_ACTIONS["forward"])
            all_actions.extend(turn_second)

        # Add a turn movement
        if abs(dyaw) >= self.angular_deadband:
            turn_action = EXPECTED_HABITAT_ACTIONS["left"] if dyaw > 0 else EXPECTED_HABITAT_ACTIONS["right"]
            all_actions.append(turn_action)

        return all_actions

    def convert_observation(
        self,
        *,
        output: HabitatOutputBundle
    ) -> UniWMInputBundle:
        start_rgb = output.start_obs[self.RGB_KEY]
        goal_image = output.start_obs[self.GOAL_KEY]
        current_rgb = output.current_obs[self.RGB_KEY]
        bundle_metadata: dict[str, object] = dict(output.metadata)

        bundle_metadata.update({
            "done": output.done,
            "step_index": output.step_index,
            "source_mode": output.source_mode,
            "action_taken": output.action_taken,
            "metrics": dict(output.metrics)
        })

        return UniWMInputBundle(
            start_observation=self._to_pil_image(start_rgb),
            goal_observation=self._to_pil_image(goal_image),
            current_observation=self._to_pil_image(current_rgb),
            start_pose_str=self.extract_start_pose(output.episode),
            metadata=bundle_metadata,
        )

    def extract_start_pose(self, episode: InstanceImageGoalNavEpisode | Episode) -> str:
        position: list[float] = episode.start_position
        rotation: list[float] = episode.start_rotation

        return self.START_POSE_TEMPLATE.format(
            x=float(position[0]),
            y=float(position[2]),
            yaw=self.yaw_from_rotation(rotation),
        )

    @classmethod
    def yaw_from_rotation(cls, rotation: list[float]) -> float:
        if len(rotation) != 4:
            raise AssertionError(
                f"Expected start_rotation as a quaternion with 4 values, got {rotation}"
            )

        x, y, z, w = rotation
        return float(math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))

    def _to_pil_image(self, image: Image.Image | torch.Tensor | np.ndarray) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert(self.IMAGE_MODE).resize(self.image_size)

        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.dim() != 3:
                raise AssertionError(f"Expected image tensor with 3 dims, got shape {tuple(tensor.shape)}")
            if tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            image = tensor.numpy()

        array: np.ndarray = image
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
        return Image.fromarray(array, mode=self.IMAGE_MODE).resize(self.image_size)