from __future__ import annotations

from typing import Optional, Sequence, Tuple, Any

import numpy as np
from PIL import Image

from uniwm_episode_runner import EpisodeAdapter
from scripts.habitat_uniwm_schemas import UniWMInputBundle
from scripts.uniwm_inference_utils import is_stop_action


class DummyEpisodeAdapter(EpisodeAdapter):
    """Minimal adapter for smoke-testing the closed-loop manager without Habitat."""

    source_mode = "dummy"

    def __init__(
        self,
        observation_levels: Optional[Sequence[int]] = None,
        *,
        start_level: int = 0,
        goal_level: int = 255,
        image_size: Tuple[int, int] = (32, 32),
        episode_id: str = "dummy_episode",
        start_pose_str: str = "Starting Point Coordinate: x=0.000, y=0.000, yaw=0.000\n",
    ) -> None:
        self.observation_levels = list(observation_levels or [32, 255])
        self.start_level = int(start_level)
        self.goal_level = int(goal_level)
        self.image_size = tuple(image_size)
        self.episode_id = episode_id
        self.start_pose_str = start_pose_str
        self._current_level = self.start_level
        self._adapter_step_idx = 0
        self._last_action_text: Optional[str] = None

    def reset(self) -> UniWMInputBundle:
        self._current_level = self.start_level
        self._adapter_step_idx = 0
        self._last_action_text = None
        return self._bundle(self._current_level, info={
            "episode_id": self.episode_id,
            "source_mode": self.source_mode,
            "observation_source": "real",
            "adapter_step_idx": self._adapter_step_idx,
        })

    def step(self, action_text: str) -> UniWMInputBundle:
        action_text = str(action_text)
        self._last_action_text = action_text

        if not is_stop_action(action_text) and self._adapter_step_idx < len(self.observation_levels):
            self._current_level = int(self.observation_levels[self._adapter_step_idx])
            self._adapter_step_idx += 1

        done = is_stop_action(action_text)
        return self._bundle(self._current_level, action_text=action_text, info={
            "episode_id": self.episode_id,
            "source_mode": self.source_mode,
            "observation_source": "real",
            "adapter_step_idx": self._adapter_step_idx,
            "received_action_text": action_text,
        })

    def _bundle(self, current_level: int, action_text: Optional[str] = None, info: dict[str, Any] = None) -> UniWMInputBundle:
        return UniWMInputBundle(
            start_observation=self._solid_image(self.start_level),
            goal_observation=self._solid_image(self.goal_level),
            current_observation=self._solid_image(current_level),
            start_pose_str=self.start_pose_str,
            action_text=action_text,
            metadata=info if info else {},
        )

    def _solid_image(self, level: int) -> Image.Image:
        width, height = self.image_size
        array = np.full((height, width, 3), int(level), dtype=np.uint8)
        return Image.fromarray(array, mode="RGB")

