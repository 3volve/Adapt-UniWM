from __future__ import annotations

from dataclasses import dataclass, field

import torch
from PIL import Image

@dataclass(frozen=True)
class UniWMInputBundle:
    start_observation: Image.Image
    goal_observation: Image.Image
    current_observation: Image.Image
    start_pose_str: str
    action_text: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def unpack(self) -> tuple[Image.Image, Image.Image, Image.Image, str, str | None]:
        return (
            self.start_observation,
            self.goal_observation,
            self.current_observation,
            self.start_pose_str,
            self.action_text
        )

@dataclass()
class StepPrediction:
    input_bundle: UniWMInputBundle
    action_text: str
    raw_act: str
    visualization: Image.Image | None
    viz_token_ids: torch.LongTensor | None

    real_viz_token_ids: torch.LongTensor | None

@dataclass(frozen=True)
class RoutePrediction:
    steps: list[StepPrediction]
    stopped: bool
    stop_reason: str

    def __len__(self):
        return len(self.steps)

@dataclass
class TransitionRecord:
    step_idx: int
    action: str
    predicted_obs: Image.Image | None
    real_obs: Image.Image | None
    divergence: float
    replanned: bool
    replan_reason: str | None
    env_info: dict | None = None

@dataclass
class RouteRecord:
    route_generation: int
    reason: str
    stopped: bool
    stop_reason: str
    step_count: int
    action_outputs: list[str]
    predicted_observations: list[Image.Image | None] = field(default_factory=list)