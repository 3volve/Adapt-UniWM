from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

@dataclass(frozen=True)
class UniWMInputBundle:
    start_observation: Image.Image
    goal_observation: Image.Image
    current_observation: Image.Image
    start_pose_str: str
    action_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def unpack(self) -> tuple[Image.Image, Image.Image, Image.Image, str, str | None]:
        return (
            self.start_observation,
            self.goal_observation,
            self.current_observation,
            self.start_pose_str,
            self.action_text
        )

@dataclass(frozen=True)
class MemorySnapshot:
    current_step: int
    stored_keys: tuple[torch.Tensor | None, ...]
    stored_values: tuple[torch.Tensor | None, ...]
    context_ema: torch.Tensor | None

@dataclass()
class StepPrediction:
    input_bundle: UniWMInputBundle

    action_text: str
    visualization: Image.Image | None
    
    act_entropy: float
    viz_entropy: float
    context_familiarity: float
    context_stability: float

    real_input_obs: Image.Image | None = None
    real_next_obs: Image.Image | None = None

@dataclass(frozen=True)
class RoutePrediction:
    steps: list[StepPrediction]
    stopped: bool
    stop_reason: str

    def __len__(self):
        return len(self.steps)

@dataclass
class TransitionRecord:
    route_id: int
    route_idx: int
    action: str
    context_familiarity: float
    context_stability: float
    divergence: float
    replanned: bool
    replan_reason: str | None
    modulator_state: dict[str, Any] | None = None
    training_logs: dict[str, Any] | None = None
    env_info: dict[str, Any] | None = None

    def to_log(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "route_idx": self.route_idx,
            "action": self.action,
            "context_familiarity": self.context_familiarity,
            "context_stability": self.context_stability,
            "divergence": self.divergence,
            "replanned": self.replanned,
            "replan_reason": self.replan_reason,
            "modulator_state": None if self.modulator_state is None else dict(self.modulator_state),
            "training_logs": None if self.training_logs is None else dict(self.training_logs),
            "env_info": None if self.env_info is None else dict(self.env_info),
        }

@dataclass
class RouteRecord:
    route_id: int
    replan_reason: str
    stop_reason: str
    planned_step_count: int
    planned_actions: list[str]

    def to_log(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "replan_reason": self.replan_reason,
            "stop_reason": self.stop_reason,
            "planned_step_count": self.planned_step_count,
            "planned_actions": list(self.planned_actions)
        }
