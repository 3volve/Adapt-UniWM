from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from PIL import Image

from habitat.core.simulator import Observations
from habitat.core.dataset import Episode
from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalNavEpisode

#------------ Abstract Classes ------------#
class SourceAdapter:
    """Small environment/replay adapter interface for the episode manager."""

    source_mode = "unknown"

    @classmethod
    def reset(cls) -> OutputBundle:
        pass

    @classmethod
    def step(cls, action_text: str) -> OutputBundle:
        pass

    @classmethod
    def close(cls) -> None:
        pass

class SourceFormatter:
    """Small environment/replay converter interface for the episode manager."""

    source_mode = "unknown"

    @classmethod
    def convert_action(cls, action: str) -> list[str]:
        pass

    @classmethod
    def convert_observation(cls, output: OutputBundle) -> UniWMInputBundle:
        pass

@dataclass(frozen=True)
class OutputBundle:
    source_mode = "unknown"



#------------ UniWM-Related Classes ------------#
@dataclass(frozen=True)
class UniWMInputBundle:
    start_observation: Image.Image
    goal_observation: Image.Image
    current_observation: Image.Image
    start_pose_str: str
    action_text: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def unpack(self) -> tuple[Image.Image, Image.Image, Image.Image, str]:
        return (
            self.start_observation,
            self.goal_observation,
            self.current_observation,
            self.start_pose_str,
        )

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



#------------ Habitat-Related Classes ------------#
@dataclass(frozen=True)
class HabitatOutputBundle(OutputBundle):
    super.source_mode = "habitat"

    start_obs: Observations
    current_obs: Observations
    done: bool
    metrics: Mapping[str, object]
    episode: InstanceImageGoalNavEpisode | Episode
    step_index: int
    action_taken: str | None
    metadata: Mapping[str, object] = field(default_factory=dict)