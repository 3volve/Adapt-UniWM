from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from habitat.core.simulator import Observations
from habitat.core.dataset import Episode
from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalNavEpisode
from runtime_scripts.uniwm_schemas import UniWMInputBundle


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
    done: bool
    episode_id: str
    source_mode = "unknown"


#------------ Datasource-Specific Classes ------------#
@dataclass(frozen=True)
class HabitatOutputBundle(OutputBundle):
    super.source_mode = "habitat"
    super.episode_id = -1
    super.done = False

    start_obs: Observations
    current_obs: Observations
    metrics: Mapping[str, object]
    episode: InstanceImageGoalNavEpisode | Episode
    step_index: int
    action_taken: str | None
    metadata: Mapping[str, object] = field(default_factory=dict)