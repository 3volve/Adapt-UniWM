from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from PIL import Image
from habitat.core.simulator import Observations
from habitat.core.dataset import Episode
from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalNavEpisode
from runtime_scripts.uniwm_schemas import UniWMInputBundle


#------------ Abstract Classes ------------#
@dataclass(frozen=True, kw_only=True)
class OutputBundle:
    source_mode: str = "unknown"
    done: bool = False
    episode_id: str = "-1"
T_OutputBundle = TypeVar("T_OutputBundle", bound=OutputBundle)

class SourceAdapter(ABC, Generic[T_OutputBundle]):
    """Small environment/replay adapter interface for the episode manager."""

    source_mode = "unknown"

    @abstractmethod
    def reset_ep(self) -> T_OutputBundle:
        pass 
    
    @abstractmethod
    def reset_src(self, data_id: str) -> None:
        pass 

    @abstractmethod
    def step(self, action: str) -> T_OutputBundle:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
T_Adapter = TypeVar("T_Adapter", bound=SourceAdapter)

class SourceFormatter(ABC, Generic[T_OutputBundle]):
    """Small environment/replay converter interface for the episode manager."""

    source_mode = "unknown"

    @abstractmethod
    def convert_action(self, action: str) -> list[str]:
        pass
    
    @abstractmethod
    def convert_observation(self, output: T_OutputBundle) -> UniWMInputBundle:
        pass
T_Formatter = TypeVar("T_Formatter", bound=SourceFormatter)


#------------ Datasource-Specific Classes ------------#
@dataclass(frozen=True, kw_only=True)
class HabitatOutputBundle(OutputBundle):
    source_mode: str = "habitat"

    start_obs: Observations
    current_obs: Observations
    metrics: Mapping[str, object]
    episode: InstanceImageGoalNavEpisode | Episode
    step_index: int
    action_taken: str | None
    metadata: Mapping[str, object] = field(default_factory=dict)

@dataclass(frozen=True, kw_only=True)
class ReplayOutputBundle(OutputBundle):
    source_mode: str = "replay"

    start_observation: Image.Image
    goal_observation: Image.Image
    current_observation: Image.Image
    start_pose: list[float]
    step_index: int
    action_taken: list[float] | None
    metadata: Mapping[str, object] = field(default_factory=dict)
