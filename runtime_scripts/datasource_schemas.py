from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar
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
    def reset_ep(self) -> list[T_OutputBundle]:
        pass 
    
    @abstractmethod
    def reset_src(self, data_id: str) -> None:
        pass 

    @abstractmethod
    def step(self, actions: list[str]) -> list[T_OutputBundle]:
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
    def convert_from_source(self, outputs: list[T_OutputBundle]) -> UniWMInputBundle:
        pass
T_Formatter = TypeVar("T_Formatter", bound=SourceFormatter)
