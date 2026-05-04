from __future__ import annotations

from uniwm_episode_runner import EpisodeAdapter
from scripts.habitat_uniwm_schemas import UniWMInputBundle


class HabitatEpisodeAdapter(EpisodeAdapter):
    """Habitat adapter for running episodes via UniWMEpisodeRunner."""

    source_mode = "dummy"

    def __init__(
        self
    ) -> None:
        super().__init__()

    def reset(self) -> UniWMInputBundle:
        return NotImplemented

    def step(self, action_text: str) -> UniWMInputBundle:
        return NotImplemented