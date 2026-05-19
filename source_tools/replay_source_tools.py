from __future__ import annotations

from scripts.action_utils import extract_bin_values
from runtime_scripts.uniwm_schemas import UniWMInputBundle
from runtime_scripts.datasource_schemas import SourceFormatter, SourceAdapter, ReplayOutputBundle


class ReplayTrajectoryAdapter(SourceAdapter):
    """Thin Replay environment adapter.

    Responsibilities:
    - build/close offline dataset env
    - reset offline dataset env to next trajectory
    - step offline dataset env with dataset-friendly actions
    - return ReplayOutputBundle
    """

    source_mode = "replay"

    def __init__(
        self
    ):
        raise NotImplemented

    def reset(self) -> ReplayOutputBundle:
        return NotImplemented

    def step(self, habitat_action: str) -> ReplayOutputBundle:

        return NotImplemented

    def close(self) -> None:
        raise NotImplemented

    def _pack_step(
        self,
    ) -> ReplayOutputBundle:
        step = ReplayOutputBundle()

        self.last_step = step
        return step


class ReplayUniWMFormatter(SourceFormatter):
    """Strict converter between Habitat output bundles and UniWM input bundles."""

    source_mode = "replay"

    def __init__(self, bin_step: float = 0.01, linear_deadband: float = 0.02, angular_deadband: float = 0.02, image_h: int = 256, image_w: int = 256):
        self.bin_step: float = float(bin_step)
        self.linear_deadband: float = float(linear_deadband)
        self.angular_deadband: float = float(angular_deadband)
        self.image_size: tuple[int, int] = (int(image_w), int(image_h))

    def convert_action(self, action_text: str) -> list[str]:
        dx, dy, dyaw, action_text = 0.0, 0.0, 0.0, action_text.strip()
        is_stop = action_text.lower() == "stop"

        dx: float = extract_bin_values(action_text, "dx", self.bin_step)
        dy: float = extract_bin_values(action_text, "dy", self.bin_step)
        dyaw: float = extract_bin_values(action_text, "dyaw", self.bin_step)

        all_actions: list[str] = []
        return all_actions

    def convert_observation(
        self,
        output: ReplayOutputBundle
    ) -> UniWMInputBundle:
        # start_rgb = output.start_obs[self.RGB_KEY]
        # goal_image = output.start_obs[self.GOAL_KEY]
        # current_rgb = output.current_obs[self.RGB_KEY]
        # bundle_metadata: dict[str, object] = dict(output.metadata)
        #
        # bundle_metadata.update({
        #     "done": output.done,
        #     "step_index": output.step_index,
        #     "source_mode": output.source_mode,
        #     "action_taken": output.action_taken,
        #     "metrics": dict(output.metrics)
        # })

        return UniWMInputBundle(
            # start_observation=self._to_pil_image(start_rgb),
            # goal_observation=self._to_pil_image(goal_image),
            # current_observation=self._to_pil_image(current_rgb),
            # start_pose_str=self.extract_start_pose(output.episode),
            # metadata=bundle_metadata,
        )