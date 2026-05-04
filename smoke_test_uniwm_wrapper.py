from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

from scripts.habitat_uniwm_schemas import UniWMInputBundle
from scripts.uniwm_wrapper import UniWMWrapper
from scripts.uniwm_inference_utils import load_config


@dataclass(frozen=True)
class StubStepPrediction:
    action_text: str
    visualization: Optional[Image.Image] = None


@dataclass(frozen=True)
class StubRoutePrediction:
    steps: List[StubStepPrediction]
    stopped: bool
    stop_reason: str


class StubModel:
    def __init__(self) -> None:
        self.reset_memory_calls = 0
        self.reset_global_memory_calls = 0

    def reset_memory_bank(self) -> None:
        self.reset_memory_calls += 1

    def reset_global_memory_bank(self) -> None:
        self.reset_global_memory_calls += 1


class StubEngine:
    def __init__(self, config_path: str = "cfg/habitat_uniwm_cfg.yaml") -> None:
        self.config = load_config(config_path)
        self.model = StubModel()
        self.predict_route_calls = []

    def predict_route(self, bundle, max_steps=None, output_dir=None):
        del max_steps, output_dir
        self.predict_route_calls.append(bundle)
        call_idx = len(self.predict_route_calls)
        if call_idx == 1:
            return StubRoutePrediction(
                steps=[
                    StubStepPrediction(
                        action_text="Move by dx: <dx_pos_bin_25>, dy: <dy_pos_bin_00>, dyaw: <dyaw_pos_bin_00>",
                        visualization=_solid_image(32),
                    ),
                    StubStepPrediction(
                        action_text="Move by dx: <dx_pos_bin_25>, dy: <dy_pos_bin_00>, dyaw: <dyaw_pos_bin_00>",
                        visualization=_solid_image(64),
                    ),
                ],
                stopped=False,
                stop_reason="max_steps",
            )
        return StubRoutePrediction(
            steps=[StubStepPrediction(action_text="stop", visualization=None)],
            stopped=True,
            stop_reason="stop_action",
        )


def _solid_image(level: int) -> Image.Image:
    return Image.fromarray(np.full((32, 32, 3), level, dtype=np.uint8), mode="RGB")


def _obs(level: int) -> np.ndarray:
    return np.full((32, 32, 3), level, dtype=np.uint8)


def _bundle(level: int, env_info=None) -> UniWMInputBundle:
    env_info = {} if env_info is None else env_info
    image = Image.fromarray(_obs(level), mode="RGB")
    env_info["episode_id"] = "smoke"
    return UniWMInputBundle(
        start_observation=Image.fromarray(_obs(0), mode="RGB"),
        goal_observation=Image.fromarray(_obs(255), mode="RGB"),
        current_observation=image,
        start_pose_str="Starting Point Coordinate: x=0.000, y=0.000, yaw=0.000\n",
        action_text=None,
        metadata=env_info
    )


def main() -> None:
    engine = StubEngine()
    wrapper = UniWMWrapper()

    snapshot = wrapper.reset_episode(_bundle(0))
    assert snapshot["route_length"] == 2
    assert engine.model.reset_memory_calls == 1
    assert engine.model.reset_global_memory_calls == 1

    action_1 = wrapper.get_next_action()
    assert action_1 == "Move by dx: <dx_pos_bin_25>, dy: <dy_pos_bin_00>, dyaw: <dyaw_pos_bin_00>"

    record_1 = wrapper.observe_transition(_bundle(32, env_info={"step": 1}))
    assert record_1.divergence == 0.0
    assert record_1.replanned is False

    action_2 = wrapper.get_next_action()
    assert action_2 == "Move by dx: <dx_pos_bin_25>, dy: <dy_pos_bin_00>, dyaw: <dyaw_pos_bin_00>"

    record_2 = wrapper.observe_transition(_bundle(255, env_info={"step": 2}))
    assert record_2.divergence > wrapper.config["full_replan_threshold"]
    assert record_2.replanned is True
    assert len(engine.predict_route_calls) == 2
    assert wrapper.get_state_snapshot()["route_generation"] == 2

    next_action = wrapper.get_next_action()
    assert next_action == "stop"

    episode_log = wrapper.get_episode_log()
    assert len(episode_log["transitions"]) == 2
    assert len(episode_log["route_history"]) == 2

    print("uniwm_wrapper smoke test passed")


if __name__ == "__main__":
    main()
