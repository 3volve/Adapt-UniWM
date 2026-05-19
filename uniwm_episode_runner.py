from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

from runtime_scripts.datasource_schemas import SourceAdapter, SourceFormatter, OutputBundle
from runtime_scripts.uniwm_schemas import UniWMInputBundle, TransitionRecord
from runtime_scripts.runtime_utils import is_stop_action, load_config, resolve_config_path_from_id, validate_config
from runtime_scripts.uniwm_wrapper import UniWMWrapper
from runtime_scripts.runtime_engine import UniWMEngine


REQUIRED_FIELDS: list[str] = [
    "max_episode_steps",
    "stop_on_wrapper_done",
    "log_every_step",
    "source_file_name",
    "adapter_params"
]

class UniWMEpisodeRunner:
    """Closed-loop episode coordinator between a wrapper and an adapter."""

    def __init__(
        self,
        data_id: str,
        seed: str = "",
        config_path: str | None = None,
        engine: UniWMEngine | None = None # Mostly for testing purposes
    ) -> None:
        if config_path is None:
            config_path = resolve_config_path_from_id(data_id)

        self.config: dict[str, Any] = load_config(config_path).get("runner", {})
        validate_config(self.config, REQUIRED_FIELDS)

        engine = UniWMEngine(config_path, data_id) if engine is None else engine
        self.wrapper = UniWMWrapper(engine, config_path)

        source_classes = self._load_source_classes(data_id)
        self.adapter: SourceAdapter = source_classes[0]
        self.converter: SourceFormatter = source_classes[1]

        self._episode_logs: list[dict[str, Any]] = []

    def run_episode(self) -> dict[str, Any]:
        # Generate new observation when resetting episode
        reset_result = self.adapter.reset()

        # convert new observation to UniWMInputBundle
        converted_reset = self.converter.convert_observation(reset_result)

        # Pass new observation to the wrapper with a reset_episode command
        wrapper_reset_state = self.wrapper.reset_episode(converted_reset, reset_result.episode_id)

        episode_index = len(self._episode_logs)
        step_logs: list[dict[str, Any]] = []
        termination_reason = "max_episode_steps"
        steps_executed = 0

        # Start running through steps with the returned UniWM Action str to start the loop
        for step_idx in range(self.config["max_episode_steps"]):
            # Retrieve the predicted next action from wrapper
            planned_action: str = self.wrapper.get_next_action()

            # Convert returned actions list[str] to source-friendly version
            converted_actions: list[str] = self.converter.convert_action(planned_action)

            # Pass new actions one at a time to the source adapter
            step_result = OutputBundle(episode_id="", done=False)
            for action in converted_actions:
                step_result: OutputBundle = self.adapter.step(action)

            # Convert adapter output obs to UniWMInputBundle
            converted_obs: UniWMInputBundle = self.converter.convert_observation(step_result)

            # Give new obs state to wrapper to update its state
            transition: TransitionRecord = self.wrapper.observe_transition(converted_obs)

            steps_executed = step_idx + 1
            wrapper_requested_stop = is_stop_action(planned_action)

            if self.config["log_every_step"]:
                step_logs.append(
                    {
                        "episode_index": episode_index,
                        "step_idx": step_idx,
                        "action_text": planned_action,
                        "wrapper_requested_stop": wrapper_requested_stop,
                        "divergence": transition.divergence,
                        "replanned": transition.replanned,
                        "replan_reason": transition.replan_reason,
                        "route_generation": self.wrapper.get_state_snapshot()["route_generation"],
                        "adapter_info": dict(converted_obs.metadata),
                    }
                )

            if step_result.done:
                termination_reason = "adapter_done"
                break
            if wrapper_requested_stop and self.config["stop_on_wrapper_done"]:
                termination_reason = "wrapper_stop_action"
                break

        episode_log = {
            "episode_index": episode_index,
            "episode_id": reset_result.episode_id,
            "adapter_source_mode": self.adapter.source_mode,
            "max_episode_steps": self.config["max_episode_steps"],
            "steps_executed": steps_executed,
            "termination_reason": termination_reason,
            "reset_info": converted_reset.metadata,
            "wrapper_reset_state": wrapper_reset_state,
            "steps": step_logs,
            "final_wrapper_state": self.wrapper.get_state_snapshot(),
            "wrapper_log": self.wrapper.get_episode_log(),
        }

        self._episode_logs.append(episode_log)
        return episode_log

    def run_episodes(self, num_episodes: int) -> list[dict[str, Any]]:
        # TODO: Fix run_episodes to intelligently target the number of episodes for a specific target data source type somehow and run all available episodes if passed a -1 for num_episodes
        return [self.run_episode() for _ in range(num_episodes)]

    def get_logs(self) -> list[dict[str, Any]]:
        return list(self._episode_logs)

    def _load_source_classes(self, data_id: str) -> tuple[SourceAdapter, SourceFormatter]:
        source_tools_name = self.config.get("source_file_name")

        # Source-tools file default naming should be an allowed simplification
        if source_tools_name is None:
            source_tools_name = f"{data_id.lower()}_source_tools"

        file_path = Path(__file__).resolve().parent / "source_tools" / f"{source_tools_name}.py"
        if not file_path.is_file():
            raise FileNotFoundError(f"Unable to find adapter file from environment config path '{file_path}'")

        module_name = f"source_tools.{source_tools_name}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)

        if spec is None or spec.loader is None:
            raise AssertionError(f"Unable to load adapter module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        adapter_class_name = f"{data_id.capitalize()}EpisodeAdapter"
        adapter_cls = getattr(module, adapter_class_name, None)

        if adapter_cls is None:
            raise AssertionError(f"Unable to find expected adapter class {adapter_class_name} from environment config path '{file_path}'")

        formatter_class_name = f"{data_id.capitalize()}UniWMFormatter"
        formatter_cls = getattr(module, formatter_class_name, None)

        if formatter_cls is None:
            raise AssertionError(f"Unable to find expected formatter class {formatter_class_name} from environment config path '{file_path}'")

        return (
            adapter_cls(**self.config["adapter_params"]),
            # TODO: I need to somehow get the habitat config or the relevant info from it to the formatter
            formatter_cls(**self.config["formatter_params"])
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_id", type=str, default="habitat")
    parser.add_argument("--num_episodes", type=int, default=-1)
    parser.add_argument("--episodes_seed", type=str, default="hab0")
    args = parser.parse_args()

    runner = UniWMEpisodeRunner(data_id=args.data_id, seed=args.episodes_seed)
    runner.run_episodes(num_episodes=args.num_episodes)