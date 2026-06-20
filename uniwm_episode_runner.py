from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Generic

from runtime_scripts.datasource_schemas import T_OutputBundle, T_Adapter, T_Formatter
from runtime_scripts.uniwm_schemas import UniWMInputBundle, TransitionRecord
from runtime_scripts.uniwm_wrapper import UniWMWrapper
from runtime_scripts.runtime_engine import UniWMEngine
from runtime_scripts.runtime_utils import (
    is_stop_action,
    load_config,
    resolve_config_path_from_id,
    validate_config,
    make_runner_output_dir
)

from runtime_scripts.test_runtime_metrics import save_runner_logs


REQUIRED_FIELDS: list[str] = [
    "max_episode_steps",
    "stop_on_wrapper_done",
    "log_every_step",
    "source_file_name",
    "adapter_params"
]

class UniWMEpisodeRunner(Generic[T_OutputBundle, T_Adapter, T_Formatter]):
    """Closed-loop episode coordinator between a wrapper and an adapter."""

    def __init__(
        self,
        data_type: str,
        data_id: str,
        full_output_path: Path,
        *,
        config_path: str | None = None,
        engine: UniWMEngine | None = None # Mostly for testing purposes
    ) -> None:
        if config_path is None:
            config_path = resolve_config_path_from_id(data_type)

        config = load_config(config_path)
        self.config: dict[str, Any] = config.get("runner", {})
        
        # Need to normalize these two to ensure proper generation and conversion
        
        if self.config["formatter_params"].get("img_size", False):
            self.config["formatter_params"]["img_size"] = config["engine"]["load_model_cfg"]["img_size"]
        if self.config["formatter_params"].get("bin_step", False):
            self.config["formatter_params"]["bin_step"] = config["engine"]["action_token_generation"]["bin_step"]
            
        validate_config(self.config, REQUIRED_FIELDS)

        self.wrapper = UniWMWrapper(
            UniWMEngine(data_id, config_path) if engine is None else engine,
            config_path,
            str(full_output_path)
        )

        source_classes = self._load_source_classes(data_type)
        self.adapter: T_Adapter = source_classes[0]
        self.formatter: T_Formatter = source_classes[1]

        self._episode_logs: list[dict[str, Any]] = []

    def run_episode(self, data_id: str) -> dict[str, Any]:
        print("[RUNNER]: Starting New Episode")
        # Generate new observation when resetting episode
        step_results: list[T_OutputBundle] = self.adapter.reset_ep()

        # convert new observation to UniWMInputBundle
        converted_obs: UniWMInputBundle = self.formatter.convert_from_source(step_results)

        # Pass new observation to the wrapper with a reset_episode command
        wrapper_reset_state: dict[str, Any] = self.wrapper.reset_episode(converted_obs, step_results[0].episode_id)

        conv_info = converted_obs.metadata
        step_logs: list[dict[str, Any]] = []
        termination_reason = "max_episode_steps"
        steps_executed = 0

        # Start running through steps with the returned UniWM Action str to start the loop
        for step_idx in range(self.config["max_episode_steps"]):
            print(f"[RUNNER]: Starting New Step #[{step_idx}]")
            # Retrieve the predicted next action from wrapper
            planned_action: str = self.wrapper.get_next_action()

            # Convert returned actions list[str] to source-friendly version
            converted_actions: list[str] = self.formatter.convert_action(planned_action)

            # Assuming there are any valid actions returned...
            if len(converted_actions) > 0:
                step_results = self.adapter.step(converted_actions)

                # Convert adapter output obs to UniWMInputBundle
                converted_obs = self.formatter.convert_from_source(step_results)

            # Give new obs state to wrapper to update its state
            transition: TransitionRecord = self.wrapper.observe_transition(converted_obs)

            steps_executed = step_idx + 1
            wrapper_requested_stop = is_stop_action(planned_action)

            if self.config["log_every_step"]:
                step_logs.append(
                    {
                        "step_idx": step_idx,
                        **transition.to_log(),
                        "wrapper_requested_stop": wrapper_requested_stop,
                    }
                )

            if converted_obs.source_done:
                termination_reason = "adapter_done"
                break
            if wrapper_requested_stop and self.config["stop_on_wrapper_done"]:
                termination_reason = "wrapper_stop_action"
                break

        episode_log = {
            "episode_index": len(self._episode_logs),
            "episode_id": step_results[0].episode_id,
            "adapter_source_mode": self.adapter.source_mode,
            "max_episode_steps": self.config["max_episode_steps"],
            "steps_executed": steps_executed,
            "termination_reason": termination_reason,
            "reset_info": conv_info,
            "wrapper_reset_state": wrapper_reset_state,
            "routes": self.wrapper.get_routes_log_for_episode(),
            "steps": step_logs,
            "final_wrapper_state": self.wrapper.get_state_snapshot(),
        }

        self._episode_logs.append(episode_log)
        return episode_log

    def run_episodes(self, num_episodes: int, data_id: str):
        """ Leaving in code to parse multiple data_ids, but the system can't actually properly swap action_cfgs between types. Don't try to run with multiple. """
        if num_episodes == -1:
            num_episodes = self.config["source_max_episodes"]
        
        data_ids = [data_id]
        if ',' in data_id:
            raise NotImplementedError("[UNFINISHED_IMPLEMENTATION] System can't actually properly swap action_cfgs between types yet. Please run them one at a time instead.")
            data_ids = [id.strip() for id in data_id.split(',')]
            
        for id in data_ids:
            self.adapter.reset_src(id)
            [self.run_episode(id) for _ in range(num_episodes)]

    def get_logs(self) -> list[dict[str, Any]]:
        return list(self._episode_logs)

    def _load_source_classes(self, data_type: str) -> tuple[T_Adapter, T_Formatter]:
        source_tools_name = self.config.get("source_file_name")

        # Source-tools file default naming should be an allowed simplification
        if source_tools_name is None:
            source_tools_name = f"{data_type.lower()}_source_tools"

        file_path = Path(__file__).resolve().parent / "source_tools" / f"{source_tools_name}.py"
        if not file_path.is_file():
            raise FileNotFoundError(f"Unable to find adapter file from environment config path '{file_path}'")

        module_name = f"source_tools.{source_tools_name}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)

        if spec is None or spec.loader is None:
            raise AssertionError(f"Unable to load adapter module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        adapter_class_name = f"{data_type.capitalize()}EpisodeAdapter"
        adapter_cls = getattr(module, adapter_class_name, None)

        if adapter_cls is None:
            raise AssertionError(f"Unable to find expected adapter class {adapter_class_name} from environment config path '{file_path}'")

        formatter_class_name = f"{data_type.capitalize()}UniWMFormatter"
        formatter_cls = getattr(module, formatter_class_name, None)
        if formatter_cls is None:
            raise AssertionError(f"Unable to find expected formatter class {formatter_class_name} from environment config path '{file_path}'")

        adapter: T_Adapter = adapter_cls(**self.config["adapter_params"])
        formatter: T_Formatter = formatter_cls(adapter, **self.config["formatter_params"])
        return adapter, formatter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="habitat")
    
    # For now, I'm just going to say only this will only work with a single data_id at a time.  I'll deal with splitting and rebuilding the runner per-data_id later if it seems necessary.
    parser.add_argument("--data_id", type=str, default="habitat")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--num_episodes", type=int, default=-1)
    args = parser.parse_args()
    
    run_dir = make_runner_output_dir(args.output_dir, args.data_id)

    runner = UniWMEpisodeRunner(args.data_type, args.data_id, run_dir)
    runner.run_episodes(args.num_episodes, args.data_id)

    save_runner_logs(runner.get_logs(), run_dir)
