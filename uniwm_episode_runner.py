from __future__ import annotations

import argparse, importlib.util, sys
from pathlib import Path
from typing import Any, Generic

from runtime_scripts.datasource_schemas import T_OutputBundle, T_Adapter, T_Formatter
from runtime_scripts.uniwm_schemas import UniWMInputBundle, TransitionRecord
from runtime_scripts.uniwm_wrapper import UniWMWrapper
from runtime_scripts.run_metadata import runtime_metadata
from runtime_scripts.event_logger import EventLogger
from runtime_scripts.runtime_engine import UniWMEngine
from runtime_scripts.runtime_utils import (
    event_values,
    image_validity,
    copy_base_config,
    is_stop_action,
    load_config,
    validate_config,
    make_runner_output_dir
)

REQUIRED_FIELDS: list[str] = [
    "max_episode_steps",
    "stop_on_wrapper_done",
    "source_file_name",
    "adapter_params",
    "save_model_weights",
]

class UniWMEpisodeRunner(Generic[T_OutputBundle, T_Adapter, T_Formatter]):
    """Closed-loop episode coordinator between a wrapper and an adapter."""

    def __init__(
        self,
        config_path: str,
        data_id: str,
        full_output_path: Path,
        *,
        event_logger: EventLogger,
        engine: UniWMEngine | None = None # Mostly for testing purposes
    ) -> None:
        self.event_logger = event_logger
        config = load_config(config_path)
        copy_base_config(config_path, full_output_path)
        self.config: dict[str, Any] = config.get("runner", {})
        engine = UniWMEngine(data_id, config_path, event_logger=event_logger) if engine is None else engine
        
        # Need to normalize these two to ensure proper generation and conversion
        
        validate_config(self.config, REQUIRED_FIELDS)
        self.full_output_path = full_output_path

        self.wrapper = UniWMWrapper(
            engine,
            config_path,
            str(full_output_path), event_logger=event_logger
        )

        source_classes = self._load_source_classes(
            self.config["source_type"],
            engine.action_vocabulary.bin_step,
            int(config["engine"]["load_model_cfg"]["img_size"])
        )
        
        self.adapter: T_Adapter = source_classes[0]
        self.formatter: T_Formatter = source_classes[1]

        self.episode_index = 0

    def run_episode(self, data_id: str) -> dict[str, Any]:
        log = self.event_logger
        episode_index = self.episode_index
        setup_key = f"{data_id}/{episode_index}"
        log.feed({"episode_setup": {setup_key: {"outcome": "unfinished"}}})
        step_results = self.adapter.reset_ep()
        episode_id = str(step_results[0].episode_id)
        converted_obs = self.formatter.convert_from_source(step_results)
        reset_state = self.wrapper.reset_episode(converted_obs, episode_id)
        log.feed({"episode_setup": {setup_key: {"outcome": "completed", "episode_id": episode_id,
            "environment": event_values(converted_obs.metadata), "wrapper": reset_state}}})
        termination_reason = "max_episode_steps"
        steps_executed = 0
        attempts = 0
        consecutive_no_ops = 0
        for step_idx in range(self.config["max_episode_steps"]):
            log.next_step({"data_id": data_id, "episode_id": episode_id,
                "episode_index": episode_index, "step_idx": step_idx,
                "source_mode": self.adapter.source_mode, "transition": {"outcome": "unfinished"}})
            attempts += 1
            planned_action = self.wrapper.get_next_action()
            wrapper_requested_stop = is_stop_action(planned_action)
            valid, reason = image_validity(converted_obs.current_observation)
            log.feed({"action": {"requested": planned_action, "stop": wrapper_requested_stop},
                "transition": {"route_id": self.wrapper.route_id, "route_idx": self.wrapper.pending_step_idx},
                "observation": {"valid": valid, "invalid_reason": reason},
                "formatter": {"action_conversion": "unfinished"}})
            converted_actions = self.formatter.convert_action(planned_action)
            log.feed({"formatter": {"action_conversion": "completed", "converted_actions": converted_actions}})
            if not converted_actions:
                consecutive_no_ops += 1
                if consecutive_no_ops >= 5:
                    termination_reason = "repeated_no_op_actions"
                    log.feed({"outcome": "no_op", "transition": {"outcome": "no_op"}})
                    break
                self.wrapper.replan_route(converted_obs, "Empty converted actions")
                log.feed({"outcome": "no_op", "transition": {"outcome": "no_op"}})
                continue
            consecutive_no_ops = 0
            log.feed({"adapter": {"outcome": "unfinished"}})
            step_results = self.adapter.step(converted_actions)
            log.feed({"adapter": {"outcome": "completed"}, "formatter": {"observation_conversion": "unfinished"}})
            converted_obs = self.formatter.convert_from_source(step_results)
            log.feed({"formatter": {"observation_conversion": "completed"},
                "environment": event_values(converted_obs.metadata)})
            self.wrapper.observe_transition(converted_obs, data_id=data_id, step_idx=step_idx)
            steps_executed += 1
            log.feed({"outcome": "completed", "transition": {"outcome": "completed"}})
            if converted_obs.source_done:
                termination_reason = "adapter_done"
                break
            if wrapper_requested_stop and self.config["stop_on_wrapper_done"]:
                termination_reason = "wrapper_stop_action"
                break
        result = {"episode_id": episode_id, "data_id": data_id, "attempts": attempts,
            "steps_executed": steps_executed, "termination_reason": termination_reason}
        log.feed({"episode_end": {setup_key: result}})
        self.episode_index += 1
        return result

    def run_episodes(self, num_episodes: int, data_id: str, full_output_path: Path, source_episode_counts: dict[str, int] | None = None) -> None:
        if num_episodes == -1:
            num_episodes = self.config["source_max_episodes"]
        for source_id in data_id.split(","):
            self.event_logger.feed({"source_setup": {source_id: {"outcome": "unfinished"}}})
            self.adapter.reset_src(source_id)
            self.event_logger.feed({"source_setup": {source_id: {"outcome": "completed"}}})
            count = source_episode_counts[source_id] if source_episode_counts is not None else num_episodes
            for _ in range(count):
                self.run_episode(source_id)
                key = str(self.episode_index - 1)
                self.event_logger.feed({"schedule_save": {key: {"outcome": "unfinished"}}})
                self.wrapper.save_learning_rate_schedule()
                self.event_logger.feed({"schedule_save": {key: {"outcome": "completed"}}})
            if self.config["save_model_weights"]:
                self.event_logger.feed({"checkpoint": {source_id: {"outcome": "unfinished"}}})
                checkpoint = self.wrapper.engine.save_online_training_state(full_output_path / "final_ckpt")
                self.event_logger.feed({"checkpoint": {source_id: {"outcome": "completed", "path": str(checkpoint)}}})
        self.event_logger.feed({"schedule_finalize": {"outcome": "unfinished"}})
        self.wrapper.finalize_learning_rate_schedule()
        self.event_logger.feed({"schedule_finalize": {"outcome": "completed"}})

    def _load_source_classes(self, data_type: str, bin_step: float, img_size: int) -> tuple[T_Adapter, T_Formatter]:
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
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        adapter_class_name = f"{data_type.capitalize()}EpisodeAdapter"
        adapter_cls = getattr(module, adapter_class_name, None)

        if adapter_cls is None:
            raise AssertionError(f"Unable to find expected adapter class {adapter_class_name} from environment config path '{file_path}'")

        formatter_class_name = f"{data_type.capitalize()}UniWMFormatter"
        formatter_cls = getattr(module, formatter_class_name, None)
        if formatter_cls is None:
            raise AssertionError(f"Unable to find expected formatter class {formatter_class_name} from environment config path '{file_path}'")

        if self.config["adapter_params"].get("bin_step", False):
            self.config["adapter_params"]["bin_step"] = bin_step

        adapter: T_Adapter = adapter_cls(**self.config["adapter_params"], event_logger=self.event_logger)
        formatter: T_Formatter = formatter_cls(bin_step, img_size)
        return adapter, formatter

if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser()    
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--data_id", type=str, default="habitat")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--num_episodes", type=int, default=-1)
    parser.add_argument("--source-episode-counts", type=json.loads,
                        help="Resolved per-dataset episode counts as a JSON object; overrides --num_episodes")
    parser.add_argument("--seed", type=int,
                        help="Seed Python, NumPy and PyTorch before model initialization.")
    args = parser.parse_args()
    if args.source_episode_counts is not None:
        counts = args.source_episode_counts
    if args.seed is not None:
        if not 0 <= args.seed < 2**32:
            parser.error("--seed must be in [0, 2**32)")
        import random
        import numpy as np
        import torch

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)  # Seeds CPU and CUDA generators.
        print(f"[RUNNER] Random seed: {args.seed}")
    print("[RUNNER] Starting New Run")
    
    if args.run_dir is None:
        run_dir = make_runner_output_dir(args.output_dir, args.data_id)
    else:
        run_dir = args.run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RUNNER] Output directory: {run_dir}")

    import torch
    with EventLogger(run_dir / "events.jsonl") as log:
        before = runtime_metadata(torch, seed=args.seed)
        log.feed({"runtime": {"before_model": event_values(before)}})
        runner = UniWMEpisodeRunner(args.config_path, args.data_id, run_dir, event_logger=log)
        after = runtime_metadata(torch, seed=args.seed, engine=runner.wrapper.engine)
        # Retain changed observations without repeating static environment/settings.
        changed = {key: value for key, value in after.items() if value != before.get(key)}
        log.feed({"runtime": {"after_model": event_values(changed)}, "outcome": "completed"})
        runner.run_episodes(args.num_episodes, args.data_id, run_dir, args.source_episode_counts)
        log.finish({"outcome": "completed", "episodes": runner.episode_index})
    print("[RUNNER] Ending Run")
