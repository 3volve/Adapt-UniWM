from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from scripts.habitat_uniwm_schemas import UniWMInputBundle
from scripts.uniwm_inference_utils import is_stop_action
from scripts.uniwm_wrapper import UniWMWrapper
from scripts.uniwm_engine import UniWMEngine

class EpisodeAdapter:
    """Small environment/replay adapter interface for the episode manager."""

    source_mode = "unknown"

    def reset(self) -> UniWMInputBundle:
        raise NotImplementedError

    def step(self, action_text: str) -> UniWMInputBundle:
        raise NotImplementedError

class UniWMEpisodeRunner:
    """Closed-loop episode coordinator between a wrapper and an adapter."""

    def __init__(
        self,
        data_id: str,
        config_path: Optional[str] = None,
        engine: Optional[UniWMEngine] = None # Mostly for testing purposes
    ) -> None:
        config_path = self._resolve_and_load_config(data_id, config_path)

        engine = UniWMEngine(config_path, data_id) if engine is None else engine
        self.wrapper = UniWMWrapper(engine, config_path)
        self.adapter = self._load_adapter(data_id)

        self.max_episode_steps = self.config["runner"].get("max_episode_steps", 100)
        self.stop_on_wrapper_done = self.config["runner"].get("stop_on_wrapper_done", True)
        self.log_every_step = self.config["runner"].get("log_every_step", True)
        self._episode_logs: List[Dict[str, Any]] = []
        self._validate_config()

    def run_episode(self) -> Dict[str, Any]:
        episode_index = len(self._episode_logs)
        reset_result = self.adapter.reset()
        wrapper_reset_state = self.wrapper.reset_episode(reset_result)
        step_logs: List[Dict[str, Any]] = []
        termination_reason = "max_episode_steps"
        steps_executed = 0

        for step_idx in range(self.max_episode_steps):
            planned_action = self.wrapper.get_next_action()
            step_result = self.adapter.step(planned_action)
            transition = self.wrapper.observe_transition(step_result)

            steps_executed = step_idx + 1
            wrapper_requested_stop = is_stop_action(planned_action)

            if self.log_every_step:
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
                        "adapter_info": dict(step_result.metadata),
                    }
                )

            if step_result.metadata.get("done", False):
                termination_reason = "adapter_done"
                break
            if wrapper_requested_stop and self.stop_on_wrapper_done:
                termination_reason = "wrapper_stop_action"
                break

        episode_log = {
            "episode_index": episode_index,
            "adapter_source_mode": getattr(self.adapter, "source_mode", type(self.adapter).__name__),
            "max_episode_steps": self.max_episode_steps,
            "steps_executed": steps_executed,
            "termination_reason": termination_reason,
            "reset_info": dict(reset_result.metadata),
            "wrapper_reset_state": wrapper_reset_state,
            "steps": step_logs,
            "final_wrapper_state": self.wrapper.get_state_snapshot(),
            "wrapper_log": self.wrapper.get_episode_log(),
        }
        self._episode_logs.append(episode_log)
        return episode_log

    def run_episodes(self, num_episodes: int) -> List[Dict[str, Any]]:
        return [self.run_episode() for _ in range(int(num_episodes))]

    def get_logs(self) -> List[Dict[str, Any]]:
        return list(self._episode_logs)

    def _validate_config(self) -> None:
        if self.max_episode_steps <= 0:
            raise AssertionError("max_episode_steps must be positive.")

    def _resolve_and_load_config(self, data_id: str, config_path: Optional[str] = None) -> str:
        root_dir = Path(__file__).resolve().parent

        config_path = Path(root_dir / "cfg" / f"{data_id}_uniwm_cfg.yaml" if config_path is None else config_path)

        if not config_path.is_file():
            raise AssertionError(f"config_path must be file or data_id must identify a config file within the local cfg folder so this is a valid path: {config_path}")

        abs_path = config_path.resolve()
        with abs_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle) or {}

        return str(abs_path)

    def _load_adapter(self, data_id: str):
        adapter_file_name = self.config["runner"]["adapter_file_name"]
        adapter_path = Path(__file__).resolve().parent / "data_adapters" / f"{adapter_file_name}.py"
        if not adapter_path.is_file():
            raise FileNotFoundError(f"Unable to find adapter file from environment config path '{adapter_path}'")

        module_name = f"data_adapters.{adapter_file_name}"
        spec = importlib.util.spec_from_file_location(module_name, adapter_path)

        if spec is None or spec.loader is None:
            raise AssertionError(f"Unable to load adapter module from {adapter_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class_name = f"{data_id.capitalize()}EpisodeAdapter"
        adapter_cls = getattr(module, class_name, None)

        if adapter_cls is None:
            raise AssertionError(f"Unable to find expected adapter class {class_name} from environment config path '{adapter_path}'")

        return adapter_cls(**self.config["adapter"])
