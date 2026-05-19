from __future__ import annotations

from pprint import pprint
from dataclasses import dataclass
import sys
import types


def _install_habitat_stubs() -> None:
    habitat_module = types.ModuleType("habitat")
    core_module = types.ModuleType("habitat.core")
    simulator_module = types.ModuleType("habitat.core.simulator")
    dataset_module = types.ModuleType("habitat.core.dataset")
    tasks_module = types.ModuleType("habitat.tasks")
    nav_module = types.ModuleType("habitat.tasks.nav")
    image_nav_module = types.ModuleType("habitat.tasks.nav.instance_image_nav_task")

    class Observations(dict):
        pass

    class Episode:
        pass

    class InstanceImageGoalNavEpisode(Episode):
        pass

    simulator_module.Observations = Observations
    dataset_module.Episode = Episode
    image_nav_module.InstanceImageGoalNavEpisode = InstanceImageGoalNavEpisode

    sys.modules.setdefault("habitat", habitat_module)
    sys.modules.setdefault("habitat.core", core_module)
    sys.modules.setdefault("habitat.core.simulator", simulator_module)
    sys.modules.setdefault("habitat.core.dataset", dataset_module)
    sys.modules.setdefault("habitat.tasks", tasks_module)
    sys.modules.setdefault("habitat.tasks.nav", nav_module)
    sys.modules.setdefault("habitat.tasks.nav.instance_image_nav_task", image_nav_module)


def _install_datasource_schema_stub() -> None:
    schema_module = types.ModuleType("runtime_scripts.datasource_schemas")

    class SourceAdapter:
        source_mode = "unknown"

    class SourceFormatter:
        source_mode = "unknown"

    @dataclass(frozen=True)
    class OutputBundle:
        done: bool = False

    schema_module.SourceAdapter = SourceAdapter
    schema_module.SourceFormatter = SourceFormatter
    schema_module.OutputBundle = OutputBundle
    sys.modules["runtime_scripts.datasource_schemas"] = schema_module


_install_habitat_stubs()
_install_datasource_schema_stub()

from smoke_test_uniwm_wrapper import StubEngine
import uniwm_episode_runner as episode_runner_module


@dataclass(frozen=True)
class FakeOutputBundle:
    done: bool = False


class DummyEpisodeAdapter:
    source_mode = "dummy"

    def __init__(self) -> None:
        self.step_idx = 0
        self.episode_id = None

    def reset(self, episode_id: str) -> object:
        self.step_idx = 0
        self.episode_id = episode_id
        return _bundle_from_level(0, {"episode_id": episode_id, "adapter_step_idx": 0, "done": False})

    def step(self, action_text: str) -> object:
        levels = [32, 255, 255]
        current_level = levels[min(self.step_idx, len(levels) - 1)]
        self.step_idx += 1
        done = action_text.strip().lower() == "stop"
        return _bundle_from_level(
            current_level,
            {
                "episode_id": self.episode_id,
                "adapter_step_idx": self.step_idx,
                "received_action_text": action_text,
                "done": done,
            },
        )


class DummyUniWMFormatter:
    def convert_action(self, action: str) -> list[str]:
        return [action]

    def convert_observation(self, output) -> object:
        return output


def _bundle_from_level(level: int, metadata: dict[str, object]) -> object:
    from smoke_test_uniwm_wrapper import _bundle

    return _bundle(level, env_info=dict(metadata))


episode_runner_module.OutputBundle = FakeOutputBundle
episode_runner_module.load_config = lambda config_path: {
    "runner": {
        "max_episode_steps": 5,
        "stop_on_wrapper_done": True,
        "log_every_step": True,
        "adapter_file_name": "dummy_episode_adapter",
        "adapter_params": {},
        "converter_params": {},
    }
}
episode_runner_module.UniWMEpisodeRunner._load_source_classes = lambda self, data_id: (DummyEpisodeAdapter(), DummyUniWMFormatter())
UniWMEpisodeRunner = episode_runner_module.UniWMEpisodeRunner


def main() -> None:
    engine = StubEngine()
    manager = UniWMEpisodeRunner(
        data_id="dummy",
        engine=engine
    )

    episode_log = manager.run_episode("smoke_episode")

    assert engine.model.reset_memory_calls == 1
    assert engine.model.reset_global_memory_calls == 1
    assert engine.reset_memory_calls == ["smoke_episode"]
    print(f"len(engine_predict_route_calls) = {len(engine.predict_route_calls)}")
    assert len(engine.predict_route_calls) == 2
    assert episode_log["steps_executed"] == 3
    assert episode_log["termination_reason"] == "adapter_done"
    assert len(episode_log["steps"]) == 3
    assert episode_log["steps"][0]["action_text"].startswith("Move by dx:")
    assert episode_log["steps"][0]["replanned"] is False
    assert episode_log["steps"][1]["replanned"] is True
    assert episode_log["steps"][2]["action_text"] == "stop"
    assert episode_log["steps"][2]["divergence"] == 0.0
    assert len(episode_log["wrapper_log"]["transitions"]) == 3
    assert len(episode_log["wrapper_log"]["route_history"]) == 2

    pprint(
        {
            "termination_reason": episode_log["termination_reason"],
            "steps_executed": episode_log["steps_executed"],
            "actions": [step["action_text"] for step in episode_log["steps"]],
            "replanned_flags": [step["replanned"] for step in episode_log["steps"]],
        }
    )
    print("uniwm_episode_manager smoke test passed")


if __name__ == "__main__":
    main()
