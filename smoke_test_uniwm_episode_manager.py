from __future__ import annotations

from pprint import pprint

from smoke_test_uniwm_wrapper import StubEngine

from data_adapters.dummy_episode_adapter import DummyEpisodeAdapter
from uniwm_episode_runner import UniWMEpisodeRunner


def main() -> None:
    engine = StubEngine()
    adapter = DummyEpisodeAdapter(observation_levels=[32, 255], episode_id="manager_smoke")
    manager = UniWMEpisodeRunner(
        data_id="habitat"
    )

    episode_log = manager.run_episode()

    assert engine.model.reset_memory_calls == 1
    assert engine.model.reset_global_memory_calls == 1
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
