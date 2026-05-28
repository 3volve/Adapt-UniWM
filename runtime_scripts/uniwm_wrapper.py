from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from PIL import Image

from runtime_scripts.runtime_engine import UniWMEngine
from runtime_scripts.uniwm_schemas import UniWMInputBundle, TransitionRecord, RouteRecord, RoutePrediction, StepPrediction
from runtime_scripts.runtime_utils import (
    image_to_array,
    is_stop_action,
    load_config, validate_config,
)


REQUIRED_FIELDS: list[str] = [
    "max_route_steps",
    "full_replan_threshold",
    "divergence_metric",
    "replan_on_route_exhausted",
    "memory_mode",
    "log_predicted_observations",
    "log_real_observations"
]

class UniWMWrapper:
    def __init__(self, engine: UniWMEngine, config_path: str = "cfg/habitat_uniwm_cfg.yaml"):
        self.engine = engine
        self.config = load_config(config_path).get("wrapper", {})
        validate_config(self.config, REQUIRED_FIELDS)
        self._reset_wrapper_state()
        self.ready_to_act = False

    def reset_episode(self, initial_bundle: UniWMInputBundle, episode_id: str | None = None) -> dict[str, Any]:
        self._reset_wrapper_state()
        self._reset_episode_memory()
        self.latest_bundle = initial_bundle
        self.engine.reset_memory(episode_id)
        self._plan_route(initial_bundle, reason="episode_reset")
        return self.get_state_snapshot()

    def get_next_action(self) -> str:
        if not self.ready_to_act:
            raise AssertionError("observe_transition(...) must be called before requesting another action.")
        self.ready_to_act = False

        if not self.current_route or self.route_index >= len(self.current_route):
            if self.config["replan_on_route_exhausted"] and self.latest_bundle is not None:
                self.replan_route(self.latest_bundle, reason="route_exhausted")
            else:
                return "stop"

        if not self.current_route or self.route_index >= len(self.current_route):
            return "stop"

        step = self.current_route.steps[self.route_index]
        self.pending_step = step
        self.pending_step_idx = self.route_index
        self.last_planned_action = step.action_text
        self.last_predicted_observation = step.visualization
        self.route_index += 1
        return step.action_text

    def observe_transition(
        self,
        observed_bundle: UniWMInputBundle
    ) -> TransitionRecord:
        if self.ready_to_act or not self.pending_step or (self.pending_step_idx < 0):
            print(f"Failing with values: ({self.ready_to_act}, {'exists' if self.pending_step else 'none'}, {self.pending_step_idx})")
            raise AssertionError("get_next_action(...) must be called before observe_transition(...).")

        pending_step = self.pending_step
        pending_step_idx = self.pending_step_idx
        real_obs = observed_bundle.current_observation
        divergence = 0
        
        if pending_step and not is_stop_action(pending_step.action_text):
            divergence = self.compute_divergence(pending_step.visualization, real_obs)

        replan_reason = None
        replanned = False
        if divergence > float(self.config["full_replan_threshold"]):
            replan_reason = f"divergence>{self.config['full_replan_threshold']:.4f}"
            self.replan_route(observed_bundle, reason=replan_reason)
            replanned = True
        else:
            self.latest_bundle = observed_bundle

        record = TransitionRecord(
            step_idx=pending_step_idx,
            action=pending_step.action_text,
            predicted_obs=self._logged_observation(pending_step.visualization, predicted=True),
            real_obs=self._logged_observation(real_obs, predicted=False),
            divergence=divergence,
            replanned=replanned,
            replan_reason=replan_reason,
            env_info=observed_bundle.metadata,
        )

        self.transition_log.append(record)
        self.last_divergence = divergence
        self.ready_to_act = True
        self.pending_step = None
        self.pending_step_idx = -1
        return record

    def replan_route(self, current_bundle: UniWMInputBundle, reason: str) -> None:
        print("[WRAPPER]: Replanning Route")
        self.latest_bundle = current_bundle
        self.pending_step = None
        self.pending_step_idx = 0
        self._plan_route(current_bundle, reason=reason)

    def compute_divergence(self, predicted_img: Any, real_img: Any) -> float:
        metric = self.config["divergence_metric"]
        if predicted_img is None and real_img is None:
            return 0.0
        if predicted_img is None or real_img is None:
            return float("inf")

        if metric == "mean_absolute_error":
            predicted_img_arr = image_to_array(predicted_img)
            real_img_arr = image_to_array(real_img)
            return float(np.abs(predicted_img_arr - real_img_arr).mean())

        raise AssertionError(f"Unsupported divergence_metric '{metric}'")

    def get_episode_log(self) -> dict[str, Any]:
        return {
            "route_history": [asdict(record) for record in self.route_history],
            "transitions": [asdict(record) for record in self.transition_log],
        }

    def get_state_snapshot(self) -> dict[str, Any]:
        return {
            "route_generation": self.route_generation,
            "route_index": self.route_index,
            "route_length": len(self.current_route),
            "pending_step_idx": self.pending_step_idx,
            "last_divergence": self.last_divergence,
            "last_planned_action": self.last_planned_action,
            "last_replan_reason": self.route_history[-1].reason if self.route_history else None,
            "route_stop_reason": self.route_history[-1].stop_reason if self.route_history else None,
        }

    def _reset_wrapper_state(self) -> None:
        self.current_route: RoutePrediction = RoutePrediction([], False, "")
        self.route_index = 0
        self.route_generation = 0
        self.route_history: list[RouteRecord] = []
        self.transition_log: list[TransitionRecord] = []
        self.pending_step: StepPrediction | None = None
        self.pending_step_idx: int = -1
        self.last_planned_action: Any = None
        self.last_predicted_observation: Any = None
        self.last_divergence: float | None = None
        self.latest_bundle: UniWMInputBundle | None = None

    def _reset_episode_memory(self) -> None:
        model = getattr(self.engine, "model", None)
        if model is None:
            return
        if hasattr(model, "reset_memory_bank"):
            model.reset_memory_bank()
        if hasattr(model, "reset_global_memory_bank"):
            model.reset_global_memory_bank()

    def _plan_route(self, bundle: UniWMInputBundle, *, reason: str) -> None:
        self.current_route = self.engine.predict_route(
            bundle,
            max_steps=self.config["max_route_steps"],
        )
        self.route_index = 0
        self.route_generation += 1
        aggregated_obs = [self._logged_observation(step.visualization, predicted=True) for step in self.current_route.steps]
        self.ready_to_act = True

        self.route_history.append(
            RouteRecord(
                route_generation=self.route_generation,
                reason=reason,
                stopped=bool(self.current_route.stopped),
                stop_reason=str(self.current_route.stop_reason),
                step_count=len(self.current_route),
                action_outputs=[str(step.action_text) for step in self.current_route.steps],
                predicted_observations=aggregated_obs,
            )
        )

        # Placeholder for later memory work: the manager can later decide whether
        # real observations, planned observations, or both should feed memory.
        if self.config["memory_mode"] != "off":
            pass

    def _logged_observation(self, observation: Image.Image | None, *, predicted: bool) -> Image.Image | None:
        key = "log_predicted_observations" if predicted else "log_real_observations"
        return observation if self.config[key] else None