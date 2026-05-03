from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from scripts.habitat_uniwm_schemas import UniWMInputBundle
from scripts.uniwm_inference_utils import load_config, image_to_array, StepPrediction, RoutePrediction


DEFAULT_WRAPPER_CONFIG: Dict[str, Any] = {
    "max_route_steps": None,
    "full_replan_threshold": 0.12,
    "divergence_metric": "mean_absolute_error",
    "replan_on_route_exhausted": True,
    "memory_mode": "off",
    "log_predicted_observations": True,
    "log_real_observations": True,
}


@dataclass
class TransitionRecord:
    step_idx: int
    action: Any
    predicted_obs: Image.Image
    real_obs: Image.Image
    divergence: float
    replanned: bool
    replan_reason: Optional[str]
    env_info: Optional[dict] = None


@dataclass
class RouteRecord:
    route_generation: int
    reason: str
    stopped: bool
    stop_reason: str
    step_count: int
    action_outputs: List[str]
    predicted_observations: List[Any] = field(default_factory=list)


class UniWMWrapper:
    def __init__(self, engine: Any, config_path: str = ""):
        self.engine = engine
        self.config = load_config(config_path).get("wrapper", DEFAULT_WRAPPER_CONFIG)
        self._validate_config()
        self._reset_wrapper_state()

    def reset_episode(self, initial_bundle: UniWMInputBundle) -> Dict[str, Any]:
        self._reset_wrapper_state()
        self._reset_episode_memory()
        self.latest_bundle = initial_bundle
        self._plan_route(initial_bundle, reason="episode_reset")
        return self.get_state_snapshot()

    def get_next_action(self) -> Any:
        if self.pending_step is not None:
            raise AssertionError("observe_transition(...) must be called before requesting another action.")

        if not self.current_route or self.route_index >= len(self.current_route):
            if self.config["replan_on_route_exhausted"] and self.latest_bundle is not None:
                self.replan_route(self.latest_bundle, reason="route_exhausted")
            else:
                return "stop"

        if not self.current_route or self.route_index >= len(self.current_route):
            return "stop"

        step = self.current_route[self.route_index]
        self.pending_step = step
        self.pending_step_idx = self.route_index
        self.last_planned_action = step.action_text
        self.last_predicted_observation = step.visualization
        self.route_index += 1
        return step.action_text

    def observe_transition(
        self,
        observed_bundle: UniWMInputBundle,
        *,
        env_info: Optional[dict] = None,
    ) -> TransitionRecord:
        if self.pending_step is None or self.pending_step_idx is None:
            raise AssertionError("get_next_action(...) must be called before observe_transition(...).")

        pending_step = self.pending_step
        pending_step_idx = self.pending_step_idx
        real_obs = observed_bundle.current_observation
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
            env_info=env_info,
        )
        self.transition_log.append(record)
        self.last_divergence = divergence
        self.pending_step = None
        self.pending_step_idx = None
        return record

    def replan_route(self, current_bundle: UniWMInputBundle, reason: str) -> None:
        self.latest_bundle = current_bundle
        self.pending_step = None
        self.pending_step_idx = None
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

    def get_episode_log(self) -> Dict[str, Any]:
        return {
            "route_history": [asdict(record) for record in self.route_history],
            "transitions": [asdict(record) for record in self.transition_log],
        }

    def get_state_snapshot(self) -> Dict[str, Any]:
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

    def _validate_config(self) -> None:
        if float(self.config["full_replan_threshold"]) < 0.0:
            raise AssertionError("wrapper.full_replan_threshold must be non-negative.")
        if self.config["divergence_metric"] != "mean_absolute_error":
            raise AssertionError("wrapper.divergence_metric must currently be 'mean_absolute_error'.")
        if self.config["memory_mode"] not in {"off", "real_only", "real_plus_plan_weighted"}:
            raise AssertionError("wrapper.memory_mode must be one of: off, real_only, real_plus_plan_weighted.")

    def _reset_wrapper_state(self) -> None:
        self.current_route: RoutePrediction = RoutePrediction([], False, "")
        self.route_index = 0
        self.route_generation = 0
        self.route_history: List[RouteRecord] = []
        self.transition_log: List[TransitionRecord] = []
        self.pending_step: Optional[StepPrediction] = None
        self.pending_step_idx: Optional[int] = None
        self.last_planned_action: Any = None
        self.last_predicted_observation: Any = None
        self.last_divergence: Optional[float] = None
        self.latest_bundle: Optional[UniWMInputBundle] = None

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
        self.route_history.append(
            RouteRecord(
                route_generation=self.route_generation,
                reason=reason,
                stopped=bool(self.current_route.stopped),
                stop_reason=str(self.current_route.stop_reason),
                step_count=len(self.current_route),
                action_outputs=[str(step.action_text) for step in self.current_route.steps],
                predicted_observations=[
                    self._logged_observation(step.visualization, predicted=True) for step in self.current_route.steps
                ],
            )
        )

        # Placeholder for later memory work: the manager can later decide whether
        # real observations, planned observations, or both should feed memory.
        if self.config["memory_mode"] != "off":
            pass

    def _logged_observation(self, observation: Any, *, predicted: bool) -> Any:
        key = "log_predicted_observations" if predicted else "log_real_observations"
        return observation if self.config[key] else None
