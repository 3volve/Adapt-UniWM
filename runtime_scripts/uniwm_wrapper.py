from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from runtime_scripts.runtime_engine import UniWMEngine
from runtime_scripts.modulator_system import ModulatorSystem
from runtime_scripts.uniwm_schemas import (
    UniWMInputBundle,
    TransitionRecord,
    RouteRecord,
    RoutePrediction,
    StepPrediction
)
from runtime_scripts.runtime_utils import (
    image_to_array,
    is_stop_action,
    load_config,
    validate_config,
    build_img_paths,
    save_img,
    ema_smoothing,
)


REQUIRED_FIELDS: list[str] = [
    "max_route_steps",
    "full_replan_threshold",
    "divergence_metric",
    "replan_on_route_exhausted",
    "log_predicted_obs",
    "log_real_obs",
    "add_global_memories",
    "eval_forced_action",
    "training_enabled",
    
    "enable_modulators",
    "min_memories_for_confidence",
    "viz_loss_slow_ema",
    "viz_loss_fast_ema"
]

class UniWMWrapper:
    def __init__(self, engine: UniWMEngine, config_path: str, absolute_output_path: str):
        self.engine = engine
        root_config = load_config(config_path)
        self.config = root_config.get("wrapper", {})
        validate_config(self.config, REQUIRED_FIELDS)
        self.output_path = absolute_output_path
        self.forced_actions = False
        
        self.modulators = ModulatorSystem(self.config["enable_modulators"], root_config["modulators"])
        
        self._reset_wrapper_state()
        self.ready_to_act = False

    def reset_episode(self, initial_bundle: UniWMInputBundle, episode_id: str) -> dict[str, Any]:
        self._reset_wrapper_state()
        self.latest_bundle = initial_bundle
        self.engine.reset_episode(episode_id, initial_bundle)
        self.episode_id = episode_id
        self._plan_route(initial_bundle, reason="episode_reset")
        return self.get_state_snapshot()

    def get_next_action(self) -> str:
        if not self.ready_to_act:
            raise AssertionError("observe_transition(...) must be called before requesting another action.")
        self.ready_to_act = False

        if not self.current_route or self.route_idx >= len(self.current_route):
            if self.config["replan_on_route_exhausted"] and self.latest_bundle is not None:
                self.replan_route(self.latest_bundle, reason="route_exhausted")
                self.ready_to_act = False

            if not self.current_route or self.route_idx >= len(self.current_route):
                return "stop"

        current_step = self.current_route.steps[self.route_idx]
        self.pending_step, self.pending_step_idx = current_step, self.route_idx
        self.last_planned_action = current_step.action_text
        self.last_predicted_observation = current_step.visualization
        self.route_idx += 1
        
        self.modulators.start_step()
        self.modulators.on_action_uncertainty(current_step.act_entropy)
            
        return current_step.action_text

    def observe_transition(
        self,
        observed_bundle: UniWMInputBundle
    ) -> TransitionRecord:
        if self.ready_to_act or not self.pending_step or (self.pending_step_idx < 0):
            print(f"Failing with values: ({self.ready_to_act}, {'exists' if self.pending_step else 'none'}, {self.pending_step_idx})")
            raise AssertionError("get_next_action(...) must be called before observe_transition(...).")
        
        pending_step, pending_step_idx = self.pending_step, self.pending_step_idx
        real_obs = observed_bundle.current_observation
        pending_step.context_familiarity, pending_step.context_stability = self.engine.get_current_context()
        save_path_real, save_path_viz, save_path_eval = build_img_paths(self.output_path, self.episode_id, self.route_id, self.pending_step_idx)
        
        if real_obs is not None and self.config["log_real_obs"]:
            save_img(real_obs, save_path_real)
        if pending_step.visualization is not None and self.config["log_predicted_obs"]:
            save_img(pending_step.visualization, save_path_viz)
        
        if len(self.current_route) > self.route_idx:
            next_step = self.current_route.steps[self.route_idx]
            next_step.real_input_obs = real_obs
        pending_step.real_next_obs = real_obs
        
        divergence = 0
        training_log: dict[str, Any] | None = None
        eval_log: dict[str, Any] | None = None
        modulator_state: dict[str, Any] | None = None
        if pending_step and not is_stop_action(pending_step.action_text):
            divergence, mod_divergence = self.compute_divergence(pending_step.visualization, pending_step.real_next_obs)
            
            self._update_modulators(pending_step, mod_divergence, observed_bundle.collision)
            lr_scalar = self.modulators.compute_visualization_update_weight()
            modulator_state = self.modulators.get_current_state()
            
            # TODO: Decide if more complicated logic for determining whether to train or not is necessary
            if not observed_bundle.collision:
                if self.config["eval_forced_action"] and observed_bundle.action_text is not None:
                    eval_log = self._run_eval_predict(observed_bundle, pending_step.real_next_obs, save_path_eval)
                    
                if self.config["training_enabled"]:
                    training_log = self.engine.train_viz_step(pending_step, lr_scalar)
                    self._update_viz_loss(training_log["base_loss"])
                    training_log["_viz_loss_slow"], training_log["_viz_loss_fast"] = self._viz_loss_slow, self._viz_loss_fast
            
                self.engine.store_and_update_working_memory(real_obs, observed_bundle.start_pose_str, self.config["add_global_memories"])
                
        replan_reason = None
        replanned = False
        route_id = self.route_id
        if not observed_bundle.source_done and (observed_bundle.collision or divergence > float(self.config["full_replan_threshold"])):
            if observed_bundle.collision:
                replan_reason = f"collision"
            else:
                replan_reason = f"divergence>{self.config['full_replan_threshold']:.3f}"
                
            self.replan_route(observed_bundle, reason=replan_reason)
            replanned = True
        else:
            self.latest_bundle = observed_bundle

        record = TransitionRecord(
            route_id=route_id,
            route_idx=pending_step_idx,
            action=pending_step.action_text,
            context_familiarity=pending_step.context_familiarity,
            context_stability=pending_step.context_stability,
            divergence=divergence,
            replanned=replanned,
            replan_reason=replan_reason,
            modulator_state=modulator_state,
            training_logs=training_log,
            eval_logs=eval_log,
            env_info=observed_bundle.metadata,
        )

        self.last_divergence = divergence
        self.ready_to_act = True
        self.pending_step, self.pending_step_idx = None, -1
        return record

    def replan_route(self, current_bundle: UniWMInputBundle, reason: str) -> None:
        if current_bundle.source_done:
            return
        
        print("[WRAPPER]: Replanning Route")
        self.latest_bundle = current_bundle
        self.pending_step = None
        self.pending_step_idx = 0
        self._plan_route(current_bundle, reason=reason)

    def compute_divergence(self, predicted_img: Any, real_img: Any) -> tuple[float, float]:
        # TODO: Figure out a cleaner solution for needing to return a divergence on abnormal inputs or ensure abnormal inputs can't be input.
        metric = self.config["divergence_metric"]
        if predicted_img is None and real_img is None:
            return 0.0, 0.0
        if predicted_img is None or real_img is None:
            return float(self.config["full_replan_threshold"]) + 100, -1.0

        if metric == "mean_absolute_error":
            predicted_img_arr = image_to_array(predicted_img)
            real_img_arr = image_to_array(real_img)
            result = float(np.abs(predicted_img_arr - real_img_arr).mean())
            return result, result

        raise AssertionError(f"Unsupported divergence_metric '{metric}'")

    def get_routes_log_for_episode(self) -> list[dict[str, Any]]:
        return [record.to_log() for record in self.route_history]

    def get_state_snapshot(self) -> dict[str, Any]:
        return {
            "route_generation": self.route_id,
            "route_index": self.route_idx,
            "route_length": len(self.current_route),
            "pending_step_idx": self.pending_step_idx,
            "last_divergence": self.last_divergence,
            "last_planned_action": self.last_planned_action,
            "last_replan_reason": self.route_history[-1].replan_reason if self.route_history else None,
            "route_stop_reason": self.route_history[-1].stop_reason if self.route_history else None,
        }

    def _reset_wrapper_state(self) -> None:
        self.current_route: RoutePrediction = RoutePrediction([], False, "")
        self.route_idx = 0
        self.route_id = 0
        self.route_history: list[RouteRecord] = []
        self.pending_step: StepPrediction | None = None
        self.pending_step_idx: int = -1
        self.last_planned_action: Any = None
        self.last_predicted_observation: Any = None
        self.last_divergence: float | None = None
        self.latest_bundle: UniWMInputBundle | None = None
        self._viz_loss_slow: float | None = None
        self._viz_loss_fast: float | None = None
        self.modulators.reset_episode()

    def _plan_route(self, bundle: UniWMInputBundle, *, reason: str) -> None:
        if bundle.source_done:
            return
        
        self.route_idx = 0
        self.route_id += 1
        self.ready_to_act = True
        
        self.current_route = self.engine.predict_route(
            bundle,
            self.config["max_route_steps"],
        )
        self.current_route.steps[0].real_input_obs = bundle.current_observation
        
        self.route_history.append(
            RouteRecord(
                route_id=self.route_id,
                replan_reason=reason,
                stop_reason=str(self.current_route.stop_reason),
                planned_step_count=len(self.current_route),
                planned_actions=[str(step.action_text) for step in self.current_route.steps]
            )
        )
        
    def _run_eval_predict(self, observed_bundle: UniWMInputBundle, target_obs: Image.Image, save_path_eval: str) -> dict[str, Any]:
        eval_log = {}
        eval_prediction = self.engine.eval_step(observed_bundle)
        eval_divergence, _ = self.compute_divergence(eval_prediction.visualization, target_obs)
        
        eval_log["forced_action"] = eval_prediction.action_text
        eval_log["divergence"] = eval_divergence
        eval_log["viz_entropy"] = eval_prediction.viz_entropy
        eval_log["context_familiarity"] = eval_prediction.context_familiarity
        eval_log["context_stability"] = eval_prediction.context_stability
        eval_log["prediction_available"] = eval_prediction.visualization is not None
        eval_log["predicted_obs_path"] = save_path_eval if eval_prediction.visualization is not None and self.config["log_predicted_obs"] else None
        
        if eval_prediction.visualization is not None and self.config["log_predicted_obs"]:
            save_img(eval_prediction.visualization, save_path_eval)
            
        return eval_log
    
    def _update_viz_loss(self, new_loss: float) -> None:
        new_loss = max(0.0, new_loss)

        if self._viz_loss_fast is None or self._viz_loss_slow is None:
            self._viz_loss_fast = new_loss
            self._viz_loss_slow = new_loss
            return
        
        self._viz_loss_slow = ema_smoothing(self._viz_loss_slow, new_loss, self.config["viz_loss_slow_ema"])
        self._viz_loss_fast = ema_smoothing(self._viz_loss_fast, new_loss, self.config["viz_loss_fast_ema"])
        
    def _update_modulators(self, pending_step: StepPrediction, divergence: float, is_collision: bool):
        """ Calculates necessary values to feed to modulators
        
            Complex input values:
                - context_familiarity = current context's highest cosine similarity to a stored memory; mean-averaged across layers
                - context_stability   = cosine similarity between the current compressed context and its EMA-smoothed historical context
                - viz_entropy         = Shannon entropy over valid visual-code choices at generated image-token positions; mean-normalized
                - viz_loss_fast      = EMA-smoothed visual loss with a lower-tau restriction to indicate current-step's trend
                - viz_loss_slow      = EMA-smoothed visual loss with a higher-tau restriction to track loss movement over time
        
            Final computed values fed to modulator events
                - memory_novelty_score = inverse familiarity weighted by confidence from the current stored-memory count
                - relative_surprise    = positive fast-over-slow EMA loss gap; normalized by largest EMA loss
                - relative_progress    = positive slow-over-fast EMA loss gap; normalized by largest EMA loss
                - persistent_error     = smallest fast vs slow EMA loss; weighted by context stability
        """
        if not self.config["enable_modulators"]:
            return
        
        self.modulators.on_collision_or_unsafe(is_collision)
        
        memory_confidence = min(self.engine.memory_count / int(self.config["min_memories_for_confidence"]), 1.0)
        memory_novelty_score = (1.0 - pending_step.context_familiarity) * memory_confidence
        self.modulators.on_memory_novelty(memory_novelty_score)
        
        if self._viz_loss_fast is not None and self._viz_loss_slow is not None:
            loss_baseline = max(self._viz_loss_fast, self._viz_loss_slow, 1e-8)
            relative_surprise = max(0.0, (self._viz_loss_fast - self._viz_loss_slow) / loss_baseline)
            relative_progress = max(0.0, (self._viz_loss_slow - self._viz_loss_fast) / loss_baseline)
            persistent_error = min(self._viz_loss_fast, self._viz_loss_slow) * pending_step.context_stability

            self.modulators.on_learning_surprise(relative_surprise)
            self.modulators.on_learning_progress(relative_progress * pending_step.context_stability)
            self.modulators.on_persistent_error(persistent_error)
            
        if not is_collision and divergence >= 0.0:
            self.modulators.on_prediction_mismatch(divergence)
            self.modulators.on_visual_uncertainty(pending_step.viz_entropy)
        
        self.modulators.ema_update_signals()
