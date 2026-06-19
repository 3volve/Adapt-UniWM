from __future__ import annotations

import math
from typing import Any
from runtime_scripts.runtime_utils import validate_config, clamp, ema_decay


REQUIRED_FIELDS: dict[str, list[str]] = {
    "viz_ema_tau": ["ach", "ne"],
    "viz_learning_mods": ["ach_base_weight", "ne_bias", "ne_range", "ne_curve_width"],
    "viz_norm_scalers": ["novelty", "learning_surprise", "learning_progress", "persistent_loss"],
}

VISUAL_EVENT_STRENGTHS: dict[str, tuple[float, float]] = {
    "pred_mismatch":     (0.15, 0.25),
    "mem_novelty":       (0.30, 0.10),
    "act_uncertainty":   (0.00, 0.10),
    "viz_uncertainty":   (0.05, 0.55),
    "learning_surprise": (0.00, 0.35),
    "learning_progress": (0.40, 0.00),
    "persistent_error":  (0.20, 0.10),
    "collision":         (0.00, 0.25),
}

ACTION_EVENT_STRENGTHS: dict[str, tuple[float, float]] = {
    "pred_mismatch":     (0.00, 0.00),
    "mem_novelty":       (0.00, 0.00),
    "act_uncertainty":   (0.00, 0.00),
    "viz_uncertainty":   (0.00, 0.00),
    "learning_surprise": (0.00, 0.00),
    "learning_progress": (0.00, 0.00),
    "persistent_error":  (0.00, 0.00),
    "collision":         (0.00, 0.00),
}

class ModulatorSystem:
    """Collect per-step events and expose current ACh/NE-derived query values.

    Trigger methods currently record their raw event values only. Mapping those
    events into aggregated ACh-like and NE-like modulation is intentionally
    deferred.
    """
    
    _global_ach_viz: float = 0.0
    _global_ne_viz: float  = 0.0
    _step_ach_viz: float   = 0.0
    _step_ne_viz: float    = 0.0
    
    # _global_ach_actor: float = 0.0
    # _global_ne_actor: float  = 0.0
    
    _step_events: dict[str, float] = {}

    def __init__(self, enabled: bool, config: dict):
        """Initialize the scaffold with optional visualization-weight settings."""
        self.enabled = enabled
        
        if enabled:
            validate_config(config, REQUIRED_FIELDS)
            self.config = config
            self.step_id: int = 0

    def reset_episode(self) -> None:
        """Clear all modulation state at the start of an episode."""
        if not self.enabled:
            return
        
        self.step_id: int    = 0
        self._global_ach_viz = 0.0
        self._global_ne_viz  = 0.0
        self._step_ach_viz   = 0.0
        self._step_ne_viz    = 0.0
        
        # self._global_ach_actor = 0.0
        # self._global_ne_actor  = 0.0
        
        self._step_events = {}

    def start_step(self) -> None:
        """Clear per-step state and optionally associate it with a step id."""
        if not self.enabled:
            return
        
        self.step_id += 1
        self._step_ach_viz = 0.0
        self._step_ne_viz  = 0.0
        
        # self._global_ach_actor = 0.0
        # self._global_ne_actor  = 0.0
        
        self._step_events = {}
        
        
#----------------- Event Triggers -----------------# 
    def on_prediction_mismatch(self, mismatch: float) -> None:
        """Record a prediction-mismatch event for future aggregation."""
        if not self.enabled:
            return
        
        self._step_events["prediction_mismatch"] = mismatch
        
        self._step_ach_viz += mismatch * VISUAL_EVENT_STRENGTHS["pred_mismatch"][0]
        self._step_ne_viz  += mismatch * VISUAL_EVENT_STRENGTHS["pred_mismatch"][1]

    def on_memory_novelty(self, novelty: float) -> None:
        """Record a memory-novelty event for future aggregation."""
        if not self.enabled:
            return
        
        self._step_events["memory_novelty"] = novelty
        novelty = 1 - math.exp(-novelty / self.config["viz_norm_scalers"]["novelty"])
        self._step_events["norm_memory_novelty"] = novelty
        
        self._step_ach_viz += novelty * VISUAL_EVENT_STRENGTHS["mem_novelty"][0]
        self._step_ne_viz  += novelty * VISUAL_EVENT_STRENGTHS["mem_novelty"][1]

    def on_action_uncertainty(self, entropy: float) -> None:
        """Record an action-uncertainty event for future aggregation."""
        if not self.enabled:
            return
        
        self._step_events["act_uncertainty"] = entropy
        entropy = clamp(entropy, 0, 1)
        self._step_events["clamped_act_uncertainty"] = entropy
        
        self._step_ach_viz += entropy * VISUAL_EVENT_STRENGTHS["act_uncertainty"][0]
        self._step_ne_viz  += entropy * VISUAL_EVENT_STRENGTHS["act_uncertainty"][1]

    def on_visual_uncertainty(self, entropy: float) -> None:
        """Record a visual-uncertainty event for future aggregation."""
        if not self.enabled:
            return
        
        self._step_events["viz_uncertainty"] = entropy
        entropy = clamp(entropy, 0, 1)
        self._step_events["clamped_viz_uncertainty"] = entropy
        
        self._step_ach_viz += entropy * VISUAL_EVENT_STRENGTHS["viz_uncertainty"][0]
        self._step_ne_viz  += entropy * VISUAL_EVENT_STRENGTHS["viz_uncertainty"][1]

    def on_learning_surprise(self, surprise: float) -> None:
        """Record a visual-learning-surprise event for future aggregation."""
        if not self.enabled:
            return
        
        self._step_events["learning_surprise"] = surprise
        surprise = 1 - math.exp(
            -max(0.0, surprise) / self.config["viz_norm_scalers"]["learning_surprise"]
        )
        self._step_events["norm_learning_surprise"] = surprise
        
        self._step_ach_viz += surprise * VISUAL_EVENT_STRENGTHS["learning_surprise"][0]
        self._step_ne_viz  += surprise * VISUAL_EVENT_STRENGTHS["learning_surprise"][1]

    def on_learning_progress(self, progress: float) -> None:
        """Record a visual-learning-progress event for future aggregation."""
        if not self.enabled:
            return
        
        self._step_events["learning_progress"] = progress
        progress = 1 - math.exp(
            -max(0.0, progress) / self.config["viz_norm_scalers"]["learning_progress"]
        )
        self._step_events["norm_learning_progress"] = progress
        
        self._step_ach_viz += progress * VISUAL_EVENT_STRENGTHS["learning_progress"][0]
        self._step_ne_viz  += progress * VISUAL_EVENT_STRENGTHS["learning_progress"][1]

    def on_persistent_error(self, error: float) -> None:
        """Record a persistent visual-learning-error event for future aggregation."""
        if not self.enabled:
            return
        
        self._step_events["persistent_error"] = error
        error = 1 - math.exp(-error / self.config["viz_norm_scalers"]["persistent_loss"])
        self._step_events["norm_persistent_error"] = error
        
        self._step_ach_viz += error * VISUAL_EVENT_STRENGTHS["persistent_error"][0]
        self._step_ne_viz  += error * VISUAL_EVENT_STRENGTHS["persistent_error"][1]

    def on_collision_or_unsafe(self, strength: float = 1.0) -> None:
        """Record a collision or unsafe-state event for future aggregation."""
        if not self.enabled or strength == 0.0:
            return
        
        self._step_events["collision"] = strength
        
        self._step_ach_viz += strength * VISUAL_EVENT_STRENGTHS["collision"][0]
        self._step_ne_viz  += strength * VISUAL_EVENT_STRENGTHS["collision"][1]
        
    def ema_update_signals(self):
        if not self.enabled:
            return
        
        self._clip_step_signals()
        
        self._global_ach_viz = ema_decay(
            self._global_ach_viz,
            self._step_ach_viz,
            self.config["viz_ema_tau"]["ach"]
        )
        
        self._global_ne_viz = ema_decay(
            self._global_ne_viz,
            self._step_ne_viz,
            self.config["viz_ema_tau"]["ne"]
        )
        
        self._clip_global_signals()

    def compute_visualization_update_weight(self) -> float:
        """Compute the current visual update weight from ACh-like and NE-like state."""
        if not self.enabled:
            return 1.0

        if self._step_events.get("collision", 0.0) > 0.0:
            self._step_events["viz_update_weight"] = 0.0
            return 0.0

        ne_mult = self._compute_ne_mult()
        
        ach_mult = self._compute_ach_mult()
        
        weight = ach_mult * ne_mult
        max_weight = 1 + self.config["viz_learning_mods"]["ach_base_weight"] * self.config["viz_learning_mods"]["ne_range"][1]
        clipped_weight = clamp(weight, 0.0, max_weight)

        self._step_events["viz_update_weight"] = clipped_weight
        return clipped_weight

    def compute_actor_update_weight(self, *args, **kwargs) -> float:
        """Return no actor weights while actor-side modulation is deferred."""
        if not self.enabled:
            return 1.0
        
        # TODO: Deferred: action-selection learning, critic/value-head DA feedback,
        # and actor update weighting are intentionally out of scope for the current
        # implementation.
        return 1.0

    def get_current_state(self) -> dict[str, Any]:
        """Return a plain dictionary of current state for logging and debugging."""
        if not self.enabled:
            return {}
        
        return {
            "step_id":      self.step_id,
            "gl_ach_viz":   self._global_ach_viz,
            "gl_ne_visual": self._global_ne_viz,
            "st_ach_viz":   self._step_ach_viz,
            "st_ne_visual": self._step_ne_viz,
            
            # "ach_actor":  self._global_ach_actor,
            # "ne_actor":   self._global_ne_actor,
            
            "step_events": self._step_events.copy()
        }
        
    def _compute_ach_mult(self) -> float:
        return 1 + self.config["viz_learning_mods"]["ach_base_weight"] * self._global_ach_viz
        
    def _compute_ne_mult(self) -> float:
        # Optimal NE marks the center of the inverted-U arousal curve for visual learning.
        optimal_ne = self.config["viz_learning_mods"]["ne_bias"]
        # NE range of learning modulation scales the Gaussian min and max height for a centered gaussian.
        ne_range = list(self.config["viz_learning_mods"]["ne_range"])
        ne_min = ne_range[0]
        ne_max = ne_range[1]

        # NE distance shifts the Gaussian peak along the 0-to-1 NE range.
        ne_distance = self._global_ne_viz - optimal_ne
        curve_width = max(
            float(self.config["viz_learning_mods"]["ne_curve_width"]),
            1e-8,
        )
        
        gaussian_gamma = math.log(2.0) / (curve_width ** 2)
        gaussian_endpoint = math.exp(-gaussian_gamma * 0.25)
        gaussian_height = math.exp(-gaussian_gamma * ne_distance**2)
        normalized_gaussian = (gaussian_height - gaussian_endpoint) / (1.0 - gaussian_endpoint)
        return ne_min + (ne_max - ne_min) * normalized_gaussian
        
    def _clip_global_signals(self):
        self._global_ach_viz = clamp(self._global_ach_viz, 0.0, 1.0)
        self._global_ne_viz  = clamp(self._global_ne_viz, 0.0, 1.0)
        
    def _clip_step_signals(self):
        self._step_ach_viz   = clamp(self._step_ach_viz, 0.0, 1.0)
        self._step_ne_viz    = clamp(self._step_ne_viz, 0.0, 1.0)
