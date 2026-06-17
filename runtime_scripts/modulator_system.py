from __future__ import annotations

import math
from runtime_utils import validate_config


REQUIRED_FIELDS: list[str] = [
    "enable_modulators",
    "max_visual_update_weight",
    "viz_ema_tau",
    "viz_learning_mods",
    "viz_scalers"
]

VISUAL_EVENT_STRENGTHS: dict[str, float] = {
    "pred_mismatch_ach":   0.70,
    "pred_mismatch_ne":    0.25,
    "mem_novelty_ach":     0.50,
    "mem_novelty_ne":      0.10,
    "act_uncertainty_ach": 0.00,
    "act_uncertainty_ne":  0.10,
    "viz_uncertainty_ach": 0.00,
    "viz_uncertainty_ne":  0.60,
    "collision_ach":       0.00,
    "collision_ne":        0.25,
}

ACTION_EVENT_STRENGTHS: dict[str, float] = {
    "pred_mismatch_ach":   0.00,
    "pred_mismatch_ne":    0.00,
    "mem_novelty_ach":     0.00,
    "mem_novelty_ne":      0.00,
    "act_uncertainty_ach": 0.00,
    "act_uncertainty_ne":  0.00,
    "viz_uncertainty_ach": 0.00,
    "viz_uncertainty_ne":  0.00,
    "collision_ach":       0.00,
    "collision_ne":        0.00,
    
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

    def __init__(self, config: dict):
        """Initialize the scaffold with optional visualization-weight settings."""
        validate_config(config, REQUIRED_FIELDS)
        self.config = config
        self.step_id: int = 0

    def reset_episode(self) -> None:
        """Clear all modulation state at the start of an episode."""
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
        self.step_id += 1
        self._step_ach_viz = 0.0
        self._step_ne_viz  = 0.0
        
        # self._global_ach_actor = 0.0
        # self._global_ne_actor  = 0.0
        
        self._step_events = {}
        
    def on_prediction_mismatch(self, mismatch: float) -> None:
        """Record a prediction-mismatch event for future aggregation."""
        self._step_events["prediction_mismatch"] = mismatch
        
        self._step_ach_viz += mismatch * VISUAL_EVENT_STRENGTHS["pred_mismatch_ach"]
        self._step_ne_viz  += mismatch * VISUAL_EVENT_STRENGTHS["pred_mismatch_ne"]

    def on_memory_novelty(self, novelty: float) -> None:
        """Record a memory-novelty event for future aggregation."""
        self._step_events["memory_novelty"] = novelty
        novelty = 1 - math.exp(-novelty / self.config["viz_scalers"]["novelty"])
        self._step_events["norm_memory_novelty"] = novelty
        
        self._step_ach_viz += novelty * VISUAL_EVENT_STRENGTHS["mem_novelty_ach"]
        self._step_ne_viz  += novelty * VISUAL_EVENT_STRENGTHS["mem_novelty_ne"]

    def on_action_uncertainty(self, entropy: float) -> None:
        """Record an action-uncertainty event for future aggregation."""
        self._step_events["act_uncertainty"] = entropy
        entropy = entropy / self.config["viz_scalers"]["act_entropy"]
        self._step_events["norm_act_uncertainty"] = entropy
        
        self._step_ach_viz += entropy * VISUAL_EVENT_STRENGTHS["act_uncertainty_ach"]
        self._step_ne_viz  += entropy * VISUAL_EVENT_STRENGTHS["act_uncertainty_ne"]

    def on_visual_uncertainty(self, entropy: float) -> None:
        """Record a visual-uncertainty event for future aggregation."""
        self._step_events["viz_uncertainty"] = entropy
        entropy = entropy / self.config["viz_scalers"]["viz_entropy"]
        self._step_events["norm_viz_uncertainty"] = entropy
        
        self._step_ach_viz += entropy * VISUAL_EVENT_STRENGTHS["viz_uncertainty_ach"]
        self._step_ne_viz  += entropy * VISUAL_EVENT_STRENGTHS["viz_uncertainty_ne"]

    def on_collision_or_unsafe(self, strength: float = 1.0) -> None:
        """Record a collision or unsafe-state event for future aggregation."""
        if strength == 0.0:
            return
        
        self._step_events["collision"] = strength
        
        self._step_ach_viz += strength * VISUAL_EVENT_STRENGTHS["collision_ach"]
        self._step_ne_viz  += strength * VISUAL_EVENT_STRENGTHS["collision_ne"]
        
    def ema_update_signals(self):
        self._clip_step_signals()
        
        self._global_ach_viz = self._apply_ema_signal(
            self._global_ach_viz,
            self._step_ach_viz,
            self.config["viz_ema_tau"]["ach"]
        )
        
        self._global_ne_viz = self._apply_ema_signal(
            self._global_ne_viz,
            self._step_ne_viz,
            self.config["viz_ema_tau"]["ne"]
        )
        
        self._clip_global_signals()

    def compute_visualization_update_weight(self) -> float:
        """Compute the current visual update weight from ACh-like and NE-like state."""
        if not self.config["enable_modulators"]:
            return 1.0

        if self._step_events.get("collision", 0.0) > 0.0:
            self._step_events["viz_update_weight"] = 0.0
            return 0.0

        # Optimal NE marks the center of the inverted-U arousal curve for visual learning.
        optimal_ne = self.config["viz_learning_mods"]["learner_preference"]
        # NE range of learning modulation scales the Gaussian min and max height for a centered gaussian.
        ne_range = list(self.config["viz_learning_mods"]["ne_range"])
        ne_min = ne_range[0]
        ne_max = ne_range[1]

        # NE distance shifts the Gaussian peak along the 0-to-1 NE range.
        ne_distance = self._global_ne_viz - optimal_ne
        # ln(2) over 0.5 squared makes a centered optimum produce 0.5x at both endpoints.
        gaussian_gamma = math.log(2.0) / (self.config["viz_learning_mods"]["learner_preference"] ** 2)
        gaussian_endpoint = math.exp(-gaussian_gamma * 0.25)
        gaussian_height = math.exp(-gaussian_gamma * ne_distance**2)
        normalized_gaussian = (gaussian_height - gaussian_endpoint) / (1.0 - gaussian_endpoint)
        ne_multiplier = ne_min + (ne_max - ne_min) * normalized_gaussian
        
        max_weight = self.config["max_visual_update_weight"]
        weight = max_weight * self._global_ach_viz * ne_multiplier
        clipped_weight = max(0.0, min(max_weight, weight))

        self._step_events["viz_update_weight"] = clipped_weight
        return clipped_weight

    def compute_actor_update_weight(self, *args, **kwargs) -> float:
        """Return no actor weights while actor-side modulation is deferred."""
        if not self.config["enable_modulators"]:
            return 1.0
        # TODO: Deferred: action-selection learning, critic/value-head DA feedback,
        # and actor update weighting are intentionally out of scope for the current
        # implementation.
        return 1.0

    def get_current_state(self) -> dict:
        """Return a plain dictionary of current state for logging and debugging."""
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
        
    def _clip_global_signals(self):
        self._global_ach_viz = max(0.0, min(1.0, self._global_ach_viz))
        self._global_ne_viz  = max(0.0, min(1.0, self._global_ne_viz))
        
    def _clip_step_signals(self):
        self._step_ach_viz   = max(0.0, min(1.0, self._step_ach_viz))
        self._step_ne_viz    = max(0.0, min(1.0, self._step_ne_viz))

    def _apply_ema_signal(self, previous_value: float, added_value: float, decay: float) -> float:
        """Mock helper for future EMA smoothing of accumulated modulation signals."""
        decay = max(0.0, min(1.0, decay))
        return decay * previous_value + added_value
