from __future__ import annotations

import math
from typing import Any
from runtime_scripts.runtime_utils import validate_config, clamp, ema_decay


REQUIRED_FIELDS: dict[str, list[str]] = {
    "decay_tau": ["ach", "ne"],
    "learning_mods": ["ach_base_weight", "ne_bias", "ne_range", "ne_curve_width"],
    "norm_scalers": ["novelty", "learning_surprise", "learning_progress", "persistent_loss"],
    "event_strengths": ["pred_mismatch", "mem_novelty", "act_uncertainty", "viz_uncertainty", "learning_surprise", "learning_progress", "persistent_error", "collision"]
}

class ModulatorSystem:
    """Collect per-step events and expose current ACh/NE-derived query values.

    Trigger methods currently record their raw event values only. Mapping those
    events into aggregated ACh-like and NE-like modulation is intentionally
    deferred.
    """
    
    _system_name: str  = "default"
    _global_ach: float = 0.0
    _global_ne: float  = 0.0
    _step_ach: float   = 0.0
    _step_ne: float    = 0.0
    
    _step_events: dict[str, float] = {}

    def __init__(self, enabled: bool, config: dict | None):
        """Initialize the scaffold with optional visualization-weight settings."""
        self.enabled = enabled
        
        if enabled and config is not None:
            validate_config(config, REQUIRED_FIELDS)
            self.config = config
            self.step_id: int = 0

    def reset_episode(self) -> None:
        """Clear all modulation state at the start of an episode."""
        if not self.enabled:
            return
        
        self.step_id: int    = 0
        self._global_ach = 0.0
        self._global_ne  = 0.0
        self._step_ach   = 0.0
        self._step_ne    = 0.0
        
        self._step_events = {}

    def start_step(self) -> None:
        """Clear per-step state and optionally associate it with a step id."""
        if not self.enabled:
            return
        
        self.step_id += 1
        self._step_ach = 0.0
        self._step_ne  = 0.0
        
        self._step_events = {}
        
        
#----------------- Event Triggers -----------------# 
    def on_prediction_mismatch(self, mismatch: float) -> None:
        """Record a prediction-mismatch event for future aggregation."""
        if not self.enabled:
            return
        
        self._step_events["prediction_mismatch"] = mismatch
        
        self._step_ach += mismatch * self.config["event_strengths"]["pred_mismatch"][0]
        self._step_ne  += mismatch * self.config["event_strengths"]["pred_mismatch"][1]

    def on_memory_novelty(self, novelty: float) -> None:
        """Record a memory-novelty event for future aggregation."""
        if not self.enabled:
            return
        
        self._step_events["memory_novelty"] = novelty
        novelty = 1 - math.exp(-novelty / self.config["norm_scalers"]["novelty"])
        self._step_events["norm_memory_novelty"] = novelty
        
        self._step_ach += novelty * self.config["event_strengths"]["mem_novelty"][0]
        self._step_ne  += novelty * self.config["event_strengths"]["mem_novelty"][1]

    def on_action_uncertainty(self, entropy: float) -> None:
        """Record an action-uncertainty event for future aggregation."""
        if not self.enabled:
            return
        
        self._step_events["act_uncertainty"] = entropy
        entropy = clamp(entropy, 0, 1)
        self._step_events["clamped_act_uncertainty"] = entropy
        
        self._step_ach += entropy * self.config["event_strengths"]["act_uncertainty"][0]
        self._step_ne  += entropy * self.config["event_strengths"]["act_uncertainty"][1]

    def on_visual_uncertainty(self, entropy: float) -> None:
        """Record a visual-uncertainty event for future aggregation."""
        if not self.enabled:
            return
        
        self._step_events["viz_uncertainty"] = entropy
        entropy = clamp(entropy, 0, 1)
        self._step_events["clamped_viz_uncertainty"] = entropy
        
        self._step_ach += entropy * self.config["event_strengths"]["viz_uncertainty"][0]
        self._step_ne  += entropy * self.config["event_strengths"]["viz_uncertainty"][1]

    def on_learning_surprise(self, surprise: float) -> None:
        """Record a visual-learning-surprise event for future aggregation."""
        if not self.enabled:
            return
        
        self._step_events["learning_surprise"] = surprise
        surprise = 1 - math.exp(
            -max(0.0, surprise) / self.config["norm_scalers"]["learning_surprise"]
        )
        self._step_events["norm_learning_surprise"] = surprise
        
        self._step_ach += surprise * self.config["event_strengths"]["learning_surprise"][0]
        self._step_ne  += surprise * self.config["event_strengths"]["learning_surprise"][1]

    def on_learning_progress(self, progress: float) -> None:
        """Record a visual-learning-progress event for future aggregation."""
        if not self.enabled:
            return
        
        self._step_events["learning_progress"] = progress
        progress = 1 - math.exp(
            -max(0.0, progress) / self.config["norm_scalers"]["learning_progress"]
        )
        self._step_events["norm_learning_progress"] = progress
        
        self._step_ach += progress * self.config["event_strengths"]["learning_progress"][0]
        self._step_ne  += progress * self.config["event_strengths"]["learning_progress"][1]

    def on_persistent_error(self, error: float) -> None:
        """Record a persistent visual-learning-error event for future aggregation."""
        if not self.enabled:
            return
        
        self._step_events["persistent_error"] = error
        error = 1 - math.exp(-error / self.config["norm_scalers"]["persistent_loss"])
        self._step_events["norm_persistent_error"] = error
        
        self._step_ach += error * self.config["event_strengths"]["persistent_error"][0]
        self._step_ne  += error * self.config["event_strengths"]["persistent_error"][1]

    def on_collision_or_unsafe(self, strength: float = 1.0) -> None:
        """Record a collision or unsafe-state event for future aggregation."""
        if not self.enabled or strength == 0.0:
            return
        
        self._step_events["collision"] = strength
        
        self._step_ach += strength * self.config["event_strengths"]["collision"][0]
        self._step_ne  += strength * self.config["event_strengths"]["collision"][1]
        
    def ema_update_signals(self):
        if not self.enabled:
            return
        
        self._clip_step_signals()
        
        self._global_ach = ema_decay(
            self._global_ach,
            self._step_ach,
            self.config["decay_tau"]["ach"]
        )
        
        self._global_ne = ema_decay(
            self._global_ne,
            self._step_ne,
            self.config["decay_tau"]["ne"]
        )
        
        self._clip_global_signals()

    def compute_step_update_weight(self) -> float:
        """Compute the current visual update weight from ACh-like and NE-like state."""
        if not self.enabled:
            return 1.0

        if self._step_events.get("collision", 0.0) > 0.0:
            self._step_events["update_weight"] = 0.0
            return 0.0

        ne_mult = self._compute_ne_mult()
        ach_mult = self._compute_ach_mult()
        
        weight = ach_mult * ne_mult
        max_weight = 1 + self.config["learning_mods"]["ach_base_weight"] * self.config["learning_mods"]["ne_range"][1]
        clipped_weight = clamp(weight, 0.0, max_weight)

        self._step_events["update_weight"] = clipped_weight
        return clipped_weight

    def get_current_state(self) -> dict[str, Any]:
        """Return a plain dictionary of current state for logging and debugging."""
        if not self.enabled:
            return {}
        
        return {
            "mod_system":   self._system_name,
            "step_id":      self.step_id,
            
            "gl_ach":       self._global_ach,
            "gl_ne_visual": self._global_ne,
            "st_ach":       self._step_ach,
            "st_ne_visual": self._step_ne,
            
            "step_events":  self._step_events.copy()
        }
        
    # TODO: Validate that ACh and NE interact with learning in the same overall enhancing/inhibiting ways for non-visual-cortex ways.  Otherwise, these compute functions will need to be unique per mod-system
    def _compute_ach_mult(self) -> float:
        return 1 + self.config["learning_mods"]["ach_base_weight"] * self._global_ach
        
    def _compute_ne_mult(self) -> float:
        # Optimal NE marks the center of the inverted-U arousal curve for visual learning.
        optimal_ne = self.config["learning_mods"]["ne_bias"]
        # NE range of learning modulation scales the Gaussian min and max height for a centered gaussian.
        ne_range = list(self.config["learning_mods"]["ne_range"])
        ne_min = ne_range[0]
        ne_max = ne_range[1]

        # NE distance shifts the Gaussian peak along the 0-to-1 NE range.
        ne_distance = self._global_ne - optimal_ne
        curve_width = max(
            float(self.config["learning_mods"]["ne_curve_width"]),
            1e-8,
        )
        
        gaussian_gamma = math.log(2.0) / (curve_width ** 2)
        gaussian_endpoint = math.exp(-gaussian_gamma * 0.25)
        gaussian_height = math.exp(-gaussian_gamma * ne_distance**2)
        normalized_gaussian = (gaussian_height - gaussian_endpoint) / (1.0 - gaussian_endpoint)
        return ne_min + (ne_max - ne_min) * normalized_gaussian
        
    def _clip_global_signals(self):
        self._global_ach = clamp(self._global_ach, 0.0, 1.0)
        self._global_ne  = clamp(self._global_ne, 0.0, 1.0)
        
    def _clip_step_signals(self):
        self._step_ach   = clamp(self._step_ach, 0.0, 1.0)
        self._step_ne    = clamp(self._step_ne, 0.0, 1.0)