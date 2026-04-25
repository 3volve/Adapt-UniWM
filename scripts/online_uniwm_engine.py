from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import yaml
from PIL import Image

from scripts.habitat_uniwm_schemas import UniWMInputBundle
from scripts.load_model import load_model
from scripts.prompt_builder import build_action_prompt, build_viz_prompt
from scripts.uniwm_inference_utils import (
    configure_action_tokenizer,
    decode_generated_image,
    decode_generated_text,
    generation_kwargs,
    processor_inputs_from_prompt,
)


@dataclass(frozen=True)
class OnlineStepPrediction:
    action_text: str
    visualization: Optional[Image.Image] = None


@dataclass(frozen=True)
class OnlineRoutePrediction:
    steps: List[OnlineStepPrediction]
    stopped: bool
    stop_reason: str


class OnlineUniWMEngine:
    """Persistent online UniWM inference engine."""

    def __init__(self, config_path: str = "cfg/online_uniwm.yaml"):
        self.config = self._load_config(config_path)
        self.device = self.config.get("load_model_args", {}).get("device")

        loaded = load_model(SimpleNamespace(**self.config.get("load_model_args", {})), None)
        self.processor = loaded["processor"]
        self.model = loaded["model"]
        configure_action_tokenizer(self.model, self.processor, self.config)

        if hasattr(self.model, "eval"):
            self.model.eval()

    def predict_step(
        self,
        bundle: UniWMInputBundle,
        save_path: Optional[str] = None,
    ) -> OnlineStepPrediction:
        unpacked = self._unpack_bundle(bundle)
        return self._predict_step(save_path=save_path, **unpacked)

    def predict_route(
        self,
        bundle: UniWMInputBundle,
        max_steps: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> OnlineRoutePrediction:
        unpacked = self._unpack_bundle(bundle)
        return self._predict_route(max_steps=max_steps, output_dir=output_dir, **unpacked)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _predict_step(
        self,
        *,
        start_observation: Any,
        goal_observation: Any,
        current_observation: Any,
        start_pose_str: str,
        save_path: Optional[str],
    ) -> OnlineStepPrediction:
        self._validate_step_inputs(
            start_observation=start_observation,
            goal_observation=goal_observation,
            current_observation=current_observation,
            start_pose_str=start_pose_str,
        )

        action_inputs = self._action_processor_inputs(
            start_observation=start_observation,
            goal_observation=goal_observation,
            current_observation=current_observation,
            start_pose_str=start_pose_str,
        )
        action_text = self._predict_action(action_inputs)

        visualization = None
        if not self._is_stop_action(action_text):
            visualization_inputs = self._visualization_processor_inputs(
                start_observation=start_observation,
                goal_observation=goal_observation,
                current_observation=current_observation,
                start_pose_str=start_pose_str,
                action_text=action_text,
            )
            visualization = self._predict_visualization(visualization_inputs, save_path=save_path)

        return OnlineStepPrediction(action_text=action_text, visualization=visualization)

    def _predict_route(
        self,
        *,
        start_observation: Any,
        goal_observation: Any,
        current_observation: Any,
        start_pose_str: str,
        max_steps: Optional[int],
        output_dir: Optional[str],
    ) -> OnlineRoutePrediction:
        self._validate_step_inputs(
            start_observation=start_observation,
            goal_observation=goal_observation,
            current_observation=current_observation,
            start_pose_str=start_pose_str,
        )
        limit = max_steps if max_steps is not None else int(self.config.get("route", {}).get("max_steps", 10))
        current = current_observation
        steps: List[OnlineStepPrediction] = []

        for step_index in range(limit):
            save_path = None if not output_dir else self._step_image_path(output_dir, step_index)
            step = self._predict_step(
                start_observation=start_observation,
                goal_observation=goal_observation,
                current_observation=current,
                start_pose_str=start_pose_str,
                save_path=save_path,
            )
            steps.append(step)
            if self._is_stop_action(step.action_text):
                return OnlineRoutePrediction(steps=steps, stopped=True, stop_reason="stop_action")
            if step.visualization is None:
                return OnlineRoutePrediction(steps=steps, stopped=False, stop_reason="missing_visualization")
            current = step.visualization

        return OnlineRoutePrediction(steps=steps, stopped=False, stop_reason="max_steps")

    def _predict_action(self, processor_inputs: Any) -> str:
        with torch.no_grad():
            outputs = self.model.generate(**processor_inputs, **generation_kwargs(self.config, "action", self.model))
        return decode_generated_text(self.processor, outputs)

    def _predict_visualization(self, processor_inputs: Any, save_path: Optional[str]) -> Optional[Image.Image]:
        with torch.no_grad():
            outputs = self.model.generate(**processor_inputs, **generation_kwargs(self.config, "visualization", self.model))
        return decode_generated_image(self.model, self.processor, outputs, save_path=save_path)

    def _action_processor_inputs(
        self,
        *,
        start_observation: Any,
        goal_observation: Any,
        current_observation: Any,
        start_pose_str: str,
    ) -> Any:
        ranges = self._action_ranges()
        input_text = build_action_prompt(
            start_pose_str=start_pose_str,
            dxy_range=ranges["dxy"],
            dyaw_range=ranges["dyaw"],
        )
        return processor_inputs_from_prompt(
            self.processor,
            input_text=input_text,
            input_images=[start_observation, goal_observation, current_observation],
            device=self.device,
        )

    def _visualization_processor_inputs(
        self,
        *,
        start_observation: Any,
        goal_observation: Any,
        current_observation: Any,
        start_pose_str: str,
        action_text: str,
    ) -> Any:
        input_text = build_viz_prompt(decoded_action=action_text, start_pose_str=start_pose_str)
        return processor_inputs_from_prompt(
            self.processor,
            input_text=input_text,
            input_images=[start_observation, goal_observation, current_observation],
            device=self.device,
        )

    def _unpack_bundle(self, bundle: UniWMInputBundle) -> Dict[str, Any]:
        return {
            "start_observation": bundle.start_observation,
            "goal_observation": bundle.goal_observation,
            "current_observation": bundle.current_observation,
            "start_pose_str": bundle.start_pose_str,
        }

    def _validate_step_inputs(
        self,
        *,
        start_observation: Any,
        goal_observation: Any,
        current_observation: Any,
        start_pose_str: str,
    ) -> None:
        if start_observation is None or goal_observation is None or current_observation is None:
            raise AssertionError("start_observation, goal_observation, and current_observation are required.")
        if not start_pose_str:
            raise AssertionError("start_pose_str is required.")

    def _action_ranges(self) -> Dict[str, Any]:
        token_cfg = self.config.get("action_token_generation", {})
        range_profile = token_cfg.get("range_profile", "habitat")
        from scripts.action_utils import get_action_ranges

        return get_action_ranges(range_profile)

    def _step_image_path(self, output_dir: Optional[str], step_index: int) -> Optional[str]:
        if not output_dir:
            return None
        return str(Path(output_dir) / f"step_{step_index + 1}_observation.png")

    def _is_stop_action(self, action_text: str) -> bool:
        return action_text.strip().lower() == "stop"
