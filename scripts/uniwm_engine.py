from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Optional

import torch
from PIL import Image

from scripts.habitat_uniwm_schemas import UniWMInputBundle
from scripts.load_model import load_model
from scripts.prompt_builder import build_action_prompt, build_viz_prompt
from scripts.action_utils import get_action_ranges
from scripts.uniwm_inference_utils import (
    configure_action_tokenizer,
    decode_generated_image,
    decode_generated_text,
    processor_inputs_from_prompt,
    step_image_output_path,
    is_stop_action,
    load_config,
    validate_config,
    StepPrediction,
    RoutePrediction
)

REQUIRED_FIELDS = {
    "load_model_args": ["model", "image_seq_length", "device", "use_memory_bank_inference"],
    "action_token_generation": ["range_profile", "bin_step"],
    "generation": {
        "action": ["multimodal_generation_mode", "current_substep", "max_new_tokens"],
        "visualization": ["multimodal_generation_mode", "current_substep"]
    },
    "route": ["max_steps"],
}

class UniWMEngine:
    """Persistent online UniWM inference engine."""

    def __init__(self, config_path: str = "cfg/habitat_uniwm_cfg.yaml", data_id = "habitat"):
        self.config = load_config(config_path)
        validate_config(self.config, REQUIRED_FIELDS)

        self.device = self.config["load_model_args"]["device"]
        self.action_ranges = get_action_ranges("habitat" if data_id == "dummy" else data_id)

        loaded = load_model(SimpleNamespace(**self.config["load_model_args"]), None)
        self.processor = loaded["processor"]
        self.model = loaded["model"]
        configure_action_tokenizer(self.model, self.processor, self.config)

        if hasattr(self.model, "eval"):
            self.model.eval()

    def set_action_ranges(self, new_data_id: str = "habitat") -> None:
        self.action_ranges = get_action_ranges(new_data_id)

    def predict_route(
        self,
        bundle: UniWMInputBundle,
        max_steps: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> RoutePrediction:
        start_observation, goal_observation, current_observation, start_pose_str = bundle.unpack()

        limit = int(max_steps) if max_steps is not None else int(self.config["route"]["max_steps"])
        current = current_observation
        steps: List[StepPrediction] = []

        for step_index in range(limit):
            save_path = step_image_output_path(output_dir, step_index)
            step = self._predict_step(
                start_observation=start_observation,
                goal_observation=goal_observation,
                current_observation=current,
                start_pose_str=start_pose_str,
                save_path=save_path,
            )
            steps.append(step)
            if is_stop_action(step.action_text):
                return RoutePrediction(steps=steps, stopped=True, stop_reason="stop_action")
            if step.visualization is None:
                return RoutePrediction(steps=steps, stopped=False, stop_reason="missing_visualization")
            current = step.visualization

        return RoutePrediction(steps=steps, stopped=False, stop_reason="max_steps")

    def predict_step(
            self,
            bundle: UniWMInputBundle,
            save_path: Optional[str] = None,
    ) -> StepPrediction:
        return self._predict_step(save_path=save_path, **(bundle.unpack()))

    def _predict_step(
        self,
        *,
        start_observation: Any,
        goal_observation: Any,
        current_observation: Any,
        start_pose_str: str,
        save_path: Optional[str],
    ) -> StepPrediction:
        if start_observation is None or goal_observation is None:
            raise AssertionError("start_observation and goal_observation are required.")
        if not start_pose_str:
            raise AssertionError("start_pose_str is required.")

        current_observation = start_observation if current_observation is None else current_observation

        action_inputs = processor_inputs_from_prompt(
            self.processor,
            input_text=build_action_prompt(
                start_pose_str=start_pose_str,
                dxy_range=self.action_ranges["dxy"],
                dyaw_range=self.action_ranges["dyaw"],
                prompt_style_idx=self.config.get("prompt_style_idx", 0),
            ),
            input_images=[start_observation, goal_observation, current_observation],
            device=self.device,
        )

        action_text = self._predict_action(action_inputs)

        visualization = None
        if not is_stop_action(action_text):
            visualization_inputs = processor_inputs_from_prompt(
                self.processor,
                input_text=build_viz_prompt(
                    decoded_action=action_text,
                    start_pose_str=start_pose_str,
                    prompt_style_idx=self.config.get("prompt_style_idx", 0),
                ),
                input_images=[start_observation, goal_observation, current_observation],
                device=self.device,
            )

            visualization = self._predict_visualization(visualization_inputs, save_path=save_path)

        return StepPrediction(action_text=action_text, visualization=visualization)

    def _predict_action(self, processor_inputs: Any) -> str:
        kwargs = dict(self.config["generation"]["action"])

        if not self.config["load_model_args"]["use_memory_bank_inference"]:
            kwargs.pop("current_substep", None)

        with torch.no_grad(), torch.amp.autocast(device_type=self.device, dtype=getattr(self.model, "dtype", None)):
            outputs = self.model.generate(**processor_inputs, **kwargs)
        return decode_generated_text(self.processor, outputs)

    def _predict_visualization(self, processor_inputs: Any, save_path: Optional[str]) -> Optional[Image.Image]:
        kwargs = dict(self.config["generation"]["visualization"])
        kwargs["max_new_tokens"] = self.model.image_token_num + 20

        if not self.config["load_model_args"]["use_memory_bank_inference"]:
            kwargs.pop("current_substep", None)

        with torch.no_grad(), torch.amp.autocast(device_type=self.device, dtype=getattr(self.model, "dtype", None)):
            outputs = self.model.generate(**processor_inputs, **kwargs)
        return decode_generated_image(self.model, self.processor, outputs, save_path=save_path)
