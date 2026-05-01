from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml

from scripts.habitat_uniwm_schemas import UniWMInputBundle
from scripts.prompt_builder import build_action_prompt
from scripts.uniwm_inference_utils import processor_inputs_from_prompt

# Swap this import to the real loader later when the real UniWM hookup is ready.
from uniwm.dummy import load_model


class UniWMBackend:
    """Small runner-facing wrapper around the current UniWM implementation."""

    def __init__(self, config_path: str = "cfg/uniwm_backend.yaml"):
        self.config = self._load_config(config_path)
        loaded = load_model(self._build_args(), None)
        self.processor = loaded["processor"]
        self.model = loaded["model"]

    def predict_action(self, bundle: UniWMInputBundle) -> str:
        ranges = self._action_ranges()
        input_text = build_action_prompt(
            start_pose_str=bundle.start_pose_str,
            dxy_range=ranges["dxy"],
            dyaw_range=ranges["dyaw"],
        )
        inputs = processor_inputs_from_prompt(
            self.processor,
            input_text=input_text,
            input_images=[
                bundle.start_observation,
                bundle.goal_observation,
                bundle.current_observation,
            ],
        )
        outputs = self.model.generate(**inputs, **self._generation_kwargs())
        return self.processor.batch_decode(outputs[0], skip_special_tokens=False)[0].strip()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _build_args(self) -> SimpleNamespace:
        flags = self.config.get("mode_flags", {})
        return SimpleNamespace(
            model=self.config.get("model", "anole"),
            model_ckpt=self.config.get("model_ckpt"),
            image_seq_length=int(self.config.get("image_seq_length", 784)),
            action_range_profile=self.config.get("action_range_profile", "habitat"),
            do_train=bool(flags.get("do_train", False)),
            do_single_step_eval=bool(flags.get("do_single_step_eval", False)),
            do_task_level_eval=bool(flags.get("do_task_level_eval", True)),
            do_rollout_eval=bool(flags.get("do_rollout_eval", False)),
            use_memory_bank_inference=bool(flags.get("use_memory_bank_inference", False)),
        )

    def _generation_kwargs(self) -> Dict[str, Any]:
        generation = self.config.get("generation", {})
        return {
            "max_new_tokens": int(generation.get("max_new_tokens", 64)),
            "multimodal_generation_mode": generation.get("multimodal_generation_mode", "interleaved-text-image"),
            "current_substep": generation.get("current_substep", "action"),
        }

    def _action_ranges(self) -> Dict[str, Any]:
        from scripts.action_utils import get_action_ranges

        return get_action_ranges(self.config.get("action_range_profile", "habitat"))
