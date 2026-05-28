from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image
from peft.peft_model import PeftModel
from peft.mixed_model import PeftMixedModel
from transformers import AdamW, PreTrainedTokenizerFast

from runtime_scripts.runtime_memory_manager import RuntimeMemoryBankManager
from runtime_scripts.uniwm_schemas import UniWMInputBundle, StepPrediction, RoutePrediction
from scripts.load_model import load_model
from scripts.prompt_builder import build_action_prompt, build_viz_prompt
from scripts.action_utils import get_action_config, ActionCfg
from runtime_scripts.runtime_utils import (
    configure_action_tokenizer,
    decode_generated_image,
    decode_generated_text,
    processor_inputs_from_prompt,
    step_image_output_path,
    is_stop_action,
    load_config,
    validate_config,
)

REQUIRED_FIELDS: dict[str, dict | list] = {
    "load_model_args": ["model", "image_seq_length", "device", "use_memory_bank_inference"],
    "action_token_generation": ["range_profile", "bin_step"],
    "generation": {
        "action": ["multimodal_generation_mode", "current_substep", "max_new_tokens"],
        "visualization": ["multimodal_generation_mode", "current_substep"]
    },
    "training": ["initial_lr"]
}

class UniWMEngine:
    """Persistent online UniWM inference engine."""
    _action_cfg: ActionCfg
    _data_id: str = "unknown"

    @property
    def data_id(self):
        return self._data_id

    @data_id.setter
    def data_id(self, new_data_id: str) -> None:
        # NOTE: There might be a better way of setting the action_cfg than this, but this seemed clean enough to me for now
        self._data_id = "habitat" if new_data_id == "dummy" else new_data_id
        self._action_cfg = get_action_config(self._data_id)

    @property
    def action_cfg(self):
        return self._action_cfg

    def __init__(self, config_path: str = "cfg/habitat_uniwm_cfg.yaml", data_id = "habitat"):
        self.config = load_config(config_path).get("engine", {})
        validate_config(self.config, REQUIRED_FIELDS)

        self.device = self.config["load_model_args"]["device"]
        self.data_id = data_id

        loaded = load_model(SimpleNamespace(**self.config["load_model_args"]), None)
        self.model: PeftModel | PeftMixedModel = loaded["model"]
        self.processor: PreTrainedTokenizerFast = loaded["processor"]

        self.trainable_params = self._online_update_parameters(include_lm_head=False)
        self.optimizer = AdamW(
            self.trainable_params,
            lr=float(self.config["training"]["initial_lr"]),
            weight_decay=0.0,
        )

        self.memory_manager = RuntimeMemoryBankManager(self.model, self.config["load_model_args"]["use_memory_bank_inference"])
        configure_action_tokenizer(self.model, self.processor, self.config)

        if hasattr(self.model, "eval"):
            self.model.eval()

    def reset_memory(self, episode_id: str | None):
        self.memory_manager.setup_for_episode(episode_id=episode_id)

    def predict_step(
            self,
            bundle: UniWMInputBundle,
            save_path: str | None = None,
    ) -> StepPrediction:
        start_observation, goal_observation, current_observation, start_pose_str, action_text = bundle.unpack()
        action, raw, viz = self._predict_step(
            start_observation=start_observation,
            goal_observation=goal_observation,
            current_observation=current_observation,
            start_pose_str=start_pose_str,
            action_text=action_text,
            save_path=save_path
        )
        return StepPrediction(bundle, action, raw, viz)

    def predict_route(
        self,
        bundle: UniWMInputBundle,
        max_steps: int,
        output_dir: str | None = None,
    ) -> RoutePrediction:
        start_observation, goal_observation, current_observation, start_pose_str, _ = bundle.unpack()

        limit = int(max_steps)
        current = current_observation
        steps: list[StepPrediction] = []

        for step_index in range(limit):
            save_path = step_image_output_path(output_dir, step_index)
            step_action, step_raw_text, step_viz = self._predict_step(
                start_observation=start_observation,
                goal_observation=goal_observation,
                current_observation=current,
                start_pose_str=start_pose_str,
                save_path=save_path,
                is_real_obs=(step_index == 0)
            )

            new_bundle = bundle if step_index == 0 else UniWMInputBundle(
                start_observation,
                goal_observation,
                current,
                start_pose_str,
                step_action,
                bundle.metadata
            )

            steps.append(StepPrediction(new_bundle, step_action, step_raw_text, step_viz))
            if is_stop_action(step_action):
                return RoutePrediction(steps=steps, stopped=True, stop_reason="stop_action")
            if step_viz is None:
                return RoutePrediction(steps=steps, stopped=False, stop_reason="missing_visualization")
            current = step_viz

        return RoutePrediction(steps=steps, stopped=False, stop_reason="max_steps")

    def train_actions_batch(
        self,
        *,
        current_input: UniWMInputBundle,
        predicted: str | Image.Image,
        actual: str | Image.Image,
        gate: float = 1.0,
        lr_scale: float = 1.0,
        loss_weights: dict | None = None,
        max_grad_norm: float | None = None,
    ) -> dict[str, float | str | bool]:
        return NotImplemented

    def train_viz_step(
        self,
        *,
        current_input: UniWMInputBundle,
        predicted: str | Image.Image,
        actual: str | Image.Image,
        gate: float = 1.0,
        lr_scale: float = 1.0,
        loss_weights: dict | None = None,
        max_grad_norm: float | None = None,
    ) -> dict[str, float | str | bool]:
        return NotImplemented

    def apply_model_update(
        self,
        *,
        current_input: UniWMInputBundle,
        predicted: str | Image.Image,
        actual: str | Image.Image,
        gate: float = 1.0,
        lr_scale: float = 1.0,
        loss_weights: dict | None = None,
        max_grad_norm: float | None = None,
    ) -> dict[str, float | str | bool]:
        """
        Apply one bounded online LoRA update.
        """

        # I'd started working out and cleaning this up from an LLM stubbed-out method, but then realized I needed to figure out handling updates based on multiple steps at a time.
        return NotImplemented

        # if gate <= 0.0 or lr_scale <= 0.0:
        #     return {
        #         "applied": False,
        #         "reason": "gate_or_lr_scale_closed",
        #     }
        #
        # if type(predicted) is not type(actual):
        #     raise ValueError(f"Given mismatching predicted vs actual types: {type(predicted)} vs {type(actual)}")
        #
        # if isinstance(predicted, str):
        #     # action weight updates path
        #     batch = self._build_update_batch(
        #         current_input=current_input,
        #         predicted=predicted,
        #         actual=actual,
        #         prompt=
        #     )
        #     loss_config = self._make_online_loss_config(
        #         include_action_loss=True,
        #         include_image_loss=False,
        #         loss_weights=loss_weights,
        #     )
        # elif isinstance(predicted, Image.Image):
        #     # visualization weight updates path?
        #     batch = self._build_image_update_batch(
        #         current_input=current_input,
        #         target_image=target,
        #     )
        #     loss_config = self._make_online_loss_config(
        #         include_action_loss=False,
        #         include_image_loss=True,
        #         loss_weights=loss_weights,
        #     )
        # else:
        #     raise ValueError(f"Unexpected predicted typing: {type(predicted)}")
        #
        # self.model.train()
        # self.optimizer.zero_grad(set_to_none=True)
        # outputs = self.model(**batch["model_inputs"])
        #
        # supervised_loss, components = compute_supervised_uniwm_loss(
        #     model=self.model,
        #     outputs=outputs,
        #     batch=batch,
        #     tokenizer=self.processor,
        #     loss_config=loss_config,
        #     label_smoother=None,
        #     action_config=self.action_cfg,
        # )
        #
        # grad_norm = self._update_weights(float(gate) * float(lr_scale), supervised_loss, max_grad_norm)
        #
        # return {
        #     "applied": True,
        #     "update_scale": update_scale,
        #     "loss": float(supervised_loss.detach().cpu()),
        #     "scaled_loss": float(scaled_loss.detach().cpu()),
        #     "grad_norm": None if grad_norm is None else float(grad_norm.detach().cpu()),
        #     **components,
        # }

    def _predict_step(
        self,
        *,
        start_observation: Image.Image,
        goal_observation: Image.Image,
        current_observation: Image.Image,
        start_pose_str: str,
        is_real_obs: bool = True,
        action_text: str | None = None,
        save_path: str | None = None,
    ) -> tuple[str, str, Image.Image | None]:
        if start_observation is None or goal_observation is None:
            raise AssertionError("start_observation and goal_observation are required.")
        if not start_pose_str:
            raise AssertionError("start_pose_str is required.")

        if is_real_obs:
            self.memory_manager.start_new_step()

        current_observation = start_observation if current_observation is None else current_observation

        if action_text is None:
            action_inputs = processor_inputs_from_prompt(
                self.processor,
                input_text=build_action_prompt(
                    start_pose_str=start_pose_str,
                    dxy_range=self.action_cfg.get_dxy_tuple(),
                    dyaw_range=self.action_cfg.get_dyaw_tuple(),
                    prompt_style_idx=self.config.get("prompt_style_idx", 0),
                ),
                input_images=[start_observation, goal_observation, current_observation],
                device=self.device,
            )

            action_text, raw_text = self._predict_action(action_inputs, is_real_obs)
        else:
            raw_text = action_text

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

            visualization = self._predict_visualization(visualization_inputs, is_real_obs, save_path)

        if is_real_obs:
            self.memory_manager.store_step_memory()

        return action_text, raw_text, visualization

    def _predict_action(self, processor_inputs: Any, is_real_obs: bool) -> tuple[str, str]:
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.model.dtype):
            kwargs = self.memory_manager.get_action_kwargs(
                action_inputs=processor_inputs,
                action_gen_kwargs=dict(self.config["generation"]["action"]),
                is_real_obs=is_real_obs
            )

            outputs = self.model.generate(**processor_inputs, **kwargs)
        return decode_generated_text(self.processor, outputs)

    def _predict_visualization(self, processor_inputs: Any, is_real_obs: bool, save_path: str | None) -> Image.Image | None:
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.model.dtype):
            kwargs = self.memory_manager.get_viz_kwargs(
                viz_gen_kwargs=dict(self.config["generation"]["visualization"]),
                is_real_obs=is_real_obs
            )

            outputs = self.model.generate(**processor_inputs, **kwargs)
        return decode_generated_image(self.model, self.processor, outputs, save_path=save_path)

    def _build_action_batch(self, predicted, actual, prompt: str):

        # This is also an LLM-stubbed method that I'm still in the middle of adjusting.
        #   It's a decent start, but I've gotta replace a bunch of self.___ values with config values,
        #   and adjust it to handle whether I give it an image tensor or text as the "actual" to handle batching it properly either way.
        return NotImplemented
        # input_images = input_images or []
        # label_images = label_images or []
        #
        # tokenized_input = self.processor(
        #     [prompt],
        #     images=input_images if input_images else None,
        #     padding="max_length",
        #     return_tensors="pt",
        #     max_length=self.input_max_length,
        # )
        #
        # tokenized_label = self.label_processor(
        #     ["<image>"],
        #     images=label_images if label_images else None,
        #     padding="max_length",
        #     return_tensors="pt",
        #     max_length=self.label_max_length,
        # )
        #
        # # Match original UniWM: omit label-side starting token.
        # tokenized_label = {
        #     key: value[:, 1:] if key in ("input_ids", "attention_mask") else value
        #     for key, value in tokenized_label.items()
        # }
        #
        # # Replace image placeholder tokens in the input side, if present.
        # if input_images:
        #     tokenized_input["input_ids"] = self._replace_image_placeholders_with_vq_tokens(
        #         input_ids=tokenized_input["input_ids"],
        #         pixel_values=tokenized_input["pixel_values"],
        #     )
        #
        # # Replace image placeholder tokens in the label side, if present.
        # if label_images:
        #     tokenized_label["input_ids"] = self._replace_image_placeholders_with_vq_tokens(
        #         input_ids=tokenized_label["input_ids"],
        #         pixel_values=tokenized_label["pixel_values"],
        #     )
        #
        # # We only need pixel_values to produce image tokens; do not forward them here.
        # tokenized_input.pop("pixel_values", None)
        # tokenized_label.pop("pixel_values", None)
        #
        # input_ids = torch.cat(
        #     [tokenized_input["input_ids"], tokenized_label["input_ids"]],
        #     dim=1,
        # )
        #
        # attention_mask = torch.cat(
        #     [tokenized_input["attention_mask"], tokenized_label["attention_mask"]],
        #     dim=1,
        # )
        #
        # input_side_ignore = torch.full_like(
        #     tokenized_input["input_ids"],
        #     fill_value=self.ignore_index,
        # )
        #
        # labels = torch.cat(
        #     [input_side_ignore, tokenized_label["input_ids"]],
        #     dim=1,
        # )
        #
        # labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_index
        #
        # batch = {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "labels": labels,
        # }
        #
        # return self._move_batch_to_model_device(batch)


    def _update_weights(self, update_scale: float, supervised_loss: torch.Tensor, max_grad_norm: float | None) -> torch.Tensor | None:
        scaled_loss = update_scale * supervised_loss
        scaled_loss.backward()

        grad_norm = None
        if max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.trainable_params,
                max_norm=max_grad_norm,
            )

        self.optimizer.step()

        return grad_norm

    def _online_update_parameters(
            self,
            *,
            include_lm_head: bool = False,
    ) -> list[torch.nn.Parameter]:
        params = []

        for name, param in self.model.named_parameters():
            is_lora = "lora_" in name
            is_lm_head = "lm_head" in name or "modules_to_save" in name

            should_train = is_lora or (include_lm_head and is_lm_head)
            param.requires_grad_(should_train)

            if should_train:
                params.append(param)

        if not params:
            raise RuntimeError("No online-trainable parameters selected.")

        return params