from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

from scripts.uniwm_losses import compute_supervised_uniwm_loss
import torch
from PIL import Image
from peft.peft_model import PeftModel
from peft.mixed_model import PeftMixedModel
from transformers import AdamW, PreTrainedTokenizerFast

from runtime_scripts.runtime_memory_manager import RuntimeMemoryBankManager
from runtime_scripts.uniwm_schemas import MemorySnapshot, UniWMInputBundle, StepPrediction, RoutePrediction
from scripts.load_model import load_model
from scripts.prompt_builder import build_action_prompt, build_viz_prompt
from scripts.action_utils import get_action_config, ActionCfg
from runtime_scripts.runtime_utils import (
    configure_action_tokenizer,
    decode_generated_image,
    decode_generated_text,
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
    "training": {
        "hyper_params": ["initial_lr"], 
        "visualization": ["use_cache"], 
        "loss": ["include_action_loss", "include_image_loss", "action_loss_weight", "image_loss_weight", "log_prefix"]
    },
}

class UniWMEngine:
    """Persistent online UniWM inference engine."""
    _action_cfg: ActionCfg = ActionCfg()
    
    @property
    def action_cfg(self):
        return self._action_cfg
    
    @action_cfg.setter
    def action_cfg(self, new_cfg_target: str) -> None:
        self._action_cfg = get_action_config(new_cfg_target)
        

    def __init__(self, config_path: str = "cfg/habitat_uniwm_cfg.yaml"):
        self.config = load_config(config_path).get("engine", {})
        validate_config(self.config, REQUIRED_FIELDS)

        self.device = self.config["load_model_args"]["device"]

        loaded = load_model(SimpleNamespace(**self.config["load_model_args"]), None)
        self.model: PeftModel | PeftMixedModel = loaded["model"]
        self.processor: PreTrainedTokenizerFast = loaded["processor"]

        self.trainable_params = self._online_update_parameters(include_lm_head=False)
        self.optimizer = AdamW(
            self.trainable_params,
            lr=float(self.config["training"]["hyper_params"]["initial_lr"]),
            weight_decay=0.0,
        )

        self.memory_manager = RuntimeMemoryBankManager(self.model, self.config["load_model_args"]["use_memory_bank_inference"])
        configure_action_tokenizer(self.model, self.processor, self.config)

        if hasattr(self.model, "eval"):
            self.model.eval()

    def reset_episode(self, episode_id: str | None, bundle: UniWMInputBundle):
        self.memory_manager.setup_for_episode(episode_id=episode_id)
        self.start_tok_obs = self._image_to_vq_bpe_tokens(bundle.start_observation)
        self.goal_tok_obs = self._image_to_vq_bpe_tokens(bundle.goal_observation)
        self.current_tok_obs = self._image_to_vq_bpe_tokens(bundle.current_observation)
        
        action_inputs = self._processor_inputs_from_prompt(
            input_text=build_action_prompt(
                start_pose_str=bundle.start_pose_str,
                dxy_range=self.action_cfg.get_dxy_tuple(),
                dyaw_range=self.action_cfg.get_dyaw_tuple(),
                prompt_style_idx=self.config["generation"]["prompt_style_idx"]
            ),
        )
        
        self.memory_manager.start_new_step()
        self.memory_manager.initialize_step_memory(action_inputs)
        
    def init_working_memory(self, real_obs: Image.Image, start_pose_str: str, store_global_memory: bool = True):
        if self.start_tok_obs is None or self.goal_tok_obs is None:
            raise AssertionError("Attempted to store step memory before encoding the goal or start observations.")
        
        if store_global_memory:
            self.memory_manager.store_step_memory()
        
        self.current_tok_obs = self._image_to_vq_bpe_tokens(real_obs)
        
        action_inputs = self._processor_inputs_from_prompt(
            input_text=build_action_prompt(
                start_pose_str=start_pose_str,
                dxy_range=self.action_cfg.get_dxy_tuple(),
                dyaw_range=self.action_cfg.get_dyaw_tuple(),
                prompt_style_idx=self.config["generation"]["prompt_style_idx"]
            ),
        )
        
        self.memory_manager.start_new_step()
        self.memory_manager.initialize_step_memory(action_inputs)
        
    def predict_step(
            self,
            bundle: UniWMInputBundle,
            save_path: str | None = None,
    ) -> StepPrediction:
        _, _, _, start_pose_str, action_text = bundle.unpack()
        action, viz = self._predict_step(
            start_pose_str=start_pose_str,
            action_text=action_text,
            save_path=save_path
        )
        return StepPrediction(bundle, action, viz)

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
        
        self.memory_manager.cache_step_state()
        cached_current_tok_obs = self.current_tok_obs
        
        for step_index in range(limit):
            save_path = step_image_output_path(output_dir, step_index)
            
            step_action, step_viz = self._predict_step(
                start_pose_str=start_pose_str,
                save_path=save_path,
            )

            new_bundle = bundle if step_index == 0 else UniWMInputBundle(
                start_observation,
                goal_observation,
                current,
                start_pose_str,
                step_action,
                bundle.metadata
            )

            steps.append(StepPrediction(new_bundle, step_action, step_viz))
            if is_stop_action(step_action):
                self.memory_manager.load_cached_state()
                self.current_tok_obs = cached_current_tok_obs
                return RoutePrediction(steps=steps, stopped=True, stop_reason="stop_action")
            if step_viz is None:
                self.memory_manager.load_cached_state()
                self.current_tok_obs = cached_current_tok_obs
                return RoutePrediction(steps=steps, stopped=False, stop_reason="missing_visualization")
            current = step_viz
            
            if step_index < limit - 1:
                self.init_working_memory(current, start_pose_str, False)
            
        self.memory_manager.load_cached_state()
        self.current_tok_obs = cached_current_tok_obs
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
        prediction: StepPrediction,
        lr_scaler: float
    ) -> dict[str, float]:
        """Note: Make sure you are calling the UniWMEngine.init_working_memory() when you want to store a prior step into global memory and update the observation in working KV memory used by the model."""
        
        if prediction.real_input_obs is None:
            raise ValueError("[UNEXPECTED ERROR] Wrapper failed to set a real input observation on the StepPrediction passed into train_viz_step")
        if prediction.real_next_obs is None:
            raise ValueError("[UNEXPECTED ERROR] Wrapper failed to set a real predicted visualization on the StepPrediction passed into train_viz_step")

        visualization_inputs = self._processor_inputs_from_prompt(
            input_text=build_viz_prompt(
                decoded_action=prediction.action_text,
                start_pose_str=prediction.input_bundle.start_pose_str,
                prompt_style_idx=self.config["generation"]["prompt_style_idx"]
            )
        )

        training_inputs: Mapping[str, Any] = self._build_image_batch(
            viz_inputs=visualization_inputs,
            target_viz_tokens=self._image_to_vq_bpe_tokens(prediction.real_next_obs)
        )

        for group in self.optimizer.param_groups:
            group["lr"] = float(self.config["training"]["hyper_params"]["initial_lr"]) * lr_scaler

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        memory_kwargs = self.memory_manager.get_viz_kwargs(
            viz_gen_kwargs=dict(self.config["training"]["visualization"])
        )

        with torch.autocast(device_type="cuda", dtype=self.model.dtype):
            outputs = self.model(
                **{key: value for key, value in training_inputs .items() if key != "labels"},
                **memory_kwargs
            )
            
            loss_cfg = self.config["training"]["loss"].copy()
            loss_cfg["include_action_loss"] = False
            loss_cfg["action_loss_weight"] = 0.0

            loss, components = compute_supervised_uniwm_loss(
                model=self.model,
                outputs=outputs,
                batch=training_inputs ,
                tokenizer=self.processor,
                loss_config=loss_cfg
            )

        loss.backward()
        self.optimizer.step()
        self.model.eval()

        return {
            **components,
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }

    def _predict_step(
        self,
        *,
        start_pose_str: str,
        action_text: str | None = None,
        save_path: str | None = None,
    ) -> tuple[str, Image.Image | None]:
        """Note: Make sure you are calling the UniWMEngine.init_working_memory() when you want to store a prior step into global memory and update the observation in working KV memory used by the model."""
        
        if not start_pose_str:
            raise AssertionError("start_pose_str is required.")

        action_inputs = self._processor_inputs_from_prompt(
            input_text=build_action_prompt(
                start_pose_str=start_pose_str,
                dxy_range=self.action_cfg.get_dxy_tuple(),
                dyaw_range=self.action_cfg.get_dyaw_tuple(),
                prompt_style_idx=self.config["generation"]["prompt_style_idx"]
            )
        )

        action_text = action_text if action_text else self._predict_action(action_inputs)

        viz = None
        if not is_stop_action(action_text):
            visualization_inputs = self._processor_inputs_from_prompt(
                input_text=build_viz_prompt(
                    decoded_action=action_text,
                    start_pose_str=start_pose_str,
                    prompt_style_idx=self.config["generation"]["prompt_style_idx"]
                )
            )
            
            viz = self._predict_visualization(visualization_inputs, save_path)
            
        return action_text, viz

    def _predict_action(self, processor_inputs: Any) -> str:
        kwargs = self.memory_manager.get_action_kwargs(
            action_gen_kwargs=dict(self.config["generation"]["action"])
        )
        
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.model.dtype):
            outputs = self.model.generate(**processor_inputs, **kwargs)
            
        return decode_generated_text(self.processor, outputs)

    def _predict_visualization(self, processor_inputs: Any, save_path: str | None) -> Image.Image | None:
        kwargs = self.memory_manager.get_viz_kwargs(
            viz_gen_kwargs=dict(self.config["generation"]["visualization"])
        )
            
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.model.dtype):
            outputs = self.model.generate(**processor_inputs, **kwargs)
            
        return decode_generated_image(self.model, self.processor, outputs, save_path=save_path)

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
    
#----------------------------- Private Processor-related Helpers -----------------------------#
    def _image_to_vq_bpe_tokens(self, image: Image.Image) -> torch.LongTensor:
        pixel_values = self.processor(text="<image>", images=image, return_tensors="pt")["pixel_values"]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.model.dtype):
            return self.model.model.model.get_image_tokens(pixel_values.to(self.model.device, dtype=self.model.dtype)).to(torch.long)
        
    def _processor_inputs_from_prompt(
        self,
        input_text: str,
        current_tok_obs: torch.LongTensor | None = None,
        start_tok_obs: torch.LongTensor | None = None,
        goal_tok_obs: torch.LongTensor | None = None
    ) -> Any:
        tokenized_images = [
            start_tok_obs if start_tok_obs is not None else self.start_tok_obs,
            goal_tok_obs if goal_tok_obs is not None else self.goal_tok_obs,
            current_tok_obs if current_tok_obs is not None else self.current_tok_obs
        ]
        
        inputs = self.processor(
            text=[input_text],
            return_tensors="pt",
        )
        
        image_mask = inputs["input_ids"] == self.processor.image_token_id
        image_tokens = torch.cat([tokens.reshape(-1) for tokens in tokenized_images])
        inputs["input_ids"].masked_scatter_(
            image_mask,
            image_tokens.to(inputs["input_ids"].device, inputs["input_ids"].dtype),
        )

        if self.device and hasattr(inputs, "to"):
            return inputs.to(self.device)
        return inputs
    
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
        
    def _build_image_batch(
        self,
        viz_inputs: Any,
        target_viz_tokens: torch.LongTensor,
    ) -> Mapping[str, Any]:
        target_inputs = self.processor(text=["<image>"], return_tensors="pt")
        target_mask = target_inputs["input_ids"] == self.processor.image_token_id
        target_inputs["input_ids"].masked_scatter_(
            target_mask,
            target_viz_tokens.reshape(-1).to(
                target_inputs["input_ids"].device,
                target_inputs["input_ids"].dtype,
            ),
        )

        ignore_idx = self.config["training"]["loss"]["ignore_index"]
        target_ids = target_inputs["input_ids"][:, 1:].to(viz_inputs["input_ids"].device)
        target_attention = target_inputs["attention_mask"][:, 1:].to(viz_inputs["attention_mask"].device)
        labels = torch.cat([torch.full_like(viz_inputs["input_ids"], ignore_idx), target_ids], dim=1)
        labels[labels == self.processor.tokenizer.pad_token_id] = ignore_idx

        return {
            "input_ids": torch.cat([viz_inputs["input_ids"], target_ids], dim=1),
            "attention_mask": torch.cat([viz_inputs["attention_mask"], target_attention], dim=1),
            "labels": labels,
        }