from __future__ import annotations

import math, shutil, re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, cast
from collections import deque

from scripts.uniwm_losses import compute_supervised_uniwm_loss
import torch
from torch.optim.adamw import AdamW
from PIL import Image
from peft.peft_model import PeftModel
from peft.mixed_model import PeftMixedModel
from transformers import ChameleonProcessor, PreTrainedTokenizerFast
from transformers.generation.utils import GenerateDecoderOnlyOutput

from runtime_scripts.runtime_memory_manager import RuntimeMemoryBankManager
from runtime_scripts.uniwm_schemas import UniWMInputBundle, StepPrediction, RoutePrediction
from scripts.load_model import load_model
from scripts.prompt_builder import build_action_prompt, build_viz_prompt
from scripts.action_utils import (
    ACTION_AXES,
    ActionTokenVocabulary,
)
from runtime_scripts.runtime_utils import (
    decode_action,
    decode_image,
    extract_generated_tokens,
    is_stop_action,
    load_config,
    validate_config,
)

REQUIRED_FIELDS: dict[str, dict | list] = {
    "load_model_args": ["model", "image_seq_length", "device", "use_memory_bank_inference", "update_model_on_save"],
    "memory_bank_args": ["top_k", "memory_context_tau"],
    "memory": ["min_memories", "similarity_threshold", "stability_margin"],
    "generation": {
        "action": ["multimodal_generation_mode", "current_substep", "max_new_tokens"],
        "visualization": ["multimodal_generation_mode", "current_substep"]
    },
    "training": {
        "hyper_params": ["initial_lr", "max_grad_norm"], 
        "visualization": ["use_cache"], 
        "loss": ["include_action_loss", "include_image_loss", "action_loss_weight", "image_loss_weight", "log_prefix"],
    },
}

class UniWMEngine:
    """Persistent online UniWM inference engine."""

    def __init__(self, data_id: str, config_path: str = "cfg/habitat_uniwm_cfg.yaml"):
        del data_id
        self.config = load_config(config_path).get("engine", {})
        validate_config(self.config, REQUIRED_FIELDS)

        self.device = self.config["load_model_args"]["device"]
        self.action_vocabulary = ActionTokenVocabulary.from_checkpoint(
            self.config["load_model_args"]["model_ckpt"]
        )
        
        self.prior_actions = deque(maxlen=3)

        loaded = load_model(SimpleNamespace(**self.config["load_model_args"]), self.config["load_model_cfg"], self.action_vocabulary)
        self.model: PeftModel | PeftMixedModel = loaded["model"]

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
            
        raw_processor = loaded["processor"]
        self.processor = cast(ChameleonProcessor, raw_processor)
        self.tokenizer = cast(PreTrainedTokenizerFast, raw_processor.tokenizer)
        self._image_token_id = self.tokenizer.convert_tokens_to_ids(raw_processor.image_token)

        using_memory = self.config["load_model_args"]["use_memory_bank_inference"]
        self._memory_manager = RuntimeMemoryBankManager(
            self.model, using_memory, **self.config["memory_bank_args"]
        )
            
        self.config["generation"]["action"]["use_memory_bank"] = using_memory
        self.config["generation"]["action"]["use_global_memory_bank"] = using_memory
        self.config["generation"]["visualization"]["use_memory_bank"] = using_memory
        self.config["generation"]["visualization"]["use_global_memory_bank"] = using_memory
        
        if self.config["training"] != False:
            self._trainable_params = self._online_update_parameters(include_lm_head=False)
            self._optimizer = AdamW(
                self._trainable_params,
                lr=float(self.config["training"]["hyper_params"]["initial_lr"]),
                weight_decay=0.0,
            )
            self.config["training"]["visualization"]["use_memory_bank"] = using_memory
            self.config["training"]["visualization"]["use_global_memory_bank"] = using_memory
        
        if hasattr(self.model, "eval"):
            self.model.eval()

    def save_online_training_state(self, output_dir: str | Path) -> Path:
        """Save the current online-adapted weights and optimizer state."""
        if self.config["load_model_args"]["update_model_on_save"]:
            output_path = Path(self.config["load_model_args"]["model_ckpt"])
        else:
            output_path = Path(output_dir)

        shutil.rmtree(output_path, ignore_errors=True)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(
            str(output_path),
            save_embedding_layers=False,
        )
        
        if hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(output_path)
            
        self.action_vocabulary.save(output_path)

        torch.save(
            {
                "optimizer": self._optimizer.state_dict(),
                "engine_training_config": self.config["training"],
            },
            output_path / "online_training_state.pt",
        )
        return output_path

    def reset_episode(self, episode_id: str, bundle: UniWMInputBundle):
        self._memory_manager.setup_for_episode(episode_id=episode_id)
        self._start_tok_obs = self._image_to_vq_bpe_tokens(bundle.start_observation)
        self._goal_tok_obs = self._image_to_vq_bpe_tokens(bundle.goal_observation)
        self.memory_count = 0
        self.prior_actions.clear()
            
        self.update_working_memory(bundle.current_observation, bundle.start_pose_str)
        
    def store_working_memory(self):
        if self._start_tok_obs is None or self._goal_tok_obs is None:
            raise AssertionError("Attempted to store step memory before encoding the goal or start observations.")
        
        if self._memory_manager.store_step_memory():
            self.memory_count += 1
            
    def update_working_memory(self, real_obs: Image.Image, start_pose_str: str, update_context_ema: bool = True):
        self._current_tok_obs = self._image_to_vq_bpe_tokens(real_obs)
        
        action_inputs = self._processor_inputs_from_prompt(
            input_text=build_action_prompt(
                start_pose_str=start_pose_str,
                dx_range=self.action_vocabulary.range_for("dx"),
                dy_range=self.action_vocabulary.range_for("dy"),
                dyaw_range=self.action_vocabulary.range_for("dyaw"),
                prior_decoded_actions=list(self.prior_actions),
            ),
        )
        
        self._memory_manager.start_new_step()
        self._context_stability = self._memory_manager.initialize_step_memory(action_inputs, update_context_ema)
        self._context_familiarity = self._memory_manager.compute_memory_familiarity()
    
    def get_current_context(self):
        return self._context_familiarity, self._context_stability 
        
    def _store_state(self) -> tuple[torch.LongTensor, float, float, int, deque]:
        self._memory_manager.cache_step_state()
        return self._current_tok_obs, self._context_familiarity, self._context_stability, self.memory_count, self.prior_actions.copy()
        
    def _restore_state(self, tok_obs: torch.LongTensor, familiarity: float, stability: float, mem_count: int, stored_actions: deque):
        self._memory_manager.load_cached_state()
        self._current_tok_obs = tok_obs
        self._context_familiarity = familiarity
        self._context_stability = stability
        self.memory_count = mem_count
        self.prior_actions = stored_actions
        
    def eval_step(
            self,
            bundle: UniWMInputBundle,
    ) -> StepPrediction:
        cached_current_tok_obs, real_mem_familiarity, real_context_stability, real_mem_count, real_prior_actions = self._store_state()
        step = self._predict_step(bundle)
        self._restore_state(cached_current_tok_obs, real_mem_familiarity, real_context_stability, real_mem_count, real_prior_actions)
        return step

    def predict_route(
        self,
        bundle: UniWMInputBundle,
        max_steps: int,
    ) -> RoutePrediction:
        current, actions = bundle.current_observation, bundle.action_text
        if isinstance(actions, str):
            actions = None
        elif isinstance(actions, list):
            actions = None if len(actions) <= 1 else actions[1:]

        steps: list[StepPrediction] = []
        cached_current_tok_obs, real_mem_familiarity, real_context_stability, real_mem_count, real_prior_actions = self._store_state()
        
        predict_range = max_steps
        if actions is not None:
            predict_range = min(max_steps, len(actions))
        
        for route_idx in range(predict_range):
            temp_bundle = replace(
                bundle,
                action_text=None if actions is None else actions[route_idx],
                current_observation=current,
                collision=bundle.collision if route_idx == 0 else False
            )
            
            step = self._predict_step(temp_bundle)
            step.context_familiarity = real_mem_familiarity
            step.context_stability = real_context_stability
            steps.append(step)
                
            if is_stop_action(step.action_text):
                self._restore_state(cached_current_tok_obs, real_mem_familiarity, real_context_stability, real_mem_count, real_prior_actions)
                return RoutePrediction(steps=steps, stopped=True, stop_reason="stop_action")
            
            if step.visualization is None:
                self._restore_state(cached_current_tok_obs, real_mem_familiarity, real_context_stability, real_mem_count, real_prior_actions)
                return RoutePrediction(steps=steps, stopped=False, stop_reason="missing_visualization")
            current = step.visualization
            
            if route_idx < max_steps - 1:
                self.update_working_memory(current, bundle.start_pose_str, False)
                
            if route_idx == max_steps - 1:
                self._restore_state(cached_current_tok_obs, real_mem_familiarity, real_context_stability, real_mem_count, real_prior_actions)
                return RoutePrediction(steps=steps, stopped=False, stop_reason="max_steps")
            
        self._restore_state(cached_current_tok_obs, real_mem_familiarity, real_context_stability, real_mem_count, real_prior_actions)
        return RoutePrediction(steps=steps, stopped=True, stop_reason="out_of_forced_actions")

    def train_viz_step(
        self,
        prediction: StepPrediction,
        lr_scaler: float,
        loss_scaler: float = 1.0,
        max_grad_norm: float | None = None,
    ) -> dict[str, Any]:
        """Note: Make sure you are calling the UniWMEngine.init_working_memory() when you want to store a prior step into global memory and update the observation in working KV memory used by the model."""

        for group in self._optimizer.param_groups:
            group["lr"] = float(self.config["training"]["hyper_params"]["initial_lr"]) * lr_scaler

        self.model.train()
        self._optimizer.zero_grad(set_to_none=True)
        loss, components = self._compute_viz_step_loss(prediction)

        grad_norm = self._update_weights(loss_scaler, loss, max_grad_norm)
        self.model.eval()

        log_prefix = self.config["training"]["loss"]["log_prefix"]
        effective_lr = float(self.config["training"]["hyper_params"]["initial_lr"]) * lr_scaler

        return {
            **components,
            "base_loss": components[f"{log_prefix}base_loss"],
            "lr_scalar": lr_scaler,
            "optimizer_lr": self._optimizer.param_groups[0]["lr"],
            "final_lr": effective_lr,
            "effective_learning_rate": effective_lr,
            "grad_norm": grad_norm,
            "gradient_clipped": (grad_norm > max_grad_norm) if max_grad_norm is not None else False,
            "optimizer_step": True,
        }

    def record_viz_step(
        self,
        prediction: StepPrediction,
        lr_scaler: float,
    ) -> dict[str, Any]:
        """Compute the online visualization loss without updating parameters."""
        self.model.train()
        with torch.no_grad():
            _, components = self._compute_viz_step_loss(prediction)
        self.model.eval()

        log_prefix = self.config["training"]["loss"]["log_prefix"]
        effective_lr = float(self.config["training"]["hyper_params"]["initial_lr"]) * lr_scaler
        return {
            **components,
            "base_loss": components[f"{log_prefix}base_loss"],
            "lr_scalar": lr_scaler,
            "optimizer_lr": None,
            "final_lr": effective_lr,
            "effective_learning_rate": effective_lr,
            "grad_norm": None,
            "gradient_clipped": False,
            "optimizer_step": False,
        }

    def _compute_viz_step_loss(
        self,
        prediction: StepPrediction,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if prediction.real_input_obs is None:
            raise ValueError("[UNEXPECTED ERROR] Wrapper failed to set a real input observation on the StepPrediction passed into a visualization loss step")
        if prediction.real_next_obs is None:
            raise ValueError("[UNEXPECTED ERROR] Wrapper failed to set a real predicted visualization on the StepPrediction passed into a visualization loss step")

        visualization_inputs = self._processor_inputs_from_prompt(
            input_text=build_viz_prompt(
                decoded_action=prediction.action_text,
                start_pose_str=prediction.input_bundle.start_pose_str
            )
        )
        training_inputs: Mapping[str, Any] = self._build_image_batch(
            viz_inputs=visualization_inputs,
            target_viz_tokens=self._image_to_vq_bpe_tokens(prediction.real_next_obs)
        )
        memory_kwargs, _ = self._get_viz_kwargs(
            dict(self.config["training"]["visualization"]),
            prediction.logging_info["viz_used_memory"],
        )

        with torch.autocast(device_type="cuda", dtype=self.model.dtype):
            outputs = self.model(
                **{key: value for key, value in training_inputs.items() if key != "labels"},
                **memory_kwargs
            )

            loss_cfg = self.config["training"]["loss"].copy()
            loss_cfg["include_action_loss"] = False
            loss_cfg["action_loss_weight"] = 0.0

            return compute_supervised_uniwm_loss(
                self.action_vocabulary,
                model=self.model,
                outputs=outputs,
                batch=training_inputs,
                tokenizer=self.processor,
                loss_config=loss_cfg
            )

    def _predict_step(
        self,
        input_bundle: UniWMInputBundle
    ) -> StepPrediction:
        """Note: Make sure you are calling the UniWMEngine.init_working_memory() when you want to store a prior step into global memory and update the observation in working KV memory used by the model."""
        
        if not input_bundle.start_pose_str:
            raise AssertionError("[UNEXPECTED ERROR] start_pose_str is required.")
        
        if isinstance(input_bundle.action_text, list):
            raise AssertionError("[UNEXPECTED ERROR] any forced actions should be converted to text by this point.")
        
        action_inputs = self._processor_inputs_from_prompt(
            input_text=build_action_prompt(
                start_pose_str=input_bundle.start_pose_str,
                dx_range=self.action_vocabulary.range_for("dx"),
                dy_range=self.action_vocabulary.range_for("dy"),
                dyaw_range=self.action_vocabulary.range_for("dyaw"),
                prior_decoded_actions=list(self.prior_actions),
            )
        )
        
        action_text, raw_action_text, act_entropy = input_bundle.action_text, input_bundle.action_text, 0
        if action_text is None:
            action_text, raw_action_text, act_entropy = self._predict_action(action_inputs)
            
            if input_bundle.collision:
                action_text = self._zero_act_translations(action_text)
                
            if action_text.count("pos_bin_00") == 3:
                action_text = f"{action_text[:-3]}20>"

        self.prior_actions.append(action_text)
        
        viz, viz_entropy, used_memory = None, 0, False
        if not is_stop_action(action_text):
            visualization_inputs = self._processor_inputs_from_prompt(
                input_text=build_viz_prompt(
                    decoded_action=action_text,
                    start_pose_str=input_bundle.start_pose_str
                )
            )
            
            viz, viz_entropy, used_memory = self._predict_visualization(visualization_inputs)
            
        step_output = StepPrediction(input_bundle, action_text, viz, act_entropy, viz_entropy, self._context_familiarity, self._context_stability)
        step_output.logging_info["raw_action_text"] = raw_action_text
        step_output.logging_info["act_entropy"] = act_entropy
        step_output.logging_info["viz_entropy"] = viz_entropy
        step_output.logging_info["viz_used_memory"] = used_memory
        return step_output
    
    def _zero_act_translations(self, action_text) -> str:
        ''' Zeroes out the x and y translations in the given action'''
        result = re.sub(
            r"<(dx|dy)_(?:pos|neg)_bin_\d+>",
            r"<\1_pos_bin_00>",
            action_text,
        )
        
        return result

    def _predict_action(self, processor_inputs: Any) -> tuple[str, str, float]:
        prompt_length = processor_inputs["input_ids"].shape[-1]
        kwargs = self._memory_manager.get_action_kwargs(
            action_gen_kwargs=dict(self.config["generation"]["action"])
        )
        
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.model.dtype):
            outputs = cast(
                GenerateDecoderOnlyOutput,
                self.model.generate(**processor_inputs, **kwargs)
            )
            
        action_token_ids = [cast(
                list[int],
                self.tokenizer.convert_tokens_to_ids(self.action_vocabulary.tokens_by_axis[axis])
            ) for axis in ACTION_AXES]
        
        scores = cast(tuple[torch.FloatTensor], outputs.scores)
        generated_tokens = extract_generated_tokens(outputs, prompt_length, len(scores)).clone()
    
        decoded_text, raw_text, token_position_info = decode_action(self.processor, generated_tokens, action_token_ids)
        
        entropy = 0.0
        if not is_stop_action(decoded_text) and len(token_position_info) > 0:
            entropy_tensor = torch.stack([
                self._calculate_entropy(generated_tokens, scores, ids, position)
                for position, ids in token_position_info
            ])
            
            entropy = float(entropy_tensor.mean().detach().cpu())

        return decoded_text, raw_text, entropy

    def _predict_visualization(self, processor_inputs: Any) -> tuple[Image.Image | None, float, bool]:
        prompt_length = processor_inputs["input_ids"].shape[-1]
        kwargs, used_memory = self._get_viz_kwargs(
            dict(self.config["generation"]["visualization"])
        )
            
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.model.dtype):
            outputs = cast(
                GenerateDecoderOnlyOutput,
                self.model.generate(**processor_inputs, **kwargs)
            )
            
        scores = cast(tuple[torch.FloatTensor], outputs.scores)
        generated_tokens = extract_generated_tokens(outputs, prompt_length, len(scores)).clone()
        
        entropy = self._calculate_entropy(
            generated_tokens,
            scores,
            token_ids=cast(list[int], self.model.model.bpe_indices),
        )
        
        entropy_value = float(entropy.mean().detach().cpu())
        del outputs, scores, entropy
        
        generated_img = decode_image(self.model, self.processor, generated_tokens)
        return generated_img, entropy_value, used_memory

    def _update_weights(self, update_scale: float, supervised_loss: torch.Tensor, max_grad_norm: float | None) -> float:
        scaled_loss = update_scale * supervised_loss
        scaled_loss.backward()

        grad_norm = 0.0
        if max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._trainable_params,
                max_norm=max_grad_norm,
            ).detach().cpu().item()

        self._optimizer.step()
        

        return float(grad_norm)
        
    def _get_viz_kwargs(self, kwargs: dict[str, Any], use_memory: bool | None = None) -> tuple[dict, bool]:         
        use_memory = use_memory if use_memory is not None else (
                self.memory_count >= self.config["memory"]["min_memories"]
            )
        
        print(f"[ENGINE] Getting visualization kwargs using use_memory set to: {use_memory}")
        
        if not use_memory:
            kwargs.pop("current_step", None)
            kwargs.pop("current_substep", None)
            kwargs["use_memory_bank"] = False
            kwargs["use_global_memory_bank"] = False
            
        return self._memory_manager.get_viz_kwargs(viz_gen_kwargs=kwargs), bool(use_memory)
        
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
            start_tok_obs if start_tok_obs is not None else self._start_tok_obs,
            goal_tok_obs if goal_tok_obs is not None else self._goal_tok_obs,
            current_tok_obs if current_tok_obs is not None else self._current_tok_obs
        ]
        
        inputs = self.processor(
            text=[input_text],
            return_tensors="pt",
        )
        
        image_mask = inputs["input_ids"] == self._image_token_id
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
        
    def _build_image_batch(
        self,
        viz_inputs: Any,
        target_viz_tokens: torch.LongTensor,
    ) -> Mapping[str, Any]:
        target_inputs = self.processor(text=["<image>"], return_tensors="pt")
        target_mask = target_inputs["input_ids"] == self._image_token_id
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
        labels[labels == self.tokenizer.pad_token_id] = ignore_idx

        return {
            "input_ids": torch.cat([viz_inputs["input_ids"], target_ids], dim=1),
            "attention_mask": torch.cat([viz_inputs["attention_mask"], target_attention], dim=1),
            "labels": labels,
        }

    def _calculate_entropy(
        self,
        generated_tokens: torch.Tensor,
        scores: tuple[torch.FloatTensor],
        token_ids: list[int],
        token_position: int = -1
    ) -> torch.Tensor:
        score_tensor = torch.stack(tuple(scores), dim=1).float()
        allowed_ids = torch.as_tensor(token_ids, device=score_tensor.device, dtype=torch.long)
        
        position_mask = None
        if token_position < 0:
            position_mask = torch.isin(generated_tokens.to(score_tensor.device), allowed_ids)
            score_tensor = torch.stack(
                tuple(score.index_select(-1, allowed_ids) for score in scores),
                dim=1,
            ).float()
        else:
            score_tensor = score_tensor[:, token_position, :].index_select(-1, allowed_ids)

        probs = torch.softmax(score_tensor, dim=-1)
        log_probs = torch.log_softmax(score_tensor, dim=-1)
        entropy_terms = torch.where(probs > 0.0, probs * log_probs, torch.zeros_like(probs))
        entropy = -entropy_terms.sum(dim=-1)
        
        if allowed_ids.numel() < 2:
            return entropy.new_zeros(1)
        else:
            entropy = entropy / math.log(allowed_ids.numel())
        
        if position_mask is not None:
            entropy = entropy[position_mask]
            
        return entropy.new_zeros(1) if entropy.numel() == 0 else entropy
