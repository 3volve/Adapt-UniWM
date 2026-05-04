from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List, Dict, Mapping, Optional
from dataclasses import dataclass

import numpy as np
import torch
import yaml
from PIL import Image

from scripts.action_utils import generate_bin_tokens, get_action_ranges


@dataclass(frozen=True)
class StepPrediction:
    action_text: str
    visualization: Optional[Image.Image]

@dataclass(frozen=True)
class RoutePrediction:
    steps: List[StepPrediction]
    stopped: bool
    stop_reason: str

    def __len__(self):
        return len(self.steps)

def processor_inputs_from_prompt(
    processor: Any,
    *,
    input_text: str,
    input_images: list[Any],
    device: Optional[str] = None,
) -> Any:
    inputs = processor(
        text=[input_text],
        images=input_images,
        return_tensors="pt",
    )
    if device and hasattr(inputs, "to"):
        return inputs.to(device)
    return inputs


def extract_generated_tokens(outputs: Any) -> torch.Tensor:
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, tuple) and outputs and torch.is_tensor(outputs[0]):
        return outputs[0]
    sequences = getattr(outputs, "sequences", None)
    if torch.is_tensor(sequences):
        return sequences
    raise TypeError(f"Unsupported UniWM generate output type: {type(outputs)}")


def decode_generated_text(processor: Any, outputs: Any) -> str:
    tokens = extract_generated_tokens(outputs)
    decoded = processor.batch_decode(tokens, skip_special_tokens=False)[0].strip()
    is_stop = decoded.lower() == "stop"
    pattern = r'(<d[^>]+>)+(<d[^>]+>)'
    decoded = re.sub(pattern, r'\2', decoded)
    return decoded


def decode_generated_image(
    model: Any,
    processor: Any,
    outputs: Any,
    save_path: Optional[str] = None,
) -> Optional[Image.Image]:
    tokens = extract_generated_tokens(outputs)
    if tokens.dim() == 1:
        tokens_for_split = tokens.unsqueeze(0)
    elif tokens.dim() == 2:
        tokens_for_split = tokens[:1]
    else:
        raise ValueError(f"Unsupported generated token shape for visualization: {tuple(tokens.shape)}")

    predicted_image_tokens = _first_image_segment(
        tokens=tokens_for_split[0],
        image_seq_length=model.image_token_num,
        boi=model.config.boi_token_id,
        eoi=model.config.eoi_token_id,
    )
    if predicted_image_tokens is None:
        return None

    with torch.no_grad():
        decoded_img = model.decode_image_tokens(predicted_image_tokens)

    processed_img = processor.postprocess_pixel_values(decoded_img)[0]
    np_img = processed_img.cpu().numpy().transpose(1, 2, 0)
    image = Image.fromarray(np_img.astype(np.uint8))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)

    return image


def generation_kwargs(config: Mapping[str, Any], mode: str, model: Any) -> dict:
    kwargs = dict(config.get("generation", {}).get(mode, {}))
    if mode == "visualization" and "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = model.image_token_num + 20
    return kwargs


def configure_action_tokenizer(model: Any, processor: Any, config: Mapping[str, Any]) -> None:
    token_cfg = dict(config.get("action_token_generation", {}))
    range_profile = token_cfg.get("range_profile", "habitat")
    bin_step = float(token_cfg.get("bin_step", 0.01))
    ranges = get_action_ranges(range_profile)

    bin_tokens = []
    bin_tokens.extend(generate_bin_tokens("dx", ranges["dxy"][0], ranges["dxy"][1], bin_step))
    bin_tokens.extend(generate_bin_tokens("dy", ranges["dxy"][0], ranges["dxy"][1], bin_step))
    bin_tokens.extend(generate_bin_tokens("dyaw", ranges["dyaw"][0], ranges["dyaw"][1], bin_step))

    existing_vocab = set(processor.tokenizer.get_vocab().keys())
    new_tokens = [token for token in bin_tokens if token not in existing_vocab]
    if not new_tokens:
        return

    processor.tokenizer.add_tokens(new_tokens, special_tokens=True)
    _resize_model_embeddings(model, processor)


def _first_image_segment(
    *,
    tokens: torch.Tensor,
    image_seq_length: int,
    boi: int,
    eoi: int,
) -> Optional[torch.Tensor]:
    in_image = False
    segment = []
    for token in tokens:
        token_id = int(token.item())
        if token_id == int(boi):
            in_image = True
            segment = []
            continue
        if token_id == int(eoi) and in_image:
            if len(segment) == image_seq_length:
                return torch.tensor([segment], dtype=tokens.dtype, device=tokens.device)
            return None
        if in_image:
            segment.append(token_id)

    if in_image and len(segment) == image_seq_length:
        return torch.tensor([segment], dtype=tokens.dtype, device=tokens.device)
    return None


def _resize_model_embeddings(model: Any, processor: Any) -> None:
    tokenizer_size = len(processor.tokenizer)

    if hasattr(model, "model") and hasattr(model.model, "lm_head"):
        lm_head = model.model.lm_head
        if hasattr(lm_head, "base_layer") or "ModulesToSaveWrapper" in str(type(lm_head)):
            from peft.utils.other import ModulesToSaveWrapper

            if hasattr(lm_head, "base_layer"):
                model.model.lm_head = lm_head.base_layer
            elif hasattr(lm_head, "original_module"):
                model.model.lm_head = lm_head.original_module

            model.model.resize_token_embeddings(tokenizer_size)
            model.model.lm_head = ModulesToSaveWrapper(model.model.lm_head, "default")
            return

    resize = getattr(model, "resize_token_embeddings", None)
    if callable(resize):
        resize(tokenizer_size)
        return

    inner_resize = getattr(getattr(model, "model", None), "resize_token_embeddings", None)
    if callable(inner_resize):
        inner_resize(tokenizer_size)


def image_to_array(observation: Image.Image) -> np.ndarray:
    array = np.asarray(observation, dtype=np.float32)
    return array / 255.0

#----------------- Direct Engine Helper Functions ------------------#
def step_image_output_path(output_dir: Optional[str], step_index: int) -> Optional[str]:
    return None if not output_dir else str(Path(output_dir) / f"step_{step_index + 1}_observation.png")

def is_stop_action(action_text: str) -> bool:
    return action_text.strip().lower() == "stop"

def load_config(config_path: str) -> Dict[str, Any]:
    if config_path is None:
        print("It is highly recommended to utilize a config for running this version of UniWM.")
        return {}

    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}

def validate_config(config_node: Any, required_fields_at_node: Any, parent_str: str = "config") -> None:
    if not isinstance(config_node, dict):
        raise AssertionError(f"{parent_str} must be a mapping.")

    else:
        for key in required_fields_at_node:
            if key not in config_node:
                raise AssertionError(f"{parent_str} is missing required key '{key}'.")

            if isinstance(required_fields_at_node, dict):
                validate_config(config_node[key], required_fields_at_node[key], f"{parent_str}.{key}")
