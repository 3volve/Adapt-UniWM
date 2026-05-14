from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from PIL import Image
from transformers import PreTrainedTokenizerFast

from scripts.postprocess_logits_utils import split_token_sequence
from scripts.action_utils import generate_bin_tokens, get_action_ranges

def image_to_array(observation: Image.Image) -> np.ndarray:
    array = np.asarray(observation, dtype=np.float32)
    return array / 255.0

def resolve_config_path_from_id(data_id: str) -> str:
    root_dir = Path(__file__).resolve().parent.parent
    config_path = Path(root_dir / "cfg" / f"{data_id}_uniwm_cfg.yaml")

    if not config_path.is_file():
        raise AssertionError(f"config_path must be file or data_id must identify a config file within the local cfg folder so this is a valid path: {config_path}")

    abs_path = str(config_path.resolve())
    return abs_path

def load_config(config_path: str) -> dict[str, Any]:
    if config_path is None:
        print("It is highly recommended to utilize a config for running this version of UniWM.")
        return {}

    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}

def validate_config(config: dict, required_fields: dict | list) -> None:
    validate_config_recursive(config, required_fields, "config_root")

def validate_config_recursive(config_node: dict | str, required_fields_at_node: dict | list, parent_str: str) -> None:
    if not isinstance(config_node, dict):
        raise AssertionError(f"{parent_str} must be a valid mapping.")

    else:
        for key in required_fields_at_node:
            if key not in config_node:
                raise AssertionError(f"{parent_str} is missing required key '{key}'.")

            if isinstance(required_fields_at_node, dict):
                validate_config_recursive(config_node[key], required_fields_at_node[key], f"{parent_str}.{key}")

def is_stop_action(action_text: str) -> bool:
    return action_text.strip().lower() == "stop"


#----------------- Direct Engine Helper Functions ------------------#
def step_image_output_path(output_dir: str | None, step_index: int) -> str | None:
    return None if not output_dir else str(Path(output_dir) / f"step_{step_index + 1}_observation.png")

def decode_generated_image(
    model: Any,
    processor: Any,
    outputs: Any,
    save_path: str | None = None,
) -> Image.Image | None:
    r_ids = extract_generated_tokens(outputs)

    if r_ids.dim() == 2:
        r_ids = r_ids[0]

    generated_results = split_token_sequence(
        tokens=r_ids.unsqueeze(0).to(model.device),
        image_seq_length=model.image_token_num,
        boi=model.config.boi_token_id,
        eoi=model.config.eoi_token_id,
        max_length=r_ids.shape[-1],
        pad_token_id=model.config.pad_token_id
    )

    if generated_results["images"]:
        generated_imgs = torch.cat(generated_results["images"], dim=0).to(model.device)
        generated_imgs = model.decode_image_tokens(generated_imgs)
        generated_imgs = processor.postprocess_pixel_values(generated_imgs)
    else:
        print(f"  Generated failed visualization tokens: {r_ids}")
        return None

    tensor_img = generated_imgs[0, :, :, :]
    print(f"[DEBUG] tensor_img.shape: {tensor_img.shape}")

    np_img = tensor_img.cpu().detach().to(torch.uint8).numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    print(f"[DEBUG] np_img.shape: {np_img.shape}")

    img = Image.fromarray(np_img.astype(np.uint8))
    print(f"[DEBUG] PIL image size: {img.size}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)

    return img

def _first_image_segment(
    *,
    tokens: torch.Tensor,
    image_seq_length: int,
    boi: int,
    eoi: int,
) -> torch.Tensor | None:
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

def processor_inputs_from_prompt(
    processor: PreTrainedTokenizerFast,
    *,
    input_text: str,
    input_images: list[Image.Image],
    device: str | None = None,
) -> Any:
    if not isinstance(input_images[0], Image.Image) or not isinstance(input_images, list):
        raise AssertionError(f"Input images given for processor inputs must be a list of Image.Image types.  Instead found: {type(input_images)}[{type(input_images[0])}]")

    inputs = processor(
        text=[input_text],
        images=input_images,
        return_tensors="pt",
    )
    if device and hasattr(inputs, "to"):
        return inputs.to(device)
    return inputs

def detach_processor_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }

def extract_generated_tokens(outputs: Any) -> torch.Tensor:
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, tuple) and outputs and torch.is_tensor(outputs[0]):
        return outputs[0]
    sequences = getattr(outputs, "sequences", None)
    if torch.is_tensor(sequences):
        return sequences
    raise TypeError(f"Unsupported UniWM generate output type: {type(outputs)}")

def decode_generated_text(processor: Any, outputs: Any) -> tuple[str, str]:
    tokens = extract_generated_tokens(outputs)
    raw_decoded = processor.batch_decode(tokens, skip_special_tokens=False)[0].strip()
    if raw_decoded.lower() == "stop":
        return "stop"

    pattern = r'(<d[^>]+>)+(<d[^>]+>)'
    decoded = re.sub(pattern, r'\2', raw_decoded)
    return decoded, raw_decoded

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