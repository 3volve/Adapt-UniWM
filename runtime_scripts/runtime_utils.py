from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from datetime import datetime

from transformers.generation.utils import GenerateDecoderOnlyOutput
from scripts.postprocess_logits_utils import split_token_sequence

VERBOSE_UTILS = False

def image_to_array(observation: Image.Image) -> np.ndarray:
    array = np.asarray(observation, dtype=np.float32)
    return array / 255.0

def root_dir() -> Path:
    return Path(__file__).resolve().parent.parent

def resolve_config_path_from_id(data_id: str) -> str:
    config_path = Path(root_dir() / "cfg" / f"{data_id}_uniwm_cfg.yaml")

    if not config_path.is_file():
        raise AssertionError(f"config_path must be file or data_id must identify a config file within the local cfg folder so this is a valid path: {config_path}")

    abs_path = str(config_path.resolve())
    return abs_path

def make_runner_output_dir(
    output_dir: str,
    data_id: str
) -> Path:
    """ Make a unique directory based on data_id.  Passed-in output_dir is assumed to be relative to the repo-root. """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root_dir() / output_dir / f"{data_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

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
                
def build_img_paths(output_dir: str, episode_id: str, route_id: int, route_step: int) -> tuple[str, str, str]:
    base_path = str(root_dir() / Path(output_dir) / f"episode_{episode_id}/route_{route_id}/step_{route_step}")
    return f"{base_path}_real.png", f"{base_path}_pred.png", f"{base_path}_eval.png"
    
def is_stop_action(action_text: str) -> bool:
    return action_text.strip().lower() == "stop"

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def ema_decay(previous_value: float, added_value: float, decay: float) -> float:
    """Helper to apply EMA smoothing of accumulating values."""
    return decay * previous_value + added_value

def ema_smoothing(previous_value: float, added_value: float, tau: float) -> float:
    """Helper to apply EMA smoothing of accumulating values."""
    return tau * previous_value + (1 - tau) * added_value


#----------------- Direct Engine Helper Functions ------------------#
def extract_generated_tokens(outputs: GenerateDecoderOnlyOutput, prompt_length: int, generated_tok_len: int) -> torch.Tensor:
    return outputs.sequences[:, prompt_length:prompt_length + generated_tok_len]

def save_img(img: Image.Image, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)

def decode_image(
    model: Any,
    processor: Any,
    tokens: torch.Tensor ,
) -> Image.Image | None:

    if tokens.dim() == 2:
        tokens = tokens[0]

    generated_results = split_token_sequence(
        tokens=tokens.unsqueeze(0).to(model.device), # type: ignore
        image_seq_length=model.image_token_num,
        boi=model.config.boi_token_id,
        eoi=model.config.eoi_token_id,
        max_length=tokens.shape[-1],
        pad_token_id=model.config.pad_token_id
    )

    if generated_results["images"]:
        raw_imgs = generated_results["images"]
        
        generated_imgs = torch.cat(
            [raw_imgs] if isinstance(raw_imgs, torch.Tensor) else raw_imgs,
            dim=0
        ).to(model.device)
        generated_imgs = model.decode_image_tokens(generated_imgs)
        generated_imgs = processor.postprocess_pixel_values(generated_imgs)
    else:
        print(f"  Generated failed visualization tokens: {tokens}")
        return None

    tensor_img = generated_imgs[0, :, :, :]
    if VERBOSE_UTILS:
        print(f"[DEBUG] tensor_img.shape: {tensor_img.shape}")

    np_img = tensor_img.cpu().detach().to(torch.uint8).numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    if VERBOSE_UTILS:
        print(f"[DEBUG] np_img.shape: {np_img.shape}")

    img = Image.fromarray(np_img.astype(np.uint8))
    if VERBOSE_UTILS:
        print(f"[DEBUG] PIL image size: {img.size}")

    return img

def decode_action(
    processor: Any,
    tokens: torch.Tensor,
    action_token_ids: list[list[int]],
) -> tuple[str, list[tuple[int, list[int]]]]:
    """Decode action text and locate selected generated action-bin token positions."""
    decoded_text = processor.batch_decode(tokens, skip_special_tokens=False)[0].strip()
    if decoded_text.lower() == "stop":
        return "stop", []
        
    if tokens.dim() == 2:
        tokens = tokens[0]
        
    selected_token_positions: list[tuple[int, list[int]]] = []
    decoded_tokens: list[str] = []

    for ids in action_token_ids:
        selected_token_id = ids[0]
        
        for position, token in enumerate(tokens.tolist()):
            token_id = int(token)
            if token_id in ids:
                selected_token_id = token_id
                selected_token_positions.append((position, ids))
                break
            
        decoded_tokens.append(
            processor.batch_decode(
                torch.tensor([[selected_token_id]], device=tokens.device),
                skip_special_tokens=False,
            )[0].strip()
        )
        
    decoded_text = f"Move by dx: {decoded_tokens[0]}, dy: {decoded_tokens[1]}, dyaw: {decoded_tokens[2]}"
        
    return decoded_text, selected_token_positions

def detach_processor_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
