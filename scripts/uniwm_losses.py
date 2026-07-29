from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import torch, importlib
from peft.peft_model import PeftModel
from torch import Tensor
import torch.nn.functional as F

from packaging import version
from transformers.utils import is_peft_available
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


from scripts.action_utils import ACTION_AXES, ActionTokenVocabulary, extract_bin_values


ACTION_SOFT_TARGET_SIGMA = {
    "dx": 0.04,
    "dy": 0.01,
    "dyaw": 0.025,
}


@lru_cache(maxsize=None)
def _action_token_values(
    axis: str,
    axis_tokens: tuple[str, ...],
    bin_step: float,
) -> tuple[float, ...]:
    return tuple(
        extract_bin_values(token, axis, bin_step)
        for token in axis_tokens
    )


def detach_loss_value(value: Tensor | float | int) -> float:
    """
    Convert a scalar loss tensor/value into a plain float for logging.

    This helper is only for components/logging. It should not be used on the actual
    loss tensor that will be backpropagated.
    """
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)

def compute_base_model_loss(
    outputs: Any,
    labels: Tensor,
    *,
    label_smoother: Any | None = None,
    model: Any | None = None,
    ignore_index: int,
) -> Tensor:
    """
    Compute the base UniWM/HF modeling loss from model outputs and labels.
    """
    if label_smoother is not None:
        model_name = _get_model_name(model)
        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values() or model_name.endswith("ConditionalGeneration"):
            return label_smoother(outputs, labels, shift_labels=True)
        return label_smoother(outputs, labels)

    output_loss = _get_output_value(outputs, "loss")
    if output_loss is not None:
        return output_loss

    logits = _get_output_value(outputs, "logits")
    model_name = _get_model_name(model)
    if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values() or model_name.endswith("ConditionalGeneration"):
        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].contiguous().view(-1)
        return F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)

    flat_logits = logits.contiguous().view(-1, logits.shape[-1])
    flat_labels = labels.contiguous().view(-1)
    return F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index)

def compute_action_token_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    tokenizer: Any,
    ignore_index: int,
    action_vocabulary: ActionTokenVocabulary,
) -> Tensor:
    """
    Compute distance-aware soft-target cross entropy for action tokens.
    """
    hf_tokenizer = _get_hf_tokenizer(tokenizer)

    shifted_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
    shifted_labels = labels[:, 1:].contiguous().view(-1)

    def compute_axis_loss(axis: str) -> Tensor | None:
        axis_tokens = tuple(action_vocabulary.tokens_by_axis[axis])
        axis_ids = hf_tokenizer.convert_tokens_to_ids(list(axis_tokens))
        assert len(set(axis_ids)) == len(axis_ids), (
            f"Action tokens for {axis} must map to unique tokenizer IDs."
        )

        axis_ids_tensor = torch.tensor(
            axis_ids,
            device=shifted_labels.device,
            dtype=shifted_labels.dtype,
        )
        mask = torch.isin(
            shifted_labels,
            axis_ids_tensor,
        )
        if not mask.any():
            return None

        target_ids = shifted_labels[mask]
        target_indices = (target_ids[:, None] == axis_ids_tensor[None, :]).to(
            torch.int64
        ).argmax(dim=-1)

        axis_values = torch.tensor(
            _action_token_values(
                axis,
                axis_tokens,
                action_vocabulary.bin_step,
            ),
            device=shifted_logits.device,
            dtype=torch.float32,
        )
        target_values = axis_values[target_indices]
        distances = (
            axis_values[None, :] - target_values[:, None]
        ) / ACTION_SOFT_TARGET_SIGMA[axis]
        soft_targets = F.softmax(-0.5 * distances.square(), dim=-1)

        axis_log_probs = F.log_softmax(
            shifted_logits[mask][:, axis_ids].float(),
            dim=-1,
        )
        return -(soft_targets * axis_log_probs).sum(dim=-1).mean()

    dx_loss = compute_axis_loss("dx")
    dy_loss = compute_axis_loss("dy")
    dyaw_loss = compute_axis_loss("dyaw")

    # The current trainer computes stop-token loss but does not add it to the total.
    loss_components = [loss for loss in (dx_loss, dy_loss, dyaw_loss) if loss is not None]
    if loss_components:
        return sum(loss_components) / len(loss_components)
    return shifted_logits.new_zeros(())

def compute_image_codebook_discrepancy_loss(
    *,
    model: Any,
    logits: Tensor,
    labels: Tensor,
    tokenizer: Any,
    ignore_index: int,
) -> Tensor:
    """
    Compute the image/codebook discrepancy loss
    """
    del tokenizer
    del ignore_index

    image_token_ids = torch.as_tensor(model.model.bpe_indices, device=logits.device, dtype=torch.long)
    shifted_labels = labels[:, 1:]
    image_mask = torch.isin(shifted_labels, image_token_ids.to(shifted_labels.device))

    if not torch.any(image_mask):
        return logits.new_zeros(())

    image_labels = shifted_labels[image_mask]
    shifted_image_logits = logits[:, :-1, :].index_select(-1, image_token_ids)
    image_logits = shifted_image_logits[image_mask, :]

    vis_img_tokens = model.model.model.convert_bpe2img_tokens(image_labels)
    image_probs = F.softmax(image_logits, dim=-1)

    num_codebook_tokens = model.model.model.vqmodel.quantize.embedding.weight.shape[0]
    label_one_hot = F.one_hot(
        vis_img_tokens.reshape(-1).to(torch.int64),
        num_classes=num_codebook_tokens,
    ).to(torch.bfloat16)
    label_sim_matrix = torch.matmul(
        label_one_hot.to(image_probs.device),
        model.model.codebook_sim_matrix,
    )
    return torch.mean(torch.sum(label_sim_matrix * image_probs.to(torch.bfloat16), dim=-1))

def compute_supervised_uniwm_loss(
    action_vocabulary: ActionTokenVocabulary,
    *,
    model: Any,
    outputs: Any,
    batch: Mapping[str, Any],
    tokenizer: Any,
    loss_config: dict[str, Any],
    label_smoother: Any | None = None
) -> tuple[Tensor, dict]:
    """
    Compute the combined UniWM supervised loss.
    """
    ignore_index = int(loss_config["ignore_index"])
    log_prefix = str(loss_config["log_prefix"])

    labels = batch["labels"]

    base_loss = compute_base_model_loss(
        outputs,
        labels,
        label_smoother=label_smoother,
        model=model,
        ignore_index=ignore_index,
    )

    total_loss = base_loss
    components: dict[str, float] = {
        f"{log_prefix}base_loss": detach_loss_value(base_loss),
    }

    if bool(loss_config["include_action_loss"]):
        action_loss = compute_action_token_loss(
            _get_output_value(outputs, "logits"),
            labels,
            tokenizer=tokenizer,
            ignore_index=ignore_index,
            action_vocabulary=action_vocabulary,
        )
        total_loss = total_loss + float(loss_config["action_loss_weight"]) * action_loss
        components[f"{log_prefix}action_loss"] = detach_loss_value(action_loss)

    if bool(loss_config["include_image_loss"]):
        image_loss = compute_image_codebook_discrepancy_loss(
            model=model,
            logits=_get_output_value(outputs, "logits"),
            labels=labels,
            tokenizer=tokenizer,
            ignore_index=ignore_index,
        )
        total_loss = total_loss + float(loss_config["image_loss_weight"]) * image_loss
        components[f"{log_prefix}image_loss"] = detach_loss_value(image_loss)

    components[f"{log_prefix}total_loss"] = detach_loss_value(total_loss)

    return total_loss, components

def _get_output_value(outputs: Any, key: str) -> Any:
    if isinstance(outputs, Mapping):
        return outputs.get(key)
    return getattr(outputs, key, None)

def _get_model_name(model) -> str:
    if model is None:
        return ""

    if _is_peft_model(model):
        return model.base_model.model._get_name()
    else:
        return model._get_name()

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft.mixed_model import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

def _get_hf_tokenizer(tokenizer: Any) -> Any:
    if hasattr(tokenizer, "tokenizer"):
        return tokenizer.tokenizer
    return tokenizer
