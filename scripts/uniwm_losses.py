from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor
import torch.nn.functional as F
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from scripts.action_utils import DEFAULT_ACTION_RANGE_PROFILE, generate_bin_tokens, get_action_ranges


LossConfig = Mapping[str, Any]
LossComponents = dict[str, float]


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

    Preserve current CustomizeSeq2SeqTrainer.compute_loss behavior as closely as possible.
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
    action_ranges: Any | None,
    ignore_index: int,
) -> Tensor:
    """
    Compute the action-token cross entropy currently embedded in
    CustomizeSeq2SeqTrainer.compute_loss.
    """
    hf_tokenizer = _get_hf_tokenizer(tokenizer)
    dxy_range, dyaw_range, bin_step = _resolve_action_token_ranges(action_ranges)

    dx_tokens = generate_bin_tokens("dx", dxy_range[0], dxy_range[1], bin_step)
    dy_tokens = generate_bin_tokens("dy", dxy_range[0], dxy_range[1], bin_step)
    dyaw_tokens = generate_bin_tokens("dyaw", dyaw_range[0], dyaw_range[1], bin_step)

    dx_ids = set(hf_tokenizer.convert_tokens_to_ids(dx_tokens))
    dy_ids = set(hf_tokenizer.convert_tokens_to_ids(dy_tokens))
    dyaw_ids = set(hf_tokenizer.convert_tokens_to_ids(dyaw_tokens))

    shifted_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
    shifted_labels = labels[:, 1:].contiguous().view(-1)

    def compute_bin_ce(bin_ids: set[int]) -> Tensor | None:
        mask = torch.isin(
            shifted_labels,
            torch.tensor(sorted(bin_ids), device=shifted_labels.device),
        )
        if mask.any():
            return F.cross_entropy(
                shifted_logits[mask],
                shifted_labels[mask],
                ignore_index=ignore_index,
            )
        return None

    dx_loss = compute_bin_ce(dx_ids)
    dy_loss = compute_bin_ce(dy_ids)
    dyaw_loss = compute_bin_ce(dyaw_ids)

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
    Compute the image/codebook discrepancy loss currently embedded in
    CustomizeSeq2SeqTrainer.compute_loss.
    """
    del tokenizer
    del ignore_index

    image_token_ids = torch.tensor(model.model.bpe_indices, device=labels.device)
    image_mask = torch.isin(labels, image_token_ids)

    if not torch.any(image_mask):
        return logits.new_zeros(())

    image_labels = labels[image_mask]
    image_logits = logits[:, :-1, :][image_mask[:, 1:], :]

    vis_img_tokens = model.model.model.convert_bpe2img_tokens(image_labels)
    image_probs = F.softmax(image_logits[:, model.model.bpe_indices], dim=-1)

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
    *,
    model: Any,
    outputs: Any,
    batch: Mapping[str, Any],
    tokenizer: Any,
    loss_config: LossConfig,
    label_smoother: Any | None = None,
    action_ranges: Any | None = None,
) -> tuple[Tensor, LossComponents]:
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
    components: LossComponents = {
        f"{log_prefix}base_loss": detach_loss_value(base_loss),
    }

    if bool(loss_config["include_action_loss"]):
        action_loss = compute_action_token_loss(
            _get_output_value(outputs, "logits"),
            labels,
            tokenizer=tokenizer,
            action_ranges=action_ranges,
            ignore_index=ignore_index,
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


def _get_model_name(model: Any | None) -> str:
    if model is None:
        return ""
    if hasattr(model, "_get_name"):
        return model._get_name()
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "_get_name"):
        return model.base_model.model._get_name()
    return type(model).__name__


def _get_hf_tokenizer(tokenizer: Any) -> Any:
    if hasattr(tokenizer, "tokenizer"):
        return tokenizer.tokenizer
    return tokenizer


def _resolve_action_token_ranges(action_ranges: Any | None) -> tuple[list[float], list[float], float]:
    if action_ranges is None:
        ranges = get_action_ranges(DEFAULT_ACTION_RANGE_PROFILE)
        return ranges["dxy"], ranges["dyaw"], 0.01

    if "dxy" in action_ranges and "dyaw" in action_ranges:
        return action_ranges["dxy"], action_ranges["dyaw"], float(action_ranges["bin_step"])

    return (
        [float(action_ranges["min_dxy"]), float(action_ranges["max_dxy"])],
        [float(action_ranges["min_dyaw"]), float(action_ranges["max_dyaw"])],
        float(action_ranges["bin_step"]),
    )
