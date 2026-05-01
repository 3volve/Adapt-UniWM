"""Standalone dummy UniWM module for interface development.

Swap real imports to this module one line at a time, for example:

    from scripts.load_model import load_model
    -> from uniwm.dummy import load_model

    from uniwm.wrapped_visualizer import AnoleforConditionalGeneration
    -> from uniwm.dummy import AnoleforConditionalGeneration

    from uniwm.memory_bank import MemoryBankAnoleForConditionalGeneration
    -> from uniwm.dummy import MemoryBankAnoleForConditionalGeneration

This module keeps the public names stable while replacing the heavy model stack
with a validating dummy implementation that emits random but shape-compatible
outputs for both action and visualization generation.
"""

import copy
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutput


_TOKEN_PATTERN = re.compile(r"<[^>]+>|[A-Za-z_]+:|[A-Za-z_]+|[^\s]")


class DummyBatch(dict):
    """Minimal dict-like batch with the `.to(...)` API used downstream."""

    def to(self, device: Optional[Union[str, torch.device]] = None):
        for key, value in self.items():
            if torch.is_tensor(value):
                self[key] = value.to(device=device)
        return self


class DummyTokenizer:
    def __init__(self, image_seq_length: int):
        self.image_seq_length = image_seq_length
        self.padding_side = "right"
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._seed_vocab()

    def _seed_vocab(self) -> None:
        for token in [
            "<pad>",
            "<bos>",
            "</s>",
            "<image>",
            "<boi>",
            "<eoi>",
            "<reserved08706>",
            "stop",
            "Move",
            "by",
            "dx:",
            "dy:",
            "dyaw:",
            ",",
        ]:
            self._add_token(token)

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return self.token_to_id[token]

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["</s>"]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.token_to_id)

    def add_tokens(self, tokens: Iterable[str], special_tokens: bool = False) -> int:
        del special_tokens
        added = 0
        for token in tokens:
            if token not in self.token_to_id:
                self._add_token(token)
                added += 1
        return added

    def convert_tokens_to_ids(self, tokens: Union[str, Sequence[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self._add_token(tokens)
        return [self._add_token(token) for token in tokens]

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        return self.convert_tokens_to_ids(_TOKEN_PATTERN.findall(text))

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        special_ids = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
            self.token_to_id["<image>"],
            self.token_to_id["<boi>"],
            self.token_to_id["<eoi>"],
        }
        parts: List[str] = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in special_ids:
                continue
            parts.append(self.id_to_token.get(token_id, f"<unk_{token_id}>"))
        return " ".join(parts).replace(" ,", ",").strip()

    def batch_decode(
        self,
        sequences: Union[torch.Tensor, Sequence[Sequence[int]]],
        skip_special_tokens: bool = False,
    ) -> List[str]:
        if torch.is_tensor(sequences):
            if sequences.dim() == 1:
                sequences = sequences.unsqueeze(0)
            iterable = sequences.tolist()
        else:
            iterable = sequences
        return [self.decode(sequence, skip_special_tokens=skip_special_tokens) for sequence in iterable]


@dataclass
class DummyImageProcessor:
    size: Dict[str, int]
    crop_size: Dict[str, int]


class DummyProcessor:
    """Processor replacement that validates UniWM-compatible text/image inputs."""

    def __init__(self, image_seq_length: int = 784, resolution: int = 448):
        self.image_seq_length = image_seq_length
        self.tokenizer = DummyTokenizer(image_seq_length=image_seq_length)
        self.image_processor = DummyImageProcessor(
            size={"shortest_edge": resolution},
            crop_size={"height": resolution, "width": resolution},
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "DummyProcessor":
        clone = type(self)(
            image_seq_length=self.image_seq_length,
            resolution=self.image_processor.size["shortest_edge"],
        )
        clone.tokenizer.token_to_id = copy.deepcopy(self.tokenizer.token_to_id, memo)
        clone.tokenizer.id_to_token = copy.deepcopy(self.tokenizer.id_to_token, memo)
        clone.tokenizer.padding_side = self.tokenizer.padding_side
        return clone

    def _normalize_image(self, value: Any) -> Image.Image:
        if isinstance(value, Image.Image):
            return value.convert("RGB")

        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.dim() != 3:
                raise AssertionError(f"Expected image tensor with 3 dims, got shape {tuple(tensor.shape)}")
            if tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            elif tensor.shape[-1] not in (1, 3):
                raise AssertionError(f"Expected image tensor with 1 or 3 channels, got shape {tuple(tensor.shape)}")
            return self._normalize_image(tensor.numpy())

        if isinstance(value, np.ndarray):
            array = value
            if array.ndim == 4 and array.shape[0] == 1:
                array = array[0]
            if array.ndim != 3:
                raise AssertionError(f"Expected image array with 3 dims, got shape {array.shape}")
            if array.shape[0] in (1, 3):
                array = np.transpose(array, (1, 2, 0))
            if array.shape[-1] not in (1, 3):
                raise AssertionError(f"Expected image array with 1 or 3 channels, got shape {array.shape}")
            if array.dtype != np.uint8:
                max_value = float(array.max()) if array.size else 0.0
                if max_value <= 1.0:
                    array = array * 255.0
                array = np.clip(array, 0, 255).astype(np.uint8)
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            return Image.fromarray(array, mode="RGB")

        if isinstance(value, dict):
            raise AssertionError(
                "Received an observation dict. Convert Habitat observations to a UniWM-compatible image "
                "(PIL image, HWC ndarray, or CHW/HWC tensor) before calling the processor."
            )

        observation_keys = [name for name in ("rgb", "image", "observation", "obs") if hasattr(value, name)]
        if observation_keys:
            raise AssertionError(
                f"Received an observation-like object exposing {observation_keys}. Extract the rendered RGB image "
                "first so the UniWM call site stays compatible with the real processor."
            )

        raise AssertionError(
            f"Unsupported image type {type(value)}. Expected PIL image, numpy array, or tensor already in "
            "a UniWM-compatible image form."
        )

    def _normalize_images(self, images: Any) -> List[Image.Image]:
        if images is None:
            return []
        if isinstance(images, (list, tuple)):
            return [self._normalize_image(image) for image in images]
        return [self._normalize_image(images)]

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        resolution = self.image_processor.size["shortest_edge"]
        resized = image.resize((resolution, resolution))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def __call__(
        self,
        text: Optional[Union[str, Sequence[str]]] = None,
        images: Optional[Any] = None,
        padding: Optional[str] = None,
        return_tensors: Optional[str] = None,
        max_length: Optional[int] = None,
        **_: Any,
    ) -> DummyBatch:
        if return_tensors not in (None, "pt"):
            raise AssertionError(f"DummyProcessor only supports return_tensors='pt', got {return_tensors}")

        if text is None:
            texts = [""]
        elif isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        if len(texts) != 1:
            raise AssertionError(f"DummyProcessor currently supports a single example at a time, got batch size {len(texts)}")

        normalized_images = self._normalize_images(images)

        token_ids = [self.tokenizer.bos_token_id]
        token_ids.extend(self.tokenizer.encode(texts[0]))
        if normalized_images:
            image_placeholder = self.tokenizer.convert_tokens_to_ids("<image>")
            token_ids.extend([image_placeholder] * (len(normalized_images) * self.image_seq_length))
        token_ids.append(self.tokenizer.eos_token_id)

        if max_length is not None:
            if len(token_ids) > max_length:
                if self.tokenizer.padding_side == "left":
                    token_ids = token_ids[-max_length:]
                else:
                    token_ids = token_ids[:max_length]
            elif padding == "max_length":
                pad_count = max_length - len(token_ids)
                pad_tokens = [self.tokenizer.pad_token_id] * pad_count
                if self.tokenizer.padding_side == "left":
                    token_ids = pad_tokens + token_ids
                else:
                    token_ids = token_ids + pad_tokens

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.long)
        batch = DummyBatch({"input_ids": input_ids, "attention_mask": attention_mask})

        if normalized_images:
            image_tensors = [self._image_to_tensor(image) for image in normalized_images]
            if len(image_tensors) == 1:
                batch["pixel_values"] = torch.stack(image_tensors, dim=0)
            else:
                batch["pixel_values"] = torch.stack(image_tensors, dim=0).unsqueeze(0)

        return batch

    def batch_decode(
        self,
        sequences: Union[torch.Tensor, Sequence[Sequence[int]]],
        skip_special_tokens: bool = False,
    ) -> List[str]:
        return self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)

    def postprocess_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(pixel_values):
            raise AssertionError(f"Expected pixel_values tensor, got {type(pixel_values)}")
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.dim() != 4:
            raise AssertionError(f"Expected pixel_values with shape [B, C, H, W], got {tuple(pixel_values.shape)}")
        if pixel_values.shape[1] != 3:
            raise AssertionError(f"Expected 3-channel pixel_values, got shape {tuple(pixel_values.shape)}")
        processed = pixel_values.detach().float()
        if processed.max() <= 1.0:
            processed = processed * 255.0
        return processed.clamp(0, 255).round().to(torch.uint8)


class _DummyVisionCore(nn.Module):
    def __init__(self, image_seq_length: int, image_bpe_start: int, codebook_size: int, resolution: int):
        super().__init__()
        self.image_seq_length = image_seq_length
        self.image_bpe_start = image_bpe_start
        self.codebook_size = codebook_size
        self.vqmodel = SimpleNamespace(
            config=SimpleNamespace(resolution=resolution, channel_multiplier=[1, 2, 4]),
            quantize=SimpleNamespace(
                embedding=nn.Embedding(codebook_size, 32),
                quant_state_dims=[resolution // 4, resolution // 4],
            ),
        )
        self.layers = nn.ModuleList()

    def _assert_pixel_values(self, pixel_values: torch.Tensor) -> None:
        if not torch.is_tensor(pixel_values):
            raise AssertionError(f"pixel_values must be a tensor, got {type(pixel_values)}")
        if pixel_values.dim() not in (4, 5):
            raise AssertionError(f"pixel_values must have 4 or 5 dims, got shape {tuple(pixel_values.shape)}")
        if pixel_values.dim() == 4 and pixel_values.shape[1] != 3:
            raise AssertionError(f"Expected pixel_values shape [B, 3, H, W], got {tuple(pixel_values.shape)}")
        if pixel_values.dim() == 5 and pixel_values.shape[2] != 3:
            raise AssertionError(f"Expected pixel_values shape [B, N, 3, H, W], got {tuple(pixel_values.shape)}")

    def _single_image_tokens(self, image_tensor: torch.Tensor) -> torch.Tensor:
        flattened = image_tensor.flatten()
        if flattened.numel() == 0:
            raise AssertionError("pixel_values must be non-empty")
        target = self.image_seq_length
        if flattened.numel() < target:
            flattened = F.pad(flattened, (0, target - flattened.numel()))
        flattened = flattened[:target]
        offsets = torch.arange(target, device=image_tensor.device, dtype=torch.long)
        bins = (flattened.float() * 997).round().to(torch.long).abs()
        return (bins + offsets) % self.codebook_size + self.image_bpe_start

    def get_image_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self._assert_pixel_values(pixel_values)
        if pixel_values.dim() == 4:
            return torch.stack([self._single_image_tokens(image) for image in pixel_values], dim=0)
        return torch.stack(
            [torch.stack([self._single_image_tokens(image) for image in batch], dim=0) for batch in pixel_values],
            dim=0,
        )

    def convert_bpe2img_tokens(self, bpe_tokens: torch.Tensor) -> torch.Tensor:
        return (bpe_tokens.to(torch.long) - self.image_bpe_start).clamp(min=0, max=self.codebook_size - 1)


class _DummyBackbone(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, image_seq_length: int, resolution: int):
        super().__init__()
        self.image_bpe_start = 16000
        self.codebook_size = 512
        self.model = _DummyVisionCore(
            image_seq_length=image_seq_length,
            image_bpe_start=self.image_bpe_start,
            codebook_size=self.codebook_size,
            resolution=resolution,
        )
        self.vqmodel = self.model.vqmodel
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bpe_indices = list(range(self.image_bpe_start, self.image_bpe_start + self.codebook_size))
        self.codebook_sim_matrix = torch.eye(self.codebook_size, dtype=torch.bfloat16)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        return self.lm_head(hidden)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def resize_token_embeddings(self, new_size: int) -> nn.Embedding:
        if new_size <= self.embed_tokens.num_embeddings:
            return self.embed_tokens

        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        old_embed = self.embed_tokens
        old_head = self.lm_head

        new_embed = nn.Embedding(new_size, old_embed.embedding_dim, device=device, dtype=dtype)
        new_head = nn.Linear(old_head.in_features, new_size, bias=False, device=device, dtype=dtype)

        with torch.no_grad():
            new_embed.weight[: old_embed.num_embeddings] = old_embed.weight
            new_head.weight[: old_head.out_features] = old_head.weight

        self.embed_tokens = new_embed
        self.lm_head = new_head
        return self.embed_tokens

    def decode_image_tokens(self, image_tokens: torch.Tensor) -> torch.Tensor:
        if image_tokens.dim() == 1:
            image_tokens = image_tokens.unsqueeze(0)
        if image_tokens.dim() != 2:
            raise AssertionError(f"Expected image token tensor with shape [B, T], got {tuple(image_tokens.shape)}")
        indices = self.model.convert_bpe2img_tokens(image_tokens).float()
        side = int(self.model.image_seq_length ** 0.5)
        base = indices[:, : side * side].reshape(-1, 1, side, side)
        base = base / max(1, self.codebook_size - 1)
        resolution = self.vqmodel.config.resolution
        upsampled = F.interpolate(base, size=(resolution, resolution), mode="bilinear", align_corners=False)
        rgb = torch.cat((upsampled, 1.0 - upsampled, 0.5 * torch.ones_like(upsampled)), dim=1)
        return (rgb * 255.0).clamp(0, 255)


class AnoleforConditionalGeneration(nn.Module):
    """Dummy replacement for the real UniWM visualizer model."""

    def __init__(self, config: Optional[Any] = None, processor: Optional[DummyProcessor] = None, **kwargs: Any):
        super().__init__()
        del kwargs
        self.processor = processor or DummyProcessor()
        resolution = self.processor.image_processor.size["shortest_edge"]
        image_token_num = self.processor.image_seq_length
        vocab_size = 20000
        self.config = config or SimpleNamespace(
            image_token_id=self.processor.tokenizer.convert_tokens_to_ids("<image>"),
            pad_token_id=self.processor.tokenizer.pad_token_id,
            boi_token_id=self.processor.tokenizer.convert_tokens_to_ids("<boi>"),
            eoi_token_id=self.processor.tokenizer.convert_tokens_to_ids("<eoi>"),
            num_beams=1,
            max_length=128,
            torch_dtype=torch.bfloat16,
        )
        self.model = _DummyBackbone(vocab_size=vocab_size, hidden_size=64, image_seq_length=image_token_num, resolution=resolution)
        self.generation_config = GenerationConfig(max_length=self.config.max_length, num_beams=self.config.num_beams)
        self.generation_config._from_model_config = False
        self.image_decoder = None
        self.generate_with_embeds = False
        self.image_postprocess = True
        self.image_token_num = image_token_num
        self.sketch_resolution = (resolution, resolution)
        self.bpe_indices = self.model.bpe_indices
        self.img_indices = list(range(self.model.codebook_size))
        self.codebook_sim_matrix = self.model.codebook_sim_matrix
        self.memory_bank_initialized = False
        self.current_step = 0

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> "AnoleforConditionalGeneration":
        del pretrained_model_name_or_path, args
        return cls(**kwargs)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def get_vis_codebook_sim(self) -> None:
        weights = self.model.vqmodel.quantize.embedding.weight.detach().to(torch.float32)
        normalized = F.normalize(weights, dim=-1)
        self.model.codebook_sim_matrix = (normalized @ normalized.t()).to(torch.bfloat16)
        self.codebook_sim_matrix = self.model.codebook_sim_matrix

    def set_global_memory_manager(self, global_memory_manager: Any) -> None:
        del global_memory_manager

    def update_step(self, step: int) -> None:
        self.current_step = step

    def reset_memory_bank(self) -> None:
        self.memory_bank_initialized = False

    def decode_image_tokens(self, bpe_tokens: torch.Tensor) -> torch.Tensor:
        return self.model.decode_image_tokens(bpe_tokens)

    def _assert_inputs(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        pixel_values: Optional[torch.Tensor],
    ) -> None:
        if input_ids is None:
            raise AssertionError("UniWM generate/forward calls require input_ids")
        if input_ids.dim() != 2:
            raise AssertionError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise AssertionError(
                f"attention_mask shape must match input_ids: {tuple(attention_mask.shape)} vs {tuple(input_ids.shape)}"
            )
        if pixel_values is not None:
            self.model.model._assert_pixel_values(pixel_values)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **_: Any,
    ) -> CausalLMOutput:
        self._assert_inputs(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        logits = self.model(input_ids.to(self.device))
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            logits = logits[:, : labels.shape[1], :]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
            if torch.isnan(loss):
                loss = logits.sum() * 0.0
        return CausalLMOutput(loss=loss, logits=logits)

    def _random_action_tokens(self, batch_size: int) -> torch.Tensor:
        vocab = self.processor.tokenizer.get_vocab()
        stop_id = self.processor.tokenizer.convert_tokens_to_ids("stop")
        dx_tokens = [token for token in vocab if token.startswith("<dx_")]
        dy_tokens = [token for token in vocab if token.startswith("<dy_")]
        dyaw_tokens = [token for token in vocab if token.startswith("<dyaw_")]

        sequences: List[List[int]] = []
        for _ in range(batch_size):
            if not (dx_tokens and dy_tokens and dyaw_tokens) or torch.rand(1, device=self.device).item() < 0.2:
                sequences.append([stop_id])
                continue

            dx_token = dx_tokens[int(torch.randint(0, len(dx_tokens), (1,), device=self.device).item())]
            dy_token = dy_tokens[int(torch.randint(0, len(dy_tokens), (1,), device=self.device).item())]
            dyaw_token = dyaw_tokens[int(torch.randint(0, len(dyaw_tokens), (1,), device=self.device).item())]
            sequence = self.processor.tokenizer.convert_tokens_to_ids(
                ["Move", "by", "dx:", dx_token, ",", "dy:", dy_token, ",", "dyaw:", dyaw_token]
            )
            sequences.append(sequence)

        max_len = max(len(sequence) for sequence in sequences)
        padded = [
            sequence + [self.processor.tokenizer.pad_token_id] * (max_len - len(sequence))
            for sequence in sequences
        ]
        return torch.tensor(padded, dtype=torch.long, device=self.device)

    def _random_visual_tokens(self, batch_size: int) -> torch.Tensor:
        start = self.config.boi_token_id
        end = self.config.eoi_token_id
        image_tokens = torch.randint(
            low=self.model.bpe_indices[0],
            high=self.model.bpe_indices[-1] + 1,
            size=(batch_size, self.image_token_num),
            device=self.device,
        )
        return torch.cat(
            (
                torch.full((batch_size, 1), start, dtype=torch.long, device=self.device),
                image_tokens,
                torch.full((batch_size, 1), end, dtype=torch.long, device=self.device),
            ),
            dim=1,
        )

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[Any] = None,
        multimodal_generation_mode: str = "interleaved-text-image",
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, None]:
        del logits_processor
        input_ids = kwargs.get("input_ids", inputs)
        attention_mask = kwargs.get("attention_mask")
        pixel_values = kwargs.get("pixel_values")
        self._assert_inputs(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

        batch_size = input_ids.shape[0]
        current_substep = kwargs.get("current_substep")
        max_new_tokens = kwargs.get("max_new_tokens")
        if max_new_tokens is None and generation_config is not None:
            max_new_tokens = generation_config.max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = self.generation_config.max_length

        should_generate_image = (
            current_substep == "visualization"
            or multimodal_generation_mode == "image-only"
            or max_new_tokens >= self.image_token_num
        )
        if should_generate_image:
            return self._random_visual_tokens(batch_size=batch_size), None
        return self._random_action_tokens(batch_size=batch_size), None


class MemoryBankAnoleForConditionalGeneration(AnoleforConditionalGeneration):
    """Dummy replacement for the memory-bank UniWM variant."""

    def __init__(self, config: Optional[Any] = None, processor: Optional[DummyProcessor] = None, **kwargs: Any):
        super().__init__(config=config, processor=processor, **kwargs)
        self.use_memory_bank_layers = {7, 13, 19, 25, 31}
        self.use_global_memory_bank = False
        self.use_memory_bank = False

    def reset_global_memory_bank(self) -> None:
        self.use_global_memory_bank = False

    def store_to_global_memory_bank(self, current_step: int) -> None:
        self.use_global_memory_bank = current_step > 0

    def enable_global_memory_bank(self) -> None:
        self.use_global_memory_bank = True

    def enable_memory_bank(self) -> None:
        self.use_memory_bank = True

    def initialize_memory_bank(self, input_ids: torch.Tensor, pixel_values: torch.Tensor, attention_mask: torch.Tensor) -> bool:
        self._assert_inputs(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        self.memory_bank_initialized = True
        return True

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[Any] = None,
        use_memory_bank: bool = False,
        is_memory_bank_init: bool = False,
        current_step: Optional[int] = None,
        current_substep: Optional[str] = None,
        use_global_memory_bank: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, None]:
        self.use_memory_bank = use_memory_bank
        self.memory_bank_initialized = self.memory_bank_initialized or is_memory_bank_init or use_memory_bank
        self.use_global_memory_bank = use_global_memory_bank
        self.current_step = current_step or self.current_step
        return super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            current_substep=current_substep,
            **kwargs,
        )


def load_model(args: Any, training_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Mirror `scripts.load_model.load_model` with a dummy implementation."""
    del training_cfg
    print("Loading dummy UniWM stack for interface development")
    image_token_num = getattr(args, "image_seq_length", 784)
    processor = DummyProcessor(image_seq_length=image_token_num)

    use_memory_bank = bool(
        getattr(args, "use_memory_bank_inference", False)
        and getattr(args, "do_task_level_eval", False)
        and not getattr(args, "do_train", False)
    )
    model_cls = MemoryBankAnoleForConditionalGeneration if use_memory_bank else AnoleforConditionalGeneration
    model = model_cls.from_pretrained("dummy/uniwm", processor=processor, codebook_sim="mse")

    processor.tokenizer.padding_side = "left"
    processor.image_processor.size = {"shortest_edge": 448}
    processor.image_processor.crop_size = {"height": 448, "width": 448}

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.model.vqmodel.config.resolution = processor.image_processor.size["shortest_edge"]
    model.model.vqmodel.quantize.quant_state_dims = [
        model.model.vqmodel.config.resolution // 2 ** (len(model.model.vqmodel.config.channel_multiplier) - 1)
    ] * 2
    model.sketch_resolution = (
        model.model.vqmodel.config.resolution,
        model.model.vqmodel.config.resolution,
    )
    model.image_token_num = image_token_num
    model.get_vis_codebook_sim()

    return {"processor": processor, "model": model}


__all__ = [
    "AnoleforConditionalGeneration",
    "MemoryBankAnoleForConditionalGeneration",
    "DummyProcessor",
    "load_model",
]
