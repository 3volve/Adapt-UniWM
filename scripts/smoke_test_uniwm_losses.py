from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.uniwm_losses import (
    compute_action_token_loss,
    compute_image_codebook_discrepancy_loss,
    compute_supervised_uniwm_loss,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab = {
            "<pad>": 0,
            "stop": 1,
            "<dx_pos_bin_00>": 2,
            "<dy_pos_bin_00>": 3,
            "<dyaw_pos_bin_00>": 4,
            "<img_0>": 5,
            "<img_1>": 6,
        }
        self.tokenizer = self

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return [self.vocab[token] for token in tokens]
        return self.vocab[tokens]


class FakeConditionalGeneration:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            bpe_indices=[5, 6],
            codebook_sim_matrix=torch.tensor(
                [[1.0, 0.2], [0.2, 1.0]],
                dtype=torch.bfloat16,
            ),
            model=SimpleNamespace(
                convert_bpe2img_tokens=lambda tokens: tokens - 5,
                vqmodel=SimpleNamespace(
                    quantize=SimpleNamespace(
                        embedding=SimpleNamespace(weight=torch.zeros(2, 1))
                    )
                ),
            ),
        )

    def _get_name(self) -> str:
        return "FakeConditionalGeneration"


def main() -> None:
    tokenizer = FakeTokenizer()
    model = FakeConditionalGeneration()

    labels = torch.tensor([[0, 2, 5, 4, 6, 3]], dtype=torch.long)
    logits = torch.zeros((1, 6, 7), dtype=torch.float32)
    logits[0, 0, 2] = 5.0
    logits[0, 1, 5] = 5.0
    logits[0, 2, 4] = 5.0
    logits[0, 3, 6] = 5.0
    logits[0, 4, 3] = 5.0

    outputs = SimpleNamespace(logits=logits)
    loss_config = {
        "include_action_loss": True,
        "include_image_loss": True,
        "action_loss_weight": 1.0,
        "image_loss_weight": 1.0,
        "ignore_index": -100,
        "log_prefix": "smoke_",
    }
    action_cfg = {
        "min_dxy": 0.0,
        "max_dxy": 0.0,
        "min_dyaw": 0.0,
        "max_dyaw": 0.0,
        "bin_step": 1.0,
    }

    action_loss = compute_action_token_loss(
        logits,
        labels,
        tokenizer=tokenizer,
        action_ranges=action_cfg,
        ignore_index=-100,
    )
    image_loss = compute_image_codebook_discrepancy_loss(
        model=model,
        logits=logits,
        labels=labels,
        tokenizer=tokenizer,
        ignore_index=-100,
    )
    total_loss, components = compute_supervised_uniwm_loss(
        model=model,
        outputs=outputs,
        batch={"labels": labels},
        tokenizer=tokenizer,
        loss_config=loss_config,
        label_smoother=None,
        action_ranges=action_cfg,
    )

    assert action_loss.ndim == 0
    assert image_loss.ndim == 0
    assert total_loss.ndim == 0
    assert "smoke_base_loss" in components
    assert "smoke_action_loss" in components
    assert "smoke_image_loss" in components
    assert "smoke_total_loss" in components

    print("uniwm_losses smoke test passed")
    print(f"action_loss={float(action_loss):.6f}")
    print(f"image_loss={float(image_loss):.6f}")
    print(f"total_loss={float(total_loss):.6f}")


if __name__ == "__main__":
    main()
