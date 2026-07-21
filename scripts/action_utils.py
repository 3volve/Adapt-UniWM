import math
import re
import json
from pathlib import Path
from typing import Any, Mapping


ACTION_TOKEN_MANIFEST_KEY = "action_token_vocabulary"
ACTION_TOKEN_CHECKPOINT_FILE = "action_tokens.json"
ACTION_AXES = ("dx", "dy", "dyaw")

# ===================================================================
# 1. Action Token Class
# ===================================================================
class ActionTokenVocabulary:
    """The single action-token vocabulary shared by training and inference."""

    def __init__(self, spec: Mapping[str, Any]):
        self.spec = spec
        self.coordinate_frame = str(spec["coordinate_frame"])
        self.bin_step = float(spec["bin_step"])
        self.axes = {
            axis: {
                "allow_negative": bool(spec["axes"][axis]["allow_negative"]),
                "max_bin": int(spec["axes"][axis]["max_bin"]),
            }
            for axis in ACTION_AXES
        }

        expected_tokens = spec.get("tokens_by_axis", False)
        if not expected_tokens:
            expected_tokens: dict[str, list[str]] = {}
            for axis, axis_spec in self.axes.items():
                max_bin = axis_spec["max_bin"]
                tokens = [f"<{axis}_pos_bin_{i:02d}>" for i in range(max_bin + 1)]
                if axis_spec["allow_negative"]:
                    tokens += [f"<{axis}_neg_bin_{i:02d}>" for i in range(1, max_bin + 1)]
                expected_tokens[axis] = tokens
            
        self.tokens_by_axis = expected_tokens

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "ActionTokenVocabulary":
        path = Path(manifest_path)
        with path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
            
        return cls(manifest[ACTION_TOKEN_MANIFEST_KEY])

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "ActionTokenVocabulary":
        path = Path(checkpoint_path)
        if path.is_dir():
            path = path / ACTION_TOKEN_CHECKPOINT_FILE
        with path.open("r", encoding="utf-8") as f:
            return cls(json.load(f))

    @property
    def all_tokens(self) -> list[str]:
        return [
            token
            for axis in ACTION_AXES
            for token in self.tokens_by_axis[axis]
        ]

    def range_for(self, axis: str) -> tuple[float, float]:
        axis_spec = self.axes[axis]
        maximum = axis_spec["max_bin"] * self.bin_step
        minimum = -maximum if axis_spec["allow_negative"] else 0.0
        return minimum, maximum

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate_frame": self.coordinate_frame,
            "bin_step": self.bin_step,
            "axes": self.axes,
            "tokens_by_axis": self.tokens_by_axis,
        }

    def save(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir) / ACTION_TOKEN_CHECKPOINT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")
        return output_path

# ===================================================================
# 2. Action Utilities
# ===================================================================


def calculate_action_delta(current_pos_yaw, next_pos_yaw):
    """Calculates the [dx, dy, dyaw] action vector between two poses."""
    delta_x = next_pos_yaw[0] - current_pos_yaw[0]
    delta_y = next_pos_yaw[1] - current_pos_yaw[1]
    delta_yaw = next_pos_yaw[2] - current_pos_yaw[2]
    return [float(delta_x), float(delta_y), float(delta_yaw)]

def action_to_text(action: list[float] | str, bin_width=0.01, epsilon=1e-5):
    """Encodes a numerical action vector [dx, dy, dyaw] into a token string."""
    if isinstance(action, str):
        return action

    def to_bin_token(val, prefix):
        idx = int(math.floor(abs(val) / bin_width))
        token_prefix = f"<{prefix}_pos_bin" if val >= 0 or idx == 0 else f"<{prefix}_neg_bin"
        return f"{token_prefix}_{idx:02d}>"

    dx_token = to_bin_token(action[0], "dx")
    dy_token = to_bin_token(action[1], "dy")
    dyaw_token = to_bin_token(action[2], "dyaw")

    return f"Move by dx: {dx_token}, dy: {dy_token}, dyaw: {dyaw_token}"

def extract_bin_values(token_str: str, prefix: str, step_val: float) -> float:
    pos_match = re.search(rf"<{prefix}_pos_bin_(\d+)>", token_str)
    neg_match = re.search(rf"<{prefix}_neg_bin_(\d+)>", token_str)
    
    if pos_match:
        bin_val = float(pos_match.group(1))
        return float(round(bin_val * step_val, 4))
    elif neg_match:
        bin_val = -float(neg_match.group(1))
        return float(round(bin_val * step_val, 4))
    else:
        return 0.0
