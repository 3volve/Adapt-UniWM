import math
import re

# ===================================================================
# 1. Action Range Constants
# ===================================================================
ACTION_RANGES = {
    "tartan_drive": {
        "dxy": (-2.05, 2.05),
        "dyaw": (-0.17, 0.17)
    },
    "recon": {
        "dxy": (-2.46, 2.46),
        "dyaw": (-1.87, 1.87)
    },
    "scand": {
        "dxy": (-0.7879, 0.8518),
        "dyaw": (-0.48, 0.48)
    },
    "sacson": {
        "dxy": (-1.35, 1.35),
        "dyaw": (-2.82, 2.82)
    },
    "stanford": {
        "dxy": (-0.18, 0.18),
        "dyaw": (-0.63, 0.63)
    },
    "go_stanford": {
        "dxy": (-0.18, 0.18),
        "dyaw": (-0.63, 0.63)
    },
    "habitat": {
        "dxy": (-0.25, 0.25),
        "dyaw": (-0.5236, 0.5236)
    }
}
DEFAULT_ACTION_RANGE_PROFILE = "go_stanford"

# ===================================================================
# 2. Action Calculation Utilities
# ===================================================================

class ActionCfg:
    """Convenient action-relevant variable data class"""
    min_dxy: float
    max_dxy: float
    min_dyaw: float
    max_dyaw: float
    bin_step: float

    def __init__(self,
        min_dxy: float | None = None,
        max_dxy: float | None = None,
        min_dyaw: float | None = None,
        max_dyaw: float | None = None,
        bin_step: float | None = None
    ):
        """
        Convenience data-class that takes in a set of min/max dxy/dyaw and bin_step
            Will instantiate with values from the DEFAULT_ACTION_RANGE_PROFILE for any values not given.
        """
        default_ranges = get_action_ranges(DEFAULT_ACTION_RANGE_PROFILE)

        self.min_dxy = min_dxy or default_ranges["dxy"][0]
        self.max_dxy = max_dxy or default_ranges["dxy"][1]
        self.min_dyaw = min_dyaw or default_ranges["dyaw"][0]
        self.max_dyaw = max_dyaw or default_ranges["dyaw"][1]
        self.bin_step = bin_step or 0.01

    @staticmethod
    def from_dict(d: dict[str, float]):
        return ActionCfg(d["min_dxy"], d["max_dxy"], d["min_dyaw"], d["max_dyaw"], d["bin_step"])

    def get_dxy_tok_params(self) -> tuple[float, float, float]:
        return self.min_dxy, self.max_dxy, self.bin_step

    def get_dyaw_tok_params(self) -> tuple[float, float, float]:
        return self.min_dyaw, self.max_dyaw, self.bin_step

    def get_dxy_tuple(self) -> tuple[float, float]:
        return self.min_dxy, self.max_dxy

    def get_dyaw_tuple(self) -> tuple[float, float]:
        return self.min_dyaw, self.max_dyaw


def get_action_ranges(range_profile: str | None) -> dict[str, tuple[float, float]]:
    if range_profile is None:
        return {
            "dxy": ACTION_RANGES[DEFAULT_ACTION_RANGE_PROFILE]["dxy"],
            "dyaw": ACTION_RANGES[DEFAULT_ACTION_RANGE_PROFILE]["dyaw"],
        }

    if range_profile not in ACTION_RANGES:
        raise KeyError(
            f"Unknown action range profile '{range_profile}'. "
            f"Known profiles: {sorted(ACTION_RANGES.keys())}"
        )

    return {
        "dxy": ACTION_RANGES[range_profile]["dxy"],
        "dyaw": ACTION_RANGES[range_profile]["dyaw"],
    }

def get_action_config(range_profile: str | None, bin_step: float | None = None) -> ActionCfg:
    min_dxy = ACTION_RANGES[DEFAULT_ACTION_RANGE_PROFILE]["dxy"][0]
    max_dxy = ACTION_RANGES[DEFAULT_ACTION_RANGE_PROFILE]["dxy"][1]
    min_dyaw = ACTION_RANGES[DEFAULT_ACTION_RANGE_PROFILE]["dyaw"][0]
    max_dyaw = ACTION_RANGES[DEFAULT_ACTION_RANGE_PROFILE]["dyaw"][1]
    bin_step = bin_step or 0.01

    if range_profile not in ACTION_RANGES:
        raise KeyError(
            f"Unknown action range profile '{range_profile}'. "
            f"Known profiles: {sorted(ACTION_RANGES.keys())}"
        )

    if range_profile is not None:
        min_dxy = ACTION_RANGES[range_profile]["dxy"][0]
        max_dxy = ACTION_RANGES[range_profile]["dxy"][1]
        min_dyaw = ACTION_RANGES[range_profile]["dyaw"][0]
        max_dyaw = ACTION_RANGES[range_profile]["dyaw"][1]

    return ActionCfg(min_dxy, max_dxy, min_dyaw, max_dyaw, bin_step)

def calculate_action_delta(current_pos_yaw, next_pos_yaw):
    """Calculates the [dx, dy, dyaw] action vector between two poses."""
    delta_x = next_pos_yaw[0] - current_pos_yaw[0]
    delta_y = next_pos_yaw[1] - current_pos_yaw[1]
    delta_yaw = next_pos_yaw[2] - current_pos_yaw[2]
    return [float(delta_x), float(delta_y), float(delta_yaw)]

# ===================================================================
# 3. Action Tokenization Toolkit (Encoder, Decoder, Generator)
# ===================================================================
def action_to_text(action: list[float] | str, bin_width=0.01, epsilon=1e-5):
    """Encodes a numerical action vector [dx, dy, dyaw] into a token string."""
    if isinstance(action, str):
        return action

    def to_bin_token(val, prefix):
        token_prefix = f"<{prefix}_pos_bin" if val >= 0 else f"<{prefix}_neg_bin"
        idx = int(math.floor(abs(val) / bin_width))
        return f"{token_prefix}_{idx:02d}>"

    dx_token = to_bin_token(action[0], "dx")
    dy_token = to_bin_token(action[1], "dy")
    dyaw_token = to_bin_token(action[2], "dyaw")

    return f"Move by dx: {dx_token}, dy: {dy_token}, dyaw: {dyaw_token}"

def generate_bin_tokens(prefix, vmin, vmax, step):
    """
    Generates positive, negative, and a zero token.
    """
    tokens = []
    
    # Calculate and generate positive bins based on vmax
    if vmax >= 0:
        nbins_pos = int(math.floor(vmax / step))
        tokens += [f"<{prefix}_pos_bin_{i:02d}>" for i in range(0, nbins_pos + 1)]
        
    # Calculate and generate negative bins based on vmin
    if vmin < 0:
        nbins_neg = int(math.floor(abs(vmin) / step))
        tokens += [f"<{prefix}_neg_bin_{i:02d}>" for i in range(0, nbins_neg + 1)]
        
    return tokens

def extract_bin_values(token_str: str, prefix: str, step_val: float) -> float:
    pos_match = re.search(f"<{prefix}_pos_bin_(\d+)>", token_str)
    neg_match = re.search(f"<{prefix}_neg_bin_(\d+)>", token_str)
    
    if pos_match:
        bin_val = float(pos_match.group(1))
        return float(round(bin_val * step_val, 4))
    elif neg_match:
        bin_val = -float(neg_match.group(1))
        return float(round(bin_val * step_val, 4))
    else:
        return 0.0
