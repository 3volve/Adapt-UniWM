# scripts/trainer/task_eval_utils/prompt_builder.py
from typing import Tuple

ACTION_PROMPT_TEMPLATES = [
    (
        "Task: Navigation Action Prediction\n"
        "Based on the current first-person observation, starting point observation and coordinate, goal point observation, predict the next action to take. The definition of actions is as follows.\n"
        "Action Definitions: \n"
        "The action can be the language command 'Stop', indicating the end of the trajectory. Alternatively, the action can be shifts composed of three components:\n"
        "- dx: displacement along the agent's facing direction),\n"
        "- dy: displacement perpendicular to the facing direction),\n"
        "- dyaw: change in heading angle (i.e., how much the agent rotates).\n"
        "All components are discretized into bin tokens: for example,\n"
        "- `dx pos bin 02`: dx = +0.02 meters,\n"
        "- `dy neg bin 23`: dy = -0.23 meters,\n"
        "- `dyaw pos bin 26`: counterclockwise rotation of +0.26 radians.\n"
        "If the agent reaches the goal or believes it has reached, it should predict 'Stop'.\n"
        "Action Format: \n"
        "-Range of dx, dy: [{dxy_min}, {dxy_max}], -Range of dyaw: [{dyaw_min}, {dyaw_max}]. -Output format: Move by dx: <dx>, dy: <dy>, dyaw: <dyaw>\n"
        "Inputs:\n"
        "- Start Observation: <image> \n"
        "- Goal Observation: <image> \n"
        "- Current Observation: <image> \n"
        "{start_pose_str} \n"
        "Goal: \n"
        "Predict the next action to approach the goal observation"
    ),
    (
        "Task: Navigation Action Prediction\n"
        "Predict exactly one next navigation action.\n"
        "Allowed outputs:\n"
        "1. Stop\n"
        "2. Move by dx: <dx_token>, dy: <dy_token>, dyaw: <dyaw_token>\n"
        "Rules:\n"
        "- Output exactly one action.\n"
        "- Output exactly one line.\n"
        "- Do not explain your reasoning.\n"
        "- Do not output any text before or after the action.\n"
        "- Valid dx/dy range: [{dxy_min}, {dxy_max}].\n"
        "- Valid dyaw range: [{dyaw_min}, {dyaw_max}].\n"
        "Example valid outputs:\n"
        "- Move by dx: <dx_pos_bin_02>, dy: <dy_neg_bin_00>, dyaw: <dyaw_pos_bin_26>\n"
        "- Stop\n"
        "Inputs:\n"
        "- Start Observation: <image>\n"
        "- Goal Observation: <image>\n"
        "- Current Observation: <image>\n"
        "{start_pose_str}\n"
        "Output:"
    )
]

VIZ_PROMPT_TEMPLATES = [
    (
        "Task: Navigation Single Step Visualization\n"
        "Description: Given the current first-person observation, predict the next first-person view observation after the agent executes a specified navigation action.\n To assist your prediction, you may refer to the start observation and pose (position: x, y and heading: yaw), as well as the goal and current observation.\n"
        "Inputs:\n"
        "Next Action: {decoded_action}.\n"
        "{start_pose_str} \n"
        "- Start Observation: <image> \n"
        "- Goal Observation: <image> \n"
        "- Current Observation: <image> \n"
        "Action Format:\n"
        "The action can be the language command 'Stop', indicating the end of the trajectory. Alternatively, the action can be shifts composed of three components:\n"
        "- dx: displacement along the agent's facing direction),\n"
        "- dy: displacement perpendicular to the facing direction),\n"
        "- dyaw: change in heading angle (i.e., how much the agent rotates).\n"
        "All components are discretized into bin tokens: for example,\n"
        "- `dx pos bin 02`: dx = +0.02 meters,\n"
        "- `dy neg bin 23`: dy = -0.23 meters,\n"
        "- `dyaw pos bin 26`: counterclockwise rotation of +0.26 radians.\n"
        "Spatial Interpretation:\n"
        "- The magnitude of [dx, dy] reflects how far the agent moves in this step — larger values indicate greater positional shift, leading to larger visual changes \n"
        "- dyaw controls the agent's rotation (change in heading). A positive dyaw indicates a left turn (counter-clockwise), while a negative dyaw indicates a right turn (clockwise). \n"
        "Goal: \n"
        "Predict the most likely next first-person observation, considering how the movement and rotation implied by `dx`, `dy`, and `dyaw` would affect what the agent sees next."
    ),
    (
        "Task: Navigation Single Step Visualization\n"
        "Predict the next first-person observation after executing the specified action.\n"
        "Rules:\n"
        "- Use the provided Next Action exactly as given.\n"
        "- Use the start pose, start observation, goal observation, and current observation as context.\n"
        "- Predict only the next observation.\n"
        "- Do not describe the image in words.\n"
        "- Output one image continuation only.\n"
        "Inputs:\n"
        "Next Action: {decoded_action}\n"
        "{start_pose_str}\n"
        "- Start Observation: <image>\n"
        "- Goal Observation: <image>\n"
        "- Current Observation: <image>\n"
        "Output the most likely next first-person observation."
    )
]

def build_action_prompt(
    start_pose_str: str,
    dxy_range: Tuple[float, float],
    dyaw_range: Tuple[float, float],
    prompt_style_idx: int = 0,
) -> str:
    if prompt_style_idx > len(ACTION_PROMPT_TEMPLATES):
        prompt_style_idx = 0

    return ACTION_PROMPT_TEMPLATES[prompt_style_idx].format(
        start_pose_str=start_pose_str,
        dxy_min=f"{dxy_range[0]:.2f}",
        dxy_max=f"{dxy_range[1]:.2f}",
        dyaw_min=f"{dyaw_range[0]:.2f}",
        dyaw_max=f"{dyaw_range[1]:.2f}",
    )


def build_viz_prompt(
    decoded_action: str,
    start_pose_str: str,
    prompt_style_idx: int = 0,
) -> str:
    if prompt_style_idx > len(VIZ_PROMPT_TEMPLATES):
        prompt_style_idx = 0

    return VIZ_PROMPT_TEMPLATES[prompt_style_idx].format(
        decoded_action=decoded_action,
        start_pose_str=start_pose_str,
    )
