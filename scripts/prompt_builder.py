# scripts/trainer/task_eval_utils/prompt_builder.py
from typing import Tuple

ACTION_PROMPT_TEMPLATE = (
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
    "- dx is forward-only and cannot be negative.\n"
    "- Range of dx: [{dx_min}, {dx_max}]. Range of dy: [{dy_min}, {dy_max}]. Range of dyaw: [{dyaw_min}, {dyaw_max}].\n"
    "- \n"
    "- Output format: Move by dx: <dx>, dy: <dy>, dyaw: <dyaw>\n"
    "Inputs:\n"
    "- Start Observation: <image>\n"
    "- Goal Observation: <image>\n"
    "- Current Observation: <image>\n"
    "- Episode's starting pose position: {start_pose_str}\n"
    "- Prior 0 to 3 actions from episode:{prior_decoded_actions}\n"
    "Goal: Predict the next action to approach the goal observation"
)

VIZ_PROMPT_TEMPLATE = (
    "Task: Navigation Single Step Visualization\n"
    "Description: Given the current first-person observation, predict the next first-person view observation after the agent executes a specified navigation action.\n To assist your prediction, you may refer to the start observation and pose (position: x, y and heading: yaw), as well as the goal and current observation.\n"
    "Inputs:\n"
    "- Next Action: {decoded_action}.\n"
    "- Episode's starting pose position: {start_pose_str} \n"
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
)

def build_action_prompt(
    start_pose_str: str,
    dx_range: Tuple[float, float],
    dy_range: Tuple[float, float],
    dyaw_range: Tuple[float, float],
    prior_decoded_actions: list[str] | None = None,
) -> str:
    prior_act_str = (
        "" if prior_decoded_actions is None 
        else "".join([f"\n  - {act}" for act in prior_decoded_actions])
    )
    
    return ACTION_PROMPT_TEMPLATE.format(
        start_pose_str        = start_pose_str,
        dx_min                = f"{dx_range[0]:.2f}",
        dx_max                = f"{dx_range[1]:.2f}",
        dy_min                = f"{dy_range[0]:.2f}",
        dy_max                = f"{dy_range[1]:.2f}",
        dyaw_min              = f"{dyaw_range[0]:.2f}",
        dyaw_max              = f"{dyaw_range[1]:.2f}",
        prior_decoded_actions = prior_act_str
    )


def build_viz_prompt(
    decoded_action: str,
    start_pose_str: str,
) -> str:
    return VIZ_PROMPT_TEMPLATE.format(
        decoded_action=decoded_action,
        start_pose_str=start_pose_str,
    )
