from __future__ import annotations

import math, torch, os, re, csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from habitat_sim import ActionSpec, ActuationSpec
import numpy as np
from PIL import Image
from collections.abc import Mapping

from habitat.config import read_write
from omegaconf import OmegaConf

# Disable some habitat-based warnings that pollute logs
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import habitat
from habitat.config.default import get_config
from habitat.config.default_structured_configs import DiscreteNavigationActionConfig
from habitat.core.simulator import Observations
from habitat.core.dataset import Episode
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalNavEpisode

from scripts.action_utils import extract_bin_values
from runtime_scripts.uniwm_schemas import UniWMInputBundle
from runtime_scripts.datasource_schemas import SourceFormatter, SourceAdapter, OutputBundle

EXPECTED_HABITAT_ACTIONS: Mapping[str, str] = {
    "stop": "stop",
    "forward": "move_forward",
    "backward": "move_backward",
    "strafe_left": "strafe_left",
    "strafe_right": "strafe_right",
    "left": "turn_left",
    "right": "turn_right"
}

MOVE_BACKWARD_ID = HabitatSimActions.extend_action_space("move_backward")
MOVE_LEFT_ID = HabitatSimActions.extend_action_space("move_left")
MOVE_RIGHT_ID = HabitatSimActions.extend_action_space("move_right")

@dataclass
class MoveBackwardActionConfig(DiscreteNavigationActionConfig):
    type: str = "MoveBackwardAction"

@dataclass
class StrafeLeftActionConfig(DiscreteNavigationActionConfig):
    type: str = "StrafeLeftAction"

@dataclass
class StrafeRightActionConfig(DiscreteNavigationActionConfig):
    type: str = "StrafeRightAction"

@registry.register_task_action
class MoveBackwardAction(SimulatorTaskAction):
    name: str = "move_backward"

    def step(self, *args: Any, **kwargs: Any) -> Observations:
        return self._sim.step(MOVE_BACKWARD_ID)

@registry.register_task_action
class StrafeLeftAction(SimulatorTaskAction):
    name: str = "strafe_left"

    def step(self, *args: Any, **kwargs: Any) -> Observations:
        return self._sim.step(MOVE_LEFT_ID)

@registry.register_task_action
class StrafeRightAction(SimulatorTaskAction):
    name: str = "strafe_right"

    def step(self, *args: Any, **kwargs: Any) -> Observations:
        return self._sim.step(MOVE_RIGHT_ID)


#-------------------- Habitat Source Tool Classes
@dataclass(frozen=True, kw_only=True)
class HabitatOutputBundle(OutputBundle):
    source_mode: str = "habitat"

    start_obs: Observations
    current_obs: Observations
    metrics: Mapping[str, Any]
    episode: InstanceImageGoalNavEpisode | Episode
    step_index: int
    action_taken: str | None
    forced_action_context: tuple[str, ...] | None = None
    is_collision: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

class HabitatEpisodeAdapter(SourceAdapter[HabitatOutputBundle]):
    """Thin Habitat environment adapter."""

    source_mode = "habitat"

    def __init__(
        self,
        config_path: str,
        split: str,
        data_path: str,
        scenes_dir: str,
        seed: int,
        bin_step: float = 0.01,
        episode_ids: list[str] | None = None,
        fixed_action_csv: str | None = None,
        extra_overrides: list[str] = [],
    ):
        self.reset_src()
        self.bin_step = float(bin_step)

        self.config = get_config(
            config_path=config_path,
            overrides=[
                f"habitat.seed={int(seed)}",
                f"habitat.dataset.split={split}",
                f"habitat.dataset.data_path={data_path}",
                f"habitat.dataset.scenes_dir={scenes_dir}",
                *extra_overrides,
            ],
        )
        
        registry.register_task_action(MoveBackwardAction, name="MoveBackwardAction")
        registry.register_task_action(StrafeLeftAction, name="StrafeLeftAction")
        registry.register_task_action(StrafeRightAction, name="StrafeRightAction")
        
        with read_write(self.config):
            actions = self.config.habitat.task.actions
            actions.move_backward = OmegaConf.structured(MoveBackwardActionConfig())
            actions.strafe_left = OmegaConf.structured(StrafeLeftActionConfig())
            actions.strafe_right = OmegaConf.structured(StrafeRightActionConfig())
            
        dataset = habitat.make_dataset(
            id_dataset=self.config.habitat.dataset.type,
            config=self.config.habitat.dataset,
        )
        
        if episode_ids is not None:
            episodes_by_id = {
                str(episode.episode_id): episode for episode in dataset.episodes
            }
            
            dataset.episodes = [
                episodes_by_id[str(episode_id)] for episode_id in episode_ids
            ]

        self.env = habitat.Env(config=self.config, dataset=dataset)
        assert self.env is not None

        if not isinstance(self.env.sim, HabitatSim):
            raise TypeError("[UNEXPECTED ERROR] Habitat environment simulator should be a HabitatSim type.")
            
        self.sim: HabitatSim = self.env.sim
        self._update_action_specs()
        
        self.fixed_actions_by_episode: dict[str, list[str]] | None = None

        if fixed_action_csv is not None:
            self.fixed_actions_by_episode = self._load_fixed_actions(
                fixed_action_csv
            )

    def reset_ep(self) -> list[HabitatOutputBundle]:
        obs: Observations = self.env.reset()
        self._update_action_specs()
        self.current_episode = self.env.current_episode
        self.step_index = 0
        self.start_obs = obs
        forced_context = None

        if self.fixed_actions_by_episode is not None:
            episode_id = str(self.current_episode.episode_id)

            if episode_id not in self.fixed_actions_by_episode:
                raise ValueError(
                    f"No fixed actions found for Habitat episode {episode_id}"
                )

            self.episode_fixed_actions = list(
                self.fixed_actions_by_episode[episode_id]
            )
            self.fixed_action_cursor = 0

            if not self.episode_fixed_actions:
                raise ValueError(
                    f"Fixed-action sequence for episode {episode_id} is empty"
                )

            forced_context = (
                "",
                *self.episode_fixed_actions,
            )

        return [
            self._pack_step(
                obs=obs,
                done=bool(self.env.episode_over),
                action_taken=None,
                is_collision=False,
                forced_action_context=forced_context,
            )
        ]
        
    def reset_src(self, data_id: str = "habitat") -> None:
        # TODO: Not high prio, but would like to have this start the episodes from the beginning again.
        self.current_episode: InstanceImageGoalNavEpisode | Episode | None = None
        self.step_index: int = 0
        self.last_step: HabitatOutputBundle | None = None
        self.start_obs: Observations = Observations({})
        self.goal_image: np.ndarray
        
        self.episode_fixed_actions: list[str] | None = None
        self.fixed_action_cursor = 0
        
        # TODO: The main thing left here is to find the function habitat uses to fully reset the env rather than step the episode.
        # self.current_episode = self.env.
        
    def step(self, actions: list[str]) -> list[HabitatOutputBundle]:
        step_results: list[HabitatOutputBundle] = []
        for action in actions:
            obs = self.env.step(action)
            is_collision = bool(self.sim.previous_step_collided)
            self.current_episode = self.env.current_episode
            self.step_index += 1
            done = bool(self.env.episode_over)

            step_results.append(self._pack_step(
                obs=obs,
                done=done,
                action_taken=action,
                is_collision=is_collision,
            ))
            
            if done or is_collision:
                break
        
        return step_results

    @property
    def action_space(self) -> object:
        return self.env.action_space

    @property
    def metrics(self) -> dict[str, object]:
        return dict(self.env.get_metrics())

    @property
    def habitat_config(self) -> Any:
        return self.config

    def close(self) -> None:
        self.env.close()

        self.current_episode = None
        self.last_step = None
        self.start_obs =  Observations({})
        self.step_index = 0
        
    def _update_action_specs(self) -> None:
        agent_id = self.sim.habitat_config.default_agent_id
        linear_spec = ActuationSpec(amount=self.bin_step)
        angular_spec = ActuationSpec(amount=math.degrees(self.bin_step))
        action_specs = {
            HabitatSimActions.move_forward: ActionSpec("move_forward", linear_spec),
            MOVE_BACKWARD_ID: ActionSpec("move_backward", linear_spec),
            MOVE_LEFT_ID: ActionSpec("move_left", linear_spec),
            MOVE_RIGHT_ID: ActionSpec("move_right", linear_spec),
            HabitatSimActions.turn_left: ActionSpec("turn_left", angular_spec),
            HabitatSimActions.turn_right: ActionSpec("turn_right", angular_spec),
        }

        self.sim.sim_config.agents[agent_id].action_space.update(action_specs)
        self.sim.get_agent(agent_id).agent_config.action_space.update(action_specs)
        
    @staticmethod
    def _load_fixed_actions(csv_path: str) -> dict[str, list[str]]:
        root_dir = Path(__file__).resolve().parent.parent
        path = Path(csv_path)

        if not path.is_absolute():
            path = root_dir / path

        with path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))

        required_columns = {"episode_id", "step_idx", "action"}
        if not rows:
            raise ValueError(f"Fixed-action CSV is empty: {path}")

        missing = required_columns - set(rows[0])
        if missing:
            raise ValueError(
                f"Fixed-action CSV {path} is missing columns: {sorted(missing)}"
            )

        indexed_rows: dict[str, list[tuple[int, str]]] = {}

        for row in rows:
            episode_id = str(row["episode_id"])
            step_idx = int(row["step_idx"])
            action = row["action"].strip()

            if not action:
                raise ValueError(
                    f"Empty action for episode {episode_id}, step {step_idx}"
                )

            indexed_rows.setdefault(episode_id, []).append(
                (step_idx, action)
            )

        actions_by_episode: dict[str, list[str]] = {}

        for episode_id, episode_rows in indexed_rows.items():
            episode_rows.sort(key=lambda item: item[0])

            step_indices = [step for step, _ in episode_rows]
            expected = list(range(len(step_indices)))

            if step_indices != expected:
                raise ValueError(
                    f"Non-contiguous fixed actions for episode {episode_id}: "
                    f"expected {expected}, got {step_indices}"
                )

            actions_by_episode[episode_id] = [
                action for _, action in episode_rows
            ]

        return actions_by_episode

    def _pack_step(
        self,
        *,
        obs: Observations,
        done: bool,
        action_taken: str | None,
        is_collision: bool,
        forced_action_context: tuple[str, ...] | None = None,
    ) -> HabitatOutputBundle:
        episode = self.current_episode
        if episode is None:
            raise AssertionError("Expected habitat.Env.current_episode to be set after reset/step.")

        step = HabitatOutputBundle(
            episode_id=episode.episode_id,
            done=done,
            start_obs=self.start_obs,
            current_obs=obs,
            metrics=self.metrics,
            episode=episode,
            step_index=self.step_index,
            action_taken=action_taken,
            forced_action_context=forced_action_context,
            is_collision=is_collision,
        )

        self.last_step = step
        return step


class HabitatUniWMFormatter(SourceFormatter[HabitatOutputBundle]):
    """Strict converter between Habitat output bundles and UniWM input bundles."""

    source_mode = "habitat"

    GOAL_KEY: str = "instance_imagegoal"
    IMAGE_MODE = "RGB"
    RGB_KEY: str = "rgb"
    START_POS_IDX: tuple[int, int] = (0, 2)
    START_POSE_TEMPLATE: str = "Starting Point Coordinate: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}\n"

    # TODO: bin_step should be set dynamically based on a universal config, not just a default which happens to match up to everything else.
    def __init__(self, bin_step: float, img_size: int = 448):
        self.image_size: tuple[int, int] = (int(img_size), int(img_size))

    def convert_action(self, action: str) -> list[str]:
        is_stop = action.strip().lower() == "stop"

        if is_stop:
            return [EXPECTED_HABITAT_ACTIONS["stop"]]
        
        dx_steps, dy_steps, dyaw_steps = self._extract_action_steps(action)

        all_actions: list[str] = []
        
        if dx_steps:
            action = "forward" if dx_steps > 0 else "backward"
            all_actions.extend([EXPECTED_HABITAT_ACTIONS[action]] * abs(dx_steps))

        if dy_steps:
            action = "strafe_left" if dy_steps > 0 else "strafe_right"
            all_actions.extend([EXPECTED_HABITAT_ACTIONS[action]] * abs(dy_steps))

        if dyaw_steps:
            action = "left" if dyaw_steps > 0 else "right"
            all_actions.extend([EXPECTED_HABITAT_ACTIONS[action]] * abs(dyaw_steps))

        return all_actions

    def convert_from_source(
        self,
        outputs: list[HabitatOutputBundle]
    ) -> UniWMInputBundle:
        output = outputs[-1]
        start_rgb = output.start_obs[self.RGB_KEY]
        goal_image = output.start_obs[self.GOAL_KEY]
        current_rgb = output.current_obs[self.RGB_KEY]
        bundle_metadata: dict[str, Any] = dict(output.metadata)

        bundle_metadata.update({
            "step_index": f"[{outputs[0].step_index}->{outputs[-1].step_index}]",
            "source_mode": output.source_mode,
            "action_taken": ', '.join(['<no action>' if o.action_taken is None else o.action_taken for o in outputs]),
            "metrics": dict(output.metrics)
        })
        
        
        had_collision = any(o.is_collision for o in outputs)

        return UniWMInputBundle(
            start_observation=self._to_pil_image(start_rgb),
            goal_observation=self._to_pil_image(goal_image),
            current_observation=self._to_pil_image(current_rgb),
            start_pose_str=self.extract_start_pose(output.episode),
            collision=had_collision,
            source_done=any(o.done for o in outputs),
            metadata=bundle_metadata,
        )
        
    def _extract_action_steps(self, action: str) -> tuple[int, int, int]:
        match = re.compile(
            r"^Move by dx: <dx_(?P<dx_sign>pos|neg)_bin_(?P<dx_steps>\d+)>, "
            r"dy: <dy_(?P<dy_sign>pos|neg)_bin_(?P<dy_steps>\d+)>, "
            r"dyaw: <dyaw_(?P<dyaw_sign>pos|neg)_bin_(?P<dyaw_steps>\d+)>$"
        ).fullmatch(action.strip())
        
        if match is None:
            raise ValueError(f"Invalid canonical action: {action!r}")

        def signed_steps(axis: str) -> int:
            steps = int(match[f"{axis}_steps"])
            return steps if match[f"{axis}_sign"] == "pos" else -steps

        return (
            signed_steps("dx"),
            signed_steps("dy"),
            signed_steps("dyaw"),
        )


    def extract_start_pose(self, episode: InstanceImageGoalNavEpisode | Episode) -> str:
        position: list[float] = episode.start_position
        rotation: list[float] = episode.start_rotation

        return self.START_POSE_TEMPLATE.format(
            x=float(position[0]),
            y=float(position[2]),
            yaw=self.yaw_from_rotation(rotation),
        )

    @classmethod
    def yaw_from_rotation(cls, rotation: list[float]) -> float:
        if len(rotation) != 4:
            raise AssertionError(
                f"Expected start_rotation as a quaternion with 4 values, got {rotation}"
            )

        x, y, z, w = rotation
        return float(math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))

    def _to_pil_image(self, image: Image.Image | torch.Tensor | np.ndarray) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert(self.IMAGE_MODE).resize(self.image_size)

        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.dim() != 3:
                raise AssertionError(f"Expected image tensor with 3 dims, got shape {tuple(tensor.shape)}")
            if tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            image = tensor.numpy()

        array: np.ndarray = image
        if array.ndim != 3:
            raise AssertionError(f"Expected image array with 3 dims, got shape {array.shape}")
        if array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            max_value = float(array.max()) if array.size else 0.0
            if max_value <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return Image.fromarray(array, mode=self.IMAGE_MODE).resize(self.image_size)
