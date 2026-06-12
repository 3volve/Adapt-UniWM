from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.action_utils import action_to_text, calculate_action_delta, extract_bin_values
from runtime_scripts.uniwm_schemas import UniWMInputBundle
from runtime_scripts.datasource_schemas import SourceFormatter, SourceAdapter, ReplayOutputBundle


class ReplayEpisodeAdapter(SourceAdapter):
    """Simple replay adapter for manifest-selected offline trajectories."""

    source_mode = "replay"

    def __init__(
        self,
        data_root: str = "eval_data",
        manifest_path: str = "cfg/eval_dataset_manifest.json",
    ):
        root_dir = Path(__file__).resolve().parent.parent
        self.data_root = root_dir / data_root
        self.manifest_path = root_dir / manifest_path
        self.manifest = json.load(self.manifest_path.open("r", encoding="utf-8"))
        
        self.data_id = "replay"
        self.episode_cursor = 0
        self.current_episode: dict[str, Any] | None = None
        self.current_traj: dict[str, Any] | None = None
        self.image_paths: list[Path] = []
        self.images: list[Image.Image] = []
        self.states_xy_yaw: list[list[float]] = []
        self.actions: list[list[float]] = []
        self.step_index = 0
        self.last_step: ReplayOutputBundle | None = None

    def reset_ep(self) -> ReplayOutputBundle:
        self.current_traj_dir = self.traj_dirs[self.episode_cursor]
        self.current_episode_id = self.manifest[self.data_id]["episodes"][self.episode_cursor]
        self.episode_cursor += 1

        traj_dir = self.current_traj_dir
        self.current_traj = pickle.load((traj_dir / "traj_data.pkl").open("rb"))
        self.image_paths = sorted(
            [p for p in traj_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]],
            key=lambda p: int(p.stem),
        )
        
        self.images = [Image.open(path).convert("RGB") for path in self.image_paths]
        self.states_xy_yaw = self._make_states_xy_yaw()
        self.actions = self._make_actions()
        self.step_index = 0

        return self._pack_step(action_taken=None, done=False)
    
    def reset_src(self, data_id: str):
        self.data_id = data_id
        self.traj_dirs: list[Path] = []
        for episode_id in self.manifest[data_id]["episodes"]:
            self.traj_dirs.append(Path(f"{self.data_root}/{data_id}/{episode_id}"))
            
        self.episode_cursor = 0
        self.current_episode: dict[str, Any] | None = None
        self.current_traj: dict[str, Any] | None = None
        self.image_paths: list[Path] = []
        self.images: list[Image.Image] = []
        self.states_xy_yaw: list[list[float]] = []
        self.actions: list[list[float]] = []
        self.step_index = 0
        self.last_step: ReplayOutputBundle | None = None

    def step(self, action: str) -> ReplayOutputBundle:
        next_action = self.actions[self.step_index]
        self.step_index += 1
        done = self.step_index >= len(self.images) - 1
        return self._pack_step(action_taken=next_action, done=done)

    def close(self) -> None:
        self.current_episode = None
        self.current_traj = None
        self.image_paths = []
        self.images = []
        self.states_xy_yaw = []
        self.actions = []
        self.step_index = 0
        self.last_step = None

    def _make_states_xy_yaw(self) -> list[list[float]]:
        if self.current_traj is None:
            raise AssertionError("Attempted to retrieve state from a non-initialized trajectory.")
        
        positions = self.current_traj["position"]
        yaws = self.current_traj["yaw"]

        if hasattr(positions, "tolist"):
            positions = positions.tolist()
        if hasattr(yaws, "reshape"):
            yaws = yaws.reshape(-1).tolist()
        elif hasattr(yaws, "tolist"):
            yaws = yaws.tolist()

        return [
            [float(position[0]), float(position[1]), float(yaws[idx])]
            for idx, position in enumerate(positions)
        ]

    def _make_actions(self) -> list[list[float]]:
        if self.current_traj is None:
            raise AssertionError("Attempted to generate actions from a non-initialized trajectory.")
        
        if "delta" in self.current_traj:
            delta_actions = self.current_traj["delta"]
            if hasattr(delta_actions, "tolist"):
                delta_actions = delta_actions.tolist()
            return [[float(value) for value in action] for action in delta_actions]

        numeric_actions = [
            calculate_action_delta(self.states_xy_yaw[idx], self.states_xy_yaw[idx + 1])
            for idx in range(len(self.states_xy_yaw) - 1)
        ]
        return numeric_actions

    def _pack_step(
        self,
        action_taken: list[float] | None,
        done: bool,
    ) -> ReplayOutputBundle:
        step = ReplayOutputBundle(
            episode_id=self.current_episode_id,
            done=done,
            start_observation=self.images[0],
            goal_observation=self.images[-1],
            current_observation=self.images[self.step_index],
            start_pose=self.states_xy_yaw[0],
            step_index=self.step_index,
            action_taken=action_taken,
            metadata={
                "data_id": self.data_id,
                "source_mode": self.source_mode,
                "trajectory_path": str(self.current_traj_dir),
                "adapter_step_idx": self.step_index,
                "done": done,
            },
        )

        self.last_step = step
        return step


class ReplayUniWMFormatter(SourceFormatter):
    """Strict converter between Habitat output bundles and UniWM input bundles."""

    source_mode = "replay"

    def __init__(self, adapter: ReplayEpisodeAdapter | None = None, bin_step: float = 0.01, image_h: int = 256, image_w: int = 256):
        self.bin_step: float = float(bin_step)
        self.image_size: tuple[int, int] = (int(image_w), int(image_h))

    def convert_action(self, action: str) -> list[str]:
        action_text = action.strip()
        extract_bin_values(action_text, "dx", self.bin_step)
        extract_bin_values(action_text, "dy", self.bin_step)
        extract_bin_values(action_text, "dyaw", self.bin_step)
        return [action_text]

    def convert_observation(
        self,
        output: ReplayOutputBundle
    ) -> UniWMInputBundle:
        bundle_metadata: dict[str, object] = dict(output.metadata)
        bundle_metadata.update({
            "done": output.done,
            "step_index": output.step_index,
            "source_mode": output.source_mode,
            "action_taken": output.action_taken,
        })
        
        pose_str = f"Starting Point Coordinate: x={output.start_pose[0]:.3f}, y={output.start_pose[1]:.3f}, yaw={output.start_pose[2]:.3f}\n"
        action_text = output.action_taken if output.action_taken is None else action_to_text(output.action_taken)

        return UniWMInputBundle(
            start_observation=output.start_observation.resize(self.image_size),
            goal_observation=output.goal_observation.resize(self.image_size),
            current_observation=output.current_observation.resize(self.image_size),
            start_pose_str=pose_str,
            action_text=action_text,
            metadata=bundle_metadata,
        )
