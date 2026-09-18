"""Generate one seeded Habitat exploration sequence inside a pipeline run.

Public API: generate_action_sequence(episode_id, target_steps, run_dir, seed=...).
Returns a repository-relative POSIX filename only after writing a complete JSON
sequence. No model loading, pipeline changes, cross-run cache, or goal-seeking.
Uses the repository's Habitat 0.2.5 adapter and checkpoint action vocabulary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FORWARD_BINS = 25
TURN_BINS = 17
CANDIDATES = 8
MIN_WAYPOINT_DISTANCE_M = 1.0


def extend_action(pending, action, forward_limit, turn_limit):
    """Pack forward then same-direction turns without changing execution order."""
    forward, turn = pending
    if action == "move_forward":
        return (forward + FORWARD_BINS, turn) if turn == 0 and forward + FORWARD_BINS <= forward_limit else None
    delta = {"turn_left": TURN_BINS, "turn_right": -TURN_BINS}[action]
    if turn * delta < 0 or abs(turn + delta) > turn_limit:
        return None
    return forward, turn + delta


def action_text(pending):
    forward, turn = pending
    return (f"Move by dx: <dx_pos_bin_{forward:02d}>, dy: <dy_pos_bin_00>, "
            f"dyaw: <dyaw_{'neg' if turn < 0 else 'pos'}_bin_{abs(turn):02d}>")


def collect_sequence(adapter, follower_factory, action_names, stop_action,
                     follower_error, target_steps, forward_limit, turn_limit,
                     max_seconds, max_waypoints):
    """Trusted internal adapter interface; count emitted actions while moving."""
    deadline = time.monotonic() + max_seconds
    visited = [list(map(float, adapter.sim.get_agent_state().position))]
    actions, collisions, waypoints = [], [], []
    pending = (0, 0)
    failed_legs = 0

    def flush(collided=False):
        nonlocal pending
        if pending != (0, 0):
            actions.append(action_text(pending))
            collisions.append(collided)
            pending = (0, 0)

    for _ in range(max_waypoints):
        if time.monotonic() >= deadline:
            break
        current = visited[-1]
        candidates = []
        for _ in range(CANDIDATES):
            point = list(map(float, adapter.sim.pathfinder.get_random_navigable_point()))
            if not all(math.isfinite(x) for x in point):
                continue
            distance = float(adapter.sim.geodesic_distance(current, point))
            if math.isfinite(distance) and distance >= MIN_WAYPOINT_DISTANCE_M:
                novelty = min(math.dist(point, previous) for previous in visited)
                candidates.append((novelty, point))
        if not candidates:
            continue
        _, target = max(candidates, key=lambda item: item[0])
        leg = {"position": target, "start_transition": len(actions), "status": "following"}
        waypoints.append(leg)
        follower = follower_factory()
        while time.monotonic() < deadline:
            try:
                choice = follower.get_next_action(target)
            except follower_error:
                failed_legs += 1
                leg["status"] = "follower_error"
                break
            if choice == stop_action:
                leg["status"] = "reached"
                break
            name = action_names[choice]
            updated = extend_action(pending, name, forward_limit, turn_limit)
            if updated is None:
                flush()
                if len(actions) == target_steps:
                    leg["status"] = "step_budget_reached"
                    return actions, collisions, waypoints, failed_legs
                updated = extend_action(pending, name, forward_limit, turn_limit)
                assert updated is not None
            pending = updated
            result = adapter.step([name])[-1]
            visited.append(list(map(float, adapter.sim.get_agent_state().position)))
            if result.is_collision:
                flush(collided=True)  # The normal adapter stops a group at collision.
                leg["status"] = "collision"
                break
            if result.done:
                raise RuntimeError("Habitat ended the episode before the requested sequence was generated")
        else:
            leg["status"] = "time_limit"
        flush()
        if len(actions) == target_steps:
            return actions, collisions, waypoints, failed_legs
    raise RuntimeError(
        f"Generated only {len(actions)}/{target_steps} transitions after "
        f"{len(waypoints)} waypoints ({failed_legs} follower failures). "
        "No action-sequence file written; inspect the episode or increase generation limits."
    )


def generate_action_sequence(
    episode_id: str,
    target_steps: int,
    run_dir: str | Path,
    *,
    seed: int = 100,
    config_path: str | Path = "cfg/habitat_uniwm_cfg_no_learning.yaml",
    checkpoint: str | Path = "checkpoints/base_ckpt",
    max_seconds: float = 120,
    max_waypoints: int = 64,
) -> str:
    """Return a complete sequence's path relative to REPO_ROOT, using '/' separators.

    run_dir may be absolute or repo-relative, and must resolve inside the repo.
    Budget counts total converted transitions, including any collision steps.
    Generation uses moderate simulator increments; this function does not certify
    physical equivalence to small-step replay on every new trajectory.
    """
    episode_id = str(episode_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", episode_id):
        raise ValueError("episode_id must contain only letters, digits, '_' or '-'")
    for name, value in (("target_steps", target_steps), ("max_waypoints", max_waypoints)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be positive and finite")
    directory = (REPO_ROOT / run_dir / "habitat_action_sequences").resolve()
    if not directory.is_relative_to(REPO_ROOT):
        raise ValueError("run_dir must resolve inside the repository")
    destination = (directory / f"episode_{episode_id}_seed_{seed}_steps_{target_steps}.json").resolve()
    if not destination.is_relative_to(directory):
        raise ValueError("Action-sequence filename must resolve inside the run's sequence directory")

    from omegaconf import OmegaConf
    from scripts.action_utils import ActionTokenVocabulary
    from source_tools.habitat_source_tools import HabitatEpisodeAdapter
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    import habitat_sim

    config = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / config_path), resolve=True)
    params = dict(config["runner"]["adapter_params"])
    vocab = ActionTokenVocabulary.from_checkpoint(REPO_ROOT / checkpoint)
    if vocab.bin_step != float(params["bin_step"]):
        raise ValueError("Checkpoint and Habitat config bin_step must agree")
    if (vocab.axes["dx"]["max_bin"] < FORWARD_BINS
            or vocab.axes["dyaw"]["max_bin"] < TURN_BINS
            or not vocab.axes["dyaw"]["allow_negative"]):
        raise ValueError("Checkpoint must support 25 forward bins and +/-17 turn bins")
    params.update(episode_ids=[episode_id], fixed_action_run_dir=None,
                  fixed_action_files_dir=None, seed=seed)
    for key in ("data_path", "scenes_dir"):
        params[key] = str((REPO_ROOT / params[key]).resolve())
    params["extra_overrides"] = [x for x in params.get("extra_overrides", []) if not x.startswith((
        "habitat.environment.max_episode_steps=", "habitat.environment.iterator_options.shuffle="
    ))] + ["habitat.environment.max_episode_steps=0", "habitat.environment.iterator_options.shuffle=false"]
    # Stable across processes and episode ordering; Python's hash() is not stable.
    waypoint_seed = int.from_bytes(hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()[:4], "big")
    started = time.monotonic()
    adapter = HabitatEpisodeAdapter(**params)
    try:
        adapter.reset_ep()
        adapter.sim.pathfinder.seed(waypoint_seed)
        agent_id = adapter.sim.habitat_config.default_agent_id
        for space in (adapter.sim.sim_config.agents[agent_id].action_space,
                      adapter.sim.get_agent(agent_id).agent_config.action_space):
            space[HabitatSimActions.move_forward].actuation.amount = FORWARD_BINS * vocab.bin_step
            for action in (HabitatSimActions.turn_left, HabitatSimActions.turn_right):
                space[action].actuation.amount = math.degrees(TURN_BINS * vocab.bin_step)
        actions, collisions, waypoints, failed_legs = collect_sequence(
            adapter,
            lambda: ShortestPathFollower(adapter.sim, 0.2, return_one_hot=False, stop_on_error=False),
            {HabitatSimActions.move_forward: "move_forward", HabitatSimActions.turn_left: "turn_left",
             HabitatSimActions.turn_right: "turn_right"}, HabitatSimActions.stop,
            habitat_sim.errors.GreedyFollowerError, target_steps,
            vocab.axes["dx"]["max_bin"], vocab.axes["dyaw"]["max_bin"], max_seconds, max_waypoints,
        )
        payload = {
            "schema_version": 1, "episode_id": episode_id,
            "scene_id": adapter.current_episode.scene_id,
            "seed": seed, "waypoint_seed": waypoint_seed, "target_steps": target_steps,
            "actions": actions, "collision_steps": [i for i, hit in enumerate(collisions) if hit],
            "waypoints": waypoints, "follower_failures": failed_legs,
            "generation_seconds": time.monotonic() - started,
            "bin_step": vocab.bin_step, "forward_bins": FORWARD_BINS, "turn_bins": TURN_BINS,
            "grouping": "forward then same-direction turn; split at vocabulary limits and waypoint/collision boundaries",
        }
    finally:
        adapter.close()
    assert len(actions) == target_steps
    directory.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination.relative_to(REPO_ROOT).as_posix()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--max-waypoints", type=int, default=64)
    args = parser.parse_args()
    print(generate_action_sequence(args.episode, args.target_steps, args.run_dir,
                                   seed=args.seed, max_seconds=args.max_seconds,
                                   max_waypoints=args.max_waypoints))
