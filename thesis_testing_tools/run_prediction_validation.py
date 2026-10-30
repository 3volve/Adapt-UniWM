"""Small development-only prediction validation using the normal episode runner.

Habitat example (seeded fixed actions are generated once per run):
  python -m thesis_testing_tools.run_prediction_validation --config cfg/habitat_uniwm_cfg.yaml \
    --episode 827 --episode 352 --steps 25 --output output/validation_827

Replay example (supply the existing online-training settings separately):
  python -m thesis_testing_tools.run_prediction_validation --config cfg/replay_uniwm_cfg.yaml \
    --training-config cfg/habitat_uniwm_cfg.yaml --episode sacson/TRAJECTORY \
    --steps 8 --output output/validation_source

--prepare-only generates Habitat actions and writes configs/commands without loading
the model (Habitat and scene data are still required unless --actions is supplied).
--report-only regenerates the report from existing condition logs. Run separately
for source and Habitat. Select examples with visible translation/rotation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import replace
import html
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("frozen_off", "frozen_on", "adapted_off", "adapted_on")


def capture(command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def select_workload(config, episodes, development, heldout):
    """Validate user-selected IDs against the repository's development/held-out lists."""
    source = config["runner"]["source_type"]
    selected = {}
    for value in episodes:
        if source == "habitat":
            dataset, episode = "habitat", value
        else:
            if "/" not in value:
                raise ValueError("Replay --episode must be DATASET/TRAJECTORY")
            dataset, episode = value.split("/", 1)
        if episode not in development.get(dataset, {}).get("test", []):
            raise ValueError(f"{value}: not in the development manifest's test list")
        if episode in heldout.get(dataset, {}).get("test", []):
            raise ValueError(f"{value}: overlaps held-out evaluation data")
        if episode in selected.setdefault(dataset, []):
            raise ValueError(f"Duplicate selected episode: {value}")
        selected[dataset].append(episode)
    return selected


def condition_config(base, condition, steps, selected, manifest, actions):
    cfg = deepcopy(base)
    adapted, memory = condition.startswith("adapted"), condition.endswith("on")
    engine, wrapper, runner = cfg["engine"], cfg["wrapper"], cfg["runner"]
    if adapted and engine["training"] is False:
        raise ValueError("Adapted conditions require a training mapping; supply --training-config")
    engine["generation"]["diagnostics"] = True
    engine["load_model_args"]["use_memory_bank_inference"] = memory
    if engine["training"] is not False:
        engine["training"]["diagnostics"] = True
    wrapper.update(max_route_steps=1, training_enabled=adapted,
                   log_real_obs=True, log_predicted_obs=True,
                   add_global_memories=memory, learning_rate_schedule=False,
                   eval_forced_action=runner["source_type"] == "replay")
    runner.update(max_episode_steps=steps, max_total_episodes=sum(map(len, selected.values())),
                  save_model_weights=False)
    if runner["source_type"] == "habitat":
        runner["adapter_params"].update(episode_ids=selected["habitat"], fixed_action_files_dir=str(actions))
    else:
        runner["adapter_params"].update(manifest_path=str(manifest), manifest_split="test")
    return cfg


def generate_validation_actions(config, config_path, episodes, output, steps, seed):
    """Use the pipeline generator once per episode, before starting any condition."""
    from scripts.generate_habitat_action_sequence import generate_action_sequence
    for episode in episodes:
        generate_action_sequence(episode, steps, output, seed=seed,
            config_path=config_path, checkpoint=config["engine"]["load_model_args"]["model_ckpt"])
    return output / "habitat_action_sequences"


def make_probe_engine(base, output, turn_bins, probe_steps=(0,)):
    """Probe selected zero-based steps of the tester's single-step routes."""
    from runtime_scripts.runtime_utils import save_img
    from scripts.action_utils import action_to_text
    from thesis_testing_tools.evaluate_memory_refresh import rng_state, restore_rng

    class ProbeEngine(base):
        def reset_episode(self, episode_id, bundle):
            self._probe_step = 0
            self._probe_episode = episode_id
            super().reset_episode(episode_id, bundle)

        def predict_route(self, bundle, max_steps):
            assert max_steps == 1, "Validation probe steps require single-step routes"
            step = self._probe_step
            self._probe_step += 1
            route = super().predict_route(bundle, max_steps)
            if step not in probe_steps or not route.steps or route.steps[0].visualization is None:
                return route
            state = rng_state()
            bin_step = self.action_vocabulary.bin_step
            actions = {"original": route.steps[0].action_text,
                       "zero": action_to_text([0, 0, 0], bin_step),
                       "turn": action_to_text([0, 0, turn_bins * bin_step], bin_step)}
            token = f"<dyaw_pos_bin_{turn_bins:02d}>"
            if token not in self.action_vocabulary.tokens_by_axis["dyaw"]:
                raise ValueError(f"Probe turn token {token} is absent from the checkpoint vocabulary")
            try:
                for name, action in actions.items():
                    restore_rng(state)
                    prediction = self.eval_step(replace(bundle, action_text=action))
                    generation_id = prediction.logging_info["generation_id"]
                    path = output / "probes" / f"{generation_id}_{name}.png"
                    if prediction.visualization is not None:
                        save_img(prediction.visualization, str(path))
                    self.event_logger.feed({"validation_probes": {str(generation_id): {
                        "name": name, "action": action, "input_source": bundle.input_source,
                        "episode_id": self._probe_episode, "step": step,
                        "completed_updates": self._diagnostic_update_count,
                        "memory_count": self.memory_count,
                        "memory_used": prediction.viz_used_memory,
                        "prediction_path": str(path) if prediction.visualization is not None else None,
                        "reference_generation_id": route.steps[0].logging_info["generation_id"],
                    }}})
            finally:
                restore_rng(state)
            return route
    return ProbeEngine


def worker(args):
    import numpy as np
    import torch
    from runtime_scripts.event_logger import EventLogger
    from runtime_scripts.runtime_engine import UniWMEngine
    from runtime_scripts.runtime_utils import load_config, event_values
    from runtime_scripts.run_metadata import runtime_metadata
    from uniwm_episode_runner import UniWMEpisodeRunner
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_config(str(args.config))
    source = cfg["runner"]["source_type"]
    with EventLogger(args.output / "events.jsonl") as log:
        engine_cls = make_probe_engine(UniWMEngine, args.output, args.turn_bins, args.probe_steps)
        engine = engine_cls(source, str(args.config), event_logger=log)
        runner = UniWMEpisodeRunner(str(args.config), source, args.output, event_logger=log, engine=engine)
        log.feed({"runtime": event_values(runtime_metadata(torch, seed=args.seed, engine=engine)), "outcome": "completed"})
        try:
            runner.run_episodes()
        finally:
            runner.adapter.close()
        log.finish({"outcome": "completed", "episodes": runner.episode_index})


def report(output, metrics):
    import numpy as np
    from PIL import Image
    from thesis_testing_tools.generate_metrics import ImageCalculator
    calculator = ImageCalculator(metrics)
    def array(path):
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    def image_tag(path):
        if not path:
            return "unavailable"
        relative = os.path.relpath(path, output).replace(os.sep, "/")
        return f'<img loading="lazy" src="{html.escape(relative, quote=True)}">'
    rows, sections, probes = [], [], []
    matched, run_status = {}, {}
    for condition in CONDITIONS:
        log = output / condition / "events.jsonl"
        if not log.exists():
            continue
        events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        run_status[condition] = {
            "outcome": events[-1].get("outcome") if events and events[-1].get("_event", {}).get("kind") == "run_summary" else "incomplete",
            "episode_ends": [end for event in events for end in event.get("episode_end", {}).values()],
        }
        generations = {}
        for event in events:
            generations.update(event.get("diagnostics", {}).get("generations", {}))
            probes.extend((condition, p) for p in event.get("validation_probes", {}).values())
        sections.append(f"<h2>{condition}</h2>")
        for event in events:
            if event.get("transition", {}).get("outcome") != "completed":
                continue
            wrapper = event.get("wrapper", {})
            prediction = wrapper.get("evaluation", {}) if event["source_mode"] == "replay" else event.get("prediction", {})
            generation_id = prediction.get("generation_id")
            generated = generations.get(str(generation_id), {})
            input_path = generated.get("input_source")
            pred_path = prediction.get("predicted_obs_path") if event["source_mode"] == "replay" else wrapper.get("predicted_obs_path")
            target_path = wrapper.get("real_obs_path")
            row = {"condition": condition, "data_id": event["data_id"], "episode": event["episode_id"],
                   "step": event["step_idx"], "generation_id": generation_id,
                   "action": prediction.get("forced_action", prediction.get("action")),
                   "completed_updates": generated.get("completed_updates"),
                   "memory_used": generated.get("use_memory_bank"),
                   "global_memory_used": generated.get("use_global_memory_bank"),
                   "input": input_path, "prediction": pred_path, "target": target_path,
                   "teacher_forced_loss": event.get("training", {}).get("visual_loss"),
                   "applied_lr": event.get("training", {}).get("applied_learning_rate"),
                   "grad_norm": event.get("training", {}).get("grad_norm_before_clip")}
            if not isinstance(input_path, str) or not pred_path or not target_path:
                row["unscored_reason"] = "Missing input reference, prediction, or target; inspect events.jsonl"
            else:
                current, predicted, target = map(array, (input_path, pred_path, target_path))
                if current.shape != target.shape or predicted.shape != target.shape:
                    raise ValueError(f"{condition}/{row['episode']}/{row['step']}: image shapes differ")
                row.update(calculator(predicted, target))
                row.update({"copy_" + key: value for key, value in calculator(current, target).items()})
                row["prediction_spatial_std"] = float(predicted.std(axis=(0, 1), dtype=np.float64).mean())
                row["target_spatial_std"] = float(target.std(axis=(0, 1), dtype=np.float64).mean())
                key = (row["data_id"], row["episode"], row["step"])
                from hashlib import sha256
                signature = (row["action"], sha256(current.tobytes()).hexdigest(), sha256(target.tobytes()).hexdigest())
                if key in matched and matched[key] != signature:
                    raise ValueError(f"Unmatched action/input/target across conditions at {key}")
                matched[key] = signature
            rows.append(row)
            sections.append(f"<h3>{html.escape(str(row['episode']))}, step {row['step']}</h3>"
                            + "<div>Input / prediction / actual next frame</div>"
                            + image_tag(input_path if isinstance(input_path, str) else None)
                            + image_tag(pred_path) + image_tag(target_path)
                            + "<pre>" + html.escape(json.dumps(row, indent=2)) + "</pre>")
    if not rows:
        raise ValueError("No completed transitions available to report")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "transitions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for condition in run_status:
        group = [row for row in rows if row["condition"] == condition]
        summary[condition] = {"transitions": len(group), "scored": sum("unscored_reason" not in row for row in group)}
        for metric in [*metrics, *("copy_" + m for m in metrics)]:
            values = [r[metric] for r in group if metric in r]
            summary[condition][metric] = sum(values) / len(values) if values else None
    for condition, probe in probes:
        sections.append(f"<h3>Action probe: {condition} / {html.escape(probe['name'])}</h3>"
                        + image_tag(probe["prediction_path"]) + "<pre>" + html.escape(json.dumps(probe, indent=2)) + "</pre>")
    review = ("Review pending: visible motion and action direction/units; checkpoint/tokenizer/preprocessing compatibility; "
              "token decoding and target shift/masking; original UniWM comparison where possible; "
              "first collapse and its before/after evidence. Spatial standard deviation is a review aid, not a collapse label. "
              "Action probes are not scored against the original-action target. Teacher-forced loss is not generation quality. "
              "Conditions may terminate early; check transition counts before comparing means.")
    (output / "summary.json").write_text(json.dumps({"conditions": summary, "run_status": run_status, "metrics": calculator.provenance,
        "review_pending": review}, indent=2), encoding="utf-8")
    (output / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Prediction validation</title>'
        '<style>body{font:16px sans-serif;margin:30px}img{width:30%;max-width:360px}pre{white-space:pre-wrap}</style>'
        '<h1>Small development validation</h1><p>' + review + '</p><pre>'
        + html.escape(json.dumps({"conditions": summary, "run_status": run_status}, indent=2)) + '</pre>' + ''.join(sections), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--actions", type=Path, help="Optional existing Habitat actions; otherwise generate once per episode")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--turn-bins", type=int, default=20, help="Positive yaw probe magnitude in action bins")
    parser.add_argument("--probe-steps", nargs="+", type=int, default=[0],
                        help="Zero-based prediction steps to probe in each episode (default: 0)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--metrics", nargs="+", choices=("mae", "ssim", "lpips"), default=["mae", "ssim", "lpips"])
    args = parser.parse_args()
    args.conditions = list(dict.fromkeys(args.conditions))
    if any(step < 0 for step in args.probe_steps):
        parser.error("--probe-steps must be nonnegative")
    args.output = args.output.resolve()
    os.chdir(ROOT)
    if args.worker:
        worker(args); return
    if args.report_only:
        report(args.output, args.metrics); return
    if args.config is None or not args.episode or args.steps < 1 or args.turn_bins < 1:
        parser.error("Provide --config, at least one --episode, and positive --steps/--turn-bins")
    if any(step >= args.steps for step in args.probe_steps):
        parser.error("--probe-steps must be smaller than --steps")
    import yaml
    from thesis_testing_tools.run_manifest import RunManifest, artifact_fingerprint
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.training_config:
        cfg["engine"]["training"] = yaml.safe_load(args.training_config.read_text(encoding="utf-8"))["engine"]["training"]
    development_path, heldout_path = ROOT / "cfg/eval_dataset_manifest.json", ROOT / "cfg/thesis_heldout_manifest.json"
    selected = select_workload(cfg, args.episode, json.loads(development_path.read_text()), json.loads(heldout_path.read_text()))
    if args.actions and cfg["runner"]["source_type"] != "habitat":
        parser.error("--actions applies only to Habitat runs")
    if args.actions:
        args.actions = args.actions.resolve()
        if not args.actions.is_dir():
            parser.error("--actions directory does not exist")
    args.output.mkdir(parents=True, exist_ok=False)
    with RunManifest(args.output, repo_root=ROOT, run_type="prediction_validation",
                     metadata={"seed": args.seed, "selected": selected, "steps": args.steps, "conditions": args.conditions,
                               "probe_steps": args.probe_steps,
                               "prepare_only": args.prepare_only}, capture=capture) as record:
        record.capture_environment({})
        record.snapshot_code()
        for path in (args.config, development_path, heldout_path):
            record.snapshot(path, "validation_input")
        if args.training_config:
            record.snapshot(args.training_config, "training_config")
        checkpoints = {key: artifact_fingerprint((ROOT / value).resolve()) for key, value in cfg["engine"]["load_model_args"].items()
                       if key in ("model_ckpt", "init_lora_ckpt", "resume_ckpt_path") and value}
        record.data["checkpoints"] = checkpoints
        if not args.prepare_only and any(v["type"] == "missing" for v in checkpoints.values()):
            raise ValueError("Configured checkpoint missing; see run_manifest.json")
        if cfg["runner"]["source_type"] == "habitat" and args.actions is None:
            record.data["phase"] = "generate_actions"
            record.save()
            args.actions = generate_validation_actions(cfg, args.config.resolve(),
                selected["habitat"], args.output, args.steps, args.seed)
        if args.actions:
            sequences = {}
            for path in sorted(args.actions.glob("*.json")):
                sequence = json.loads(path.read_text(encoding="utf-8"))
                episode = sequence.get("episode_id")
                if episode in sequences:
                    raise ValueError(f"Duplicate fixed action sequence for {episode}")
                sequences[episode] = sequence
            for episode in selected["habitat"]:
                sequence = sequences.get(episode)
                if sequence is None or len(sequence.get("actions", [])) < args.steps:
                    raise ValueError(f"Episode {episode}: fixed actions must cover all {args.steps} requested steps")
            for path in sorted(args.actions.glob("*.json")):
                record.snapshot(path, "fixed_actions")
            # Run against the snapshotted directory, not mutable originals.
            action_files = [p for p in record.data["inputs"] if p["role"] == "fixed_actions"]
            if not action_files:
                raise ValueError("--actions contains no JSON sequences")
            args.actions = (args.output / action_files[0]["snapshot"]).parent
        manifest = args.output / "development_subset.json"
        manifest.write_text(json.dumps({dataset: {"test": episodes} for dataset, episodes in selected.items()}, indent=2))
        commands = []
        for condition in args.conditions:
            condition_cfg = condition_config(cfg, condition, args.steps, selected, manifest, args.actions)
            path = args.output / f"{condition}.yaml"
            path.write_text(yaml.safe_dump(condition_cfg, sort_keys=False), encoding="utf-8")
            directory = args.output / condition
            directory.mkdir()
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
                       "--module", "thesis_testing_tools.run_prediction_validation", "--worker", "--config", str(path),
                       "--output", str(directory), "--seed", str(args.seed), "--turn-bins", str(args.turn_bins),
                       "--probe-steps", *map(str, args.probe_steps)]
            commands.append(command)
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2))
        if not args.prepare_only:
            for condition, command in zip(args.conditions, commands):
                with record.stage(condition, command, args.output / condition, args.output / f"{condition}.yaml", cfg["runner"]["source_type"]):
                    subprocess.run(command, cwd=ROOT, check=True)
            report(args.output, args.metrics)
    print(args.output)


if __name__ == "__main__":
    main()
