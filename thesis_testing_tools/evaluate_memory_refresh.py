"""Standalone memory-staleness evaluation.

Example (run in the normal UniWM GPU environment):
  python -m thesis_testing_tools.evaluate_memory_refresh --config cfg/memory_refresh_development.yaml --output output/memory_refresh --contexts 50

Uses the config's Habitat/replay source, initial checkpoint and base learning rate.
Missing Habitat action sequences are generated automatically for configured
episode IDs. Existing matching sequences are reused; replay uses its own actions.
Runs one-step, fixed-LR adaptation, not the controller or five-step rollout pipeline.
At each sampled real context, compares memory bypassed, cached history encoded
before intervening updates, and the same history re-encoded after those updates.
All three predictions use identical current weights, current working memory and
action. Predictions are generated BEFORE observing or training on that target;
saved predictions are scored after the inference run finishes.

Outputs report.html, scores.csv, summary.json and per-context images/metadata.
With --probe-step 19, --contexts counts paired generation seeds at one frozen
transition in the first configured episode, after 19 real history transitions.
Each invocation creates a unique run_<UTC timestamp>_<UUID> directory under --output.
No changes to the normal runner, checkpoint files or memory implementation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import csv
import html
import importlib
import json
from pathlib import Path
import random
import shutil
from statistics import mean, median
import sys
from uuid import uuid4

# Support direct execution as well as `python -m thesis_testing_tools...`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rng_state():
    import numpy as np
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def make_engine_class(base):
    """Keep provenance in this evaluator, without modifying the runtime engine."""
    from scripts.prompt_builder import build_action_prompt

    class EvaluationEngine(base):
        weight_version = 0

        def attentions(self):
            model = self.model.get_base_model()
            return [model.model.layers[i].self_attn for i in sorted(model.use_memory_bank_layers)]

        def reset_episode(self, episode_id, bundle):
            self.history_inputs = []
            super().reset_episode(episode_id, bundle)

        def update_working_memory(self, observation, pose, update_context_ema=True, *, input_source=None):
            super().update_working_memory(observation, pose, update_context_ema, input_source=input_source)
            inputs = self._processor_inputs_from_prompt(build_action_prompt(
                start_pose_str=pose, dx_range=self.action_vocabulary.range_for("dx"),
                dy_range=self.action_vocabulary.range_for("dy"),
                dyaw_range=self.action_vocabulary.range_for("dyaw"),
                prior_decoded_actions=list(self.prior_actions)))
            self.working_input = {
                "inputs": {k: inputs[k].detach().cpu().clone() for k in ("input_ids", "attention_mask")},
                "version": self.weight_version,
            }

        def store_working_memory(self):
            count = self.memory_count
            super().store_working_memory()
            if self.memory_count > count:
                self.history_inputs.append(self.working_input)
                assert all(len(a.global_stored_keys) == len(self.history_inputs) for a in self.attentions())

        def _update_weights(self, *args, **kwargs):
            result = super()._update_weights(*args, **kwargs)
            self.weight_version += 1
            return result

        def refresh_history(self):
            import torch
            attentions = self.attentions()
            rebuilt = [[] for _ in attentions]
            for record in self.history_inputs:
                inputs = {k: v.to(self.device) for k, v in record["inputs"].items()}
                with torch.autocast(device_type="cuda", dtype=self.model.dtype):
                    ok = self.model.initialize_memory_bank_no_pixels(**inputs)
                if not ok:
                    raise RuntimeError("Historical memory initialization failed")
                for bucket, attention in zip(rebuilt, attentions):
                    bucket.append((attention.stored_keys, attention.stored_values))
            for attention, bucket in zip(attentions, rebuilt):
                attention.global_stored_keys = [pair[0] for pair in bucket]
                attention.global_stored_values = [pair[1] for pair in bucket]

    return EvaluationEngine


def compare_memory(engine, bundle):
    """Return paired predictions; restore state and RNG even on inference failure."""
    assert not engine.model.training
    attentions = engine.attentions()
    original_history = [(a.global_stored_keys, a.global_stored_values) for a in attentions]
    original_kwargs = engine.config["generation"]["visualization"].copy()
    original_rng = rng_state()
    predictions, metadata = {}, {}
    try:
        for mode in ("bypass", "cached", "refreshed"):
            state = engine._store_state()
            try:
                if mode == "refreshed":
                    engine.refresh_history()
                    # Re-encoding changes working memory as a side effect. Preserve
                    # the exact same CURRENT working K/V for all three conditions.
                    saved = engine._memory_manager._cached_step_state
                    for i, attention in enumerate(attentions):
                        attention.stored_keys = saved.stored_keys[i]
                        attention.stored_values = saved.stored_values[i]
                engine.config["generation"]["visualization"].update(
                    use_memory_bank=mode != "bypass", use_global_memory_bank=mode != "bypass")
                restore_rng(original_rng)
                predictions[mode] = engine._predict_step(bundle)
                selections = []
                for attention in attentions:
                    indices, similarities = attention.compute_similarity_and_select_topk(
                        attention.stored_keys, k=attention.memory_top_k)
                    selections.append({"indices": indices, "similarities": similarities})
                metadata[mode] = {"available": predictions[mode].visualization is not None,
                                  "retrieval_candidates": selections}
            finally:
                engine._restore_state(*state)
                for attention, (keys, values) in zip(attentions, original_history):
                    attention.global_stored_keys, attention.global_stored_values = keys, values
    finally:
        engine.config["generation"]["visualization"] = original_kwargs
        restore_rng(original_rng)
    return predictions, metadata


def next_forced_action(bundle):
    actions = bundle.action_text
    if isinstance(actions, list):
        # Both repository source formatters place the previous action at index 0.
        return actions[1] if len(actions) > 1 else None
    if isinstance(actions, str) and actions:
        return actions
    raise ValueError("Source must supply fixed actions: set Habitat fixed_action_files_dir, or use a replay config")


def sampling_steps(max_steps, warmup, count):
    """Spread probes across the episode so we include older memory banks."""
    available = max_steps - warmup
    count = min(count, available)
    if count <= 0:
        return []
    if count == 1:
        return [warmup]
    return [warmup + round(i * (available - 1) / (count - 1)) for i in range(count)]


def repeat_transition(engine, current, *, episode_id, step, count, seed, output, observe):
    """Sample fixed weights/history; observe the target once, after every prediction."""
    from runtime_scripts.runtime_utils import prediction_seed

    if len(engine.history_inputs) < 10:
        raise ValueError(f"Probe step {step}: only {len(engine.history_inputs)} stored memories; "
                         "need at least 10. Choose a later step or a different episode.")
    rows = []
    for repeat in range(count):
        # Vary only the generation seed, not initialization, history, or training.
        trial = replace(current, prediction_seed=prediction_seed(seed + repeat, episode_id, step))
        predictions, conditions = compare_memory(engine, trial)
        folder = output / f"context_{repeat:03d}"
        folder.mkdir()
        current.current_observation.save(folder / "input.png")
        for mode, prediction in predictions.items():
            if prediction.visualization is not None:
                prediction.visualization.save(folder / f"{mode}.png")
        rows.append({"context": repeat, "episode_id": episode_id, "step": step,
                     "repeat_index": repeat, "prediction_seed": trial.prediction_seed,
                     "weight_version": engine.weight_version,
                     "history_update_ages": [engine.weight_version - r["version"] for r in engine.history_inputs],
                     "action": current.action_text, "conditions": conditions})
        print(f"[MEMORY COMPARISON] seed {repeat + 1}/{count}: {trial.prediction_seed}", flush=True)
    observed = observe()
    for row in rows:
        folder = output / f"context_{row['context']:03d}"
        observed.current_observation.save(folder / "target.png")
        row["collision"] = observed.collision
        (folder / "context.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return rows


def prepare_habitat_actions(config, config_path, output, seed):
    """Reuse compatible files and generate only missing configured episodes."""
    source = config["runner"]
    if source["source_type"] != "habitat":
        return {"generated": [], "reused": []}
    params = source["adapter_params"]
    episodes = params.get("episode_ids")
    if not isinstance(episodes, list) or not episodes or any(not isinstance(ep, str) or not ep for ep in episodes):
        raise ValueError("Automatic action preparation requires explicit nonempty string episode_ids")
    root = Path(__file__).resolve().parents[1]
    directory = (root / params["fixed_action_files_dir"]).resolve()
    steps = source["max_episode_steps"]
    action_seed = params.get("seed", seed)
    existing = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError(f"{path}: expected action sequence schema_version 1")
        episode = payload.get("episode_id")
        if not isinstance(episode, str) or not episode:
            raise ValueError(f"{path}: episode_id must be a nonempty string")
        if episode in existing:
            raise ValueError(f"{path}: duplicate episode_id {episode}")
        existing[episode] = path
        actions = payload.get("actions")
        if (not isinstance(actions, list) or not actions or any(not isinstance(a, str) or not a for a in actions)
                or len(actions) != payload.get("target_steps")):
            raise ValueError(f"{path}: actions must be nonempty strings matching target_steps")
        if episode in episodes and (len(actions) < steps or payload.get("seed") != action_seed):
            raise ValueError(f"{path}: need seed {action_seed} and at least {steps} actions; use a separate action directory for a different setup")
    result = {"generated": [], "reused": []}
    for episode in dict.fromkeys(episodes):
        if episode in existing:
            result["reused"].append(str(existing[episode]))
            continue
        from scripts.generate_habitat_action_sequence import generate_action_sequence
        print(f"[MEMORY COMPARISON] Generating {steps} fixed actions for episode {episode}")
        generated = root / generate_action_sequence(episode, steps, output, seed=action_seed,
            config_path=config_path, checkpoint=config["engine"]["load_model_args"]["model_ckpt"])
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / generated.name
        if generated.resolve() != destination.resolve():
            # Exclusive creation prevents overwriting an existing sequence.
            with generated.open("rb") as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        result["generated"].append(str(destination))
    print(f"[MEMORY COMPARISON] Actions: {len(result['reused'])} reused, {len(result['generated'])} generated")
    return result


def summarize(rows, metrics):
    paired = {}
    for left, right in (("refreshed", "cached"), ("cached", "bypass"), ("refreshed", "bypass")):
        comparisons = {}
        for metric in metrics:
            pairs = [(r["episode_id"], r["conditions"][left]["scores"][metric] - r["conditions"][right]["scores"][metric])
                     for r in rows if r["conditions"][left]["scores"] is not None
                     and r["conditions"][right]["scores"] is not None]
            episode_means = {ep: mean(d for e, d in pairs if e == ep) for ep, _ in pairs}
            comparisons[metric] = {"paired_contexts": len(pairs),
                "mean_delta": mean(d for _, d in pairs) if pairs else None,
                "median_delta": median(d for _, d in pairs) if pairs else None,
                "left_better_count": sum(d > 0 if metric == "ssim" else d < 0 for _, d in pairs),
                "tie_count": sum(d == 0 for _, d in pairs),
                "episode_mean_deltas": episode_means,
                "equal_episode_mean_delta": mean(episode_means.values()) if episode_means else None}
        paired[f"{left}_minus_{right}"] = comparisons
    return paired


def write_report(output, rows, metrics, requested, provenance):
    summary = {"requested_contexts": requested, "evaluated_contexts": len(rows),
        "status": "complete" if len(rows) == requested else "insufficient_contexts",
        "missing_predictions": {mode: sum(r["conditions"][mode]["scores"] is None for r in rows)
                                for mode in ("bypass", "cached", "refreshed")},
        "paired": summarize(rows, metrics), "provenance": provenance,
        "unique_transitions": len({(r["episode_id"], r["step"]) for r in rows}),
        "interpretation": "Negative refreshed-minus-cached MAE/LPIPS, or positive SSIM, favors refreshed history. Repeated seeds measure generation variability at one fixed transition, not generalization across contexts. A null result is not proof of no staleness effect."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["context", "episode_id", "step", "weight_version", "repeat_index", "prediction_seed", "condition", "available", *metrics])
        writer.writeheader()
        for r in rows:
            for mode, condition in r["conditions"].items():
                writer.writerow({k: r[k] for k in ("context", "episode_id", "step", "weight_version")} |
                    {k: r.get(k) for k in ("repeat_index", "prediction_seed")} |
                    {"condition": mode, "available": condition["scores"] is not None} | (condition["scores"] or {}))
    body = ["<!doctype html><meta charset='utf-8'><title>Memory refresh comparison</title>",
        "<style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:8px;vertical-align:top}img{width:180px}pre{white-space:pre-wrap}section{margin:30px 0}</style>",
        "<h1>Memory refresh comparison</h1>",
        f"<p>{len(rows)} / {requested} comparisons; {summary['unique_transitions']} unique transitions. Fixed-LR, one-step adaptation. Same current weights and action across conditions.</p>",
        "<p>Repeated-seed probes hold history and weights fixed across all draws. They measure generation variability, not independent scene coverage.</p>",
        "<p>Cached history was encoded before intervening updates; refreshed history was re-encoded afterward. Bypass disables memory injection. Lower MAE/LPIPS and higher SSIM are better. Contexts within episodes are correlated.</p>",
        "<pre>" + html.escape(json.dumps(summary["paired"], indent=2)) + "</pre>"]
    for r in rows:
        body.append(f"<section><h2>Context {r['context']} — episode {html.escape(str(r['episode_id']))}, step {r['step']}</h2>")
        body.append(f"<p>Prediction seed: {r.get('prediction_seed', 'not recorded')}; repeat: {r.get('repeat_index', 'not a repeated probe')}</p>")
        body.append(f"<p>Weight version {r['weight_version']}; memory update ages: {r['history_update_ages']}</p><table><tr>")
        for name in ("input", "target", "bypass", "cached", "refreshed"):
            body.append(f"<th>{name}</th>")
        body.append("</tr><tr>")
        for name in ("input", "target", "bypass", "cached", "refreshed"):
            available = name in ("input", "target") or r["conditions"][name]["available"]
            cell = f"<img src='context_{r['context']:03d}/{name}.png'>" if available else "Missing prediction"
            if name in r["conditions"]:
                cell += "<pre>" + html.escape(json.dumps(r["conditions"][name]["scores"], indent=2)) + "</pre>"
            body.append("<td>" + cell + "</td>")
        body.append("</tr></table></section>")
    (output / "report.html").write_text("\n".join(body), encoding="utf-8")
    return summary


def run(args):
    import numpy as np
    import torch
    from runtime_scripts.event_logger import EventLogger
    from runtime_scripts.runtime_engine import UniWMEngine
    from runtime_scripts.uniwm_schemas import StepPrediction
    from runtime_scripts.runtime_utils import is_stop_action, load_config, prediction_seed
    from thesis_testing_tools.generate_metrics import ImageCalculator

    config = load_config(str(args.config))
    if config["engine"]["training"] is False or not config["engine"]["load_model_args"]["use_memory_bank_inference"]:
        raise ValueError("Evaluation requires engine.training and use_memory_bank_inference enabled")
    source = config["runner"]
    if source["source_type"] not in ("habitat", "replay"):
        raise ValueError("Use a Habitat or replay source config")
    if source["source_type"] == "habitat" and not source["adapter_params"].get("fixed_action_files_dir"):
        raise ValueError("Set runner.adapter_params.fixed_action_files_dir to the development action sequences")
    if args.probe_step is not None:
        if not 10 <= args.probe_step < source["max_episode_steps"]:
            raise ValueError("--probe-step must be at least 10 and below runner.max_episode_steps")
        if not config["engine"]["generation"]["visualization"].get("do_sample", False):
            raise ValueError("Repeated-seed probes require engine.generation.visualization.do_sample=true")
        source["max_total_episodes"] = 1
        if source["source_type"] == "habitat":
            source["adapter_params"]["episode_ids"] = source["adapter_params"]["episode_ids"][:1]
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.config, args.output / "config.yaml")
    action_setup = prepare_habitat_actions(config, args.config, args.output, args.seed)
    # Habitat generation can consume RNG; reseed before model initialization.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = []
    with EventLogger(args.output / "events.jsonl") as log:
        engine = make_engine_class(UniWMEngine)(source["source_type"], str(args.config), event_logger=log)
        hyper_params = engine.config["training"]["hyper_params"]
        learning_rate = float(args.learning_rate if args.learning_rate is not None else hyper_params["initial_lr"])
        hyper_params["initial_lr"] = learning_rate
        module = importlib.import_module("source_tools." + source["source_file_name"])
        params = source["adapter_params"].copy()
        if "bin_step" in params:
            params["bin_step"] = engine.action_vocabulary.bin_step
        adapter = getattr(module, source["source_type"].capitalize() + "EpisodeAdapter")(**params, event_logger=log)
        formatter = getattr(module, source["source_type"].capitalize() + "UniWMFormatter")(
            engine.action_vocabulary.bin_step, int(engine.config["load_model_cfg"]["img_size"]))
        try:
            for episode_index in range(source["max_total_episodes"]):
                initial = adapter.next_episode()
                if initial is None or len(rows) >= args.contexts:
                    break
                episode_id = str(initial[0].episode_id)
                bundle = formatter.convert_from_source(initial)
                engine.reset_episode(episode_id, bundle)
                sampled = 0
                sample_after = sampling_steps(source["max_episode_steps"],
                    max(1, engine.config["memory"]["min_memories"]), args.per_episode)
                for step in range(source["max_episode_steps"]):
                    if bundle.source_done or len(rows) >= args.contexts:
                        break
                    action = next_forced_action(bundle)
                    if action is None or is_stop_action(action):
                        break
                    log.next_step({"episode_id": episode_id, "step": step})
                    current = replace(bundle, action_text=action,
                        prediction_seed=prediction_seed(config["wrapper"]["seed"], episode_id, step))
                    use_memory = engine.memory_count >= engine.config["memory"]["min_memories"]
                    eligible = (use_memory and bool(engine.history_inputs) and sampled < len(sample_after)
                                and step >= sample_after[sampled])
                    eligible = eligible and any(r["version"] < engine.weight_version for r in engine.history_inputs)
                    if args.probe_step is not None:
                        if step == args.probe_step:
                            converted = formatter.convert_action(action)
                            if not converted:
                                raise ValueError(f"Source action produced no executable primitives: {action}")
                            rows = repeat_transition(engine, current, episode_id=episode_id, step=step, count=args.contexts,
                                seed=args.seed, output=args.output,
                                observe=lambda: formatter.convert_from_source(adapter.step(converted)))
                            # Do not train on the probe target or continue this trajectory.
                            break
                        eligible = False
                    if eligible:
                        predictions, conditions = compare_memory(engine, current)
                        training_prediction = predictions["cached"]
                    else:
                        training_prediction = StepPrediction(current, action, None, 0, 0,
                            *engine.get_current_context(), viz_used_memory=use_memory)
                    converted = formatter.convert_action(action)
                    if not converted:
                        raise ValueError(f"Source action produced no executable primitives: {action}")
                    observed = formatter.convert_from_source(adapter.step(converted))
                    if eligible:
                        folder = args.output / f"context_{len(rows):03d}"
                        folder.mkdir()
                        current.current_observation.save(folder / "input.png")
                        observed.current_observation.save(folder / "target.png")
                        for mode, prediction in predictions.items():
                            if prediction.visualization is not None:
                                prediction.visualization.save(folder / f"{mode}.png")
                        row = {"context": len(rows), "episode_id": episode_id, "step": step,
                            "prediction_seed": current.prediction_seed,
                            "weight_version": engine.weight_version,
                            "history_update_ages": [engine.weight_version - r["version"] for r in engine.history_inputs],
                            "action": action, "collision": observed.collision, "conditions": conditions}
                        rows.append(row)
                        (folder / "context.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
                        sampled += 1
                        print(f"[MEMORY COMPARISON] {len(rows)}/{args.contexts}: episode {episode_id}, step {step}")
                    # Real transition learning only, after all diagnostic predictions.
                    engine.prior_actions.append(action)
                    training_prediction.real_input_obs = current.current_observation
                    training_prediction.real_next_obs = observed.current_observation
                    if not observed.collision:
                        engine.train_viz_step(training_prediction, 1.0,
                            max_grad_norm=engine.config["training"]["hyper_params"]["max_grad_norm"])
                        engine.store_working_memory()
                    engine.update_working_memory(observed.current_observation, observed.start_pose_str, not observed.collision)
                    bundle = observed
                if len(rows) >= args.contexts:
                    break
        finally:
            adapter.close()
        provenance = {"seed": args.seed, "torch": torch.__version__, "initial_checkpoint": engine.config["load_model_args"]["model_ckpt"],
            "learning_rate": engine.config["training"]["hyper_params"]["initial_lr"], "updates": engine.weight_version,
            "sampling": "Evenly spaced slots over configured episode length; deferred until memory is eligible", "per_episode_cap": args.per_episode,
            "controller": False, "horizon": 1, "action_sequences": action_setup}
        if args.probe_step is not None:
            provenance.update(sampling="Repeated generation seeds at one frozen transition in the first configured episode",
                probe_step=args.probe_step, per_episode_cap=None,
                prediction_seed_rule="prediction_seed(seed + repeat_index, episode_id, probe_step)",
                target_used_for_training=False)
        log.feed({"evaluation": provenance})
    # Initialize metrics AFTER inference: the repository metric backend changes
    # Torch deterministic settings, which should not affect model generation.
    calculator = ImageCalculator(args.metrics)
    from PIL import Image
    for row in rows:
        folder = args.output / f"context_{row['context']:03d}"
        with Image.open(folder / "target.png") as image:
            target = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        for mode, condition in row["conditions"].items():
            condition["scores"] = None
            if condition["available"]:
                with Image.open(folder / f"{mode}.png") as image:
                    predicted = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                if predicted.shape != target.shape:
                    raise ValueError(f"{folder}/{mode}.png: prediction/target shapes differ")
                condition["scores"] = calculator(predicted, target)
        (folder / "context.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    provenance["metrics"] = calculator.provenance
    summary = write_report(args.output, rows, args.metrics, args.contexts, provenance)
    print(json.dumps(summary, indent=2))
    print(f"Report: {args.output / 'report.html'}")
    return 0 if summary["status"] == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output root; each invocation creates a unique run subdirectory")
    parser.add_argument("--contexts", type=int, default=50)
    parser.add_argument("--per-episode", type=int, default=5)
    parser.add_argument("--probe-step", type=int,
                        help="Repeat --contexts paired seeds at this zero-based step in the first episode; requires >=10 stored memories")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, help="Default: engine base LR from config")
    parser.add_argument("--metrics", nargs="+", choices=("mae", "ssim", "lpips"), default=["mae", "ssim", "lpips"])
    args = parser.parse_args()
    if args.contexts <= 0 or args.per_episode <= 0 or not 0 <= args.seed < 2**32:
        parser.error("Use positive context/episode counts and a seed in [0, 2**32)")
    if args.learning_rate is not None and not 0 < args.learning_rate < float("inf"):
        parser.error("--learning-rate must be finite and positive")
    run_id = f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid4().hex}"
    args.config, args.output = args.config.resolve(), args.output.resolve() / run_id
    print(f"[MEMORY COMPARISON] Run ID: {run_id}", flush=True)
    print(f"[MEMORY COMPARISON] Output: {args.output}", flush=True)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
