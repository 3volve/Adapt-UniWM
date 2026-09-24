"""Read-only accounting of a thesis pipeline run. See docs/reconciliation.md."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import yaml

from thesis_testing_tools.generate_metrics import read_events, resolve_image
from thesis_testing_tools import generate_metrics

VERSION = "1.0.0"


class InvalidRunData(ValueError):
    """The input cannot be interpreted as pipeline execution evidence."""


class Reconciliation:
    def __init__(self, root, path_maps=()):
        self.root = Path(root).resolve()
        self.path_maps = list(path_maps)
        self.checks = []
        self.inputs = {}

    def check(self, name, expected, observed, *, stage=None, location=None, known=True):
        status = ("pass" if expected == observed else "fail") if known else "unknown"
        self.checks.append(dict(check=name, status=status, stage=stage,
                                location=location, expected=expected, observed=observed))

    def note(self, name, status, reason, stage=None):
        self.checks.append(dict(check=name, status=status, stage=stage, reason=reason))

    def path(self, value, base=None):
        return resolve_image(value, (base or self.root) / "_anchor", self.path_maps)

    def fingerprint(self, path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        item = dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())
        if str(path) in self.inputs and self.inputs[str(path)] != item:
            raise ValueError(f"Input changed during reconciliation: {path}")
        self.inputs[str(path)] = item
        return item

    def document(self, path, stage=None, *, yaml_format=False):
        try:
            self.fingerprint(path)
            text = path.read_text(encoding="utf-8")
            if yaml_format:
                value = yaml.safe_load(text)
            else:
                value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError("expected a mapping")
            return value
        except (ValueError, yaml.YAMLError) as error:
            raise InvalidRunData(f"{path}: {error}") from error
        except OSError as error:
            self.note("input_readable", "fail", f"{path}: {error}", stage)
            return None

    def worker(self, stage, metadata):
        name = stage["stage_id"]
        path = self.path(stage["events"])
        self.check("stage_completed", "completed", stage.get("status"), stage=name)
        try:
            self.fingerprint(path)
            records, evidence = read_events(path, allow_incomplete=True)
        except ValueError as error:
            raise InvalidRunData(str(error)) from error
        except OSError as error:
            self.note("event_stream", "fail", str(error), name)
            return
        self.check("run_summary", True, evidence["finalized"], stage=name, location=str(path))
        self.check("worker_outcome", "completed", evidence["outcome"], stage=name)
        bad = [r["_event"]["id"] for r in records if r["outcome"] not in ("completed", "no_op")]
        self.check("event_outcomes", [], bad, stage=name)
        self.check("events_sealed", [], [r["_event"]["id"] for r in records if not r["_event"].get("finished_at")], stage=name)
        config = self.document(self.path(stage["config_snapshot"]), name, yaml_format=True) if stage.get("config_snapshot") else None
        if config is None:
            self.note("configuration", "unknown", "No readable saved configuration", name)
        config = config or {}
        if any(not isinstance(config.get(key, {}), dict) for key in ("runner", "wrapper")):
            self.note("configuration", "fail", "runner and wrapper must be mappings", name)
            return
        runner = config.get("runner", {})
        wrapper_config = config.get("wrapper", {})
        planned_ceiling = metadata.get("habitat_max_episode_steps" if stage["name"] == "habitat" else "source_max_episode_steps")
        if planned_ceiling is not None:
            self.check("planned_step_ceiling", planned_ceiling, runner.get("max_episode_steps"), stage=name)
        setups, ends, steps = {}, {}, defaultdict(list)
        for record in records:
            for section, target in (("episode_setup", setups), ("episode_end", ends)):
                for key, value in record.get(section, {}).items():
                    self.check(f"{section}_unique", False, key in target, stage=name, location=key)
                    target[key] = value
            if record["_event"]["kind"] == "step":
                steps[f"{record['data_id']}/{record['episode_index']}"].append(record)
        self.check("episode_end_coverage", list(setups), list(ends), stage=name)
        self.check("attempt_episode_coverage", [], sorted(set(steps) - set(setups)), stage=name)
        self.check("episode_indices", list(range(len(setups))),
                   [int(key.rsplit("/", 1)[1]) for key in setups], stage=name)
        self.check("summary_episode_count", len(ends), records[-1].get("episodes"), stage=name,
                   known=evidence["finalized"] and "episodes" in records[-1])
        expected = metadata.get("source_episode_order") if stage["name"] != "habitat" else None
        if stage["name"] == "habitat":
            ids = metadata.get("habitat_episode_order", runner.get("adapter_params", {}).get("episode_ids"))
            expected = {stage["data_id"]: ids} if ids is not None else None
        observed = defaultdict(list)
        for key, setup in setups.items():
            observed[key.rsplit("/", 1)[0]].append(setup.get("episode_id"))
            self.check("episode_setup_completed", "completed", setup.get("outcome"), stage=name, location=key)
        self.check("selected_episode_order", expected, dict(observed), stage=name, known=expected is not None)
        for key, setup in setups.items():
            attempts = steps[key]
            end = ends.get(key, {})
            if end:
                self.check("episode_end_identity", setup.get("episode_id"), end.get("episode_id"), stage=name, location=key)
                reason = end.get("termination_reason")
                self.check("termination_reason", True, reason in ("max_episode_steps", "adapter_done", "wrapper_stop_action", "repeated_no_op_actions"), stage=name, location=key)
                if reason == "adapter_done":
                    self.check("adapter_done_evidence", True, bool(attempts) and attempts[-1].get("wrapper", {}).get("source_done") is True, stage=name, location=key)
                if reason == "wrapper_stop_action":
                    self.check("stop_evidence", True, bool(attempts) and attempts[-1].get("action", {}).get("stop") is True and runner.get("stop_on_wrapper_done") is True, stage=name, location=key)
                if reason == "repeated_no_op_actions":
                    self.check("no_op_termination", ["no_op"] * 5, [r["transition"]["outcome"] for r in attempts[-5:]], stage=name, location=key)
            self.check("attempt_indices", list(range(len(attempts))), [r["step_idx"] for r in attempts], stage=name, location=key)
            self.check("attempt_episode_identity", [setup.get("episode_id")] * len(attempts),
                       [r["episode_id"] for r in attempts], stage=name, location=key)
            for field, count in (("attempts", len(attempts)), ("steps_executed", sum(r["transition"]["outcome"] == "completed" for r in attempts))):
                self.check("episode_" + field, count, end.get(field), stage=name, location=key, known=bool(end))
            ceiling = runner.get("max_episode_steps")
            if ceiling is not None:
                self.check("attempt_ceiling", True, len(attempts) <= ceiling, stage=name, location=key)
                if end.get("termination_reason") == "max_episode_steps":
                    self.check("ceiling_termination", ceiling, len(attempts), stage=name, location=key)
            else:
                self.note("attempt_ceiling", "unknown", f"{key}: missing configured ceiling", name)
            for record in attempts:
                self.transition(record, wrapper_config, name)
        self.actions(runner, steps, name)
        engine_training = config.get("engine", {}).get("training", False)
        base_lr = engine_training.get("hyper_params", {}).get("initial_lr") if engine_training else None
        # PyYAML leaves YAML's bare 1e-4 notation as text; OmegaConf accepts it.
        if isinstance(base_lr, str):
            try:
                base_lr = float(base_lr)
            except ValueError:
                self.note("configured_learning_rate", "fail", f"Not a numeric learning rate: {base_lr}", name)
                base_lr = None
        self.schedule(wrapper_config, records, path.parent, name, base_lr)
        for record in records:
            self.artifacts(record, path.parent, name)
        if runner.get("save_model_weights"):
            saved = {key for record in records for key, item in record.get("checkpoint", {}).items() if item.get("path")}
            self.check("checkpoint_coverage", sorted(observed), sorted(saved), stage=name)

    def transition(self, record, config, stage):
        location = f"event:{record['_event']['id']}"
        wrapper = record.get("wrapper", {})
        training = record.get("training", {})
        outcome = record["transition"]["outcome"]
        def check(name, expected, observed, known=True):
            self.check(name, expected, observed, stage=stage, location=location, known=known)
        check("transition_terminal", True, outcome in ("completed", "no_op", "failed", "interrupted"))
        if outcome == "no_op":
            check("no_op_actions", [], record.get("formatter", {}).get("converted_actions"))
            check("no_op_not_executed", False, "adapter" in record or "training" in record)
        if outcome == "completed":
            check("adapter_completed", "completed", record.get("adapter", {}).get("outcome"))
            check("wrapper_completed", "completed", wrapper.get("phase"))
        optimizer = training.get("optimizer_step")
        if "optimizer_step" in training:
            check("optimizer_outcome_known", True, optimizer is not None)
        if config.get("training_enabled") is False or wrapper.get("update_eligible") is False:
            check("optimizer_not_applied", True, "training" not in record or optimizer is False,
                  known="training" not in record or optimizer is not None)
        elif config.get("training_enabled") is True and wrapper.get("update_eligible") is True and outcome == "completed":
            check("eligible_optimizer_applied", True, optimizer)
        elif "training_enabled" not in config:
            self.note("optimizer_policy", "unknown", f"{location}: missing wrapper training setting", stage)
        if "update_eligible" in wrapper:
            reason = "stop_action" if wrapper.get("stop_action") else "collision" if wrapper.get("collision") else None
            check("update_eligibility", reason is None, wrapper["update_eligible"])
            check("eligibility_skip_reason", reason, wrapper.get("eligibility_skip_reason"))
        if record["source_mode"] == "habitat" and "adapter" in record:
            primitives = record["adapter"].get("primitives", {})
            check("primitive_indices", [str(i) for i in range(len(primitives))], list(primitives))
            executed = [p.get("action") for p in primitives.values() if p.get("outcome") == "completed"]
            check("primitive_completion", len(primitives), len(executed))
            environment = record.get("environment", {})
            check("primitive_accounting", executed, environment.get("primitive_actions_executed"),
                  known="primitive_actions_executed" in environment)
            check("primitive_count", len(executed), environment.get("primitive_action_count"), known="primitive_action_count" in environment)
            converted = record.get("formatter", {}).get("converted_actions")
            if isinstance(converted, list):
                check("primitive_action_prefix", converted[:len(executed)], executed)
                last = list(primitives.values())[-1] if primitives else {}
                check("primitive_early_stop", True, len(executed) == len(converted) or last.get("collision") is True or last.get("source_done") is True)
            check("primitive_collision", any(p.get("collision") is True for p in primitives.values()), wrapper.get("collision"), known="collision" in wrapper)

    def actions(self, runner, steps, stage):
        value = runner.get("adapter_params", {}).get("fixed_action_files_dir")
        if not value:
            self.note("fixed_actions", "not_applicable" if runner else "unknown", "No configured standalone action directory", stage)
            return
        directory = self.path(value)
        files = sorted(directory.glob("*.json"))
        if not files:
            self.note("fixed_actions", "fail", f"No action files in {directory}; use --path-map for relocated paths", stage)
            return
        artifacts = {}
        for path in files:
            artifact = self.document(path, stage)
            if artifact is not None:
                episode = str(artifact.get("episode_id"))
                self.check("action_episode_unique", False, episode in artifacts, stage=stage, location=str(path))
                artifacts[episode] = artifact
        for key, records in steps.items():
            episode = records[0]["episode_id"]
            artifact = artifacts.get(episode)
            if artifact is None:
                self.note("action_episode", "fail", f"Missing action artifact for {key} ({episode})", stage)
                continue
            expected = artifact.get("actions")
            actual = [r.get("action", {}).get("requested") for r in records]
            self.check("action_sequence", expected, actual, stage=stage, location=key)
            self.check("action_target_count", artifact.get("target_steps"), len(actual), stage=stage, location=key)

    def schedule(self, config, records, worker_dir, stage, base_lr=None):
        settings = config.get("learning_rate_schedule")
        if not settings:
            self.note("learning_rate_schedule", "not_applicable" if "learning_rate_schedule" in config else "unknown", "No configured LR schedule", stage)
            return
        if not isinstance(settings, dict) or settings.get("mode") not in ("record", "replay"):
            self.note("schedule_configuration", "fail", "Expected record/replay schedule mapping", stage)
            return
        mode = settings["mode"]
        if mode == "replay" and not isinstance(settings.get("input_path"), str):
            self.note("schedule_configuration", "fail", "Replay requires input_path", stage)
            return
        directory = worker_dir if mode == "record" else self.path(settings["input_path"])
        filename = "learning_rate_schedule_shuffled.json" if settings.get("shuffled") else "learning_rate_schedule.json"
        artifact = self.document(directory / filename, stage)
        if artifact is None:
            return
        entries = artifact.get("entries")
        if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
            self.note("schedule_entries", "fail", "Expected a list of entry mappings", stage)
            return
        required = {"data_id", "episode_id", "step_idx", "action", "collision", "update_eligible", "skip_reason", "lr_scalar", "effective_learning_rate"}
        if any(required - entry.keys() for entry in entries):
            self.note("schedule_fields", "fail", "Schedule entries lack required transition fields", stage)
            return
        key = lambda e: (e.get("data_id"), e.get("episode_id"), e.get("step_idx"))
        observed = [r for r in records if r["_event"]["kind"] == "step" and r.get("wrapper", {}).get("phase") == "completed"]
        self.check("schedule_schema", 1, artifact.get("schema_version"), stage=stage)
        self.check("schedule_shuffled", bool(settings.get("shuffled")), artifact.get("shuffled"), stage=stage)
        self.check("schedule_base_lr", base_lr, artifact.get("base_learning_rate"), stage=stage, known=base_lr is not None)
        self.check("schedule_consumption", [key(e) for e in entries], [key(r) for r in observed], stage=stage)
        for entry, record in zip(entries, observed):
            wrapper = record["wrapper"]
            location = f"event:{record['_event']['id']}"
            for field, value in (("action", record.get("action", {}).get("requested")), ("collision", wrapper.get("collision")),
                                 ("update_eligible", wrapper.get("update_eligible")), ("skip_reason", wrapper.get("eligibility_skip_reason"))):
                self.check("schedule_" + field, entry.get(field), value, stage=stage, location=location)
            if entry.get("update_eligible"):
                self.check("schedule_scalar", entry.get("lr_scalar"), wrapper.get("update_weight"), stage=stage, location=location)
                applied = record.get("training", {}).get("applied_learning_rate")
                if mode == "replay":
                    expected = entry.get("effective_learning_rate")
                    close = isinstance(applied, (int, float)) and isinstance(expected, (int, float)) and math.isclose(applied, expected, rel_tol=1e-9, abs_tol=1e-12)
                    self.check("schedule_applied_lr", True, close, stage=stage, location=dict(event=location, expected_lr=expected, applied_lr=applied))
        if mode == "record" or settings.get("shuffled"):
            self.shuffle(directory, stage)

    def shuffle(self, directory, stage):
        aligned = self.document(directory / "learning_rate_schedule.json", stage)
        shuffled = self.document(directory / "learning_rate_schedule_shuffled.json", stage)
        if aligned is None or shuffled is None:
            return
        entries = aligned.get("entries")
        seed = shuffled.get("shuffled_seed")
        if not isinstance(entries, list) or type(seed) is not int or any(not isinstance(e, dict) or not {"data_id", "episode_id", "update_eligible", "lr_scalar", "effective_learning_rate"} <= e.keys() for e in entries):
            self.note("schedule_shuffle", "fail", "Missing shuffle seed or valid aligned entries", stage)
            return
        expected = [dict(entry) for entry in entries]
        groups = defaultdict(list)
        for i, entry in enumerate(entries):
            if entry["update_eligible"]:
                groups[(entry["data_id"], entry["episode_id"])].append(i)
        rng = random.Random(seed)
        for indices in groups.values():
            pairs = [(entries[i]["lr_scalar"], entries[i]["effective_learning_rate"]) for i in indices]
            rng.shuffle(pairs)
            for i, (scalar, rate) in zip(indices, pairs):
                expected[i].update(lr_scalar=scalar, effective_learning_rate=rate)
        self.check("schedule_shuffle", expected, shuffled.get("entries"), stage=stage)

    def artifacts(self, record, directory, stage):
        wrapper = record.get("wrapper", {})
        refs = [wrapper.get("real_obs_path"), wrapper.get("predicted_obs_path"), wrapper.get("evaluation", {}).get("predicted_obs_path")]
        for checkpoint in record.get("checkpoint", {}).values():
            self.check("checkpoint_completed", "completed", checkpoint.get("outcome"), stage=stage)
            refs.append(checkpoint.get("path"))
        for value in filter(None, refs):
            path = self.path(value, directory)
            self.check("artifact_exists", True, path.exists(), stage=stage, location=str(path))
            if path.is_file():
                self.fingerprint(path)
        for section in ("schedule_save", "source_setup"):
            for key, details in record.get(section, {}).items():
                self.check(section + "_completed", "completed", details.get("outcome"), stage=stage, location=key)
        if "schedule_finalize" in record:
            self.check("schedule_finalized", "completed", record["schedule_finalize"].get("outcome"), stage=stage)

    def run(self):
        manifest = self.document(self.root / "run_manifest.json")
        if manifest is None:
            raise InvalidRunData(f"No readable run_manifest.json in {self.root}")
        if manifest.get("schema_version") != 1:
            raise InvalidRunData("run_manifest.json requires schema_version 1")
        if not isinstance(manifest.get("stages"), list) or not isinstance(manifest.get("metadata", {}), dict) or not isinstance(manifest.get("references", {}), dict):
            raise InvalidRunData("run_manifest.json requires stages list and metadata/references mappings")
        for stage in manifest["stages"]:
            required = ("stage_id", "name", "data_id", "run_dir", "events")
            if not isinstance(stage, dict) or any(not isinstance(stage.get(key), str) or not stage[key] for key in required):
                raise InvalidRunData(f"Each stage requires nonempty string fields: {required}")
        self.check("manifest_schema", 1, manifest.get("schema_version"))
        self.check("pipeline_completed", "completed", manifest.get("status"))
        stages = manifest.get("stages", [])
        self.check("stage_ids_unique", len(stages), len({s["stage_id"] for s in stages}))
        references = manifest.get("references", {})
        plan_ref = references.get("seed_plan") or references.get("pipeline_plan")
        plan = self.document(self.path(plan_ref)) if plan_ref else None
        if plan is None:
            self.note("planned_worker_coverage", "unknown", "No readable pipeline plan; launched stages alone cannot establish coverage")
        else:
            expected = []
            if "conditions" in plan:
                expected.append(self.path(plan["source_pre"]["run_dir"]))
                for condition in plan["conditions"]:
                    directory = self.path(condition["condition_dir"])
                    expected.append(directory / "habitat")
                    if condition.get("source_post_reused_from_source_pre"):
                        reuse = self.document(directory / "source_post/reuse_manifest.json")
                        if reuse is not None:
                            self.check("source_post_reuse_status", "reused", reuse.get("status"))
                            self.check("source_post_reuse_target", str(expected[0]), str(self.path(reuse.get("source_pre_reference"))))
                    else:
                        expected.append(directory / "source_post")
            else:
                expected = [self.path(s["run_dir"]) for s in plan.get("stages", []) if s.get("command")]
            self.check("planned_worker_coverage", sorted(map(str, expected)), sorted(str(self.path(s["run_dir"])) for s in stages))
        for entry in manifest.get("inputs", []):
            path = self.path(entry["snapshot"])
            if path.is_file():
                self.check("snapshot_hash", entry["sha256"], self.fingerprint(path)["sha256"], location=str(path))
            else:
                self.note("snapshot_exists", "fail", str(path))
        for stage in stages:
            self.worker(stage, manifest.get("metadata", {}))
        if plan and "conditions" in plan:
            for condition in plan["conditions"]:
                if condition.get("source_post_reused_from_source_pre"):
                    directory = self.path(condition["condition_dir"]) / "habitat"
                    matching = [s for s in stages if self.path(s["run_dir"]) == directory]
                    if matching:
                        cfg = self.document(self.path(matching[0]["config_snapshot"]), yaml_format=True)
                        if cfg is not None:
                            self.check("reuse_requires_frozen_condition", False, cfg.get("wrapper", {}).get("training_enabled"), stage=matching[0]["stage_id"])
        return self.report()

    def report(self):
        counts = dict(Counter(c["status"] for c in self.checks))
        status = "fail" if counts.get("fail") else "unknown" if counts.get("unknown") else "pass"
        return dict(schema_version=1, method_version=VERSION, generated_at=datetime.now(timezone.utc).isoformat(),
                    run_dir=str(self.root), status=status, counts=counts, path_maps=self.path_maps,
                    implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    event_reader_sha256=hashlib.sha256(Path(generate_metrics.__file__).read_bytes()).hexdigest(),
                    inputs=list(self.inputs.values()), checks=self.checks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, help="Default: RUN_DIR/reconciliation.json; must not exist")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    args = parser.parse_args(argv)
    mappings = []
    for value in args.path_map:
        old, separator, new = value.partition("=")
        if not separator or not old or not new:
            parser.error("--path-map requires nonempty OLD=NEW")
        mappings.append((old, new))
    output = args.output or args.run_dir / "reconciliation.json"
    if output.exists():
        parser.error(f"Output already exists: {output}; choose a new --output")
    try:
        report = Reconciliation(args.run_dir, mappings).run()
    except (InvalidRunData, KeyError, TypeError, AttributeError) as error:
        # This CLI boundary also rejects unsupported shapes in external artifacts.
        print(f"Reconciliation skipped: invalid output data ({error}). No report written.")
        return 3
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Reconciliation {report['status']}: {report['counts']}\n{output}")
    for check in report["checks"]:
        if check["status"] in ("fail", "unknown"):
            print(f"  {check['status']}: {check.get('stage') or 'pipeline'}: {check['check']} {check.get('reason', '')}")
    return {"pass": 0, "fail": 1, "unknown": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
