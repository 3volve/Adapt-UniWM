"""Submit one C0-C5 seed batch, or execute one of its Slurm-owned operations."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from thesis_testing_tools import pipeline
from thesis_testing_tools.run_manifest import RunManifest, write_json, now, index_artifacts


def job_graph():
    jobs = {"prepare": [], "source_pre": ["prepare"]}
    for condition, _ in pipeline.CORE_CONDITIONS:
        jobs[f"{condition}/habitat"] = ["c0_frozen/habitat"] if condition in ("c3_aligned_replay", "c4_shuffled_replay") else ["prepare"]
        if condition != "c0_frozen":
            jobs[f"{condition}/source_post"] = [f"{condition}/habitat"]
    jobs["finalize"] = list(jobs)
    return jobs


def restore_run(value):
    fields = dict(value)
    for key in ("source_config", "habitat_base_config", "fixed_mean_calibration", "initial_checkpoint", "source_manifest", "seed_dir"):
        if fields[key] is not None:
            fields[key] = Path(fields[key])
    return pipeline.SeedRun(**fields)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def snapshot_code(root):
    """Freeze executable project sources; datasets/checkpoints stay at their shared paths."""
    destination = root / "slurm/code"
    files = [REPO_ROOT / "uniwm_episode_runner.py"]
    for name in ("thesis_testing_tools", "runtime_scripts", "scripts", "source_tools", "uniwm", "cfg"):
        files.extend(p for p in (REPO_ROOT / name).rglob("*") if p.is_file() and p.suffix in (".py", ".html", ".yaml", ".json", ".sh"))
    inventory = []
    for path in files:
        relative = path.relative_to(REPO_ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        inventory.append({"path": str(relative), "sha256": pipeline.sha256_file(target)})
    write_json(root / "slurm/code_inventory.json", {"files": inventory})
    return destination


def sbatch_command(name, parents, ids, root, code_root, resources):
    args = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--no-requeue", "--export=ALL",
            f"--job-name={root.name}-{name.replace('/', '-')}", f"--chdir={resources['repository_root']}",
            f"--output={root / 'slurm' / (name.replace('/', '-') + '-%j.out')}",
            f"--error={root / 'slurm' / (name.replace('/', '-') + '-%j.err')}",
            f"--cpus-per-task={resources['cpus']}", f"--mem={resources['memory']}",
            f"--time={resources['finalize_time'] if name == 'finalize' else resources['time']}"]
    if resources.get("account"):
        args.append(f"--account={resources['account']}")
    partition = resources.get("finalize_partition") if name == "finalize" else resources.get("partition")
    if partition:
        args.append(f"--partition={partition}")
    if name != "finalize":
        args.append(f"--gres={resources['gres']}")
    if name == "prepare":
        args.append("--hold")
    if parents:
        kind = "afterany" if name == "finalize" else "afterok"
        args += [f"--dependency={kind}:" + ":".join(ids[p] for p in parents), "--kill-on-invalid-dep=yes"]
    command = [sys.executable, str(code_root / "thesis_testing_tools/run_slurm.py"), "worker",
               "--run-dir", str(root), "--stage", name]
    args.append("--wrap=exec " + shlex.join(command))
    return args


def submit(run, counts, resources, *, dry_run=False, runner=subprocess.run):
    root = run.seed_dir
    graph = job_graph()
    code_root = root / "slurm/code"
    ids = {}
    preview = []
    for name, parents in graph.items():
        preview.append({"stage": name, "depends_on": parents,
                        "command": sbatch_command(name, parents, ids, root, code_root, resources)})
        ids[name] = f"JOB_{len(ids) + 1}"
    if dry_run:
        print(json.dumps({"run_dir": str(root), "source_selection": run.metadata["source_episode_order"],
                          "habitat_selection": run.habitat_ids, "source_counts": counts,
                          "checkpoint": str(run.initial_checkpoint), "seed": run.seed,
                          "schedule_producer": "c0_frozen/habitat", "jobs": preview}, indent=2))
        return
    (root / "slurm").mkdir(parents=True, exist_ok=False)
    code_root = snapshot_code(root)
    registry = {"schema_version": 1, "status": "submitting", "started_at": now(), "run": asdict(run),
                "source_counts": counts, "resources": resources, "jobs": {}}
    registry["input_origins"] = {}
    for key in ("source_config", "habitat_base_config", "source_manifest", "fixed_mean_calibration"):
        source = registry["run"][key]
        if source is not None:
            target = root / "slurm/inputs" / key / Path(source).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            registry["input_origins"][key] = {"path": str(source), "sha256": pipeline.sha256_file(target)}
            registry["run"][key] = str(target)
    path = root / "submission.json"
    write_json(path, registry)
    ids = {}
    try:
        for name, parents in graph.items():
            command = sbatch_command(name, parents, ids, root, code_root, resources)
            result = runner(command, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdecimal():
                raise ValueError(f"Unexpected sbatch job ID: {result.stdout!r}")
            ids[name] = job_id
            registry["jobs"][name] = {"job_id": job_id, "parents": parents, "submission_command": command}
            write_json(path, registry)
        registry["status"] = "submitted"
        write_json(path, registry)
        runner(["scontrol", "release", ids["prepare"]], check=True)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        registry.update(status="submission_failed", error=str(error))
        write_json(path, registry)
        print(f"Submission did not finish. Preparation has not been intentionally released; inspect {path} and the recorded job IDs.", flush=True)
        raise
    print(f"Submitted {len(ids)} jobs: {root}\n{subprocess.list2cmdline(['squeue', '-j', ','.join(ids.values())])}")


def distributed_stage(stage, code_root):
    stage = dict(stage)
    original = stage["command"]
    script_index = next(i for i, part in enumerate(original) if part.endswith("uniwm_episode_runner.py"))
    stage["command"] = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=1",
                        str(code_root / "uniwm_episode_runner.py"), *original[script_index + 1:]]
    return stage


def prepare(root, registry):
    run = restore_run(registry["run"])
    original_repo = Path(registry["resources"]["repository_root"])
    # Dataset/resource paths retain their original shared-filesystem meaning.
    pipeline.REPO_ROOT = original_repo
    import scripts.generate_habitat_action_sequence as action_generator
    action_generator.REPO_ROOT = original_repo
    with RunManifest(root, repo_root=REPO_ROOT, run_type="core_condition_seed_batch",
                     metadata=run.metadata, capture=pipeline._captured_command) as record:
        prepared = pipeline.prepare_seed_run(run, record)
        write_json(prepared.provenance_path, prepared.provenance)
        stages = {}
        for name in job_graph():
            if name in ("prepare", "finalize"):
                continue
            stage = distributed_stage(pipeline.seed_stage(prepared, name), REPO_ROOT)
            snapshot = record.snapshot(Path(stage["config_path"]), "stage_config")
            stage["command"][stage["command"].index("--config_path") + 1] = str(snapshot)
            stage.update(stage_id=name, events=name + "/events.jsonl", config_snapshot=str(snapshot.relative_to(root)))
            stages[name] = stage
        plan_path = root / "seed_manifest.json"
        plan = read_json(plan_path)
        plan["workload"] = prepared.workload
        plan["stages"] = stages
        plan["source_pre"] = stages["source_pre"]
        for condition in plan["conditions"]:
            cid = condition["condition_id"]
            condition["habitat_command"] = stages[cid + "/habitat"]["command"]
            if cid != "c0_frozen":
                condition["source_post_command"] = stages[cid + "/source_post"]["command"]
        write_json(plan_path, plan)
        for key, value in (("submission", root / "submission.json"), ("reconciliation", root / "reconciliation.json"),
                           ("run_report", root / "run_report.html"), ("run_report_metrics", root / "run_report_metrics")):
            record.reference(key, value)
        record.data["status"] = "prepared"


def execute(root, name, registry):
    stage = read_json(root / "seed_manifest.json")["stages"][name]
    status_path = root / name / "stage_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    if status_path.exists():
        raise FileExistsError(f"Stage already attempted: {status_path}")
    record = {**stage, "status": "running", "started_at": now(), "job_id": os.environ.get("SLURM_JOB_ID"),
              "hostname": os.environ.get("SLURMD_NODENAME"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    write_json(status_path, record)
    started = time.perf_counter()
    try:
        result = pipeline.execute_seed_stage(stage, repo_root=Path(registry["resources"]["repository_root"]))
        record.update(result)
        if stage["name"] == "habitat":
            record["artifacts"] = pipeline.collect_habitat_artifacts(stage)
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      failure={"type": type(error).__name__, "message": str(error), "returncode": getattr(error, "returncode", None)})
        raise
    finally:
        record.update(finished_at=now(), duration_seconds=time.perf_counter() - started,
                      events_available=(root / stage["events"]).is_file())
        write_json(status_path, record)


def scheduler_outcomes(registry):
    ids = [job["job_id"] for name, job in registry["jobs"].items() if name != "finalize"]
    command = ["sacct", "-n", "-P", "-X", "-j", ",".join(ids), "--format=JobIDRaw,State,ExitCode"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        return {}, result.stderr.strip()
    outcomes = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 3 and parts[0] in ids:
            outcomes[parts[0]] = {"state": parts[1], "exit_code": parts[2]}
    return outcomes, None


def finalize(root, registry, *, outcomes=None):
    if outcomes is None:
        try:
            outcomes, error = scheduler_outcomes(registry)
        except OSError as exc:
            outcomes, error = {}, str(exc)
    else:
        error = None
    if any(item["state"].split()[0] in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED") for item in outcomes.values()):
        raise RuntimeError("Experiment jobs are still active; finalization requires terminal jobs")
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists() and read_json(manifest_path).get("phase") == "finalized":
        raise FileExistsError(f"Run already finalized: {manifest_path}")
    manifest = read_json(manifest_path) if manifest_path.exists() else {
        "schema_version": 1, "run_id": root.name, "run_type": "core_condition_seed_batch", "metadata": registry["run"]["metadata"],
        "inputs": [], "references": {}, "stages": [], "artifact_index": "provenance/artifact_index.json"}
    plan_path = root / "seed_manifest.json"
    plan = read_json(plan_path) if plan_path.exists() else None
    stages = []
    for name, job in registry["jobs"].items():
        if name in ("prepare", "finalize"):
            continue
        status_path = root / name / "stage_status.json"
        stage = read_json(status_path) if status_path.exists() else dict(
            (plan or {}).get("stages", {}).get(name, {}), stage_id=name, name=name.split("/")[-1],
            data_id=pipeline.SOURCE_DATA_IDS if name.split("/")[-1] != "habitat" else pipeline.HABITAT_DATA_ID,
            run_dir=str(root / name), events=name + "/events.jsonl", status="not_started")
        scheduler = outcomes.get(job["job_id"], {})
        stage.update(job_id=job["job_id"], scheduler=scheduler, status_file=str(status_path.relative_to(root)))
        if scheduler and (scheduler["state"] != "COMPLETED" or scheduler["exit_code"] != "0:0"):
            stage["status"] = "failed" if status_path.exists() else "not_started"
        elif stage["status"] == "running":
            stage["status"] = "incomplete"
        stages.append(stage)
    manifest.update(stages=stages, phase="finalized", finished_at=now(),
                    status="completed" if plan and all(s["status"] == "completed" for s in stages) else "failed",
                    slurm={"outcomes": outcomes, "accounting_error": error})
    by_id = {stage["stage_id"]: stage for stage in stages}
    if plan:
        provenance = read_json(root / "provenance.json")
        for stage in stages:
            artifacts = stage.get("artifacts", {})
            if "output_checkpoint" in artifacts:
                provenance["output_checkpoints"].append(artifacts["output_checkpoint"])
            if "controller_schedule" in artifacts:
                provenance["inputs"].append(artifacts["controller_schedule"])
                manifest["references"]["controller_schedule"] = stage["stage_id"] + "/learning_rate_schedule.json"
        write_json(root / "provenance.json", provenance)
        run = restore_run(registry["run"])
        prepared = pipeline.PreparedSeedRun(run, plan["source_pre"], plan["conditions"], plan["workload"],
                                            provenance, root / "provenance.json", Path(plan["source_manifest"]))
        summaries = []
        for condition in plan["conditions"]:
            cid = condition["condition_id"]
            post = by_id.get(cid + "/source_post")
            if cid == "c0_frozen":
                if by_id["source_pre"]["status"] == by_id[cid + "/habitat"]["status"] == "completed":
                    post = pipeline.record_source_pre_reuse(root / "source_pre", root / cid / "source_post")
                else:
                    post = {"name": "source_post", "status": "unavailable", "reason": "C0 or source-pre did not complete"}
            condition_records = [by_id["source_pre"], by_id[cid + "/habitat"], post]
            (root / cid).mkdir(exist_ok=True)
            summary = pipeline.write_condition_summary(prepared, condition, condition_records)
            summary["status"] = "completed" if all(s["status"] in ("completed", "reused") for s in condition_records) else "failed"
            write_json(root / cid / "pipeline_summary.json", summary)
            summaries.append(summary)
        summary = pipeline.finalize_seed_run(prepared, by_id["source_pre"], summaries, duration_seconds=
                                            (datetime.fromisoformat(now()) - datetime.fromisoformat(registry["started_at"])).total_seconds())
        summary["status"] = manifest["status"]
        write_json(root / "seed_summary.json", summary)
        manifest["references"]["summary"] = "seed_summary.json"
    manifest["references"].update(submission="submission.json", reconciliation="reconciliation.json", run_report="run_report.html")
    (root / "provenance").mkdir(exist_ok=True)
    artifacts = index_artifacts(root)
    manifest["artifact_count"] = len(artifacts)
    write_json(manifest_path, manifest)
    write_json(root / "provenance/artifact_index.json", {"updated_at": now(), "files": artifacts})
    pipeline.generate_run_outputs(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    launch = sub.add_parser("submit")
    pipeline.add_experiment_arguments(launch)
    launch.add_argument("--partition")
    launch.add_argument("--finalize-partition")
    launch.add_argument("--account")
    launch.add_argument("--gres", default="gpu:1")
    launch.add_argument("--cpus", type=int, default=4)
    launch.add_argument("--memory", default="32G")
    launch.add_argument("--time", default="12:00:00")
    launch.add_argument("--finalize-time", default="02:00:00")
    launch.add_argument("--dry-run", action="store_true")
    worker = sub.add_parser("worker")
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--stage", choices=list(job_graph()), required=True)
    args = parser.parse_args(argv)
    if args.operation == "worker":
        root = args.run_dir.resolve()
        registry = read_json(root / "submission.json")
        if args.stage == "prepare":
            prepare(root, registry)
        elif args.stage == "finalize":
            finalize(root, registry)
        else:
            execute(root, args.stage, registry)
        return
    if args.cpus <= 0:
        parser.error("cpus must be positive")
    if not args.initial_checkpoint.exists():
        parser.error(f"Checkpoint does not exist: {args.initial_checkpoint}")
    if args.fixed_mean_calibration is not None and not args.fixed_mean_calibration.is_file():
        parser.error(f"Calibration file does not exist: {args.fixed_mean_calibration}")
    if not args.output_root.resolve().is_relative_to(REPO_ROOT):
        parser.error("output-root must be inside the repository for the Habitat action generator")
    run = pipeline.resolve_seed_run(**pipeline.experiment_arguments(args))
    resources = {key: getattr(args, key) for key in ("partition", "finalize_partition", "account", "gres", "cpus", "memory", "time", "finalize_time")}
    resources["repository_root"] = str(REPO_ROOT)
    submit(run, run.source_episode_counts, resources, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
