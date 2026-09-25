"""Sequential entry point for the shared experiment pipeline."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from thesis_testing_tools import pipeline

@contextmanager
def reconciled_run(root, **kwargs):
    """Finalize execution evidence before running the independent offline check."""
    record = None
    try:
        with pipeline.RunManifest(root, **kwargs) as record:
            record.reference("reconciliation", record.root / "reconciliation.json")
            record.reference("run_report", record.root / "run_report.html")
            record.reference("run_report_metrics", record.root / "run_report_metrics")
            yield record
    finally:
        if record is not None:
            pipeline.generate_run_outputs(record.root)


def run_seed_batch(
    *,
    seed: int,
    fixed_mean_lr: float,
    source_config: Path = pipeline.REPLAY_CONFIG,
    habitat_base_config: Path = pipeline.HABITAT_CONFIG,
    fixed_mean_calibration: Path | None = None,
    initial_checkpoint: Path = pipeline.BASE_CHECKPOINT,
    source_manifest: Path = pipeline.DEVELOPMENT_MANIFEST,
    schedule_shuffle_seed: int = 20260827,
    source_episodes: int | None = None,
    habitat_episodes: int | None = None,
    max_episode_steps: int | None = None,
    source_max_episode_steps: int | None = None,
    habitat_max_episode_steps: int | None = None,
    max_route_steps: int | None = None,
    smoke_test: bool = False,
    output_root: Path = pipeline.DEFAULT_OUTPUT_ROOT,
    timestamp: str | None = None,
    subprocess_runner: pipeline.SubprocessRunner | None = None,
    metric_calculator: pipeline.MetricCalculator | None = None,
) -> Path:
    """Run the shared C0-C5 experiment sequentially."""
    seed_started = time.perf_counter()
    run = pipeline.resolve_seed_run(
        seed=seed,
        fixed_mean_lr=fixed_mean_lr,
        source_config=source_config,
        habitat_base_config=habitat_base_config,
        fixed_mean_calibration=fixed_mean_calibration,
        initial_checkpoint=initial_checkpoint,
        source_manifest=source_manifest,
        schedule_shuffle_seed=schedule_shuffle_seed,
        source_episodes=source_episodes,
        habitat_episodes=habitat_episodes,
        max_episode_steps=max_episode_steps,
        source_max_episode_steps=source_max_episode_steps,
        habitat_max_episode_steps=habitat_max_episode_steps,
        max_route_steps=max_route_steps,
        smoke_test=smoke_test,
        output_root=output_root,
        timestamp=timestamp,
    )
    run.seed_dir.mkdir(parents=True, exist_ok=False)
    with reconciled_run(
        run.seed_dir, repo_root=pipeline.REPO_ROOT, run_type="core_condition_seed_batch",
        metadata=dict(run.metadata), capture=pipeline._captured_command,
    ) as run_record:
        prepared = pipeline.prepare_seed_run(run, run_record)
        source_pre_record = pipeline.execute_seed_stage(prepared.source_pre, subprocess_runner, run_manifest=run_record)
        condition_summaries = []
        for plan in prepared.conditions:
            condition_id = plan["condition_id"]
            habitat_stage = pipeline.seed_stage(prepared, f"{condition_id}/habitat")
            habitat_record = pipeline.execute_seed_stage(habitat_stage, subprocess_runner, run_manifest=run_record)
            pipeline.record_habitat_artifacts(prepared, habitat_stage, run_record)
            if plan["source_post_command"] is None:
                post_record = pipeline.record_source_pre_reuse(Path(prepared.source_pre["run_dir"]), Path(plan["source_post_dir"]))
            else:
                post_record = pipeline.execute_seed_stage(
                    pipeline.seed_stage(prepared, f"{condition_id}/source_post"),
                    subprocess_runner, run_manifest=run_record,
                )
            stage_records = [
                {"name": "source_pre", "status": "reused", "run_dir": prepared.source_pre["run_dir"]},
                habitat_record, post_record,
            ]
            condition_summaries.append(pipeline.write_condition_summary(prepared, plan, stage_records))
        summary = pipeline.finalize_seed_run(
            prepared, source_pre_record, condition_summaries,
            duration_seconds=time.perf_counter() - seed_started,
        )
        run_record.reference("summary", run.seed_dir / "seed_summary.json")
        run_record.data["status"] = summary["status"]
        run_record.data["metadata"]["workload"] = prepared.workload
        return run.seed_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one C0-C5 seed batch sequentially.")
    pipeline.add_experiment_arguments(parser)
    args = parser.parse_args(argv)
    result_dir = run_seed_batch(**pipeline.experiment_arguments(args))
    print(f"[PIPELINE] Finished: {result_dir}")


if __name__ == "__main__":
    main()
