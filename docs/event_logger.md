# EventLogger

`runtime_scripts/event_logger.py` implements the shared collector used by the
runner and pipeline coordinator. Existing writer implementations remain in the
repository but are disconnected. Analysis readers have not been migrated, and
automatic metric generation is disabled. No reconciliation checks were added.

## Runtime integration

The runner passes one collector through component construction. Only the runner
advances events. Each attempted transition has `transition.outcome`; setup,
episode completion, schedule saving and checkpoint saving contribute their own
top-level sections, keyed by episode or source where they may repeat. A save
failure marks the overall event failed while preserving a completed transition.
An unfinished nested section means its final outcome was never supplied.

The coordinator writes its own `events.jsonl`; `run_manifest.json` indexes worker
event files. Static provenance snapshots and operational schedules, actions,
images and checkpoints remain separate artifacts. Runtime execution never reads
the collector or its output.

`log_every_step` no longer suppresses event collection. Legacy `episode_logs.json`,
`runner_events.jsonl`, `transition_events.jsonl`, per-step/per-episode CSVs and
runtime-metadata JSON files are no longer produced. Before-model metadata and
changed after-model observations are in the startup record. Old metric helpers
are temporarily incompatible; pipeline summaries explicitly mark analysis disabled.
Smoke workloads still execute, but metric-based coverage is `not_evaluated`.

Past-run action replay and `--habitat-action-run` are retired. The all-conditions
pipeline generates standalone action sequences and supplies `fixed_action_files_dir`.

The module uses only the Python standard library. Each worker owns one logger
and one new JSONL file. There is no cross-worker shared event or combined output.

## Usage

```python
from runtime_scripts.event_logger import EventLogger

with EventLogger("output/example/events.jsonl", {"run_id": "example", "seed": 100}) as log:
    # The constructor has already opened a startup event.
    log.feed({"outcome": "completed"})

    log.next_step({"episode_id": "ep-1", "step_idx": 0})
    log.feed({"training": {"state": "entered"}})
    log.feed({"training": {"state": "completed", "loss": 0.72, "optimizer_step": True}})
    log.feed({"outcome": "completed"})

    log.next_step({"episode_id": "ep-1", "step_idx": 1})
    log.feed({"action": {"converted": []}, "outcome": "no_op"})

    # Final global details go in their own summary, not into the last step.
    log.finish({"outcome": "completed", "checkpoint": "output/example/final_ckpt"})
```

This produces four lines: startup, step 0, step 1, and run summary. The context
manager does not finish a second time after an explicit `finish()`.

## Lifecycle and ownership

- Construction creates a new file and an in-memory startup event. Existing files
  are rejected, never truncated or appended to by another invocation.
- `feed(dictionary)` updates only the current in-memory event.
- `next_step(dictionary=None)` writes the previous event, then opens the next.
  Its optional dictionary supplies identifiers or other initial fields.
- `finish(run_details=None)` writes the last event and a run-summary record.
  Further calls to `feed`, `next_step`, or `finish` raise `RuntimeError`.
- Every record defaults to `outcome: "unfinished"`, including startup and summary.
  Neither advancing nor normal context exit infers successful completion.
  Set event outcomes and the overall summary outcome explicitly when known.
- An exception escaping the context marks the current event and summary `failed`
  (`interrupted` for `KeyboardInterrupt`), records its type/message, finishes, and
  re-raises it. Component progress fields remain intact. Errors caught inside the
  context are the caller's responsibility to record.

The top-level `_event` dictionary is reserved for the logger: schema version,
sequential ID, kind, opening/closing timestamps, and any escaping exception.
IDs are local to the file; identify a record using its worker file and ID.
Timestamps use UTC. Run-summary timestamps describe writing that summary;
the startup record carries the beginning of the run.

All other field meanings belong to producers. The collector has no public
interface for reading the current event, and experiment execution must use its
own runtime objects. Their `to_dict()` results can be fed into the collector.

## Merge rules

- Dictionaries merge recursively, preserving untouched sibling fields.
- New scalar values overwrite earlier scalar values, including `None`.
- Lists replace earlier lists; they are copied, not concatenated.
- A dictionary/non-dictionary conflict at an existing path raises `AssertionError`
  with the field path. Feeding reserved `_event` metadata also raises an assertion.
  These indicate producer programming errors, not data-repair opportunities.
- Contributions are deep-copied. Mutating an input dictionary, list, or nested
  value after feeding it cannot alter the event.
- A merge conflict leaves the entire prior event unchanged, including fields
  visited earlier in the rejected contribution.

Producers should feed an entry marker before any early return if section absence
is intended to mean that component was not reached. The logger does not invent
missing sections or interpret absence as false. Shared roots are allowed; the
producers remain responsible for consistent field meanings and ownership.

## Persistence limits

Only finalized events are written. Advancing or finishing serializes and appends
records, then closes the file handle. Later feeds cannot modify written records.
Unsupported objects raise the normal JSON serialization error; the collector
does not silently stringify tensors, images, or arbitrary runtime objects.
Producers should supply serializable snapshots.

Use the context manager or an explicit outer finalization path. Process kills or
node loss can lose the current in-memory event, and filesystem errors propagate.
There is no recovery, replay, retry, or cross-process synchronization machinery.

Run the standalone tests with:

```bash
python -m unittest runtime_scripts.test_event_logger
```
