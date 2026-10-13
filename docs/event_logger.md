# EventLogger

[`runtime_scripts/event_logger.py`](../runtime_scripts/event_logger.py) collects
JSON-serializable dictionaries into one process-owned `events.jsonl`. It uses
only the Python standard library. The runner and coordinator each own a separate
logger. [Metrics](metric_generation.md) and [reconciliation](reconciliation.md)
consume saved evidence; experiment execution uses its own runtime state.

## Runtime integration

Components feed the current event; the runner alone advances worker event
boundaries. Each attempted transition has `transition.outcome`. Episode loading,
setup, completion, schedule saving, and checkpoint saving contribute separate
sections. A save failure can mark the event failed while its transition remains
completed. An unfinished section means its final outcome was not supplied.

Startup records capture model/environment metadata. Event collection is
independent of `log_every_step`. Images, checkpoints, actions, schedules, and
provenance snapshots remain separate artifacts linked from the run evidence.

## Collector example

This demonstrates the collector API; field names other than `_event` are supplied
by the caller. The runner's producers define the fields required by analysis.

```python
from runtime_scripts.event_logger import EventLogger

with EventLogger("output/example/events.jsonl", {"seed": 100}) as log:
    log.feed({"outcome": "completed"})  # Startup event.
    log.next_step({"episode_id": "ep-1", "step_idx": 0})
    log.feed({"training": {"state": "entered"}})
    log.feed({"training": {"state": "completed", "loss": 0.72}})
    log.feed({"outcome": "completed"})
    log.finish({"outcome": "completed"})
```

This writes startup, step, and run-summary records. Explicit `finish()` prevents
a second finish on context exit. Use a new output path for each invocation.

## Lifecycle and ownership

- Construction creates a new file and an in-memory startup event. Existing files
  are rejected, never truncated or appended to by another invocation.
- `feed(fields)` updates only the current in-memory event.
- `next_step(fields=None)` writes the previous event, then opens the next.
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
