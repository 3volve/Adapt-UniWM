"""A standalone, single-writer event collector with no experiment dependencies.

Events accumulate in memory until next_step() or finish(). Components supply
JSON-serializable dictionaries; only the top-level `_event` key is reserved.
The runner and pipeline coordinator each own a separate collector.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _merge(target: dict[str, Any], incoming: dict[str, Any], prefix: str = "") -> None:
    """Merge into an owned candidate; a conflicting shape is a programmer error."""
    for key, value in incoming.items():
        location = f"{prefix}.{key}" if prefix else key
        if key in target:
            old_is_dict = isinstance(target[key], dict)
            new_is_dict = isinstance(value, dict)
            assert old_is_dict == new_is_dict, (
                f"EventLogger.feed: dictionary/value conflict at {location!r} "
                f"({type(target[key]).__name__} -> {type(value).__name__})"
            )
            if new_is_dict:
                _merge(target[key], value, location)
                continue
        target[key] = deepcopy(value)


def _with_fields(event: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    assert "_event" not in fields, "EventLogger: '_event' is reserved for logger metadata"
    candidate = deepcopy(event)
    _merge(candidate, fields)
    return candidate


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventLogger:
    """One current event and one JSONL file per worker; no runtime read interface.

    Construction opens a startup event. next_step() writes the previous event
    without changing its outcome, then opens a step event. finish() writes the
    final event and a run-summary event. All outcomes default to 'unfinished'.

    feed() recursively merges dictionaries, replaces scalar/list values, and
    copies incoming data. The caller owns field meanings and final outcomes.
    Shape conflicts leave the current event unchanged.

    Use as a context manager to finish on normal exit or a caught exception.
    Normal exit does not infer success. Exceptional exit records the exception
    under `_event.exception`, sets failed/interrupted, finishes, and propagates it.
    A process kill can lose the current in-memory event.
    """

    def __init__(self, path: str | Path, startup: dict[str, Any] | None = None):
        self.path = Path(path)
        self._closed = False
        self._current = self._new_event("startup", 0, _now(), startup)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A new invocation must not overwrite or append to an earlier run.
        with self.path.open("x", encoding="utf-8"):
            pass

    @staticmethod
    def _new_event(kind: str, event_id: int, started_at: str,
                   fields: dict[str, Any] | None) -> dict[str, Any]:
        event = {
            "_event": {"schema_version": 1, "id": event_id, "kind": kind,
                       "started_at": started_at, "finished_at": None},
            "outcome": "unfinished",
        }
        return event if fields is None else _with_fields(event, fields)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("EventLogger is finished; no further events or feeds are allowed")

    @staticmethod
    def _encode(event: dict[str, Any], finished_at: str) -> str:
        frozen = {**event, "_event": {**event["_event"], "finished_at": finished_at}}
        return json.dumps(frozen, ensure_ascii=False) + "\n"

    def _append(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text)

    def feed(self, fields: dict[str, Any]) -> None:
        """Contribute a dictionary to the current event; do not write a record yet."""
        self._require_open()
        self._current = _with_fields(self._current, fields)

    def next_step(self, fields: dict[str, Any] | None = None) -> None:
        """Freeze/write the current event, then open a new unfinished step."""
        self._require_open()
        boundary = _now()
        following = self._new_event("step", self._current["_event"]["id"] + 1, boundary, fields)
        self._append(self._encode(self._current, boundary))
        self._current = following

    def finish(self, run_details: dict[str, Any] | None = None) -> None:
        """Write the last event and run summary; summary fields do not alter the step."""
        self._require_open()
        boundary = _now()
        summary = self._new_event("run_summary", self._current["_event"]["id"] + 1, boundary, run_details)
        # Serialize both first so unsupported summary values cannot leave a
        # written final step that would be duplicated by a corrected finish().
        text = self._encode(self._current, boundary) + self._encode(summary, boundary)
        self._append(text)
        self._closed = True

    def __enter__(self) -> EventLogger:
        self._require_open()
        return self

    def __exit__(self, exc_type, error, traceback) -> bool:
        if not self._closed:
            if error is None:
                self.finish()
            else:
                outcome = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                self._current["outcome"] = outcome
                self._current["_event"]["exception"] = {
                    "type": type(error).__name__, "message": str(error),
                }
                self.finish({"outcome": outcome})
        return False
