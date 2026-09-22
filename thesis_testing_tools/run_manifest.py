"""Run-local provenance and lifecycle records; no experiment policy lives here."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback


ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "NCCL_P2P_DISABLE",
    "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
    "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST",
)


def now():
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value):
    """Replace a record atomically so readers never see half-written JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_fingerprint(path: Path):
    path = Path(path).resolve()
    if path.is_file():
        return {"path": str(path), "type": "file", "bytes": path.stat().st_size,
                "sha256": sha256_file(path)}
    if not path.is_dir():
        return {"path": str(path), "type": "missing"}
    files = []
    tree_digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        size, digest = file_path.stat().st_size, sha256_file(file_path)
        files.append({"relative_path": relative, "bytes": size, "sha256": digest})
        tree_digest.update(f"{relative}\0{size}\0{digest}\n".encode())
    return {"path": str(path), "type": "directory", "bytes": sum(f["bytes"] for f in files),
            "file_count": len(files), "tree_sha256": tree_digest.hexdigest(), "files": files}


def index_artifacts(root: Path):
    """Index files without rehashing checkpoints or embedding thousands of images."""
    return [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size}
            for p in sorted(root.rglob("*")) if p.is_file()
            and p not in (root / "run_manifest.json", root / "provenance/artifact_index.json")
            and not p.name.endswith(".tmp")]


class RunManifest:
    """One owner writes this record; concurrent jobs will need separate owners."""

    def __init__(self, root: Path, *, repo_root: Path, run_type: str, metadata: dict, capture):
        self.root = root.resolve()
        self.repo_root = repo_root.resolve()
        self.capture = capture
        self.started = time.perf_counter()
        self.data = {
            "schema_version": 1, "run_id": self.root.name, "run_type": run_type,
            "command": list(sys.argv),
            "status": "running", "phase": "preparation", "started_at": now(),
            "finished_at": None, "metadata": metadata, "inputs": [], "stages": [],
            "references": {}, "environment": "provenance/environment.json",
            "artifact_index": "provenance/artifact_index.json", "failure": None,
        }

    def save(self):
        write_json(self.root / "run_manifest.json", self.data)

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        # Never silently replace the audit record for an earlier invocation.
        with (self.root / "run_manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2, default=str)
        (self.root / "provenance").mkdir()
        from runtime_scripts.event_logger import EventLogger
        self.event_logger = EventLogger(self.root / "events.jsonl", {
            "pipeline": {"run_type": self.data["run_type"], "run_id": self.data["run_id"]}})
        self.data["references"]["events"] = "events.jsonl"
        return self

    def capture_environment(self, overrides: dict):
        environment = {
            "recorded_at": now(), "hostname": platform.node(), "platform": platform.platform(),
            "python": sys.version, "python_executable": sys.executable,
            "cpu_count": os.cpu_count(), "machine": platform.machine(),
            "packages": sorted([
                {"name": d.metadata["Name"], "version": d.version}
                for d in importlib.metadata.distributions()
            ], key=lambda d: (d["name"] or "", d["version"])),
            "environment_variables": {k: os.environ.get(k) for k in ENVIRONMENT_KEYS},
            "stage_environment_overrides": overrides,
            "gpu_query": self.capture(["nvidia-smi"]),
            "pip_freeze": self.capture([sys.executable, "-m", "pip", "freeze"]),
        }
        write_json(self.root / "provenance/environment.json", environment)
        self.data["git"] = {
            "commit": self.capture(["git", "rev-parse", "HEAD"]),
            "branch": self.capture(["git", "branch", "--show-current"]),
            "status": self.capture(["git", "status", "--short"]),
        }
        diff = self.capture(["git", "diff", "--binary", "HEAD"])
        write_json(self.root / "provenance/working_tree_diff.json", diff)
        self.data["git"]["working_tree_diff"] = "provenance/working_tree_diff.json"
        self.save()

    def snapshot(self, source: Path, role: str) -> Path:
        source = Path(source).resolve()
        if source.is_relative_to(self.root / "provenance/snapshots"):
            return source
        for entry in self.data["inputs"]:
            if entry["original_path"] == str(source):
                return self.root / entry["snapshot"]
        if source.is_relative_to(self.repo_root):
            relative = source.relative_to(self.repo_root)
        else:
            relative = Path("external") / hashlib.sha256(str(source).encode()).hexdigest()[:12] / source.name
        destination = self.root / "provenance/snapshots" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        fingerprint = artifact_fingerprint(destination)
        self.data["inputs"].append({"role": role, "original_path": str(source),
                                    "snapshot": destination.relative_to(self.root).as_posix(),
                                    "sha256": fingerprint["sha256"], "bytes": fingerprint["bytes"]})
        self.save()
        return destination

    def snapshot_code(self):
        paths = [self.repo_root / "uniwm_episode_runner.py"]
        for directory in ("thesis_testing_tools", "runtime_scripts", "scripts", "source_tools", "uniwm"):
            paths.extend(sorted((self.repo_root / directory).glob("*.py")))
        paths.extend(sorted((self.repo_root / "thesis_testing_tools").glob("*.sh")))
        for path in paths:
            self.snapshot(path, "implementation")

    def reference(self, name: str, path: Path):
        path = path.resolve()
        self.data["references"][name] = (path.relative_to(self.root).as_posix()
                                          if path.is_relative_to(self.root) else str(path))
        self.save()

    @contextmanager
    def stage(self, name, command, run_dir, config_path, data_id):
        config_snapshot = self.snapshot(config_path, "stage_config")
        stage_id = (run_dir.resolve().relative_to(self.root).as_posix()
                    if run_dir.resolve().is_relative_to(self.root) else str(run_dir.resolve()))
        record = {"name": name, "stage_id": stage_id, "status": "running", "data_id": data_id,
                  "run_dir": str(run_dir.resolve()), "config_path": str(config_path),
                  "config_snapshot": config_snapshot.relative_to(self.root).as_posix(),
                  "command": list(command), "started_at": now(), "finished_at": None,
                  "events": stage_id + "/events.jsonl", "event_schema_version": 1}
        self.data["stages"].append(record)
        self.event_logger.feed({"outcome": "completed"})
        self.event_logger.next_step({"stage": {"id": stage_id, "name": name, "outcome": "unfinished",
                                              "events": record["events"]}})
        self.data["phase"] = name
        self.save()
        started = time.perf_counter()
        try:
            yield record
        except BaseException as error:
            record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          failure={"type": type(error).__name__, "message": str(error),
                                   "returncode": getattr(error, "returncode", None)})
            raise
        else:
            record["status"] = "completed"
        finally:
            record.update(finished_at=now(), duration_seconds=time.perf_counter() - started)
            record["events_available"] = (run_dir / "events.jsonl").is_file()
            self.event_logger.feed({"outcome": record["status"], "stage": {
                "outcome": record["status"], "duration_seconds": record["duration_seconds"],
                "events_available": record["events_available"]}})
            self.data["phase"] = "stage_finished" if record["status"] == "completed" else name
            self.save()

    def __exit__(self, exc_type, error, tb):
        if error is not None:
            self.data["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            self.data["failure"] = {"type": type(error).__name__, "message": str(error),
                                    "phase": self.data["phase"], "returncode": getattr(error, "returncode", None),
                                    "traceback": "provenance/failure.txt"}
            (self.root / "provenance/failure.txt").write_text(
                "".join(traceback.format_exception(exc_type, error, tb)), encoding="utf-8")
        elif self.data["status"] == "running":
            self.data["status"] = "completed"
        self.data.update(finished_at=now(), duration_seconds=time.perf_counter() - self.started)
        if error is not None:
            self.event_logger.feed({"outcome": self.data["status"], "failure": {
                "type": type(error).__name__, "message": str(error), "phase": self.data["phase"]}})
        else:
            self.event_logger.feed({"outcome": "completed"})
        self.event_logger.finish({"outcome": self.data["status"],
                                  "manifest": "run_manifest.json"})
        self.save()
        artifacts = index_artifacts(self.root)
        write_json(self.root / self.data["artifact_index"], {"updated_at": now(), "files": artifacts})
        self.data["artifact_count"] = len(artifacts)
        self.save()
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Refresh a run's artifact index after generating tables or figures.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    artifacts = index_artifacts(root)
    write_json(root / manifest["artifact_index"], {"updated_at": now(), "files": artifacts})
    manifest["artifact_count"] = len(artifacts)
    write_json(root / "run_manifest.json", manifest)
