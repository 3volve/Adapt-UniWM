"""Analytic image fixtures exercise the complete offline MAE reporting path."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

from PIL import Image

from runtime_scripts.event_logger import EventLogger
from thesis_testing_tools.generate_metrics import generate, read_events, sha256


class GenerateMetricsTests(unittest.TestCase):
    def test_spatial_variation_is_within_channels(self):
        path = self.stage("variance", [[0]])
        Image.new("RGB", (16, 16), (255, 0, 0)).save(path.parent / "ep-0-0-pred.png")
        output = self.root / "variance_metrics"
        generate([path], output, metrics=["mae"])
        row = self.rows(output, "step_metrics.csv")[0]
        self.assertEqual(float(row["prediction_spatial_std"]), 0)
        self.assertAlmostEqual(float(row["mae"]), 1 / 3)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def stage(self, name, episodes, *, mode="habitat", no_op=False, failed_save=False):
        folder = self.root / name
        folder.mkdir()
        path = folder / "events.jsonl"
        with EventLogger(path) as log:
            log.feed({"outcome": "completed"})
            for ep_index, values in enumerate(episodes):
                ep = f"ep-{ep_index}"
                log.feed({"episode_setup": {f"dataset/{ep_index}": {"outcome": "completed", "episode_id": ep}}})
                for index, value in enumerate(values):
                    pred = folder / f"{ep}-{index}-pred.png"
                    real = folder / f"{ep}-{index}-real.png"
                    Image.new("RGB", (16, 16), (value,) * 3).save(pred)
                    Image.new("RGB", (16, 16), (0, 0, 0)).save(real)
                    wrapper = {"real_obs_path": str(real), "collision": False, "replanned": False}
                    if mode == "habitat":
                        wrapper["predicted_obs_path"] = str(pred)
                    else:
                        wrapper["evaluation"] = {"predicted_obs_path": str(pred)}
                    log.next_step({"data_id": "dataset", "episode_id": ep, "episode_index": ep_index,
                        "step_idx": index, "source_mode": mode, "transition": {"outcome": "completed"},
                        "wrapper": wrapper, "outcome": "completed"})
                if no_op:
                    log.next_step({"data_id": "dataset", "episode_id": ep, "episode_index": ep_index,
                        "step_idx": len(values), "source_mode": mode, "transition": {"outcome": "no_op"}, "outcome": "no_op"})
                log.feed({"episode_end": {f"dataset/{ep_index}": {"episode_id": ep, "termination_reason": "adapter_done"}}})
            if failed_save:
                log.feed({"checkpoint": {"outcome": "unfinished"}, "outcome": "failed"})
            log.finish({"outcome": "failed" if failed_save else "completed"})
        return path

    def rows(self, report, filename):
        with (report / filename).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def rewrite(self, path, change):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        change(records)
        path.write_text("".join(json.dumps(r) + "\n" for r in records))

    def test_analytic_means_counts_hashes_and_repeatability(self):
        path = self.stage("stage", [[0, 255], [255]], no_op=True)
        first, second = self.root / "report-1", self.root / "report-2"
        manifest = generate([path], first, metrics=["mae"])
        generate([path], second, metrics=["mae"])
        episodes = self.rows(first, "episode_metrics.csv")
        self.assertEqual([float(e["mae"]) for e in episodes], [0.5, 1.0])
        self.assertEqual([int(e["no_op_count"]) for e in episodes], [1, 1])
        stage = self.rows(first, "stage_metrics.csv")[0]
        self.assertEqual(float(stage["mae_episode_mean"]), 0.75)
        self.assertAlmostEqual(float(stage["mae_transition_mean"]), 2 / 3)
        self.assertEqual(manifest["pair_status_counts"], {"scored": 3, "transition_not_completed": 2})
        self.assertEqual(manifest["event_inputs"][0]["sha256"], sha256(path.read_bytes()))
        for item in manifest["image_inputs"]:
            self.assertEqual(item["sha256"], sha256(Path(item["path"]).read_bytes()))
        for item in manifest["outputs"]:
            self.assertEqual(item["sha256"], sha256((first / item["path"]).read_bytes()))
            self.assertEqual((first / item["path"]).read_bytes(), (second / item["path"]).read_bytes())

    def test_late_failure_does_not_discard_completed_transition(self):
        path = self.stage("stage", [[255]], failed_save=True)
        report = self.root / "report"
        manifest = generate([path], report, metrics=["mae"])
        self.assertEqual(manifest["event_inputs"][0]["outcome"], "failed")
        row = self.rows(report, "episode_metrics.csv")[0]
        self.assertEqual(row["failed_event_count"], "1")
        self.assertEqual(row["mae"], "1.0")

    def test_missing_images_are_excluded_not_zero_filled(self):
        path = self.stage("stage", [[255, 0]])
        (path.parent / "ep-0-1-pred.png").unlink()
        report = self.root / "report"
        generate([path], report, metrics=["mae"])
        row = self.rows(report, "episode_metrics.csv")[0]
        self.assertEqual(row["mae"], "1.0")
        self.assertEqual(row["scored_pair_count"], "1")
        self.assertEqual(self.rows(report, "step_metrics.csv")[1]["pair_status"], "image_file_missing")

    def test_path_remapping_and_shape_exclusion(self):
        path = self.stage("stage", [[255]])
        def change(records):
            records[1]["wrapper"]["predicted_obs_path"] = "/cluster/run/ep-0-0-pred.png"
        self.rewrite(path, change)
        report = self.root / "mapped"
        generate([path], report, metrics=["mae"], path_maps=[("/cluster/run", str(path.parent))])
        self.assertEqual(self.rows(report, "step_metrics.csv")[0]["mae"], "1.0")
        Image.new("RGB", (10, 10)).save(path.parent / "ep-0-0-real.png")
        report = self.root / "mismatch"
        generate([path], report, metrics=["mae"], path_maps=[("/cluster/run", str(path.parent))])
        self.assertEqual(self.rows(report, "step_metrics.csv")[0]["pair_status"], "image_shape_mismatch")

    def test_retention_pairs_only_identical_target_coverage(self):
        pre = self.stage("pre", [[0], [0]], mode="replay")
        post = self.stage("post", [[255], [255]], mode="replay")
        Image.new("RGB", (16, 16), (20, 20, 20)).save(post.parent / "ep-1-0-real.png")
        report = self.root / "report"
        generate([], report, metrics=["mae"], source_pre=pre, source_posts=[post])
        rows = self.rows(report, "source_comparison.csv")
        self.assertEqual(rows[0]["mae_post_minus_pre"], "1.0")
        self.assertEqual(rows[1]["status"], "image_coverage_mismatch")
        self.assertEqual(rows[1]["mae_post_minus_pre"], "")
        summary = self.rows(report, "source_retention.csv")[0]
        self.assertEqual(summary["paired_episode_count"], "1")
        self.assertEqual(summary["excluded_episode_count"], "1")

    def test_incomplete_and_invalid_streams_are_explicit(self):
        path = self.stage("stage", [[255]])
        self.rewrite(path, lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "missing run_summary"):
            read_events(path)
        _, evidence = read_events(path, allow_incomplete=True)
        self.assertFalse(evidence["finalized"])
        with path.open("a") as handle:
            handle.write('{"truncated":')
        with self.assertRaisesRegex(ValueError, "invalid worker event"):
            read_events(path, allow_incomplete=True)

    def test_existing_report_and_coordinator_are_rejected(self):
        path = self.stage("stage", [[0]])
        report = self.root / "report"
        generate([path], report, metrics=["mae"])
        with self.assertRaises(FileExistsError):
            generate([path], report, metrics=["mae"])
        self.rewrite(path, lambda rows: rows[0].update(pipeline={"run_type": "batch"}))
        with self.assertRaisesRegex(ValueError, "coordinator"):
            read_events(path)

    def test_malformed_nested_evidence_and_duplicate_steps_are_rejected(self):
        path = self.stage("malformed", [[0]])
        self.rewrite(path, lambda rows: rows[1]["wrapper"].update(collision="false"))
        with self.assertRaisesRegex(ValueError, "wrapper.collision must be boolean"):
            read_events(path)
        path = self.stage("duplicate", [[0, 0]])
        self.rewrite(path, lambda rows: rows[2].update(step_idx=0))
        with self.assertRaisesRegex(ValueError, "duplicate step index"):
            generate([path], self.root / "duplicate-report", metrics=["mae"])

    def test_incomplete_episode_is_not_in_stage_mean(self):
        path = self.stage("stage", [[255]])
        self.rewrite(path, lambda rows: rows[1].pop("episode_end"))
        report = self.root / "report"
        generate([path], report, metrics=["mae"])
        self.assertEqual(self.rows(report, "episode_metrics.csv")[0]["mae"], "1.0")
        self.assertEqual(self.rows(report, "stage_metrics.csv")[0]["mae_episode_mean"], "")

    def test_empty_failed_run_produces_headers_and_failure_provenance(self):
        path = self.root / "events.jsonl"
        with EventLogger(path) as log:
            log.feed({"outcome": "failed"})
            log.finish({"outcome": "failed"})
        report = self.root / "report"
        manifest = generate([path], report, metrics=["mae"])
        self.assertEqual(manifest["event_inputs"][0]["outcome"], "failed")
        self.assertEqual(self.rows(report, "step_metrics.csv"), [])
        self.assertTrue((report / "step_metrics.csv").read_text().startswith("event_file,event_id"))

    def test_cli_generates_a_reproducible_report(self):
        path = self.stage("stage", [[255]])
        report = self.root / "cli-report"
        command = [sys.executable, "-m", "thesis_testing_tools.generate_metrics", "--events", str(path),
                   "--output", str(report), "--metrics", "mae"]
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((report / "analysis_manifest.json").read_text())
        self.assertEqual(manifest["settings"]["metrics"], ["mae"])
        self.assertEqual(self.rows(report, "stage_metrics.csv")[0]["mae_episode_mean"], "1.0")


if __name__ == "__main__":
    unittest.main()
