import json
from pathlib import Path
import pickle
import random
import runpy
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from thesis_testing_tools.run_manifest import RunManifest, artifact_fingerprint, sha256_file
from thesis_testing_tools.pipeline import run_stage
from runtime_scripts.run_metadata import runtime_metadata
from runtime_scripts.event_logger import EventLogger


class RunManifestTests(unittest.TestCase):
    def test_stage_uses_snapshot_and_indexes_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.yaml"
            config.write_text("setting: original\n")
            output = root / "run"
            with RunManifest(output, repo_root=root, run_type="test", metadata={}, capture=lambda c: {}) as record:
                saved = record.snapshot(config, "config")
                config.write_text("setting: edited\n")

                def runner(command, **kwargs):
                    self.assertEqual(Path(command[2]), saved)
                    self.assertEqual(saved.read_text(), "setting: original\n")
                    (output / "stage").mkdir()
                    (output / "stage/result.txt").write_text("result")

                stage = run_stage("stage", ["runner", "--config_path", str(config)],
                                  output / "stage", config, "source", runner, run_manifest=record)
                stage["input_checkpoint_path"] = "checkpoint"
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["stages"][0]["input_checkpoint_path"], "checkpoint")
            self.assertEqual(manifest["inputs"][0]["sha256"], sha256_file(saved))
            self.assertEqual(len(manifest["inputs"]), 1)
            self.assertFalse(manifest["stages"][0]["events_available"])
            index = json.loads((output / manifest["artifact_index"]).read_text())
            self.assertIn("stage/result.txt", [item["path"] for item in index["files"]])
            self.assertTrue(all(not Path(item["path"]).is_absolute() for item in index["files"]))
            before = (output / "run_manifest.json").read_bytes()
            with self.assertRaises(FileExistsError), RunManifest(
                output, repo_root=root, run_type="test", metadata={}, capture=lambda c: {}
            ):
                pass
            self.assertEqual((output / "run_manifest.json").read_bytes(), before)
            (output / "figure.svg").write_text("<svg/>")
            with patch("sys.argv", ["run_manifest", str(output)]):
                runpy.run_path(str(Path(__file__).with_name("run_manifest.py")), run_name="__main__")
            refreshed = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(refreshed["status"], manifest["status"])
            self.assertEqual(refreshed["artifact_count"], manifest["artifact_count"] + 1)

    def test_stage_failure_interrupt_and_analysis_failure_are_preserved(self):
        for failure in (subprocess.CalledProcessError(7, ["worker"]), KeyboardInterrupt(), RuntimeError("bad metric")):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = root / "config.yaml"
                config.write_text("setting: 1")
                output = root / "run"
                with self.assertRaises(type(failure)):
                    with RunManifest(output, repo_root=root, run_type="test", metadata={}, capture=lambda c: {}) as record:
                        with record.stage("habitat", ["worker"], output / "habitat", config, "habitat"):
                            if not isinstance(failure, RuntimeError):
                                raise failure
                        raise failure
                manifest = json.loads((output / "run_manifest.json").read_text())
                self.assertEqual(manifest["status"], "interrupted" if isinstance(failure, KeyboardInterrupt) else "failed")
                self.assertEqual(manifest["failure"]["returncode"], 7 if isinstance(failure, subprocess.CalledProcessError) else None)
                if isinstance(failure, RuntimeError):
                    self.assertEqual(manifest["stages"][0]["status"], "completed")
                    self.assertEqual(manifest["failure"]["phase"], "stage_finished")
                self.assertTrue((output / manifest["failure"]["traceback"]).is_file())
                self.assertIsNotNone(manifest["finished_at"])

    def test_environment_allowlist_and_code_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "uniwm_episode_runner.py").write_text("# runner")
            (root / "scripts").mkdir()
            (root / "scripts/helper.py").write_text("# helper")
            with patch.dict("os.environ", {"SLURM_JOB_ID": "123", "TEST_SECRET": "do-not-save"}), patch(
                "importlib.metadata.distributions", return_value=[]
            ), RunManifest(root / "run", repo_root=root, run_type="test", metadata={}, capture=lambda c: {"command": c}) as record:
                record.capture_environment({"NCCL_P2P_DISABLE": "1"})
                record.snapshot_code()
            environment = json.loads((root / "run/provenance/environment.json").read_text())
            self.assertEqual(environment["environment_variables"]["SLURM_JOB_ID"], "123")
            self.assertNotIn("TEST_SECRET", environment["environment_variables"])
            self.assertEqual(len(record.data["inputs"]), 2)
            self.assertEqual(artifact_fingerprint(root / "scripts")["file_count"], 1)



class RuntimeMetadataTests(unittest.TestCase):
    def test_observation_does_not_change_rng_and_records_actual_model_settings(self):
        state = NS(numpy=lambda: NS(tobytes=lambda: b"state"))
        torch = NS(
            __version__="test", version=NS(cuda="test"), initial_seed=lambda: 321,
            cuda=NS(is_initialized=lambda: False), get_rng_state=lambda: state,
            are_deterministic_algorithms_enabled=lambda: True,
            is_deterministic_algorithms_warn_only_enabled=lambda: False,
            get_float32_matmul_precision=lambda: "highest",
            backends=NS(cudnn=NS(version=lambda: 99, deterministic=True, benchmark=False, allow_tf32=False),
                        cuda=NS(matmul=NS(allow_tf32=False))),
            nn=NS(Dropout=type("Dropout", (), {})),
        )
        dropout = torch.nn.Dropout()
        dropout.p, dropout.training = 0.1, False
        parameter = NS(requires_grad=True, shape=(8, 16), dtype="bfloat16", numel=lambda: 128)
        model = NS(training=False, config=NS(_name_or_path="base", _commit_hash="revision"),
                   peft_config={"default": NS(to_dict=lambda: {"r": 8})},
                   named_modules=lambda: [("lora_dropout", dropout)],
                   named_parameters=lambda: [("adapter.weight", parameter)])
        engine = NS(model=model, config={"load_model_args": {}, "training": {}, "generation": {}},
                    initialization_metadata={"adapter_initialization": "loaded_trainable_adapter"},
                    _optimizer=NS(defaults={"lr": 1e-4}, param_groups=[{"lr": 1e-4, "params": [parameter]}]))
        numpy = NS(random=NS(get_state=lambda: ("test", (1, 2, 3))))
        before = pickle.dumps(random.getstate())
        with patch.dict("sys.modules", {"numpy": numpy}):
            result = runtime_metadata(torch, seed=321, engine=engine)
            unseeded = runtime_metadata(torch, seed=None)
        self.assertEqual(pickle.dumps(random.getstate()), before)
        self.assertTrue(result["determinism"]["algorithms_enabled"])
        self.assertEqual(result["model"]["initialization"]["adapter_initialization"], "loaded_trainable_adapter")
        self.assertEqual(result["model"]["dropout_modules"][0]["p"], 0.1)
        self.assertFalse(result["model"]["dropout_modules"][0]["training"])
        self.assertEqual(result["model"]["trainable_parameters"][0]["elements"], 128)
        self.assertNotIn("params", result["model"]["optimizer"]["parameter_groups"][0])
        self.assertFalse(unseeded["seeds"]["explicitly_seeded"])
        self.assertEqual(result["rng_state_sha256"], unseeded["rng_state_sha256"])
        self.assertEqual(result["cuda_devices"], [])  # No CUDA initialization just for metadata.


if __name__ == "__main__":
    unittest.main()
