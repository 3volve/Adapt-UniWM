"""Exercise download extraction and subset building with the committed manifests."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
SOURCES = ("go_stanford", "recon", "sacson", "scand")


class SourcePreparationTests(unittest.TestCase):
    def test_committed_manifests(self):
        shell = (REPO / "download_eval_dataset.sh").read_text()
        extractor = shell.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
        for name in ("eval_dataset_manifest", "thesis_heldout_manifest"):
            with self.subTest(manifest=name), tempfile.TemporaryDirectory(
                prefix="adapt_source_test_"
            ) as temporary:
                root = Path(temporary).resolve()
                assert root.is_relative_to(Path(tempfile.gettempdir()).resolve())
                manifest_path = REPO / "cfg" / f"{name}.json"
                manifest = json.loads(manifest_path.read_text())
                expected = {}
                for dataset, entry in manifest.items():
                    if dataset in {"habitat", "action_token_vocabulary"}:
                        continue
                    expected[dataset] = [ref for split in ("train", "validation", "test")
                                         for ref in entry.get(split, [])]
                requested = {source: set() for source in SOURCES}
                for dataset, refs in expected.items():
                    for ref in refs:
                        source, episode = ref.split("/", 1) if "/" in ref else (dataset, ref)
                        requested[source].add(episode)
                for source, episodes in requested.items():
                    archive = root / f"{source}.tar"
                    with tarfile.open(archive, "w") as handle:
                        for episode in episodes | {"unselected_episode"}:
                            for filename in ("traj_data.pkl", "0.jpg"):
                                member = tarfile.TarInfo(f"{source}/{episode}/{filename}")
                                member.size = 1
                                handle.addfile(member, io.BytesIO(b"x"))
                    result = subprocess.run(
                        [sys.executable, "-", source, str(manifest_path),
                         str(archive), str(root / "data")], input=extractor,
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertFalse((root / "data" / source / "unselected_episode").exists())
                output = root / "subset"
                result = subprocess.run(
                    [sys.executable, str(REPO / "scripts/create_manifest_dataset.py"),
                     "--manifest", str(manifest_path), "--source-data-dir", str(root / "data"),
                     "--output-data-dir", str(output), "--copy"],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual({p.name for p in output.iterdir()}, set(expected))
                for dataset, refs in expected.items():
                    for ref in refs:
                        self.assertTrue((output / dataset / ref / "traj_data.pkl").is_file())
                        self.assertTrue((output / dataset / ref / "0.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
