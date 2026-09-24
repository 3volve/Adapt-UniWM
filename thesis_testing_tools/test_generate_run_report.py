import json
from pathlib import Path
import re
import tempfile
import unittest
import shutil
import subprocess
from unittest.mock import patch

from PIL import Image

from runtime_scripts.event_logger import EventLogger
from thesis_testing_tools.generate_run_report import build_report
from thesis_testing_tools import generate_metrics as gm


class RunReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stages = []
        for name, mode, values in (("source_pre", "replay", [0, 0]), ("c1/habitat", "habitat", [64, 128]),
                                    ("c1/source_post", "replay", [255, 255])):
            directory = self.root / name
            directory.mkdir(parents=True)
            target = directory / "target.png"
            Image.new("RGB", (16, 16), (0, 0, 0)).save(target)
            with EventLogger(directory / "events.jsonl", {"outcome": "completed", "episode_setup": {
                "dataset/0": {"episode_id": "ep0", "outcome": "completed"}}}) as log:
                for i, value in enumerate(values):
                    image = directory / f"pred{i}.png"
                    Image.new("RGB", (16, 16), (value, value, value)).save(image)
                    wrapper = {"real_obs_path": str(target), "collision": i == 0, "update_weight": 0.5}
                    if mode == "replay":
                        wrapper["evaluation"] = {"predicted_obs_path": str(image)}
                    else:
                        wrapper["predicted_obs_path"] = str(image)
                    log.next_step(dict(data_id="dataset", episode_id="ep0", episode_index=0, step_idx=i,
                                       source_mode=mode, outcome="completed", transition={"outcome": "completed"}, wrapper=wrapper,
                                       training={"optimizer_step": i == 1, "applied_learning_rate": 0.001,
                                                 "grad_norm_before_clip": 2, "gradient_clipped": True}))
                log.feed({"episode_end": {"dataset/0": {"episode_id": "ep0", "termination_reason": "adapter_done"}}})
                log.finish({"outcome": "completed", "episodes": 1})
            self.stages.append(dict(name=name.split('/')[-1], stage_id=name, events=f"{name}/events.jsonl", status="completed"))
        self.manifest = dict(schema_version=1, run_id="fixture </script><script>alert(1)</script>", status="completed",
                             stages=self.stages, references={"source_pre": "source_pre"},
                             metadata={"seed": 17, "habitat_episode_order": ["ep0"], "source_episode_order": {"dataset": ["ep0"]}})
        (self.root / "run_manifest.json").write_text(json.dumps(self.manifest))

    def data(self, path):
        document = path.read_text(encoding="utf-8")
        return document, json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>', document, re.S)[1])

    def test_full_report_official_metrics_and_gallery(self):
        originals = {p: p.read_bytes() for p in self.root.rglob("events.jsonl")}
        path = build_report(self.root, metrics=("mae",))
        document, data = self.data(path)
        self.assertEqual(data["retention"][0]["mae"], 1)
        self.assertEqual(len(data["gallery"]), 3)
        self.assertTrue(data["gallery"][0]["images"][0].startswith("data:image/png;base64,"))
        self.assertEqual(data["workers"][0]["steps"][0]["lr"], None)
        self.assertEqual(data["workers"][0]["steps"][1]["lr"], 0.001)
        self.assertNotIn('</script><script>alert(1)</script>', document)
        self.assertNotIn('@@', document)
        for p, raw in originals.items():
            self.assertEqual(p.read_bytes(), raw)
        with self.assertRaises(FileExistsError):
            build_report(self.root, metrics=("mae",))
        # Existing metric outputs are verified and reused without any recalculation.
        with patch.object(gm, "generate", side_effect=AssertionError("must not recompute")):
            build_report(self.root, output=self.root / "second.html", analysis_dir=self.root / "run_report_metrics")

    def test_backend_failure_retains_diagnostics_and_no_fake_scores(self):
        with patch.object(gm, "generate", side_effect=RuntimeError("backend unavailable")):
            document, data = self.data(build_report(self.root))
        self.assertIn("backend unavailable", document)
        self.assertEqual(data["retention"], [])
        self.assertIsNone(data["workers"][0]["episodes"][0]["mae"])
        self.assertEqual(data["workers"][0]["steps"][1]["lr"], 0.001)

    def test_missing_worker_is_visible(self):
        (self.root / "c1/habitat/events.jsonl").unlink()
        document, data = self.data(build_report(self.root, metrics=("mae",)))
        self.assertIn("Worker unavailable", document)
        self.assertEqual(data["workers"][0]["episodes"], [])

    def test_tampered_analysis_not_presented_as_results(self):
        build_report(self.root, metrics=("mae",))
        (self.root / "run_report_metrics/stage_metrics.csv").write_text("tampered")
        document, data = self.data(build_report(self.root, output=self.root / "tampered.html", analysis_dir=self.root / "run_report_metrics"))
        self.assertIn("hash mismatch", document)
        self.assertEqual(data["retention"], [])

    def test_stale_reconciliation_and_c0_reuse_are_explicit(self):
        (self.root / "reconciliation.json").write_text(json.dumps({"status": "pass", "checks": [], "inputs": [
            {"path": str(self.root / "c1/habitat/events.jsonl"), "sha256": "wrong"}]}))
        (self.root / "seed_manifest.json").write_text(json.dumps({"conditions": [
            {"condition_id": "c0_frozen", "source_post_reused_from_source_pre": True}]}))
        self.manifest["references"]["seed_plan"] = "seed_manifest.json"
        (self.root / "run_manifest.json").write_text(json.dumps(self.manifest))
        document, data = self.data(build_report(self.root, metrics=("mae",)))
        self.assertIn("Reconciliation: unavailable", document)
        self.assertIn("c0_frozen: source-post reuses source-pre", document)
        self.assertFalse(any(row["name"].startswith("c0") for row in data["retention"]))

    @unittest.skipUnless(shutil.which("node"), "Node is needed for the embedded control test")
    def test_embedded_controls_render_with_dom_stub(self):
        document, data = self.data(build_report(self.root, metrics=("mae",)))
        script = document.split('<script>')[1].split('</script>')[0]
        harness = r'''
const fs=require('fs'), vm=require('vm');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.value='';this.style={setProperty(){}};this.listeners={};}
 append(...items){this.children.push(...items);if(this.tag==='select' && this.children.length===items.length && items.length)this.value=items[0].value;}
 replaceChildren(...items){this.children=items;}
 setAttribute(){}
 addEventListener(name,callback){this.listeners[name]=callback;}
}
const elements={};for(const id of ['metric','worker','gallery-worker'])elements[id]=new Element('select');
elements.metric.value='mae';elements['report-data']=new Element('script');elements['report-data'].textContent=JSON.stringify(input.data);
global.document={getElementById(id){return elements[id]||(elements[id]=new Element('div'))},createElement:t=>new Element(t),createElementNS:(ns,t)=>new Element(t)};
vm.runInThisContext(input.script);
if(!elements.adaptation.children.some(e=>e.tag==='svg') || elements.gallery.children.length!==3)throw Error('Initial charts/gallery missing');
elements.metric.value='lpips';elements.metric.listeners.change();
if(elements.adaptation.children[0].tag!=='p')throw Error('Unavailable metric was not labelled');
elements.worker.listeners.change();elements['gallery-worker'].listeners.change();
console.log('Controls rendered and changed successfully');
'''
        result = subprocess.run([shutil.which("node"), "-e", harness], input=json.dumps({"data": data, "script": script}),
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
