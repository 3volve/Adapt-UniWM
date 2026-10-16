# Adapt-UniWM: proposed run request

**Ask:** Help run my online world-model adaptation comparison on your GPUs.
Start with one pilot; agree the full batch after checking its results and cost.

**Version:** Use the accompanying handoff commit. Setup and data instructions:
[README](../README.md). [Recorded environment](working_environment.md): Linux,
RTX 6000 Ada 48 GiB, Torch 2.4.0/CUDA 12.1. Fresh cluster setup remains unverified.

**Pilot:** With setup complete, run from the repo root:

```bash
bash thesis_testing_tools/run_sequential.sh \
  --seed 100 --debug-source-episodes 1 --debug-habitat-episodes 2 \
  --max-episode-steps 5 --smoke-test
```

This runs all six conditions. Extra GPUs can run independent jobs; each worker
uses one GPU. A previous ten-step pilot took 8.7 hours on the recorded hardware.

**Success:** All stages complete, reconciliation passes, and the report has
the expected metric coverage. I will review those checks before scaling up.

**Return:** Share the complete seed output directory, plus scheduler logs if used.
Keep its files together for reproducibility.

**Before starting:** Confirm GPU/VRAM, scheduler/wall-time limits, shared storage,
and download access. I will provide the full-run seeds and calibrated settings
after the pilot.
