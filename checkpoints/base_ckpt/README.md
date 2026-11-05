---
base_model: leloy/Anole-7b-v0.1-hf
library_name: peft
license: other
license_name: chameleon-research-license
---

# Adapt-UniWM thesis base checkpoint

This directory contains the common PEFT adapter and processor artifacts used by
all C0-C5 conditions in the adaptation experiment. It does
not contain the full 7B-parameter base model.

## Provenance and license

Model lineage: Meta Chameleon -> GAIR `Anole-7b-v0.1` ->
`leloy/Anole-7b-v0.1-hf` conversion -> separate Adapt-UniWM fine-tuning using
modified UniWM code. **This is not the original UniWM checkpoint or an adapter
inherited from the UniWM research release.** The local
`adapter_config.json` records the Hugging Face conversion as the base model.

The [model-provenance table](../../docs/model_provenance.md) records the adapter
hash, saved training settings, dataset counts, and modifications requested
for the thesis provenance deliverable. The accepted training commit is
`60d52a4cdd0b8ca81646df3b29a9c56bcf9ae35c` (29 July 2026), reconstructed from
matching saved artifacts and Git history and accepted by the researcher.
The current loader pins base revision
`96df52301e844d8a624a13953051ead4c008343b`; the base revision used during offline
training has not been established.

GAIR states that Anole weights follow the Chameleon Research License. The same
terms conservatively govern this model-derived adapter. Any release containing
it must include [`../../docs/CHAMELEON_RESEARCH_LICENSE.txt`](../../docs/CHAMELEON_RESEARCH_LICENSE.txt)
and the attribution in [`../../NOTICE`](../../NOTICE).

## Intended use

This adapter is provided solely to reproduce the noncommercial Adapt-UniWM
thesis experiments. The pipeline selects it with `--initial-checkpoint` and
loads the full base model separately. Uses outside those
allowed by the Chameleon Research License are out of scope.

## Data, evaluation, and limitations

The accompanying training records identify a local four-source training run
(Go Stanford, SACSoN, RECON, and SCAND), with this checkpoint recorded at 3,000
steps / approximately 5.43 epochs. Saved arguments specify initial LR `8e-5`,
seed 42, per-device batch size 1, gradient accumulation 8, and a 10-epoch plan;
the plan must not be confused with completed training. The accepted manifest
configures 208 training, 24 validation, and 24 test trajectories; saved evaluation
outputs match all 24 validation trajectories (908 examples). See the provenance
table for the requested information and supporting evidence.
Adapt-UniWM evaluates it on the source domains and Habitat protocol documented
in [running experiments](../../docs/running_experiments.md). Reported behavior is specific to the pinned model
conversion, data manifests, Habitat assets, and software stack, and should not
be interpreted as a general safety or deployment evaluation.
## Loading

Follow the [repository setup](../../README.md#setup) and
[experiment guide](../../docs/running_experiments.md). The PEFT dependency is
specified in [`environment.yml`](../../environment.yml).
