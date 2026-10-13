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

Model lineage: Meta Chameleon → GAIR `Anole-7b-v0.1` →
`leloy/Anole-7b-v0.1-hf` conversion → UniWM fine-tuning. The local
`adapter_config.json` records the Hugging Face conversion as the base model.

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

The adapter was inherited from the UniWM research release. Its complete
fine-tuning history, duration, and energy use are not available in this fork.
Adapt-UniWM evaluates it on the source domains and Habitat protocol documented
in [running experiments](../../docs/running_experiments.md). Reported behavior is specific to the pinned model
conversion, data manifests, Habitat assets, and software stack, and should not
be interpreted as a general safety or deployment evaluation.

## Loading

Follow the [repository setup](../../README.md#setup) and
[experiment guide](../../docs/running_experiments.md). The PEFT dependency is
specified in [`environment.yml`](../../environment.yml).
