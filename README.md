# Adapt-UniWM

Online world-model adaptation experiments for UniWM.

Adapt-UniWM is Evo Lamont's research fork of
[UniWM](https://github.com/F1y1113/UniWM), studying stability and plasticity during
online visual-navigation adaptation. It combines a Habitat runner, online model
updates, a neuromodulator-inspired learning controller, and source-retention
evaluation.

Each experiment runs six conditions (C0-C5) from the same checkpoint on shared
Habitat episodes and generated action sequences: frozen, fixed base learning
rate, fixed mean learning rate, aligned schedule replay, shuffled schedule
replay, and the full online controller. Source evaluations before and after
adaptation measure retention on Go Stanford, SACSoN, SCAND, and RECON.

## Setup

Use Linux x86_64 with an NVIDIA GPU. The [recorded working environment](docs/working_environment.md)
identifies the tested hardware/software and remaining setup uncertainties.
From the repository root, create and activate the base environment:

```bash
conda env create -f environment.yml
conda activate uniwm
```

The model loader uses FlashAttention 2, installed separately from the base
environment:

```bash
conda install -c nvidia/label/cuda-12.1.1 cuda=12.1.1
pip install flash-attn==2.5.9.post1 --no-build-isolation
```

CUDA 12.1.1 is the proposed build-toolkit target for the recorded Torch CUDA 12.1
runtime; the workstation's actual compiler/toolkit was not captured. On a cluster,
an administrator-provided CUDA 12.1 toolkit can serve this role. This revised
installation recipe still needs a fresh-machine pilot.

Then prepare the experiment inputs:

1. Install Habitat using the [Habitat setup guide](docs/habitat025_uniwm_setup.md).
   HM3D scenes and InstanceImageNav episodes must be acquired separately and
   placed at the paths configured in
   [`cfg/habitat_uniwm_cfg.yaml`](cfg/habitat_uniwm_cfg.yaml).
2. Prepare source trajectories following [source-data setup](docs/source_data_setup.md),
   including the download command and manifest selection.
3. Use the included [base adapter](checkpoints/base_ckpt/README.md). The model
   loader also needs access to `leloy/Anole-7b-v0.1-hf`; the full base model is
   downloaded separately through Hugging Face at the revision pinned in
   [`scripts/load_model.py`](scripts/load_model.py).

## Run an experiment

With the environment active and inputs available, run one sequential seed batch:

```bash
bash thesis_testing_tools/run_sequential.sh
```

The launcher supplies seed 100, the development manifest, and a development mean
learning rate of `6.98e-5`. Review these defaults for your experiment. Outputs go
under `output/sequential_runs/thesis_seed_<seed>_<timestamp>/`.

See [running experiments](docs/running_experiments.md) for condition definitions,
calibration, configuration, workload selection, and smaller debug runs. For
concurrent conditions on a cluster, use the [Slurm guide](docs/slurm_pipeline.md).

## Inspect results

Start with `run_report.html` in the seed directory. It presents image metrics,
source retention, execution coverage, adaptation diagnostics, and prediction
examples. Check its notices and `reconciliation.json` before interpreting results.
`run_manifest.json` links the execution evidence and saved inputs.

| Guide | Purpose |
| --- | --- |
| [Run reports](docs/run_report.md) | Interpret or regenerate the offline HTML report. |
| [Metric generation](docs/metric_generation.md) | Measurement definitions, exclusions, pairing, and reproducibility. |
| [Reconciliation](docs/reconciliation.md) | Check planned workload against recorded execution. |
| [Pipeline structure](docs/pipeline_structure.md) | Find the main execution components when modifying the code. |
| [EventLogger](docs/event_logger.md) | Understand the event collector's API and record lifecycle. |

## Attribution and licensing

For UniWM-native training and evaluation workflows, refer to the
[upstream repository](https://github.com/F1y1113/UniWM). This fork also builds on
[Anole](https://arxiv.org/abs/2407.06135) and [MVOT](https://arxiv.org/abs/2501.07542).

This fork does not assert a repository-wide MIT license. Third-party code, data,
models, and Habitat assets retain their respective terms. Model attribution is
in [NOTICE](NOTICE); the bundled model terms are in the
[Chameleon Research License](docs/CHAMELEON_RESEARCH_LICENSE.txt). See the
[checkpoint README](checkpoints/base_ckpt/README.md) for adapter provenance.

## Citation

For Adapt-UniWM experiments, cite Evo Lamont's *Adapt-UniWM: Online Adaptation
Experiments for UniWM* and identify the repository commit used. For reported
metrics, also identify the analysis manifest as described in the
[metric guide](docs/metric_generation.md#citing-the-implementation).

Please also cite the upstream UniWM paper:

```bibtex
@misc{dong2025unifiedworldmodelsmemoryaugmented,
      title={Unified World Models: Memory-Augmented Planning and Foresight for Visual Navigation}, 
      author={Yifei Dong and Fengyi Wu and Guangyu Chen and Zhi-Qi Cheng and Qiyu Hu and Yuxuan Zhou and Jingdong Sun and Jun-Yan He and Qi Dai and Alexander G Hauptmann},
      year={2025},
      eprint={2510.08713},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.08713}, 
}
```
