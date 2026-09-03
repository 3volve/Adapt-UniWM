# Adapt-UniWM

**Online world-model adaptation experiments for UniWM**

Adapt-UniWM is the research code and reproducibility artifact for Evo Lamont's
thesis experiment on stability and plasticity during online visual-navigation
adaptation. It is a research fork of
[UniWM](https://github.com/F1y1113/UniWM), not the official UniWM repository. It
adds a closed-loop Habitat runner, online training, a neuromodulator-inspired
learning controller, fixed-action evaluation, and a source-retention pipeline.

The published experiment compares three conditions on the same Habitat episodes, drawing actions from the same fixed-action reference:

- **Frozen / no learning:** online parameter updates are disabled.
- **Fixed learning:** online updates use a fixed learning rule without modulators.
- **Modulated learning:** the same online learning path is controlled by the neuromodulator-inspired signals.

The fixed action reference used by all three configurations is tracked in [`thesis_artifacts/habitat_fixed_actions/no_learning`](thesis_artifacts/habitat_fixed_actions/no_learning/). This replaces the original reference to an ignored local output directory.

### Run the published Habitat configurations

After completing the base and Habitat setup below, launch each condition from the repository root. Use a distinct free `--master_port` if runs overlap.

```bash
torchrun --nproc_per_node=1 --master_port=20001 uniwm_episode_runner.py \
  --config_path cfg/habitat_uniwm_cfg_no_learning.yaml \
  --data_id habitat \
  --run_dir output/thesis/no_learning/habitat

torchrun --nproc_per_node=1 --master_port=20003 uniwm_episode_runner.py \
  --config_path cfg/habitat_uniwm_cfg_fixed_learning.yaml \
  --data_id habitat \
  --run_dir output/thesis/fixed_learning/habitat

torchrun --nproc_per_node=1 --master_port=20005 uniwm_episode_runner.py \
  --config_path cfg/habitat_uniwm_cfg_modulated_learning.yaml \
  --data_id habitat \
  --run_dir output/thesis/modulated_learning/habitat
```

The Habitat no-learning configuration also records the controller-derived
learning-rate schedule without taking optimizer steps. It writes
`learning_rate_schedule.json` and the deterministically within-episode-shuffled
`learning_rate_schedule_shuffled.json` in the Habitat run directory.

To train with either saved schedule, use a learning configuration with
modulators disabled and point `input_path` at that Habitat run directory:

```yaml
wrapper:
  training_enabled: true
  enable_modulators: false
  learning_rate_schedule:
    mode: replay
    input_path: output/thesis/no_learning/habitat
    shuffled: false  # Set true for the within-episode-shuffled schedule.
```

Configurations that do not record or replay a schedule use
`learning_rate_schedule: false`. Replay requires the configured base learning
rate to match the one recorded in the schedule.

### Run the complete thesis pipeline

`thesis_testing_tools/run_thesis_pipeline.py` runs the three-stage protocol:

1. **Source pre:** evaluate the common base checkpoint on the source datasets.
2. **Habitat:** run one online-adaptation condition and save its final adapter.
3. **Source post:** evaluate that adapted checkpoint on the same source domains
   to measure retention.

For the paired core experiment, queue all six conditions for one seed with:

```bash
python thesis_testing_tools/run_thesis_pipeline.py \
  --all-conditions \
  --seed 100 \
  --fixed-mean-lr 6.25e-5 \
  --habitat-action-run thesis_artifacts/habitat_fixed_actions/no_learning
```

`--fixed-mean-lr` must be the preregistered Full-controller mean estimated from
the development stream; the held-out batch does not recompute it. The batch
runs source-pre once, starts every C0-C5 Habitat condition from the same initial
checkpoint, and runs a separate source-post evaluation for C1-C5. Because C0
does not update the model, its source-post metrics are reused exactly from
source-pre without launching another model process. C0 records both aligned and
within-episode-shuffled learning-rate schedules, which C3 and C4 consume from
C0's Habitat output directory.

The batch is written below `output/thesis_seed_<seed>_<timestamp>/`.
`seed_manifest.json` records the full planned queue and generated configuration
paths, while `seed_summary.json` links the completed per-condition summaries.

Run one complete pipeline per condition:

```bash
python thesis_testing_tools/run_thesis_pipeline.py \
  --habitat-cfg habitat_uniwm_cfg_no_learning.yaml \
  --habitat-port 20001

python thesis_testing_tools/run_thesis_pipeline.py \
  --habitat-cfg habitat_uniwm_cfg_fixed_learning.yaml \
  --habitat-port 20003

python thesis_testing_tools/run_thesis_pipeline.py \
  --habitat-cfg habitat_uniwm_cfg_modulated_learning.yaml \
  --habitat-port 20005
```

Each run is written below `output/thesis_pipeline_<timestamp>/`. The primary
reproducibility artifacts are:

- `pipeline_manifest.json`: commands, configurations, inputs, and checkpoint flow.
- `source_pre/thesis_episode_metrics.csv`: source performance before adaptation.
- `habitat/thesis_episode_metrics.csv`: Habitat navigation, learning, and diagnostic metrics.
- `habitat/final_ckpt/`: the condition's final PEFT adapter.
- `source_post/thesis_episode_metrics.csv`: source performance after adaptation.
- `source_episode_comparison.csv`: paired pre/post source-retention comparison.
- `pipeline_summary.json`: compact run-level Habitat and retention summary.

The thesis defines and interprets the reported metrics. The README records where
the implementation produces them so a reader can trace thesis tables back to
run artifacts without duplicating the manuscript's methods section.

### Record the experiment workstation

The thesis experiments ran on this reference workstation:

| Component | Recorded configuration |
| --- | --- |
| OS | 64-bit Linux, kernel `7.0.0-28-generic` |
| CPU | AMD Ryzen Threadripper PRO 7965WX, 24 cores / 48 threads |
| System memory | 125 GiB RAM, 8 GiB swap |
| GPUs | 2× NVIDIA RTX 6000 Ada Generation, 48 GiB VRAM each |
| NVIDIA stack | driver 580.173.02; CUDA compiler 12.4 |
| PyTorch stack | Python 3.10.20; PyTorch 2.4.0+cu121; cuDNN 9.1 |

Two concurrent experiment processes occupied approximately 22 GiB on each GPU
when the report was collected. That is an observation, not a measured peak or a
guaranteed minimum-memory requirement; a 24 GiB GPU may be borderline depending
on allocator state and configuration. The published commands use one process
per condition.

### Fork provenance and licensing status

Adapt-UniWM contains substantial modifications by Evo Lamont; the main additions are summarized above and are visible in this fork's Git history. The upstream UniWM repository does not currently provide a `LICENSE` file, despite previously displaying an MIT badge. Accordingly, this fork does not assert a repository-wide MIT license. Third-party code, datasets, model weights, and Habitat assets remain subject to their respective terms. Resolve the upstream licensing grant and asset redistribution terms before treating a tagged release as redistributable software.

The included PEFT checkpoint descends from Meta Chameleon through GAIR's Anole
and the Hugging Face conversion `leloy/Anole-7b-v0.1-hf`. GAIR states that the
Anole weights follow the Chameleon Research License. Releases containing the
checkpoint must therefore include that license and the attribution in `NOTICE`,
and are limited to the uses permitted by those terms.

The exact third-party model terms are included in
[`docs/CHAMELEON_RESEARCH_LICENSE.txt`](docs/CHAMELEON_RESEARCH_LICENSE.txt).

In this repository, provenance means recording what each external artifact is,
where it came from, why the experiment needs it, and which terms govern reuse:

| Component | Experimental role | Source and redistribution status |
| --- | --- | --- |
| UniWM code | Upstream model and navigation implementation | [UniWM](https://github.com/F1y1113/UniWM); no explicit repository license currently published |
| Thesis base checkpoint | Common starting PEFT adapter for all three conditions | `checkpoints/base_ckpt`; Chameleon → Anole → Hugging Face conversion → UniWM adapter; Chameleon Research License |
| Source-domain data | Source-pre/source-post retention evaluation | Downloaded directly from the MIT-labeled [UniWM dataset release](https://huggingface.co/datasets/fly1113/UniWM_Dataset) at a pinned revision; not redistributed by this fork |
| Habitat-Lab and Habitat-Sim | Online navigation environment | Installed from pinned upstream revisions by `scripts/setup_habitat025_uniwm.sh` |
| HM3D and InstanceImageNav data | Habitat scenes and evaluation episodes | Acquired separately under the providers' access terms; not vendored here |

## Installation

```bash
conda env create -f environment.yml
conda activate uniwm
```

To update an existing base environment in place:

```bash
conda env update -f environment.yml --prune
```

### Optional: Install flash-attn
If your environment supports `flash-attn`, then you can install it separately:

```bash
# While the uniwm conda environment is active
conda install -c nvidia cuda
pip install flash-attn==2.5.9.post1 --no-build-isolation
```

## Habitat Setup

For the current Habitat setup inside the existing `uniwm` Python 3.10 environment, see [docs/habitat025_uniwm_setup.md](docs/habitat025_uniwm_setup.md). The reproducible files live in [envs/uniwm-habitat025-addons.yml](envs/uniwm-habitat025-addons.yml), [envs/uniwm-habitat025-pip.txt](envs/uniwm-habitat025-pip.txt), and [scripts/setup_habitat025_uniwm.sh](scripts/setup_habitat025_uniwm.sh).

The base `environment.yml` intentionally excludes Habitat. The setup scripts clone `habitat-lab/` and `habitat-sim/` into the repo root as local ignored checkouts rather than vendoring those upstream trees into this fork.

## Implementation

### Data

Source data is intentionally not committed. The source-pre and source-post stages
use `go_stanford`, `recon`, `sacson`, and `scand` trajectories published in the
MIT-labeled [UniWM Hugging Face dataset](https://huggingface.co/datasets/fly1113/UniWM_Dataset).
Download the manifest-selected subset from the pinned upstream revision and
build the expected `eval_data/` tree with:

```bash
bash download_eval_dataset.sh
```

The complete archives are several gigabytes. See
[`docs/source_data_setup.md`](docs/source_data_setup.md) for provenance, expected
layout, preprocessing behavior, and which pipeline stages require this data.

### Training

To train the model on multiple datasets, use the following `torchrun` command.
The frozen thesis helper is [`thesis_testing_tools/train_offline.sh`](thesis_testing_tools/train_offline.sh).

```bash
torchrun --nproc_per_node={GPU_NUM_PER_NODE} train.py \
    --model anole \
    --data go_stanford,scand,sacson,recon \
    --data_dir ./eval_data \
    --decoder_type anole \
    --image_seq_length 784 \
    --input_format anole \
    --output /path/to/save/output \
    --note {experiment_note} \
    --report_to none \
    --do_train \
    --bfloat16
```

### Evaluation

To evaluate a trained model, use the command below. The scripts under
[`thesis_testing_tools`](thesis_testing_tools/) contain the frozen thesis
examples.

``` bash
torchrun --nproc_per_node=<GPU_NUM_PER_NODE> train.py \
    --model anole \
    --model_ckpt /path/to/your/checkpoint \
    --data go_stanford,scand,sacson,recon \
    --data_dir ./eval_data \
    --decoder_type anole \
    --image_seq_length 784 \
    --input_format anole \
    --output /path/to/save/eval_results \
    --note {experiment_note} \
    --report_to none \
    \
    # Choose ONE of the following evaluation flags for different eval mode:
    --do_single_step_eval
    # --do_task_level_eval
    # --do_rollout_eval

    # Optional: --use_memory_bank_inference
```
#### Evaluation Flags (choose one):

`--do_single_step_eval`: Evaluates the model's performance on a single step of prediction.

`--do_task_level_eval`: Evaluates the model on the full end-to-end task across an entire trajectory. You can optionally enable the memory bank mechanism by adding the `--use_memory_bank_inference` flag to the command. If this flag is omitted, the evaluation runs with the memory bank disabled.

`--do_rollout_eval`: Generates a full trajectory autoregressively (i.e., the model uses its own previous predictions and ground truth actions as input for the next step) and evaluates the result.

## Relationship to upstream work

Adapt-UniWM extends the official
[UniWM implementation](https://github.com/F1y1113/UniWM) with the online
adaptation runner, fixed-action comparison protocol, learning modulators, and
source-retention pipeline described above. UniWM's original authors and paper
remain credited through the upstream citation below.

<p align="center">
  <img src="assists/comparison.png" alt="UniWM task comparison" width="660"/>
</p>

The implementation also builds on
[Anole](https://arxiv.org/abs/2407.06135) and
[MVOT](https://arxiv.org/abs/2501.07542).

## Scope and limitations

This is research-grade code for reproducing a specific thesis experiment, not a
general navigation framework or production system. The published configurations
target the recorded software stack, checkpoint, episode set, and fixed-action
protocol; results may change with different model conversions, Habitat assets,
hardware, or dependency versions. Broader methodological limitations and the
interpretation of results belong in the thesis.

## 🌟 Citation

For the adaptation software and thesis experiment, cite the `thesis-final` repository release:

```bibtex
@software{lamont2026adaptuniwm,
  author  = {Evo Lamont},
  title   = {Adapt-UniWM: Online Adaptation Experiments for UniWM},
  year    = {2026},
  version = {thesis-final},
  url     = {https://github.com/3volve/Adapt-UniWM/tree/thesis-final}
}
```

Adapt-UniWM is derived from UniWM. Please also cite the upstream paper:
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
