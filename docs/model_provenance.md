# Starting-model provenance

Prepared 3 November 2026 for **Checklist Item 1: Document the actual starting model**.

| Requested information | Recorded provenance | Evidence |
| --- | --- | --- |
| Base model | `leloy/Anole-7b-v0.1-hf` (Anole/Chameleon lineage). Current evaluation revision: `96df52301e844d8a624a13953051ead4c008343b`. | [Model loader](../scripts/load_model.py); [adapter configuration](../checkpoints/base_ckpt/adapter_config.json). |
| Adapter checkpoint | `checkpoints/base_ckpt/adapter_model.safetensors`, recorded as training checkpoint **3000**. SHA-256: `31502dc5817e3eeab366e647dc3062fe055491c1c70980672224a5a8aa2a2aa1`. | [Checkpoint directory](../checkpoints/base_ckpt/); [trainer state](../checkpoints/base_ckpt/trainer_state.json). |
| Training repository / commit | [Adapt-UniWM](https://github.com/3volve/Adapt-UniWM), commit **`60d52a4cdd0b8ca81646df3b29a9c56bcf9ae35c`**, 29 July 2026. Attribution reconstructed from matching saved artifacts and Git history and accepted by the researcher. | [Supporting Git audit](model_provenance_git_audit.md). Historical configuration: `cfg/interleaved.yaml`; launcher: `train_offline.sh` at that commit. |
| Datasets and trajectory counts | **Go Stanford, SACSoN, RECON, SCAND**. Training manifest specifies **208 training trajectories** (52 per dataset), plus **24 validation and 24 test trajectories** (6 each per dataset). | `cfg/eval_dataset_manifest.json` and launcher at the accepted training commit. |
| Completed training | **3,000 optimizer steps**, approximately **5.43 epochs**, at the selected checkpoint. The configured training plan was 10 epochs. | [Trainer state](../checkpoints/base_ckpt/trainer_state.json). |
| Optimization hyperparameters | AdamW; initial LR **8e-5**, linear decay, no warmup; weight decay 0; gradient clipping 1.0. Batch size 1 per device, accumulation 8, two processes (effective batch 16); seed **42**; BF16. | [Saved training arguments](../checkpoints/base_ckpt/training_args.bin). |
| Adapter / input hyperparameters | LoRA rank **8**, alpha **16**, dropout **0.1**, q/k/v/o attention projections; selective training of **519 action-token rows** in input/output embeddings. Images **448x448**, **784 image tokens**. | Adapter and processor configurations in the checkpoint directory. |
| Offline training objective | Causal language-model loss with **0.1 label smoothing**, plus action soft-target and image-codebook discrepancy losses, each weighted **1.0**. Offline training uses the model implementation without memory injection. | Saved arguments/logs and loss/trainer/loader code at the accepted commit. |
| Modifications relative to original UniWM | **Training:** discrete agent-relative action encoding and selective action-token row training; revised dataset/example construction, auxiliary losses, and VQ reconstruction augmentation of input images. **Adaptation/evaluation:** Habitat and source-replay integration, online memory management, controller-modulated learning rates and fixed/replayed controls, and experiment logging/metrics. | Historical training code; [pipeline overview](pipeline_structure.md). Online adaptations are distinct from offline checkpoint training. |
| Model identity and claim scope | **Separately trained Adapt-UniWM derivative; not the original UniWM checkpoint.** Experimental claims concern this derivative and the tested protocol. Recovering or fully reproducing original UniWM is not a prerequisite for this provenance deliverable. | [Corrected checkpoint README](../checkpoints/base_ckpt/README.md). |

**Status against the professor's four requirements:** base/adapter identified;
repository, commit, datasets, trajectory counts, training steps and relevant
hyperparameters recorded; modifications listed; derivative identity stated.
**No required information gap remains for Item 1 at this requested scope.**

One retained limitation: the exact base-model revision downloaded during offline
training was not recorded; the revision above identifies the current evaluation
base. This does not prevent identifying the base model and adapter as requested.
The training commit is retrospectively attributed as noted above.

The detailed audit is supporting material only. Loaded-example counts, exact
transition totals, hardware history, duration, energy, and further historical
forensics are not prerequisites for this one-page provenance deliverable.
Predictive-capability verification remains under Checklist Items 2-3.
