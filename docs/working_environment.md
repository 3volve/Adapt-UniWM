# Recorded working environment

Recovered from completed run `thesis_seed_100_20261002_110940`.
This records an earlier successful execution; the handoff installation recipe
has not been tested on a fresh machine.

| Component | Recorded value |
| --- | --- |
| Platform | Linux x86_64, glibc 2.39 |
| GPU / driver | RTX 6000 Ada, 48 GiB / 580.173.02 |
| Python | 3.10.20 |
| PyTorch / CUDA runtime | 2.4.0+cu121 / 12.1 |
| FlashAttention / PEFT | 2.5.9.post1 / 0.15.2 |
| Habitat-Lab / Habitat-Sim | 0.2.5 / 0.2.5 |
| NumPy | 1.24.3 |
| Transformers | Custom fork, commit `dae439cb93e7c2ef61943554b7a2f4590c2118cd` |
| Anole base model | Revision `96df52301e844d8a624a13953051ead4c008343b` |

The [machine-readable record](../envs/recorded/2026-10-02.json) contains all
recorded Python package versions, source revisions, evidence hashes, and
validation results. It omits workstation usernames/paths. It is an inventory,
not an installable lockfile. The committed adapter and processor artifacts match
the saved run's hashes; the checkpoint README has since changed.

**What remains unknown:** installed CUDA compiler (`nvcc`), host C/C++ compiler,
complete Conda/native-library versions, Habitat-Sim Git revision/build flags,
and peak RAM/VRAM. Python's compiler string and `nvidia-smi`'s CUDA version do not
establish the toolkit used to build extensions.

**Installation target:** follow [README setup](../README.md#setup). Python,
PEFT and the Torch CUDA build now have exact pins. CUDA toolkit 12.1.1 is a proposed
build target based on that runtime, following [NVIDIA's versioned installation
instructions](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-installation-guide-linux/index.html#installing-previous-cuda-releases)
and [PyTorch's CUDA 12.1 wheels](https://pytorch.org/get-started/previous-versions/#v240).
Other transitive dependencies remain unlocked. The setup uses setuptools below
80, wheel 0.45.1 and CMake below 4 for the legacy Habitat build; the saved runtime
instead had setuptools 82.0.1 and wheel 0.46.3, and did not record CMake.
These build constraints are setup fixes, not recovered historical versions.

**Known exception:** Habitat-Sim declares NumPy below 1.24, while this completed
run used 1.24.3. Record dependency-check output and verify actual scene loading;
the successful run does not establish general compatibility.

**Validation:** the saved six-condition run completed in 8.7 hours using one
source episode per dataset, two Habitat episodes, and ten steps per episode.
Current reconciliation passed 5,203 checks (15 not applicable). That timing and
48 GiB device capacity are observations, not minimum requirements or a forecast.
The next validation step is a small run on the collaborator's hardware.
