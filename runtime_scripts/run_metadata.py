"""Observe the actual worker environment and model settings without changing them."""
from __future__ import annotations

import hashlib
import os
import pickle
import random
from pathlib import Path
import platform
import sys

from thesis_testing_tools.run_manifest import ENVIRONMENT_KEYS, now, write_json


def runtime_metadata(torch, *, seed: int | None, engine=None):
    import numpy as np
    cuda_initialized = torch.cuda.is_initialized()
    result = {
        "recorded_at": now(), "phase": "before_model_initialization" if engine is None else "model_initialized",
        "hostname": platform.node(), "python": sys.version, "python_executable": sys.executable,
        "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
        "command": list(sys.argv),
        "cudnn_version": torch.backends.cudnn.version(),
        "environment_variables": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
        "seeds": {"requested": seed, "python": seed, "numpy": seed, "torch_cpu": seed,
                  "torch_cuda": seed, "explicitly_seeded": seed is not None,
                  "torch_initial_seed": torch.initial_seed()},
        "determinism": {
            "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "cuda_initialized": cuda_initialized,
        "cuda_devices": ([{"index": i, "name": torch.cuda.get_device_name(i),
                           "capability": list(torch.cuda.get_device_capability(i)),
                           "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                          for i in range(torch.cuda.device_count())] if cuda_initialized else []),
    }
    # Reading RNG states does not draw samples or reseed the model.
    result["rng_state_sha256"] = {
        "python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "torch_cuda": ([hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
                        for state in torch.cuda.get_rng_state_all()] if cuda_initialized else []),
    }
    if engine is not None:
        model = engine.model
        result["model"] = {
            "class": type(model).__name__, "training": model.training,
            "resolved_engine_configuration": engine.config,
            "initialization": engine.initialization_metadata,
            "base_model": {"name_or_path": getattr(model.config, "_name_or_path", None),
                           "revision": getattr(model.config, "_commit_hash", None)},
            "peft_config": {name: config.to_dict() for name, config in model.peft_config.items()},
            "dropout_modules": [{"name": name, "class": type(module).__name__,
                                 "p": module.p, "training": module.training}
                                for name, module in model.named_modules()
                                if isinstance(module, torch.nn.Dropout)],
            "trainable_parameters": [{"name": name, "shape": list(parameter.shape),
                                      "dtype": str(parameter.dtype), "elements": parameter.numel()}
                                     for name, parameter in model.named_parameters() if parameter.requires_grad],
            "optimizer": ({"class": type(engine._optimizer).__name__,
                           "defaults": engine._optimizer.defaults,
                           "parameter_groups": [{k: v for k, v in group.items() if k != "params"}
                                                for group in engine._optimizer.param_groups]}
                          if engine.config["training"] is not False else None),
            "mode_policy": "Prediction uses eval(); visualization loss uses train(), then returns to eval().",
        }
    return result


def write_runtime_metadata(output_dir: Path, *, seed: int | None, engine=None):
    import torch
    filename = "runtime_metadata_before_model.json" if engine is None else "runtime_metadata.json"
    write_json(output_dir / filename, runtime_metadata(torch, seed=seed, engine=engine))
