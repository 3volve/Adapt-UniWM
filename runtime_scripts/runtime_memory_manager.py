
import torch
from torch import nn

from runtime_scripts.uniwm_schemas import MemorySnapshot


class RuntimeMemoryBankManager:
    _cached_step_state: MemorySnapshot | None = None

    def __init__(self, model: nn.Module, use_memory_bank_inference: bool, verbose: bool = False):
        self.model = model
        self.is_enabled = use_memory_bank_inference
        self.current_step = 0
        self._current_flashback_idx = self.current_step
        self.verbose = verbose

    def setup_for_episode(self, episode_id: str | None = None):
        """Prepares the memory bank for a new episode."""
        if not self.is_enabled:
            return

        # Reset memory bank for each episode to ensure independence
        if hasattr(self.model, 'reset_memory_bank'):
            self.model.reset_memory_bank()
            if self.verbose:
                print(f"  Intra-step memory bank reset for episode {episode_id}")
        elif hasattr(self.model, 'memory_bank_initialized'):
            # Fallback: manually reset memory bank state
            self.model.memory_bank_initialized = False # type: ignore
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'model') and hasattr(self.model.model.model,
                                                                                               'layers'):
                for layer in self.model.model.model.layers:
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'reset_memory_bank'):
                        layer.self_attn.reset_memory_bank()
            if self.verbose:
                print(f"  Fallback intra-step memory bank reset for episode {episode_id}")

        # Reset global cross-step memory bank for each episode
        if hasattr(self.model, 'reset_global_memory_bank'):
            self.model.reset_global_memory_bank()
            if self.verbose:
                print(f"  Cross-step memory bank reset for episode {episode_id}")

        # Enable global memory bank functionality
        if hasattr(self.model, 'enable_global_memory_bank'):
            self.model.enable_global_memory_bank()
            if self.verbose:
                print(f"  Global memory bank enabled for episode {episode_id}")

        # Enable memory bank functionality if available
        if hasattr(self.model, 'enable_memory_bank'):
            self.model.enable_memory_bank()
            if self.verbose:
                print(f"  Memory bank functionality enabled for episode {episode_id}")

        self.current_step = 0

    def start_new_step(self):
        """Prepares for a new step within an episode."""
        if not self.is_enabled:
            return

        self.current_step += 1
        if self.verbose:
            print(f"\n=== Step {self.current_step} Action Prediction Substep ===")

        # Reset intra memory bank for action prediction substep (but keep cross memory bank)
        if hasattr(self.model, 'reset_memory_bank'):
            self.model.reset_memory_bank()
            if self.verbose:
                print(f"  Step {self.current_step}: intra memory bank reset for action prediction")

    def load_cached_state(self) -> None:
        if self._cached_step_state is None:
            raise AssertionError("Attempted to load a cached memory state that hadn't been set.")

        layers = [
            self.model.model.model.layers[index]
            for index in sorted(self.model.model.use_memory_bank_layers)
        ]

        for index, layer in enumerate(layers):
            attention = layer.self_attn
            attention.memory_bank_initialized = self._cached_step_state.stored_keys[index] is not None
            attention.stored_keys = self._cached_step_state.stored_keys[index]
            attention.stored_values = self._cached_step_state.stored_values[index]

        self.current_step = self._cached_step_state.current_step

        self._cached_step_state = None

    def cache_step_state(self) -> None:
        if self._cached_step_state is not None:
            raise AssertionError("Attempted to cache a memory state while one was still cached.")

        attentions = [
            self.model.model.model.layers[index].self_attn
            for index in sorted(self.model.model.use_memory_bank_layers)
        ]

        self._cached_step_state = MemorySnapshot(
            current_step=self.current_step,
            stored_keys=tuple(attention.stored_keys for attention in attentions),
            stored_values=tuple(attention.stored_values for attention in attentions)
        )

    def reset_memory_state_to_step(self, target_step: int):
        self.cache_step_state()

        self.current_step = target_step
        self._current_flashback_idx = target_step

    def restore_memory_state_to_current(self):
        self.load_cached_state()
        self._current_flashback_idx = self.current_step

    def initialize_step_memory(self, processor_inputs):
        if not self.is_enabled or self.model.memory_bank_initialized:
            return
        
        with torch.autocast(device_type='cuda', dtype=self.model.dtype):
            self.model.initialize_memory_bank_no_pixels(
                input_ids=processor_inputs['input_ids'],
                attention_mask=processor_inputs['attention_mask']
            )

    def get_action_kwargs(self, action_gen_kwargs):
        if not self.is_enabled:
            action_gen_kwargs.pop("current_step", None)
            action_gen_kwargs.pop("current_substep", None)
            return action_gen_kwargs

        action_gen_kwargs_with_memory = action_gen_kwargs.copy()

        use_global_mb = self.current_step > 1
        action_gen_kwargs_with_memory.update({
            'use_memory_bank': True,
            'is_memory_bank_init': False,
            'current_step': self.current_step,
            'current_substep': 'action',
            'use_global_memory_bank': use_global_mb
        })

        return action_gen_kwargs_with_memory

    def get_viz_kwargs(self, viz_gen_kwargs):
        if not self.is_enabled:
            viz_gen_kwargs.pop("current_step", None)
            viz_gen_kwargs.pop("current_substep", None)
            return viz_gen_kwargs

        if self.verbose:
            print(f"\n=== Step {self.current_step} Visualization Substep ===")

        # Enable memory bank for visualization generation
        viz_gen_kwargs_with_memory = viz_gen_kwargs.copy()
        
        # Use global memory bank for visualization (always available since we're in step >= 1)
        use_global_mb_viz = self.current_step >= 1
        viz_gen_kwargs_with_memory.update({
            'use_memory_bank': True,
            'is_memory_bank_init': False,  # Use existing memory bank for visualization
            'current_step': self.current_step,
            'current_substep': 'visualization',
            'use_global_memory_bank': use_global_mb_viz
        })
        return viz_gen_kwargs_with_memory

    def store_step_memory(self):
        """Stores the current step's K,V pairs into the global memory bank."""
        if not self.is_enabled or not self.model.memory_bank_initialized:
            return

        # Store current step's intra-step K,V to global cross-step memory bank
        # This happens after both action prediction and visualization substeps are completed
        if hasattr(self.model, 'store_to_global_memory_bank'):
            self.model.store_to_global_memory_bank(self.current_step)
            if self.verbose:
                print(f"  Step {self.current_step}: Stored intra-step K,V to global memory bank")

                if hasattr(self.model, 'model') and hasattr(self.model.model, 'model') and hasattr(
                        self.model.model.model, 'layers'):
                    for layer_idx, layer in enumerate(self.model.model.model.layers):
                        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'global_stored_keys'):
                            if len(layer.self_attn.global_stored_keys) > 0:
                                print(
                                    f"    - Layer {layer_idx}: Global memory bank now has {len(layer.self_attn.global_stored_keys)} steps")
                                print(
                                    f"    - Layer {layer_idx}: Latest stored K shape: {layer.self_attn.global_stored_keys[-1].shape}")
                                print(
                                    f"    - Layer {layer_idx}: Latest stored V shape: {layer.self_attn.global_stored_values[-1].shape}")
                                break  # Only print for first layer to avoid spam
