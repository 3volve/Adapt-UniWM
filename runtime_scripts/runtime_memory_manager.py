
import torch
from torch import nn


class RuntimeMemoryBankManager:
    def __init__(self, model: nn.Module, use_memory_bank_inference: bool, verbose: bool = False):
        self.model = model
        self.is_enabled = use_memory_bank_inference
        self.current_step = 0
        self.verbose = verbose

    def setup_for_episode(self, episode_id: str = None):
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
            self.model.memory_bank_initialized = False
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

    def get_action_kwargs(self, action_inputs, action_gen_kwargs, is_real_obs=True):
        if not self.is_enabled or not is_real_obs:
            action_gen_kwargs.pop("current_step", None)
            action_gen_kwargs.pop("current_substep", None)
            return action_gen_kwargs

        step = self.current_step - 1

        # Check for memory bank initialization (third pair of 8197 and 8196 tokens)
        input_ids_list = action_inputs['input_ids'][0].tolist()
        # Always try to initialize memory bank for each step (intra-step memory bank)
        if hasattr(self.model, 'initialize_memory_bank') and not getattr(self.model, 'memory_bank_initialized', False):

            # Count pairs of 8197 and 8196 tokens
            pairs_count = 0
            i = 0
            while i < len(input_ids_list) - 1:
                if input_ids_list[i] == 8197:
                    for j in range(i + 1, len(input_ids_list)):
                        if input_ids_list[j] == 8196:
                            pairs_count += 1
                            i = j
                            break
                    else:
                        break
                i += 1

            # Initialize memory bank if we have at least 3 pairs, or fallback to any image tokens
            should_initialize = False
            init_method = ""

            if pairs_count >= 3:
                should_initialize = True
                init_method = f"special token pairs (found {pairs_count})"

            if should_initialize:
                if self.verbose:
                    print(f"  Step {step + 1}: Initializing memory bank using {init_method}")

                # try:
                self.model.initialize_memory_bank(
                    input_ids=action_inputs['input_ids'],
                    pixel_values=action_inputs['pixel_values'],
                    attention_mask=action_inputs['attention_mask']
                )
                if self.verbose:
                    print(f"  Step {step + 1}: Memory bank initialization completed successfully")
                    # Print memory bank storage details if available
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'model') and hasattr(
                            self.model.model.model, 'layers'):
                        for layer_idx, layer in enumerate(self.model.model.model.layers):
                            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'stored_keys'):
                                if layer.self_attn.stored_keys is not None:
                                    print(
                                        f"    - Layer {layer_idx}: Stored keys shape: {layer.self_attn.stored_keys.shape}")
                                    print(
                                        f"    - Layer {layer_idx}: Stored values shape: {layer.self_attn.stored_values.shape}")
                                    print(
                                        f"    - Layer {layer_idx}: Memory bank storage size: {layer.self_attn.stored_keys.numel() + layer.self_attn.stored_values.numel()} elements")
                # except Exception as e:
                #     if self.verbose:
                #         print(f"  Warning: Memory bank initialization failed: {e}")
            else:
                if self.verbose:
                    print(f"  Step {step + 1}: Warning - No suitable tokens found for memory bank initialization")

        # Print memory bank usage details before generation
        if hasattr(self.model, 'memory_bank_initialized') and self.model.memory_bank_initialized:
            if self.verbose:
                print(f"  Step {step + 1}: Using memory bank for action generation")

                # Print stored K,V sizes if available
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'model') and hasattr(
                        self.model.model.model, 'layers'):
                    for layer_idx, layer in enumerate(self.model.model.model.layers):
                        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'stored_keys'):
                            if layer.self_attn.stored_keys is not None:
                                print(
                                    f"    - Layer {layer_idx}: Using stored K shape: {layer.self_attn.stored_keys.shape}")
                                print(
                                    f"    - Layer {layer_idx}: Using stored V shape: {layer.self_attn.stored_values.shape}")

        with torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
            # Two-phase action generation with memory bank
            action_gen_kwargs_with_memory = action_gen_kwargs.copy()
            # Remove any existing memory bank parameters to avoid conflicts
            action_gen_kwargs_with_memory.pop('use_memory_bank', None)
            action_gen_kwargs_with_memory.pop('is_memory_bank_init', None)
            action_gen_kwargs_with_memory.pop('current_step', None)
            action_gen_kwargs_with_memory.pop('current_substep', None)
            action_gen_kwargs_with_memory.pop('use_global_memory_bank', None)

            # Use global memory bank if we have previous steps (current_step > 1)
            use_global_mb = self.current_step > 1

            # Phase 1: Initialize memory bank (dummy generation to extract K,V)
            init_kwargs = action_gen_kwargs_with_memory.copy()
            init_kwargs.update({
                'use_memory_bank': True,
                'is_memory_bank_init': True,  # Initialize memory bank
                'current_step': self.current_step,
                'current_substep': 'action',
                'use_global_memory_bank': False,  # Don't use global during init
                'max_new_tokens': 1  # Minimal generation for initialization
            })

            if self.verbose:
                print(f"  Step {step + 1}: Memory bank initialization phase")

            _ = self.model.generate(**action_inputs, **init_kwargs)

            # Phase 2: Actual action generation using initialized memory bank
            gen_kwargs = action_gen_kwargs_with_memory.copy()
            gen_kwargs.update({
                'use_memory_bank': True,
                'is_memory_bank_init': False,  # Use existing memory bank
                'current_step': self.current_step,
                'current_substep': 'action',
                'use_global_memory_bank': use_global_mb
            })

            if self.verbose:
                print(f"  Step {step + 1}: Action generation using memory bank (global: {use_global_mb})")
            return gen_kwargs

    def get_viz_kwargs(self, viz_gen_kwargs, is_real_obs=True):
        if not self.is_enabled or not is_real_obs:
            viz_gen_kwargs.pop("current_step", None)
            viz_gen_kwargs.pop("current_substep", None)
            return viz_gen_kwargs

        if self.verbose:
            print(f"\n=== Step {self.current_step} Visualization Substep ===")

        with torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
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
        if not self.is_enabled:
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