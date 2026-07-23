import torch
from peft import LoraConfig, get_peft_model
from peft.peft_model import PeftModel

from transformers import AutoProcessor
from scripts.action_utils import ActionTokenVocabulary

def load_model(args, training_cfg, action_vocabulary: ActionTokenVocabulary):
    print("image_seq_length received:", args.image_seq_length)
    model_name = args.model

    model_ckpt_path = args.model_ckpt

    if model_name in ['anole']:
        image_token_num = args.image_seq_length

        # Use MemoryBankAnoleForConditionalGeneration for prediction tasks
        if args.use_memory_bank_inference and args.do_task_level_eval and not args.do_train:
            from uniwm.memory_bank import MemoryBankAnoleForConditionalGeneration
            print("Loading MemoryBankAnoleForConditionalGeneration for prediction task")
            model = MemoryBankAnoleForConditionalGeneration.from_pretrained(
                "leloy/Anole-7b-v0.1-hf",
                device_map="cuda",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                codebook_sim="mse"
            )
        else:
            from uniwm.wrapped_visualizer import AnoleforConditionalGeneration
            model = AnoleforConditionalGeneration.from_pretrained(
                "leloy/Anole-7b-v0.1-hf",
                device_map="cuda",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                codebook_sim="mse"
            )

        # Conditionally load processor from ckpt in inference to match extended vocab size
        # NEW: Now also conditionally load just the lora weights for training whether inference or not
        init_lora_ckpt_path = getattr(args, "init_lora_ckpt", None)
        resume_ckpt_path = getattr(args, "resume_ckpt_path", None)
        peft_ckpt_path = resume_ckpt_path or model_ckpt_path

        is_inference_only = (
            args.do_single_step_eval or args.do_task_level_eval or args.do_rollout_eval
        ) and not args.do_train and peft_ckpt_path

        processor_ckpt_path = init_lora_ckpt_path or peft_ckpt_path

        if processor_ckpt_path:
            print(f"Loading processor from checkpoint: {processor_ckpt_path}")
            processor = AutoProcessor.from_pretrained(
                processor_ckpt_path,
                image_seq_length=image_token_num,
            )
        else:
            print("Loading processor from base.")
            processor = AutoProcessor.from_pretrained(
                "leloy/Anole-7b-v0.1-hf",
                image_seq_length=image_token_num,
            )

        # NEW: Always resize base model if tokenizer size doesn't match (handles inference case)
        tokenizer_size = len(processor.tokenizer)
        model_embedding_size = model.get_input_embeddings().weight.shape[0]
        print(f"Tokenizer size: {tokenizer_size}. Base model embedding size: {model_embedding_size}")
        if model_embedding_size != tokenizer_size:
            print("Resizing model embeddings to match tokenizer size.")
            model.resize_token_embeddings(tokenizer_size)

        # NEW: Set padding_side attribute (required for correct generation)
        processor.tokenizer.padding_side = "left"

        # NEW: Monkey patch to ignore 'padding_side' kwarg in tokenizer (workaround for library bug)
        def patched_batch_encode_plus(self, *args, **kwargs):
            kwargs.pop('padding_side', None)  # Remove if library passes it
            return self.__original_batch_encode_plus(*args, **kwargs)

        if not hasattr(processor.tokenizer, '__original_batch_encode_plus'):
            processor.tokenizer.__original_batch_encode_plus = processor.tokenizer._batch_encode_plus
            processor.tokenizer._batch_encode_plus = patched_batch_encode_plus.__get__(processor.tokenizer)

        # Rest of your original code unchanged
        img_size = training_cfg.get("img_size", 448)
        processor.image_processor.size = {"shortest_edge": img_size}
        processor.image_processor.crop_size = {
            "height": img_size,
            "width": img_size
        }

        model.config.pad_token_id = processor.tokenizer.pad_token_id
        
        model.model.vqmodel.config.resolution = processor.image_processor.size["shortest_edge"]
        model.model.vqmodel.quantize.quant_state_dims = [
            model.model.vqmodel.config.resolution // 2 ** (len(model.model.vqmodel.config.channel_multiplier) - 1)
        ] * 2

        args.sketch_resolution = model.model.vqmodel.config.resolution
        model.sketch_resolution = (args.sketch_resolution, args.sketch_resolution)
        model.image_token_num = image_token_num

        model.get_vis_codebook_sim()
        
        action_token_ids, input_module_name, output_module_name = register_action_tokens(
            model,
            processor,
            action_vocabulary,
            loading_checkpoint=processor_ckpt_path is not None,
        )

        config = config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",

            modules_to_save=None,

            trainable_token_indices={
                "model.embed_tokens": action_token_ids,
                "lm_head": action_token_ids,
            },
        )

        if init_lora_ckpt_path:
            print(f"Initializing trainable LoRA from checkpoint: {init_lora_ckpt_path}")
            lora_model = PeftModel.from_pretrained(
                model,
                init_lora_ckpt_path,
                is_trainable=True,
            )
        elif is_inference_only:
            lora_model = PeftModel.from_pretrained(
                model,
                peft_ckpt_path,
                is_trainable=False,
            )
        else:
            lora_model = get_peft_model(model, config)
                
        return {
            'processor': processor,
            'model': lora_model
        }
    else:
        raise ValueError("Unsupported model type. ")
    
    
def register_action_tokens(
    model,
    processor,
    action_vocabulary: ActionTokenVocabulary,
    *,
    loading_checkpoint: bool,
) -> tuple[list[int], str, str]:
    tokenizer = processor.tokenizer
    allowed_tokens = action_vocabulary.all_tokens

    if len(allowed_tokens) != len(set(allowed_tokens)):
        raise ValueError("Action vocabulary contains duplicate tokens.")

    existing_vocab = tokenizer.get_vocab()
    missing_tokens = [
        token for token in allowed_tokens
        if token not in existing_vocab
    ]

    # Checkpoint token IDs are part of its trained model contract. Never add
    # missing tokens in a potentially different order.
    if loading_checkpoint and missing_tokens:
        raise ValueError(
            f"Checkpoint tokenizer is missing {len(missing_tokens)} action "
            f"tokens. First missing tokens: {missing_tokens[:5]}"
        )

    if missing_tokens:
        added_count = tokenizer.add_tokens(
            missing_tokens,
            special_tokens=True,
        )
        if added_count != len(missing_tokens):
            raise RuntimeError(
                f"Expected to add {len(missing_tokens)} action tokens, "
                f"but tokenizer added {added_count}."
            )

    tokenizer_size = len(tokenizer)
    input_size = model.get_input_embeddings().weight.shape[0]
    output_size = model.get_output_embeddings().weight.shape[0]

    if input_size != tokenizer_size or output_size != tokenizer_size:
        model.resize_token_embeddings(tokenizer_size)

    input_size = model.get_input_embeddings().weight.shape[0]
    output_size = model.get_output_embeddings().weight.shape[0]
    if input_size != tokenizer_size or output_size != tokenizer_size:
        raise RuntimeError(
            "Embedding resize failed: "
            f"tokenizer={tokenizer_size}, input={input_size}, output={output_size}"
        )

    resolved_vocab = tokenizer.get_vocab()
    action_token_ids = [resolved_vocab[token] for token in allowed_tokens]

    if len(action_token_ids) != len(set(action_token_ids)):
        raise ValueError("Action tokens do not map to unique token IDs.")

    def find_module_name(target_module) -> str:
        for name, module in model.named_modules():
            if module is target_module:
                return name
        raise RuntimeError("Could not find embedding module name.")

    input_module_name = find_module_name(model.get_input_embeddings())
    output_module_name = find_module_name(model.get_output_embeddings())

    if input_module_name == output_module_name:
        raise RuntimeError(
            "Expected Chameleon input and output embeddings to be untied."
        )

    print(f"Registered {len(action_token_ids)} action tokens.")
    print(f"Input action rows:  {input_module_name}")
    print(f"Output action rows: {output_module_name}")

    return action_token_ids, input_module_name, output_module_name