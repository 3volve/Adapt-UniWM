export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1

TARGET_GPU=0,1
MANIFEST="./cfg/eval_dataset_manifest.json"
CKPT="./checkpoints/base_ckpt"
DATASET="go_stanford,sacson,recon,scand"
Run_ID="$DATASET"_offline_training

# --init_lora_ckpt "$CKPT" \
safe_run --gpu "$TARGET_GPU" torchrun --nproc_per_node=2 --master_port=20009 train.py \
  --model anole \
  --data "$DATASET" \
  --model_ckpt "$CKPT" \
  --data_dir ./eval_data \
  --action_token_manifest "$MANIFEST" \
  --decoder_type anole \
  --image_seq_length 784 \
  --input_format anole \
  --output ./output/"$Run_ID" \
  --note "$Run_ID" \
  --report_to None \
  --do_single_step_eval \
  --do_train \
  --train_bz 1 \
  --val_bz 1 \
  --bfloat16

timeout 1

OUTPUT_FILE="$HOME/safe_run_gpu_${TARGET_GPU//,/_}.log"
touch "$OUTPUT_FILE"
tail -f "$OUTPUT_FILE"

#stop_run "$TARGET_GPU"
