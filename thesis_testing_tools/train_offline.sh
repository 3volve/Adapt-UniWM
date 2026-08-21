export CUDA_VISIBLE_DEVICES=0,1

TARGET_GPU=0,1
MANIFEST="./cfg/eval_dataset_manifest.json"
CKPT="./checkpoints/base_ckpt"
DATASET="go_stanford,sacson,recon,scand"
Run_ID="$DATASET"_offline_training

# --init_lora_ckpt "$CKPT" \
torchrun --nproc_per_node=2 --master_port=20009 ../train.py \
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
