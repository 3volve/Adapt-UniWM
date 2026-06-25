CKPT="./checkpoints/main_ckpt"
DATASET="recon,scand"
Run_ID="$DATASET"_offline_training

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1 safe_run torchrun --nproc_per_node=1 train.py \
  --model anole \
  --model_ckpt "$CKPT" \
  --data "$DATASET" \
  --data_dir ./eval_data \
  --decoder_type anole \
  --image_seq_length 784 \
  --input_format anole \
  --output ./output/"$Run_ID" \
  --note "$Run_ID" \
  --report_to none \
  --do_train \
  --train_bz 1 \
  --val_bz 1 \
  --grad_acc 2 \
  --bfloat16

timeout 0.25

tail -f ~/safe_run.log