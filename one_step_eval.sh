CKPT="./checkpoints/main_ckpt"
DATASET="go_stanford"
Run_ID="$DATASET"_offline_eval

mkdir ./output/"$Run_ID"

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 safe_run torchrun --nproc_per_node=2 train.py \
  --model anole \
  --model_ckpt "$CKPT" \
  --data "$DATASET" \
  --data_dir ./eval_data \
  --decoder_type anole \
  --image_seq_length 784 \
  --input_format anole \
  --output ./output \
  --note "$Run_ID"_onestep \
  --report_to none \
  --val_bz 1 \
  --do_single_step_eval \
  --bfloat16

timeout 0.25
  
tail -f ~/safe_run.log