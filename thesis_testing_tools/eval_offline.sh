CKPT="./checkpoints/base_ckpt"

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=20000 ../train.py \
  --model anole \
  --model_ckpt "$CKPT" \
  --data go_stanford \
  --data_dir ./eval_data \
  --decoder_type anole \
  --image_seq_length 784 \
  --input_format anole \
  --output ./outputs/prelim_eval \
  --note gostanford_ckpt_singlestep \
  --report_to none \
  --val_bz 1 \
  --do_single_step_eval \
  --bfloat16