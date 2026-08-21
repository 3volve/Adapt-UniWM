CKPT="./checkpoints/base_ckpt"
DATASET="go_stanford"
RUN_ID="$DATASET"_offline_eval
TARGET_GPU=1

mkdir ./output/"$RUN_ID"

torchrun --nproc_per_node=1 --master_port=20007 ../train.py \
  --model anole \
  --model_ckpt "$CKPT" \
  --data "$DATASET" \
  --data_dir ./eval_data \
  --decoder_type anole \
  --image_seq_length 784 \
  --input_format anole \
  --output ./output \
  --note "$RUN_ID"_onestep \
  --report_to none \
  --val_bz 1 \
  --do_single_step_eval \
  --bfloat16