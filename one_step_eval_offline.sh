CKPT="./checkpoints/base_ckpt"
DATASET="go_stanford"
Run_ID="$DATASET"_offline_eval
TARGET_GPU=1

mkdir ./output/"$Run_ID"

export NCCL_P2P_DISABLE=1

safe_run --gpu "$TARGET_GPU" torchrun --nproc_per_node=1 --master_port=20007 train.py \
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

sleep 1

OUTPUT_FILE="$HOME/safe_run_gpu_${TARGET_GPU//,/_}.log"
touch "$OUTPUT_FILE"
tail -f "$OUTPUT_FILE"

stop_run "$TARGET_GPU"