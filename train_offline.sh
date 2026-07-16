export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1

TARGET_GPU=0,1
CKPT="./output/recon,scand_offline_training/recon,scand_offline_trainingimage_seq_len-784-train-anole-hyper-train1val1lr0.0002-recon,scand-prompt_anole-42/checkpoint-4200"
DATASET="recon,scand"
Run_ID="$DATASET"_offline_training


safe_run --gpu "$TARGET_GPU" torchrun --nproc_per_node=2 --master_port=20009 train.py \
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

timeout 1

OUTPUT_FILE="$HOME/safe_run_gpu_${TARGET_GPU//,/_}.log"
touch "$OUTPUT_FILE"
tail -f "$OUTPUT_FILE"

stop_run "$TARGET_GPU"