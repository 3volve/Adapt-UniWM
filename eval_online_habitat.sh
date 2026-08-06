export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1
TARGET_GPU=0

CFG_PATH="cfg/habitat_uniwm_cfg.yaml"
DATASET="habitat"
OUTPUT_DIR="./output"
NUM_EPISODES=3
safe_run --gpu "$TARGET_GPU" torchrun --nproc_per_node=1 --master_port=20001 uniwm_episode_runner.py \
  --config_path "$CFG_PATH" \
  --data_id "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --num_episodes "$NUM_EPISODES"

sleep 1

OUTPUT_FILE="$HOME/safe_run_gpu_${TARGET_GPU//,/_}.log"
touch "$OUTPUT_FILE"
tail -f "$OUTPUT_FILE"

stop_run "$TARGET_GPU"