DATA_TYPE="habitat"
DATASET="habitat"
OUTPUT_DIR="./output"
NUM_EPISODES=3
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=1

safe_run torchrun --nproc_per_node=1 uniwm_episode_runner.py \
  --data_type "$DATA_TYPE" \
  --data_id "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --num_episodes "$NUM_EPISODES"

timeout 1
  
tail -f ~/safe_run.log