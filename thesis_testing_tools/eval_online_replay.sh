export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0,1

CFG_PATH="cfg/replay_uniwm_cfg.yaml"
DATASET="sacson"
OUTPUT_DIR="./output"
NUM_EPISODES=3
torchrun --nproc_per_node=1 --master_port=20002 ../runtime_scripts/uniwm_episode_runner.py \
  --config_path "$CFG_PATH" \
  --data_id "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --num_episodes "$NUM_EPISODES"