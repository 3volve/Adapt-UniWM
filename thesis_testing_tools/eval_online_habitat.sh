export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0,1


CFG_PATH="cfg/habitat_uniwm_cfg_no_learning.yaml"
DATASET="habitat"
OUTPUT_DIR="./output"
RUN_DIR="output/thesis_preflight_2e4/no_learning/habitat"
NUM_EPISODES=3
torchrun --nproc_per_node=1 --master_port=20011 ../runtime_scripts/uniwm_episode_runner.py \
  --config_path "$CFG_PATH" \
  --data_id "$DATASET" \
  --run_dir "$RUN_DIR" \
  --num_episodes "$NUM_EPISODES"