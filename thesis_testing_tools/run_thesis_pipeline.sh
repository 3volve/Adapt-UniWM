MONITORED_GPU=1
run="output/thesis_pipeline_20260812_235656"
action_run="output/thesis_pipeline_20260804_181441/ablation_no_learning/habitat"


if command -v safe_run &> /dev/null; then
    safe_run --gpu 0 python thesis_testing_tools/run_thesis_pipeline.py --habitat-cfg habitat_uniwm_cfg_fixed_learning.yaml --existing-run "$run" --habitat-action-run "$action_run" --habitat-port 20001
    #sleep 1

    safe_run --gpu 1 python thesis_testing_tools/run_thesis_pipeline.py --habitat-cfg habitat_uniwm_cfg_modulated_learning.yaml --existing-run "$run" --habitat-action-run "$action_run" --habitat-port 20003
    #sleep 1

    OUTPUT_FILE="$HOME/safe_run_gpu_${MONITORED_GPU//,/_}.log"
    touch "$OUTPUT_FILE"
    tail -f "$OUTPUT_FILE"

    stop_run "$MONITORED_GPU"
else
    CUDA_VISIBLE_DEVICES=0 python thesis_testing_tools/run_thesis_pipeline.py --habitat-cfg habitat_uniwm_cfg_fixed_learning.yaml --existing-run "$run" --habitat-action-run "$action_run" --habitat-port 20001
    CUDA_VISIBLE_DEVICES=1 python thesis_testing_tools/run_thesis_pipeline.py --habitat-cfg habitat_uniwm_cfg_modulated_learning.yaml --existing-run "$run" --habitat-action-run "$action_run" --habitat-port 20003
fi