run="<repo-relative path to an existing prior run's directory>"
action_run="<repo-relative path to a run with a series of actions to repeat"

CUDA_VISIBLE_DEVICES=0 python thesis_testing_tools/run_thesis_pipeline.py \
    --habitat-cfg habitat_uniwm_cfg_fixed_learning.yaml \
    --existing-run "$run" \
    --habitat-action-run "$action_run" \
    --habitat-port 20001
