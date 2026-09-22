run="<repo-relative path to an existing prior run's directory>"

CUDA_VISIBLE_DEVICES=0 python thesis_testing_tools/run_thesis_pipeline.py \
    --habitat-cfg habitat_uniwm_cfg_fixed_learning.yaml \
    --existing-run "$run" \
    --habitat-port 20001
