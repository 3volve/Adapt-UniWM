# Fixed Habitat action reference

`episode_logs.json` is copied from the no-learning Habitat stage of thesis run
`thesis_pipeline_20260804_181441/ablation_no_learning/habitat`.

The three published Habitat configurations use its action sequences so that the
no-learning, fixed-learning, and modulated-learning conditions execute the same
episodes and actions. `HabitatEpisodeAdapter._load_fixed_actions_from_run` reads
only `episode_id` and each step's `action` from this file; the rest of the source
run is not required to reproduce the forced-action protocol.
