{
    episode_generator+: {
        value_estimation_inference_strategy+: {
            samples: 10,

            node_expander+: {
                type: 'confidence_interval_aware_efficient_iid',
                acceptable_ci_length_threshold: 0.05,
                num_new_rollouts: 4,
                max_num_rollouts: 30,
            },
        },
    },
}
