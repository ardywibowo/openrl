{
    episode_generator+: {
        generation_pipeline+: {
            type: "generation_pipeline"
            vllm_server+: {
                swap_space: 20,
            },
        },
        
        dataset_pipeline+: {
            type: "dataset_pipeline",
            shuffle_on_each_iteration: true,
            shuffle_before_portion: true,
        }
    }
}
