{
    episode_generator+: {
        inference_server_handler+: {
            inference_server+: {
                type: "sglang",
                swap_space: 8,
            },
        },
        dataset_shuffle_on_each_iteration: true,
        dataset_shuffle_before_portion: true,
    }
}
