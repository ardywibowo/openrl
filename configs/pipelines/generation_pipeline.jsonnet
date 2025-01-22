{
    type: 'generation_pipeline',
    inference_strategy: Lazy[InferenceStrategy],
    vllm_server: {
        swap_space: 16,
    },
    model_name_or_path: "",
    vllm_gpu_memory_utilization: 'auto',
    vllm_min_available_gpu_memory_mb: 20 * 1024,
    save_generations_every_n_iteration: 50,
    wait_until_memory_release: true,
}
