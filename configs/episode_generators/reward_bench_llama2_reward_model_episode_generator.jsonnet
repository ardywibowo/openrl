local hf_model_name = "experiments/reward_modeling_llama2_20250123_210003/checkpoints/ckpt--iter_0002--epoch_1.00--step_0047/hf_pretrained/";

local reward_bench_task = (import '../tasks/reward_bench.jsonnet');

local prompt_library = (import '../prompt_library/reward_bench_step_by_step_sft.jsonnet');
local question_template = prompt_library.prompt_library.tree.question_template;

{
    episode_generator+: {
        type: 'episode_generator_with_reward_function',

        append_bos_to_query: true,
        append_eos_to_response: true,

        dataset_shuffle_on_each_iteration: true,
        dataset_shuffle_before_portion: true,
        dataset_sample_with_replacement: false,
        
        vllm_server_handler+: {
            vllm_server+: {
                swap_space: 16,
            },
            gpu_memory_utilization: 'auto',
            min_available_gpu_memory_mb: 20 * 1024,
            wait_until_memory_release: true,
        },

        reward_function: {
            type: 'model_based_reward_function',
            reward_model+: {
                type: 'pretrained_causal_lm_with_value_head',
                pretrained_backbone_model: {
                    type: 'pretrained_causal_lm',
                    hf_model_name: hf_model_name,
                },
                value_head_dropout: null
            },

            penalize_unfinished_response: true,
            unfinished_response_penalty: 0.0,
        },

        // max_sequence_length: 2048,
        max_sequence_length: 4096,
        max_question_length: 1512,
        question_template: question_template,

        fill_missing_episodes: true,

        task: reward_bench_task,
    },
}
