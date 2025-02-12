local math_task = (import '../tasks/math.jsonnet') + {
    prepend_in_context_few_shot: false,
    ensure_fit_in_context_size: false,
};

local prompt_library = (import '../prompt_library/MATH_step_by_step_sft.jsonnet');
local question_template = prompt_library.prompt_library.tree.question_template;

{
    episode_generator+: {
        type: 'math_episode_generator',

        append_bos_to_query: true,
        append_eos_to_response: true,

        dataset_shuffle_on_each_iteration: true,
        dataset_shuffle_before_portion: true,
        dataset_sample_with_replacement: false,
        
        inference_server+: {
            type: "sglang",
            gpu_memory_utilization: 0.9,
            swap_space: 8,
        },

        reward_function: {
            type: 'math_reward_function',
            penalize_unfinished_response: true,
            unfinished_response_penalty: 0.0,
            math_task: $.episode_generator.task,
        },
        reasoning_step_delimiter: '\n',
        answer_prefix: '\n\n# Answer\n',

        // max_sequence_length: 2048,
        max_sequence_length: 4096,
        max_question_length: 1512,
        question_template: question_template,

        fill_missing_episodes: true,

        task: math_task,
    },
}
