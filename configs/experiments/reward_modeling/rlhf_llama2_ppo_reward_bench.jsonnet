local hf_model_name = 'meta-llama/Llama-3.1-8B';

local actor_tokenizer = {
    type: 'pretrained',
    hf_model_name: hf_model_name,
};

local task = (import 'tasks/reward_modeling/reward_bench.jsonnet');

local num_episodes_per_iteration = 512;
local num_rollouts_per_sample = 8;
local num_dataset_samples_per_iteration = num_episodes_per_iteration / num_rollouts_per_sample;
local total_num_iterations = 1000;

local sampling_temperature = 0.6;

(import 'gvar.jsonnet')
+ (import 'prompt_library/reward_bench_step_by_step_sft.jsonnet')
+ (import 'runtimes/policy_iteration.jsonnet')
+ (import 'episode_generators/reward_bench_llama2_reward_model_episode_generator.jsonnet')
+ (import 'trainers/ppo_reward_bench.jsonnet')
+ {
    episode_generator+: {
        // Override the task
        task: task,
        reasoning_step_delimiter: '',

        initial_model_name_or_path: hf_model_name,

        dataset_sample_with_replacement: true,
        dataset_num_samples_per_iteration: num_dataset_samples_per_iteration,
        total_num_iterations: $.num_iterations,

        save_generations_every_n_iteration: 50,

        inference_strategy: {
            type: 'cot',

            max_concurrent_programs: 128,
            max_concurrent_generations: 64,

            samples: num_rollouts_per_sample,
            max_depth: 100,  // Deprecated parameter. Doesn't do anything.

            node_expander: {
                type: 'efficient_iid',
                program: $.prompt_library.tree.expansion.iid,
                program_kwargs+: {
                    temperature: sampling_temperature,
                    top_p: 0.9,
                    max_tokens: 1024,
                },
                node_text_template: '{chain_of_thought}',

                // Needed to compute max_tokens on the fly
                model_context_size: 4095,
                tokenizer: $.tokenizer,
            },

            answer_extractor: {
                type: 'identity',
                node_key_name: 'text',
            },

            guidance_llm: (import 'guidance_llms/deepseekmath7b-sft-MATH-v2.jsonnet') + { api_base: 'none' },

            question_field: 'query',
            question_template: $.prompt_library.tree.question_template,

            no_cache: true,
        },
    },

    tokenizer: actor_tokenizer,
    use_deepspeed: true,

    num_iterations: total_num_iterations,
    num_episodes_per_iteration: num_episodes_per_iteration,
    episodes_cloud_log_steps: 50,

    trainer+: {
        params+: { temperature: $.episode_generator.inference_strategy.node_expander.program_kwargs.temperature },

        actor_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path },
        critic_model+: { pretrained_backbone_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path } },
        reference_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path },

        // To prevent OOM errors
        report_entropy: false,

        general_training_args+: {
            save_steps: 30,
            checkpoint_keep_steps: 60,
        },
    },

    analyzers: [
        (import 'analyzers/valnet_prediction.jsonnet') + {
            task: $.episode_generator.task,
            tokenizer: $.tokenizer,
            vllm_server+: { swap_space: 64 },

            reward_function: $.episode_generator.reward_function,

            max_num_requests: 512,

            inference_strategy+: {
                guidance_llm: $.episode_generator.inference_strategy.guidance_llm,

                max_concurrent_programs: 32,
                max_concurrent_generations: 16,

                node_expander+: {
                    program_kwargs+: { temperature: $.episode_generator.inference_strategy.node_expander.program_kwargs.temperature },
                    model_context_size: $.episode_generator.max_sequence_length,
                    tokenizer: $.tokenizer,
                },
            },
        },

        (import 'analyzers/ppo_grad_variance.jsonnet') + {
            per_device_batch_size: $.trainer.general_training_args.per_device_train_batch_size,
        },

        (import 'analyzers/valnet_action_ranking.jsonnet') + {
            task: $.episode_generator.task,
            tokenizer: $.tokenizer,
            vllm_server+: { swap_space: 64 },

            reward_function: $.episode_generator.reward_function,

            max_num_requests: 512,
            max_num_states: 128,

            append_bos_to_query: $.episode_generator.append_bos_to_query,

            inference_strategy+: {
                guidance_llm: $.episode_generator.inference_strategy.guidance_llm,

                // Small model. Can afford more concurrent programs.
                max_concurrent_programs: 32,
                max_concurrent_generations: 16,

                node_expander+: {
                    program_kwargs+: { temperature: $.episode_generator.inference_strategy.node_expander.program_kwargs.temperature },
                    model_context_size: $.episode_generator.inference_strategy.node_expander.model_context_size,
                    tokenizer: $.tokenizer,
                },
            },
        },
    ],
}
+ (import 'trainers/lam1.jsonnet')
+ (import 'trainers/refKl0.0001.jsonnet')
+ (import 'trainers/klLoss.jsonnet')
// + (import 'sft_deepseekmath_for_MATH_eval.jsonnet')
