local hf_model_name = 'meta-llama/Llama-3.2-1B-Instruct';

local tokenizer = {
    type: 'pretrained',
    hf_model_name: hf_model_name,
};

local num_episodes_per_iteration = 512;
local num_rollouts_per_sample = 8;
local num_dataset_samples_per_iteration = num_episodes_per_iteration / num_rollouts_per_sample;
local total_num_iterations = 1000;
local sampling_temperature = 0.8;

local ds_stage_2_w_cpu_optimizer = (import 'deepspeed/zero_2.jsonnet') + {
    zero_optimization+: {
        offload_optimizer+: {
            device: 'cpu',
            pin_memory: true,
        },
    },
};

local math_task = (import 'tasks/math.jsonnet') + {
    answer_prefix: null,
    inplace_split_solution: true,
    prepend_in_context_few_shot: false,
    ensure_fit_in_context_size: false,
};

local system_prompt = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
';

(import 'gvar.jsonnet')
+ {
    type: 'policy_iteration',
    tokenizer: tokenizer,
    use_deepspeed: true,

    num_iterations: total_num_iterations,
    num_episodes_per_iteration: num_episodes_per_iteration,
    episodes_cloud_log_steps: 50,
    
    episode_generator+: {
        type: 'grpo_episode_generator',
        
        // Override the task
        task: math_task,
        reasoning_step_delimiter: '',
        answer_prefix: null,
        
        initial_model_name_or_path: hf_model_name,
        dataset_num_samples_per_iteration: num_dataset_samples_per_iteration,
        
        save_generations_every_n_iteration: 50,
        append_bos_to_query: true,
        append_eos_to_response: true,
        
        inference_strategy: {
            type: 'reasoning',
            num_samples: num_rollouts_per_sample,
            
            sampling_parameters+: {
                temperature: sampling_temperature,
                top_p: 0.9,
                max_tokens: 1024,
                stop: "\n\n\nProblem:",
            },
            system_prompt: system_prompt,
            question_template: "{query}",
            tokenizer: tokenizer,
        },
        
        inference_server+: {
            type: "sglang",
            gpu_memory_utilization: 0.9,
            swap_space: 8,
        },

        reward_function: {
            type: 'simple_math_reward_function',
        },
    },
    
    trainer+: {
        type: 'grpo',
        
        // To prevent OOM errors
        report_entropy: false,

        actor_model+: {
            type: 'pretrained_causal_lm',
            hf_model_name: hf_model_name,
            disable_dropout: true,
            pretrained_args+: {
                use_flash_attention_2: true,
            },
        },
        actor_deepspeed_config: ds_stage_2_w_cpu_optimizer,
        
        reference_model+: {
            type: 'pretrained_causal_lm',
            hf_model_name: hf_model_name,
            pretrained_args+: {
                use_flash_attention_2: true,
            },
        },
        reference_deepspeed_config: {
            bf16: { enabled: true },
            wall_clock_breakdown: false,
            prescale_gradients: false,
            gradient_accumulation_steps: 'auto',
            train_batch_size: 'auto',
            train_micro_batch_size_per_gpu: 'auto',
        },
        
        params+: {
            temperature: sampling_temperature,
            
            adap_kl_ctrl: false,
            init_kl_coef: 0.0001,

            gamma: 1.0,
            lam: 1.0,
            
            kl_penalty_loss_type: 'control_variate',
            kl_penalty_loss_clip_max: 10,
            kl_penalty_loss_clip_min: 0,

            cliprange: 0.2,
            cliprange_value: 0.2,
        },

        general_training_args: {
            save_steps: 30,
            checkpoint_keep_steps: 60,
            target_train_batch_size: 64,

            per_device_train_batch_size: 8,

            learning_rate: 1e-6,
            weight_decay: 0.00,
            warmup_ratio: 0.03,

            max_grad_norm: 1.0,

            dataloader_num_workers: 1,
            dataloader_pin_memory: false,

            gradient_checkpointing: true,
            bf16: true,

            logging_steps: 1,
            seed: std.parseInt(std.extVar('APP_SEED')),
        },

        num_epochs_per_iteration: 2,
        cache_deepspeed_engines: true,
        move_reference_model_to_cpu: true,
        save_hf_critic_checkpoint: true,
    },
}