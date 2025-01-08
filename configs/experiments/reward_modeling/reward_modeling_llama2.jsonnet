local hf_model_name = 'meta-llama/Llama-3.1-8B';



local query_template = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{query}

### Response:';

local chosen_template = '
{chosen}';

local rejected_template = '
{rejected}';

(import 'gvar.jsonnet')
+ (import 'runtimes/policy_iteration.jsonnet')
+ (import 'trainers/reward_modeling.jsonnet')
+ (import 'episode_generators/reward_modeling.jsonnet')
+ {
    episode_generator+: {
        query_template: query_template,
        chosen_template: chosen_template,
        rejected_template: rejected_template,
        append_bos_to_query: true,
        append_eos_to_response: true,
        task: (import 'tasks/reward_modeling/reward_bench.jsonnet')
    },

    model+: {
        type: 'pretrained_causal_lm_with_value_head',
        pretrained_backbone_model: {
            type: 'pretrained_causal_lm',
            hf_model_name: hf_model_name,
        },
        value_head_dropout: null
    },

    tokenizer+: {
        type: 'pretrained',
        hf_model_name: hf_model_name,
    },

    trainer+: {
        type: 'reward_modeling',
        num_epochs_per_iteration: 4,
        training_args+: {
            learning_rate: 1e-6,
            weight_decay: 0.0001,
            warmup_ratio: 0.10,

            save_steps: 400,
            checkpoint_keep_steps: 50,

            max_seq_len: 2048,

            // Total batch size for training = 64 (4 GPUs)
            per_device_train_batch_size: 2,
            gradient_accumulation_steps: 8,
            gradient_checkpointing: false,
        },
        finetune_iterations: 3,
        deepspeed_config: (import 'deepspeed/zero_2.jsonnet'),
    },
    use_deepspeed: true,

    num_episodes_per_iteration: null,
    num_iterations: 6,
}
