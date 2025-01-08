local hf_model_name = 'meta-llama/Llama-3.1-8B';



local query_template = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You will receive a context paragraph and a question related to the paragraph. Provide a list of answers to the question which are provided in the paragraph, even if those answers are incorrect. Separate each answer with a semicolon. If there are no answers to the question in the paragraph, write "No Answer".

### Context Paragraph:
{generated_paragraph}

### Question:
{question}

### Response:';

local response_template = '
{target}';


(import 'gvar.jsonnet')
+ (import 'runtimes/policy_iteration.jsonnet')
+ (import 'trainers/mle.jsonnet')
+ (import 'episode_generators/sft.jsonnet')
+ {
    episode_generator+: {
        query_template: query_template,
        response_template: response_template,
        append_bos_to_query: true,
        append_eos_to_response: true,
        task: (import 'tasks/linguistic_calibration/answer_extraction.jsonnet')
    },

    model+: {
        type: 'pretrained_causal_lm',
        hf_model_name: hf_model_name,
    },

    tokenizer+: {
        type: 'pretrained',
        hf_model_name: hf_model_name,
    },

    trainer+: {
        type: 'mle',
        num_epochs_per_iteration: 2,
        training_args+: {
            learning_rate: 5e-5,
            weight_decay: 0.00,
            warmup_ratio: 0.03,

            save_steps: 50,
            checkpoint_keep_steps: 50,

            max_seq_len: 2048,

            // Total batch size for training = 64 (4 GPUs)
            per_device_train_batch_size: 2,
            gradient_accumulation_steps: 8,
            gradient_checkpointing: false,
        },
        loss_reduction_mode: 'per_instance_non_pad_tokens_then_batch',
        deepspeed_config: (import 'deepspeed/zero_2.jsonnet'),
    },
    use_deepspeed: true,

    num_episodes_per_iteration: null,
    num_iterations: 1,
}
