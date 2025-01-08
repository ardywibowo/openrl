local hf_model_name = 'meta-llama/Llama-3.1-8B';



local query_template = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
I\'m going to give you a Question, Context, and an Additional Question. Based only on the information in the Context, and no other background knowledge that you possess, you will answer the Additional Question by providing a probability from 0 to 1.

Question:
{question}

Context:
{generated_paragraph}

### Input:
Additional Question: Based on the Context, what is the probability that "{ground_truth_top_answer}" is a correct answer to the Question? Pay close attention to the information in the Context salient to the Question. Answer solely based on the above information, ignoring entirely what you know about the world and the commonsense knowledge you have.

### Response:';

(import 'gvar.jsonnet')
+ (import 'runtimes/policy_iteration.jsonnet')
+ (import 'trainers/binary_classification.jsonnet')
+ (import 'episode_generators/binary_classification.jsonnet')
+ {
    episode_generator+: {
        query_template: query_template,
        append_bos_to_query: true,
        append_eos_to_response: true,
        task: (import 'tasks/linguistic_calibration/probability_forecasting.jsonnet')
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
        type: 'binary_classification',
        num_epochs_per_iteration: 4,
        training_args+: {
            learning_rate: 1e-6,
            weight_decay: 0.00,
            warmup_ratio: 0.03,

            save_steps: 400,
            checkpoint_keep_steps: 50,

            max_seq_len: 2048,

            // Total batch size for training = 64 (4 GPUs)
            per_device_train_batch_size: 8,
            gradient_accumulation_steps: 4,
            gradient_checkpointing: false,
        },
        deepspeed_config: (import 'deepspeed/zero_2.jsonnet'),
    },
    use_deepspeed: true,

    num_episodes_per_iteration: null,
    num_iterations: 1,
}
