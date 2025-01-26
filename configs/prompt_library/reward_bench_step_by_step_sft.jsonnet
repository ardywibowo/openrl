local tree_expansion_iid = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} top_p={top_p} max_tokens={max_tokens} seed={seed} save_stop_text="stop_text" n={num_samples}}}';
local tree_question_template = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{query}

### Response:';

{
    prompt_library+: {
        tree+: {
            expansion+: {
                iid: tree_expansion_iid,
            },
            question_template: tree_question_template,
        },
    },
}
