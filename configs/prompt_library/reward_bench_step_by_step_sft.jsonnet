local tree_question_template = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{query}

### Response:';

{
    prompt_library+: {
        tree+: {
            question_template: tree_question_template,
        },
    },
}
