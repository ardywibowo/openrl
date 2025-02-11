local tree_question_template = '<|system|> You are a helpful AI assistant.<|end|><|user|> {query}
Please reason step by step, and put your final answer within \\boxed{{}}.<|end|><|assistant|>';

{
    prompt_library+: {
        tree+: {
            question_template: tree_question_template,
        },
    },
}
