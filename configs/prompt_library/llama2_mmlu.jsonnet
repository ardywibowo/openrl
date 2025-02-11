local tree_expansion_iid = '{prefix}';

local tree_question_template = '[INST] <<SYS>>
You are a helpful assistant. Always answer in most accurate way.
<</SYS>>
Your task is to answer the following question. First write down your thinking process, then choose the correct choice between A, B, C, and D.

Q: {question}
[/INST]';

local tree_answer_extract_next_chat_turn = '{prefix}</s><s>[INST]So, based on your thought processes and your choice above, which choice did you choose? Only output A, B, C, D.[/INST] Sure! I chose:
';

{
    prompt_library: {
        tree: {
            expansion: {
                iid: tree_expansion_iid,
            },
            question_template: tree_question_template,
            answer_extract: {
                next_chat_turn: tree_answer_extract_next_chat_turn,
            },
        },
    },
}
