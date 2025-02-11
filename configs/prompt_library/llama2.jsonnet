local tree_expansion_iid = '{prefix}\nStep ';

local tree_expansion_high_low_temperature_iid = '{prefix}\nStep ';

local tree_answer_extract_next_chat_turn = '{prefix}</s><s>[INST]So, what is the final answer? Only output digits.[/INST] Sure! The final answer is:
';

local tree_question_template = '[INST] <<SYS>>
You are a helpful assistant solving math questions. Always answer in most accurate way.
<</SYS>>
Answer the following middle school math word problems, which require multi-step arithmetic reasoning. At the beginning of each step write “Step”.
Here is an example:
Step i:
...some thinking process...
Step i+1:
...some thinking process...

Q:{question}[/INST]
A:';

{
    prompt_library: {
        tree: {
            expansion: {
                iid: tree_expansion_iid,
                high_low_temperature_iid: tree_expansion_high_low_temperature_iid,
            },
            answer_extract: {
                next_chat_turn: tree_answer_extract_next_chat_turn,
            },
            question_template: tree_question_template,
        },
    },
}
