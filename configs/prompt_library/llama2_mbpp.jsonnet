local tree_question_template = '[INST] <<SYS>>
You are a helpful assistant. Always answer in most accurate way.
<</SYS>>
Your task is to write a Python function to solve a programming problem. You are given example test cases (between [TESTS] and [/TESTS]) from which you can infere the function signature.
First explain your thinking process, then produce the final and completed function. The Python code should go between ``` and ``` tags.

Q: {question}
[/INST]';

local tree_answer_extract_next_chat_turn = '{prefix}</s><s>[INST]Great! Repeat only the completed function.[/INST] Sure! The completed function I wrote is:
```
def ';

{
    prompt_library: {
        tree: {
            question_template: tree_question_template,
            answer_extract: {
                next_chat_turn: tree_answer_extract_next_chat_turn,
            },
        },
    },
}
