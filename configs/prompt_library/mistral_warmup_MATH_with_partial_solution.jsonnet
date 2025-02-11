local tree_question_template = '[MATH_TASK] Problem:
{query}

Solution:
{partial_solution}
';

{
    prompt_library+: {
        tree+: {
            question_template: tree_question_template,
        },
    },
}
