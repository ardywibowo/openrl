local orig_library = (import 'llama2.jsonnet');
local tree_expansion_iid = '{prefix}';
local tree_question_template = '{question}';

orig_library + {
    prompt_library+: {
        tree+: {
            expansion+: {
                iid: tree_expansion_iid,
            },

            question_template: tree_question_template,
        },
    },
}
