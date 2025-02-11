local orig_library = (import 'llama2.jsonnet');
local tree_expansion_iid = '{prefix}';

orig_library + {
    prompt_library+: {
        tree+: {
            expansion+: {
                iid: tree_expansion_iid,
            },
        },
    },
}
