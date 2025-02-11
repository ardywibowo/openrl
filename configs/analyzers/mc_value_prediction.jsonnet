local num_expansion_rounds = 16;

{
    type: 'mc_value_prediction',

    max_num_checkpoints: 10,
    max_num_requests: 100,

    inference_strategy: {
        type: 'cot',

        samples: 256 / num_expansion_rounds,
        max_depth: 100,

        node_expander: {
            type: 'efficient_iid',
            sampling_parameters+: {
                temperature: 1,
                top_p: 0.9,
                max_tokens: 1024,
                stop: "\n\n\nProblem:",
            },
            node_text_template: '{chain_of_thought}',
            num_expansion_rounds: num_expansion_rounds,
        },


        answer_extractor: {
            type: 'identity_with_solution_prefix',
            node_key_name: 'full_text',
            solution_prefix: '\nSolution:',
        },

        question_field: 'query',
        question_template: '{query}',

        no_cache: true,
    },

    inference_server+: {
        type: "sglang",
        swap_space: 8,
        enable_prefix_caching: true,
    },
}
