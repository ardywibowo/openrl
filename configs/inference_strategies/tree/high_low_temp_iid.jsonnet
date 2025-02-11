{
    inference_strategy+: {
        node_expander+: {
            type: 'high_low_temperature_iid',
            high_temp_params: {
                temperature: 1.3,
                top_p: 0.9,
                max_tokens: 10
            },
            low_temp_params: {
                temperature: 0.9,
                top_p: 0.95,
                top_k: 50,
                max_tokens: 256,
                stop: "\nStep"
            }
            node_text_template: 'Step {chain_of_thought_1}{chain_of_thought_2}',
        },
    },
}
