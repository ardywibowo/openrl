{
    inference_strategy+: {
        answer_extractor+: {
            type: 'next_chat_turn',
            sampling_parameters: {
                temperature: 0,
                max_tokens: 20,
            },
        },
    },
}
