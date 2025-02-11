(import 'iid_expander.jsonnet')
+
{
    inference_strategy+: {
        node_expander+: {
            sampling_parameters+: {
                temperature: 0.5,
            },
        },
    },
}
