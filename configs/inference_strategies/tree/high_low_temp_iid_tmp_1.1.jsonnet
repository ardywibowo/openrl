(import 'high_low_temp_iid.jsonnet')
+
{
    inference_strategy+: {
        node_expander+: {
            high_temp_params+: {
                temperature: 1.1
            }
        },
    },
}
