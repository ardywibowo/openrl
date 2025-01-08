{
    type: 'WarmupDecayLR',
    params: {
        last_batch_iteration: -1,
        total_num_steps: 'auto',
        warmup_min_lr: 1e-7,
        warmup_max_lr: 'auto',
        warmup_num_steps: 'auto',
    },
}
