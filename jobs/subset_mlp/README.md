# Subset-only MLP local GPU jobs

`run_subset_mlp_task.sh` runs one registered MLP-only paper task on the local
CUDA GPU. Each run trains and evaluates only:

- `gnn_subset`
- `gnn_subset_rnn`
- `gnn_subset_lstm`

The separately tracked `run_subset_mlp_horizon_task.sh` instead runs
`gnn_subset_rnn_horizon` and `gnn_subset_lstm_horizon`, writing everything
below `subset_mlp_horizon_adam_results/`. Run it with:

```bash
jobs/subset_mlp/submit_subset_mlp_horizon_jobs.sh --task mnist_to_fashion_mnist_test
```

The horizon variants reset only recurrent memory every 100 evaluation steps,
randomize short memory/unroll horizons during training, use chunk-relative
progress, and damp updates as raw gradients vanish. They do not use an Adam
base step or residual.

LSTM-DM is disabled. The benchmark's classical reference metrics are still
recorded for comparison.

Run one task from the workspace root with:

```bash
jobs/subset_mlp/submit_subset_mlp_jobs.sh --task mnist_to_fashion_mnist_test
```

Both commands require `--task` and reject names outside the supported MLP task
list. They use an active virtual environment when present, otherwise they fall
back to `/srv/scratch/z5591496/newStart/venv`, and fail fast when CUDA is not
available.

If California Housing is not cached yet, prefetch all curriculum/evaluation
scopes before running that task:

```bash
python download_housing_data.py
```

All artifacts are written beneath `subset_mlp_results/`:

- `checkpoints/<task>/`
- `metrics/`
- `plots/<task>/`
- `logs/<task>.log`
