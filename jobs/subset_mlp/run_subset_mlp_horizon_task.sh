#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_ACTIVATE="${VENV_ACTIVATE:-/srv/scratch/z5591496/newStart/venv/bin/activate}"

TASKS=(
  mnist_to_fashion_mnist_test
  fashion_mnist_to_mnist_test
  mnist_nn_family_ood_test
  fashion_nn_family_ood_test
  mnist_to_relu_test
  mnist_to_large_test
  mnist_deep_narrow_test
  mnist_noisy_test
  fashion_mnist_noisy_test
  fashion_mnist_linear_test
  covertype_test
  housing_test
)

usage() {
  echo "Usage: $0 --task <task-name>" >&2
}

TASK=""
while (( $# > 0 )); do
  case "$1" in
    --task)
      if (( $# < 2 )); then
        echo "ERROR: --task requires a task name." >&2
        usage
        exit 2
      fi
      TASK="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$TASK" ]]; then
  echo "ERROR: --task is required." >&2
  usage
  exit 2
fi

VALID_TASK=0
for CANDIDATE in "${TASKS[@]}"; do
  if [[ "$TASK" == "$CANDIDATE" ]]; then
    VALID_TASK=1
    break
  fi
done
if (( VALID_TASK == 0 )); then
  echo "ERROR: unsupported task: ${TASK}" >&2
  printf 'Valid tasks:\n  %s\n' "${TASKS[@]}" >&2
  exit 2
fi

cd "$WORKDIR"
if [[ -z "${VIRTUAL_ENV:-}" && -f "$VENV_ACTIVATE" ]]; then
  source "$VENV_ACTIVATE"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! "$PYTHON_BIN" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "ERROR: CUDA is not available to ${PYTHON_BIN}." >&2
  exit 1
fi

RESULT_ROOT="subset_mlp_horizon_adam_results"
mkdir -p "${RESULT_ROOT}"/{checkpoints,logs,metrics,plots}

OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
"$PYTHON_BIN" -u benchmark_harnessMeta2.py paper_compare \
  --problems "${TASK}" \
  --device cuda \
  --gnn_variants gnn_subset_rnn_horizon gnn_subset_lstm_horizon \
  --classical_baselines Adam \
  --no_dm \
  --retrain_per_problem \
  --retrain_num_workers 2 \
  --retrain_threads_per_worker 1 \
  --retrain_output_dir "${RESULT_ROOT}/checkpoints" \
  --csv "${RESULT_ROOT}/metrics/${TASK}_results.csv" \
  --json "${RESULT_ROOT}/metrics/${TASK}_results.json" \
  --plot_dir "${RESULT_ROOT}/plots/${TASK}" \
  --meta_seeds 101 202 303 404 505 606 707 808 909 1010 \
  --eval_checkpoint_epochs 50 \
  --step_debug_every 200 \
  --retrain_gnn_grad_clip 0.8 \
  --scale_curriculum \
  --retrain_gnn_layers 3 \
  --retrain_gnn_meta_lr 0.0002 \
  >> "${RESULT_ROOT}/logs/${TASK}.log" 2>&1
