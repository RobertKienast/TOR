#!/usr/bin/env bash
set -euo pipefail

# Compare PyTorch harness results with Open-L2O TensorFlow evaluator results.
#
# Usage:
#   bash compare_benchmarks.sh
#
# Optional environment overrides:
#   PYTORCH_PY=/custom/path/to/python
#   TF_PY=/custom/path/to/python
#   TF_LEGACY_PY=/custom/path/to/python
#   TF_DM_PY=/custom/path/to/python
#   TF_RNNPROP_PY=/custom/path/to/python
#   TF_SWARM_PY=/custom/path/to/python
#   TF_SCALE_PY=/custom/path/to/python
#   RUN_TF=1
#   AUTO_TRAIN_TF=1
#   CHECKPOINT=gnn_meta.pt
#   STEPS=50
#   SEEDS="0 1 2"
#   PROBLEMS="quadratic_test lasso_test rastrigin_test_small rastrigin_test_large mnist_test mnist_relu_test mnist_deeper_test mnist_conv_test cifar_conv_test lenet_test nas_test"
#   TF_DM_CKPT="Open-L2O/Model_Free_L2O/L2O-DM and L2O-RNNProp/trained_models/dm/cw.l2l-0"
#   TF_RNNPROP_CKPT="Open-L2O/Model_Free_L2O/L2O-DM and L2O-RNNProp/trained_models_cl_il/rnnprop/rp.l2l-0"
#   TF_SWARM_SAVE="Open-L2O/Model_Free_L2O/L2O-Swarm/src/harness_swarm"
#   TF_SCALE_TRAIN_DIR="harness_scale_train"
#   TF_SWARM_TRAIN_TIMEOUT=900
#   TF_SCALE_TRAIN_TIMEOUT=1200
#   TF_SWARM_TRAIN_EPOCHS=50
#   TF_SWARM_TRAIN_STEPS=50
#   TF_SCALE_META_ITERATIONS=20

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTORCH_PY="$ROOT_DIR/venv/Scripts/python.exe"
DEFAULT_TF_PY="$ROOT_DIR/tf_venv/Scripts/python.exe"
DEFAULT_TF1_PY="$ROOT_DIR/tf_venv_ver1/Scripts/python.exe"

PYTORCH_PY="${PYTORCH_PY:-$DEFAULT_PYTORCH_PY}"
TF_PY="${TF_PY:-$DEFAULT_TF_PY}"
TF_LEGACY_PY="${TF_LEGACY_PY:-$DEFAULT_TF1_PY}"
TF_DM_PY="${TF_DM_PY:-$TF_LEGACY_PY}"
TF_RNNPROP_PY="${TF_RNNPROP_PY:-$TF_LEGACY_PY}"
TF_SWARM_PY="${TF_SWARM_PY:-$TF_LEGACY_PY}"
TF_SCALE_PY="${TF_SCALE_PY:-$TF_LEGACY_PY}"
RUN_TF="${RUN_TF:-1}"
AUTO_TRAIN_TF="${AUTO_TRAIN_TF:-1}"
CHECKPOINT="${CHECKPOINT:-gnn_meta.pt}"
STEPS="${STEPS:-50}"
SEEDS="${SEEDS:-0 1 2}"
PROBLEMS="${PROBLEMS:-quadratic_test lasso_test rastrigin_test_small rastrigin_test_large mnist_test mnist_relu_test mnist_deeper_test mnist_conv_test cifar_conv_test lenet_test nas_test}"
TF_TRAIN_PROBLEM="${TF_TRAIN_PROBLEM:-mnist}"
TF_TRAIN_EPOCHS="${TF_TRAIN_EPOCHS:-100}"
TF_TRAIN_STEPS="${TF_TRAIN_STEPS:-100}"
TF_SWARM_PROBLEM="${TF_SWARM_PROBLEM:-quadratic}"
TF_SWARM_TRAIN_EPOCHS="${TF_SWARM_TRAIN_EPOCHS:-50}"
TF_SWARM_TRAIN_STEPS="${TF_SWARM_TRAIN_STEPS:-50}"
TF_SCALE_TRAIN_DIR="${TF_SCALE_TRAIN_DIR:-harness_scale_train}"
TF_SCALE_META_ITERATIONS="${TF_SCALE_META_ITERATIONS:-20}"
TF_SCALE_UNROLL_LENGTH="${TF_SCALE_UNROLL_LENGTH:-20}"
TF_SWARM_TRAIN_TIMEOUT="${TF_SWARM_TRAIN_TIMEOUT:-900}"
TF_SCALE_TRAIN_TIMEOUT="${TF_SCALE_TRAIN_TIMEOUT:-1200}"

if [[ ! -f "$PYTORCH_PY" ]]; then
  echo "[error] PyTorch python not found: $PYTORCH_PY"
  echo "        Expected default venv at: $DEFAULT_PYTORCH_PY"
  exit 1
fi

if [[ "$RUN_TF" == "1" && ! -f "$TF_DM_PY" ]]; then
  echo "[error] TensorFlow DM python not found: $TF_DM_PY"
  echo "        Set TF_DM_PY, or provide TF_LEGACY_PY/TF_PY with a valid interpreter."
  exit 1
fi

if [[ "$RUN_TF" == "1" && ! -f "$TF_RNNPROP_PY" ]]; then
  echo "[error] TensorFlow RNNProp python not found: $TF_RNNPROP_PY"
  echo "        Set TF_RNNPROP_PY, or provide TF_LEGACY_PY/TF_PY with a valid interpreter."
  exit 1
fi

if [[ "$RUN_TF" == "1" && ! -f "$TF_SWARM_PY" ]]; then
  echo "[error] TensorFlow Swarm python not found: $TF_SWARM_PY"
  echo "        Set TF_SWARM_PY, or provide TF_LEGACY_PY/TF_PY with a valid interpreter."
  exit 1
fi

if [[ "$RUN_TF" == "1" && ! -f "$TF_SCALE_PY" ]]; then
  echo "[error] TensorFlow Scale python not found: $TF_SCALE_PY"
  echo "        Set TF_SCALE_PY, or provide TF_LEGACY_PY/TF_PY with a valid interpreter."
  exit 1
fi

OPENL2O_TF_DIR="$ROOT_DIR/Open-L2O/Model_Free_L2O/L2O-DM and L2O-RNNProp"
TF_DM_CKPT="${TF_DM_CKPT:-$OPENL2O_TF_DIR/trained_models/dm/cw.l2l-0}"
TF_RNNPROP_CKPT="${TF_RNNPROP_CKPT:-$OPENL2O_TF_DIR/trained_models_cl_il/rnnprop/rp.l2l-0}"
OPENL2O_SWARM_DIR="$ROOT_DIR/Open-L2O/Model_Free_L2O/L2O-Swarm/src"
TF_SWARM_SAVE="${TF_SWARM_SAVE:-$OPENL2O_SWARM_DIR/harness_swarm}"
TF_SWARM_MARKER="$TF_SWARM_SAVE/loss_record.pickle"
# Swarm evaluate.py expects --path to be relative when run inside OPENL2O_SWARM_DIR.
if [[ "$TF_SWARM_SAVE" == "$OPENL2O_SWARM_DIR"/* ]]; then
  TF_SWARM_SAVE_ARG="${TF_SWARM_SAVE#"$OPENL2O_SWARM_DIR"/}"
else
  TF_SWARM_SAVE_ARG="$TF_SWARM_SAVE"
fi
OPENL2O_SCALE_DIR="$ROOT_DIR/Open-L2O/Model_Free_L2O/L2O-Scale/L2O-Scale-Training"
TF_SCALE_FULL_TRAIN_DIR="$OPENL2O_SCALE_DIR/$TF_SCALE_TRAIN_DIR"
TF_SCALE_CKPT_GLOB="$TF_SCALE_FULL_TRAIN_DIR/model.ckpt-*"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/compare_outputs/$TS"
PT_JSON="$OUT_DIR/pytorch_results.json"
PT_CSV="$OUT_DIR/pytorch_results.csv"
TF_DM_DIR="$OUT_DIR/tf_dm"
TF_RNN_DIR="$OUT_DIR/tf_rnnprop"
TF_SWARM_DIR="$OUT_DIR/tf_swarm"
TF_SCALE_DIR="$OUT_DIR/tf_scale"
SUMMARY_CSV="$OUT_DIR/compare_summary.csv"
SUMMARY_MD="$OUT_DIR/compare_summary.md"
SUMMARY_TXT="$OUT_DIR/compare_summary.txt"
TRAINING_CSV="$OUT_DIR/training_summary.csv"
TRAINING_MD="$OUT_DIR/training_summary.md"
TRAINING_TXT="$OUT_DIR/training_summary.txt"
LIVE_LOG="$OUT_DIR/run_live.log"
LIVE_TABLE="$OUT_DIR/live_results_table.txt"

export OUT_DIR PT_JSON TF_DM_DIR TF_RNN_DIR TF_SWARM_DIR TF_SCALE_DIR SUMMARY_CSV SUMMARY_MD SUMMARY_TXT TRAINING_CSV TRAINING_MD TRAINING_TXT LIVE_TABLE PROBLEMS TF_SCALE_FULL_TRAIN_DIR TF_SWARM_SAVE

mkdir -p "$OUT_DIR" "$TF_DM_DIR" "$TF_RNN_DIR" "$TF_SWARM_DIR" "$TF_SCALE_DIR"

: > "$LIVE_LOG"
cat > "$LIVE_TABLE" <<'TXT'
Benchmark is starting...
This table will update in real time.
TXT

cat > "$TRAINING_CSV" <<'CSV'
timestamp,framework,model,checkpoint_path,missing_before,auto_train,status,train_command,notes
CSV

csv_escape() {
  local s="$1"
  s="${s//\"/\"\"}"
  printf '"%s"' "$s"
}

append_training_row() {
  local ts="$1"
  local framework="$2"
  local model="$3"
  local ckpt="$4"
  local missing_before="$5"
  local auto_train="$6"
  local status="$7"
  local cmd="$8"
  local notes="$9"

  {
    csv_escape "$ts"; printf ','
    csv_escape "$framework"; printf ','
    csv_escape "$model"; printf ','
    csv_escape "$ckpt"; printf ','
    csv_escape "$missing_before"; printf ','
    csv_escape "$auto_train"; printf ','
    csv_escape "$status"; printf ','
    csv_escape "$cmd"; printf ','
    csv_escape "$notes"; printf '\n'
  } >> "$TRAINING_CSV"
}

probe_tf_env() {
  local py="$1"
  "$py" - <<'PY' 2>/dev/null || true
try:
    import tensorflow as tf
    version = getattr(tf, "__version__", "unknown")
    has_contrib = hasattr(tf, "contrib")
    print(f"ok={1 if has_contrib else 0}")
    print(f"version={version}")
except Exception:
    print("ok=0")
    print("version=unimportable")
PY
}

tf_model_env_ok() {
  local py="$1"
  local out
  out="$(probe_tf_env "$py")"
  local ok
  ok="$(printf '%s\n' "$out" | grep -E '^ok=' | tail -n1 | cut -d= -f2 || true)"
  if [[ "$ok" == "1" ]]; then
    return 0
  fi
  return 1
}

tf_model_env_version() {
  local py="$1"
  local out
  out="$(probe_tf_env "$py")"
  printf '%s\n' "$out" | grep -E '^version=' | tail -n1 | cut -d= -f2 || true
}

run_logged() {
  # Runs a command, streams to terminal, and appends to live log.
  "$@" 2>&1 | tee -a "$LIVE_LOG"
}

run_logged_timeout() {
  # Runs a command with optional timeout (seconds), logging to terminal and live log.
  local timeout_s="$1"
  shift

  if [[ -z "$timeout_s" || "$timeout_s" == "0" ]]; then
    run_logged "$@"
    return $?
  fi

  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=30 "${timeout_s}s" "$@" 2>&1 | tee -a "$LIVE_LOG"
    return ${PIPESTATUS[0]}
  fi

  echo "  [warn] 'timeout' command not found; running without time cutoff."
  run_logged "$@"
}

refresh_live_table() {
  "$PYTORCH_PY" - <<'PY'
import json
import os
import pickle
from pathlib import Path

pt_json = Path(os.environ.get("PT_JSON", ""))
tf_dm_dir = Path(os.environ.get("TF_DM_DIR", ""))
tf_rnn_dir = Path(os.environ.get("TF_RNN_DIR", ""))
tf_swarm_dir = Path(os.environ.get("TF_SWARM_DIR", ""))
tf_scale_dir = Path(os.environ.get("TF_SCALE_DIR", ""))
problems = os.environ.get("PROBLEMS", "").split()
live_table = Path(os.environ.get("LIVE_TABLE", ""))

pt = {}
if pt_json.exists():
  with pt_json.open("r", encoding="utf-8") as f:
    pt = json.load(f)

pt_opts = sorted({opt for curves in pt.values() for opt in curves.keys()})
if not pt_opts:
  pt_opts = ["GNN-Meta", "Adam", "RMSProp", "SGD"]

def load_tf_final(tf_dir: Path, base_name: str):
  p = tf_dir / f"L2L_eval_loss_record.pickle-{base_name}"
  if not p.exists():
    return ""
  try:
    with p.open("rb") as f:
      arr = pickle.load(f)
    if not arr:
      return ""
    return float(arr[-1])
  except Exception:
    return ""

def load_swarm_final_refresh(tf_dir: Path, base_name: str):
  # Swarm saves evaluate_record.pickle directly in the --path save directory
  swarm_save = os.environ.get("TF_SWARM_SAVE", "")
  p = Path(swarm_save) / "evaluate_record.pickle" if swarm_save else Path("/nonexistent")
  if not p.exists():
    return ""
  try:
    with p.open("rb") as f:
      data = pickle.load(f)
    if isinstance(data, dict) and "min_loss_record" in data:
      arr = data["min_loss_record"]
      if arr:
        return float(arr[-1])
  except Exception:
    pass
  return ""

def load_scale_final_refresh(tf_dir: Path, base_name: str):
  scale_dir = os.environ.get("TF_SCALE_FULL_TRAIN_DIR", "")
  p = Path(scale_dir) / "L2o_eval_loss_record.pickle" if scale_dir else Path("/nonexistent")
  if not p.exists():
    return ""
  try:
    with p.open("rb") as f:
      arr = pickle.load(f)
    if arr:
      return float(arr[-1])
  except Exception:
    pass
  return ""

def load_swarm_final(tf_dir: Path, base_name: str):
  # Swarm saves to {path}/evaluate_record.pickle as dict with 'min_loss_record' key
  p = Path(tf_dir) / "evaluate_record.pickle"
  if not p.exists():
    return ""
  try:
    with p.open("rb") as f:
      data = pickle.load(f)
    if isinstance(data, dict) and "min_loss_record" in data:
      arr = data["min_loss_record"]
      if arr:
        return float(arr[-1])
  except Exception:
    pass
  return ""

def load_scale_final(tf_dir: Path, base_name: str):
  scale_train_base = Path(os.environ.get("OPENL2O_SCALE_TRAIN_DIR", ""))
  if scale_train_base and scale_train_base.exists():
    p = scale_train_base / "L2o_eval_loss_record.pickle"
    if p.exists():
      try:
        with p.open("rb") as f:
          arr = pickle.load(f)
        if arr:
          return float(arr[-1])
      except Exception:
        pass
  return ""

headers = ["problem"] + [f"pt_{o}" for o in pt_opts] + ["tf_dm", "tf_rnnprop", "tf_swarm", "tf_scale"]
rows = []
for p in problems:
  base = p[:-5] if p.endswith("_test") else p
  curves = pt.get(p, pt.get(base, {}))
  row = {"problem": p}
  for o in pt_opts:
    v = ""
    if o in curves and curves[o]:
      v = float(curves[o][-1])
    row[f"pt_{o}"] = v
  row["tf_dm"] = load_tf_final(tf_dm_dir, base)
  row["tf_rnnprop"] = load_tf_final(tf_rnn_dir, base)
  row["tf_swarm"] = load_swarm_final_refresh(tf_swarm_dir, base)
  row["tf_scale"] = load_scale_final_refresh(tf_scale_dir, base)
  rows.append(row)

display_rows = []
for r in rows:
  line = []
  for h in headers:
    v = r[h]
    if isinstance(v, float):
      line.append(f"{v:.6f}")
    else:
      line.append(str(v))
  display_rows.append(line)

widths = [len(h) for h in headers]
for row in display_rows:
  for i, cell in enumerate(row):
    widths[i] = max(widths[i], len(cell))

def fmt_row(cells):
  return " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells)))

sep = "-+-".join("-" * w for w in widths)
lines = [fmt_row(headers), sep]
lines += [fmt_row(r) for r in display_rows]

live_table.parent.mkdir(parents=True, exist_ok=True)
with live_table.open("w", encoding="utf-8") as f:
  f.write("\n".join(lines) + "\n")
PY
}

echo "[1/4] Running PyTorch benchmark harness..."
if [[ ! -e "$CHECKPOINT" ]]; then
  append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "pytorch" "gnn_meta" "$CHECKPOINT" "yes" "yes" "delegated_to_harness" "benchmark_harness.py eval --auto_train_gnn_if_missing" "Checkpoint missing before run; harness handles training if needed."
fi
run_logged "$PYTORCH_PY" benchmark_harness.py eval \
  --checkpoint "$CHECKPOINT" \
  --problems $PROBLEMS \
  --steps "$STEPS" \
  --seeds $SEEDS \
  --csv "$PT_CSV" \
  --json "$PT_JSON" \
  --auto_train_gnn_if_missing \
  --no_auto_train_openl2o_if_missing
refresh_live_table

if [[ "$RUN_TF" == "1" ]]; then
  echo "[2/4] Running TensorFlow Open-L2O evaluators (DM + RNNProp) where supported..."

  DM_ENV_OK=0
  RNN_ENV_OK=0
  SWARM_ENV_OK=0
  SCALE_ENV_OK=0
  DM_ENV_VER="$(tf_model_env_version "$TF_DM_PY")"
  RNN_ENV_VER="$(tf_model_env_version "$TF_RNNPROP_PY")"
  SWARM_ENV_VER="$(tf_model_env_version "$TF_SWARM_PY")"
  SCALE_ENV_VER="$(tf_model_env_version "$TF_SCALE_PY")"

  if tf_model_env_ok "$TF_DM_PY"; then
    DM_ENV_OK=1
  else
    echo "  [warn] DM env is incompatible (needs tensorflow.contrib / TF1.x)."
    echo "         TF_DM_PY=$TF_DM_PY (detected TF: ${DM_ENV_VER:-unknown})"
  fi

  if tf_model_env_ok "$TF_RNNPROP_PY"; then
    RNN_ENV_OK=1
  else
    echo "  [warn] RNNProp env is incompatible (needs tensorflow.contrib / TF1.x)."
    echo "         TF_RNNPROP_PY=$TF_RNNPROP_PY (detected TF: ${RNN_ENV_VER:-unknown})"
  fi

  if tf_model_env_ok "$TF_SWARM_PY"; then
    SWARM_ENV_OK=1
  else
    echo "  [warn] Swarm env is incompatible (needs tensorflow.contrib / TF1.x)."
    echo "         TF_SWARM_PY=$TF_SWARM_PY (detected TF: ${SWARM_ENV_VER:-unknown})"
  fi

  if tf_model_env_ok "$TF_SCALE_PY"; then
    SCALE_ENV_OK=1
  else
    echo "  [warn] Scale env is incompatible (needs tensorflow.contrib / TF1.x)."
    echo "         TF_SCALE_PY=$TF_SCALE_PY (detected TF: ${SCALE_ENV_VER:-unknown})"
  fi

  if [[ "$AUTO_TRAIN_TF" == "1" ]]; then
    if [[ "$DM_ENV_OK" != "1" && ! -e "$TF_DM_CKPT" ]]; then
      append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "dm" "$TF_DM_CKPT" "yes" "yes" "env_incompatible" "" "Skipped: TF_DM_PY has no tensorflow.contrib (TF=${DM_ENV_VER:-unknown})."
      refresh_live_table
    elif [[ ! -e "$TF_DM_CKPT" ]]; then
      echo "  [dm] checkpoint missing, training first..."
      if run_logged "$TF_DM_PY" "$OPENL2O_TF_DIR/train_dm.py" \
        --save_path "$(dirname "$TF_DM_CKPT")" \
        --problem "$TF_TRAIN_PROBLEM" \
        --if_cl false \
        --if_mt false \
        --num_epochs "$TF_TRAIN_EPOCHS" \
        --num_steps "$TF_TRAIN_STEPS"; then
        append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "dm" "$TF_DM_CKPT" "yes" "yes" "trained" "train_dm.py --problem $TF_TRAIN_PROBLEM --num_epochs $TF_TRAIN_EPOCHS --num_steps $TF_TRAIN_STEPS" "Auto-trained using TF_DM_PY=$TF_DM_PY."
      else
        echo "  [warn] DM training failed"
        append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "dm" "$TF_DM_CKPT" "yes" "yes" "train_failed" "train_dm.py --problem $TF_TRAIN_PROBLEM --num_epochs $TF_TRAIN_EPOCHS --num_steps $TF_TRAIN_STEPS" "Auto-training failed with TF_DM_PY=$TF_DM_PY."
      fi
      refresh_live_table
    fi

    if [[ "$RNN_ENV_OK" != "1" && ! -e "$TF_RNNPROP_CKPT" ]]; then
      append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "rnnprop" "$TF_RNNPROP_CKPT" "yes" "yes" "env_incompatible" "" "Skipped: TF_RNNPROP_PY has no tensorflow.contrib (TF=${RNN_ENV_VER:-unknown})."
      refresh_live_table
    elif [[ ! -e "$TF_RNNPROP_CKPT" ]]; then
      echo "  [rnnprop] checkpoint missing, training first..."
      if run_logged "$TF_RNNPROP_PY" "$OPENL2O_TF_DIR/train_rnnprop.py" \
        --save_path "$(dirname "$TF_RNNPROP_CKPT")" \
        --problem "$TF_TRAIN_PROBLEM" \
        --if_cl false \
        --if_mt false \
        --num_epochs "$TF_TRAIN_EPOCHS" \
        --num_steps "$TF_TRAIN_STEPS"; then
        append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "rnnprop" "$TF_RNNPROP_CKPT" "yes" "yes" "trained" "train_rnnprop.py --problem $TF_TRAIN_PROBLEM --num_epochs $TF_TRAIN_EPOCHS --num_steps $TF_TRAIN_STEPS" "Auto-trained using TF_RNNPROP_PY=$TF_RNNPROP_PY."
      else
        echo "  [warn] RNNProp training failed"
        append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "rnnprop" "$TF_RNNPROP_CKPT" "yes" "yes" "train_failed" "train_rnnprop.py --problem $TF_TRAIN_PROBLEM --num_epochs $TF_TRAIN_EPOCHS --num_steps $TF_TRAIN_STEPS" "Auto-training failed with TF_RNNPROP_PY=$TF_RNNPROP_PY."
      fi
      refresh_live_table
    fi
  else
    if [[ ! -e "$TF_DM_CKPT" ]]; then
      append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "dm" "$TF_DM_CKPT" "yes" "no" "missing_not_trained" "" "AUTO_TRAIN_TF disabled."
    fi
    if [[ ! -e "$TF_RNNPROP_CKPT" ]]; then
      append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "rnnprop" "$TF_RNNPROP_CKPT" "yes" "no" "missing_not_trained" "" "AUTO_TRAIN_TF disabled."
    fi
  fi

  if [[ "$AUTO_TRAIN_TF" == "1" ]]; then
    if [[ "$SWARM_ENV_OK" != "1" && ! -e "$TF_SWARM_MARKER" ]]; then
      append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "swarm" "$TF_SWARM_MARKER" "yes" "yes" "env_incompatible" "" "Skipped: TF_SWARM_PY has no tensorflow.contrib (TF=${SWARM_ENV_VER:-unknown})."
      refresh_live_table
    elif [[ ! -e "$TF_SWARM_MARKER" ]]; then
      echo "  [swarm] checkpoint missing, training first..."
      mkdir -p "$TF_SWARM_SAVE"
      if run_logged_timeout "$TF_SWARM_TRAIN_TIMEOUT" "$TF_SWARM_PY" "$OPENL2O_SWARM_DIR/train.py" \
        --problem "$TF_SWARM_PROBLEM" \
        --num_epochs "$TF_SWARM_TRAIN_EPOCHS" \
        --num_steps "$TF_SWARM_TRAIN_STEPS" \
        --save_path "$TF_SWARM_SAVE"; then
        append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "swarm" "$TF_SWARM_MARKER" "yes" "yes" "trained" "train.py --problem $TF_SWARM_PROBLEM --num_epochs $TF_SWARM_TRAIN_EPOCHS --num_steps $TF_SWARM_TRAIN_STEPS --save_path $TF_SWARM_SAVE" "Auto-trained using TF_SWARM_PY=$TF_SWARM_PY."
      elif [[ $? -eq 124 ]]; then
        echo "  [warn] Swarm training timed out after ${TF_SWARM_TRAIN_TIMEOUT}s"
        append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "swarm" "$TF_SWARM_MARKER" "yes" "yes" "train_timeout" "train.py --problem $TF_SWARM_PROBLEM --num_epochs $TF_SWARM_TRAIN_EPOCHS --num_steps $TF_SWARM_TRAIN_STEPS --save_path $TF_SWARM_SAVE" "Auto-training timed out after ${TF_SWARM_TRAIN_TIMEOUT}s with TF_SWARM_PY=$TF_SWARM_PY."
      else
        echo "  [warn] Swarm training failed"
        append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "swarm" "$TF_SWARM_MARKER" "yes" "yes" "train_failed" "train.py --problem $TF_SWARM_PROBLEM --num_epochs $TF_SWARM_TRAIN_EPOCHS --num_steps $TF_SWARM_TRAIN_STEPS --save_path $TF_SWARM_SAVE" "Auto-training failed with TF_SWARM_PY=$TF_SWARM_PY."
      fi
      refresh_live_table
    fi

    if [[ "$SCALE_ENV_OK" != "1" ]] && ! compgen -G "$TF_SCALE_CKPT_GLOB" > /dev/null; then
      append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "scale" "$TF_SCALE_CKPT_GLOB" "yes" "yes" "env_incompatible" "" "Skipped: TF_SCALE_PY has no tensorflow.contrib (TF=${SCALE_ENV_VER:-unknown})."
      refresh_live_table
    elif ! compgen -G "$TF_SCALE_CKPT_GLOB" > /dev/null; then
      echo "  [scale] checkpoint missing, training first..."
      mkdir -p "$TF_SCALE_FULL_TRAIN_DIR"
      if run_logged_timeout "$TF_SCALE_TRAIN_TIMEOUT" "$TF_SCALE_PY" "$OPENL2O_SCALE_DIR/metarun.py" \
        --train_dir "$TF_SCALE_TRAIN_DIR" \
        --regularize_time none \
        --alpha 1e-4 \
        --reg_optimizer True \
        --reg_option hessian-esd \
        --include_mnist_mlp_problems True \
        --num_problems 1 \
        --num_meta_iterations "$TF_SCALE_META_ITERATIONS" \
        --fix_unroll True \
        --fix_unroll_length "$TF_SCALE_UNROLL_LENGTH" \
        --evaluation_period 1 \
        --evaluation_epochs 1 \
        --use_second_derivatives False \
        --if_cl False \
        --if_mt False \
        --mt_ratio 0.1 \
        --mt_k 1; then
        append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "scale" "$TF_SCALE_CKPT_GLOB" "yes" "yes" "trained" "metarun.py --train_dir $TF_SCALE_TRAIN_DIR --num_meta_iterations $TF_SCALE_META_ITERATIONS --fix_unroll_length $TF_SCALE_UNROLL_LENGTH" "Auto-trained using TF_SCALE_PY=$TF_SCALE_PY."
      elif [[ $? -eq 124 ]]; then
        echo "  [warn] Scale training timed out after ${TF_SCALE_TRAIN_TIMEOUT}s"
        append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "scale" "$TF_SCALE_CKPT_GLOB" "yes" "yes" "train_timeout" "metarun.py --train_dir $TF_SCALE_TRAIN_DIR --num_meta_iterations $TF_SCALE_META_ITERATIONS --fix_unroll_length $TF_SCALE_UNROLL_LENGTH" "Auto-training timed out after ${TF_SCALE_TRAIN_TIMEOUT}s with TF_SCALE_PY=$TF_SCALE_PY."
      else
        echo "  [warn] Scale training failed"
        append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "scale" "$TF_SCALE_CKPT_GLOB" "yes" "yes" "train_failed" "metarun.py --train_dir $TF_SCALE_TRAIN_DIR --num_meta_iterations $TF_SCALE_META_ITERATIONS --fix_unroll_length $TF_SCALE_UNROLL_LENGTH" "Auto-training failed with TF_SCALE_PY=$TF_SCALE_PY."
      fi
      refresh_live_table
    fi
  else
    if [[ ! -e "$TF_SWARM_MARKER" ]]; then
      append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "swarm" "$TF_SWARM_MARKER" "yes" "no" "missing_not_trained" "" "AUTO_TRAIN_TF disabled."
    fi
    if ! compgen -G "$TF_SCALE_CKPT_GLOB" > /dev/null; then
      append_training_row "$(date +%Y-%m-%dT%H:%M:%S)" "tensorflow" "scale" "$TF_SCALE_CKPT_GLOB" "yes" "no" "missing_not_trained" "" "AUTO_TRAIN_TF disabled."
    fi
  fi

  # Open-L2O util.get_config supports these base names.
  # We map *_test names to their base names and evaluate only supported tasks.
  SUPPORTED_BASE=(quadratic lasso rastrigin mnist mnist_relu mnist_deeper mnist_conv cifar_conv lenet nas)

  # shellcheck disable=SC2206
  PROBLEM_ARR=($PROBLEMS)

  pushd "$OPENL2O_TF_DIR" >/dev/null
  for p in "${PROBLEM_ARR[@]}"; do
    base="${p%_test}"

    supported=0
    for s in "${SUPPORTED_BASE[@]}"; do
      if [[ "$base" == "$s" ]]; then
        supported=1
        break
      fi
    done

    if [[ "$supported" -ne 1 ]]; then
      echo "  [skip] TF evaluator does not support: $p"
      continue
    fi

    if [[ "$DM_ENV_OK" == "1" && -e "$TF_DM_CKPT" ]]; then
      echo "  [dm] problem=$base"
      run_logged "$TF_DM_PY" evaluate_dm.py \
        --path "$TF_DM_CKPT" \
        --problem "$base" \
        --num_steps "$STEPS" \
        --num_epochs 1 \
        --output_path "$TF_DM_DIR" \
        --seed 0 || echo "  [warn] DM eval failed for $base"
      refresh_live_table
    elif [[ "$DM_ENV_OK" != "1" ]]; then
      echo "  [skip] DM eval skipped due to incompatible TF_DM_PY"
    else
      echo "  [warn] DM checkpoint not found: $TF_DM_CKPT"
    fi

    if [[ "$RNN_ENV_OK" == "1" && -e "$TF_RNNPROP_CKPT" ]]; then
      echo "  [rnnprop] problem=$base"
      run_logged "$TF_RNNPROP_PY" evaluate_rnnprop.py \
        --path "$TF_RNNPROP_CKPT" \
        --problem "$base" \
        --num_steps "$STEPS" \
        --num_epochs 1 \
        --output_path "$TF_RNN_DIR" \
        --seed 0 || echo "  [warn] RNNProp eval failed for $base"
      refresh_live_table
    elif [[ "$RNN_ENV_OK" != "1" ]]; then
      echo "  [skip] RNNProp eval skipped due to incompatible TF_RNNPROP_PY"
    else
      echo "  [warn] RNNProp checkpoint not found: $TF_RNNPROP_CKPT"
    fi

    if [[ "$SWARM_ENV_OK" == "1" && -e "$TF_SWARM_MARKER" ]]; then
      echo "  [swarm] problem=$base"
      pushd "$OPENL2O_SWARM_DIR" >/dev/null
      run_logged "$TF_SWARM_PY" evaluate.py \
        --problem "$base" \
        --optimizer "L2L" \
        --path "$TF_SWARM_SAVE_ARG" || echo "  [warn] Swarm eval failed for $base"
      popd >/dev/null
      refresh_live_table
    elif [[ "$SWARM_ENV_OK" != "1" ]]; then
      echo "  [skip] Swarm eval skipped due to incompatible TF_SWARM_PY"
    else
      echo "  [warn] Swarm checkpoint not found: $TF_SWARM_MARKER"
    fi

    if [[ "$SCALE_ENV_OK" == "1" && -e "$TF_SCALE_FULL_TRAIN_DIR/model.ckpt-0.meta" ]]; then
      echo "  [scale] problem=$base"
      pushd "$OPENL2O_SCALE_DIR" >/dev/null
      run_logged "$TF_SCALE_PY" metatest.py \
        --train_dir "$TF_SCALE_TRAIN_DIR" \
        --num_testing_itrs "$STEPS" || echo "  [warn] Scale eval failed for $base"
      popd >/dev/null
      refresh_live_table
    elif [[ "$SCALE_ENV_OK" != "1" ]]; then
      echo "  [skip] Scale eval skipped due to incompatible TF_SCALE_PY"
    else
      echo "  [warn] Scale checkpoint not found"
    fi
  done
  popd >/dev/null
else
  echo "[2/4] Skipping TensorFlow evaluators (RUN_TF=$RUN_TF)."
fi

echo "[3/4] Building merged comparison files..."
$PYTORCH_PY - <<'PY'
import csv
import json
import os
import pickle
from pathlib import Path

root = Path(os.environ.get("OUT_DIR", ""))
pt_json = Path(os.environ.get("PT_JSON", ""))
tf_dm_dir = Path(os.environ.get("TF_DM_DIR", ""))
tf_rnn_dir = Path(os.environ.get("TF_RNN_DIR", ""))
tf_swarm_dir = Path(os.environ.get("TF_SWARM_DIR", ""))
tf_scale_dir = Path(os.environ.get("TF_SCALE_DIR", ""))
tf_scale_full_train_dir = Path(os.environ.get("TF_SCALE_FULL_TRAIN_DIR", ""))
tf_swarm_save = Path(os.environ.get("TF_SWARM_SAVE", ""))
os.environ["OPENL2O_SCALE_TRAIN_DIR"] = str(Path(os.environ.get("TF_SCALE_FULL_TRAIN_DIR", "")).parent / "L2o_eval_loss_record.pickle").replace("/L2o_eval_loss_record.pickle", "") if os.environ.get("TF_SCALE_FULL_TRAIN_DIR") else ""
summary_csv = Path(os.environ.get("SUMMARY_CSV", ""))
summary_md = Path(os.environ.get("SUMMARY_MD", ""))
summary_txt = Path(os.environ.get("SUMMARY_TXT", ""))
training_csv = Path(os.environ.get("TRAINING_CSV", ""))
training_md = Path(os.environ.get("TRAINING_MD", ""))
training_txt = Path(os.environ.get("TRAINING_TXT", ""))
problems = os.environ.get("PROBLEMS", "").split()

if not pt_json.exists():
    raise SystemExit(f"Missing PyTorch results json: {pt_json}")

with pt_json.open("r", encoding="utf-8") as f:
    pt = json.load(f)

pt_opts = sorted({opt for curves in pt.values() for opt in curves.keys()})

def load_tf_final(tf_dir: Path, base_name: str):
    p = tf_dir / f"L2L_eval_loss_record.pickle-{base_name}"
    if not p.exists():
        return ""
    with p.open("rb") as f:
        arr = pickle.load(f)
    if not arr:
        return ""
    return float(arr[-1])

def load_swarm_final(tf_dir: Path, base_name: str):
    # Swarm saves evaluate_record.pickle directly in the --path save directory
    p = Path(os.environ.get("TF_SWARM_SAVE", "")) / "evaluate_record.pickle"
    if not p.exists():
        return ""
    try:
        with p.open("rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "min_loss_record" in data:
            arr = data["min_loss_record"]
            if arr:
                return float(arr[-1])
    except Exception:
        pass
    return ""

def load_scale_final(tf_dir: Path, base_name: str):
    # Scale saves {train_dir_basename}/{test_optimizer}_eval_loss_record.pickle
    # next to metatest.py, i.e. inside TF_SCALE_FULL_TRAIN_DIR
    p = tf_scale_full_train_dir / "L2o_eval_loss_record.pickle"
    if not p.exists():
        return ""
    try:
        with p.open("rb") as f:
            arr = pickle.load(f)
        if arr:
            return float(arr[-1])
    except Exception:
        pass
    return ""

headers = ["problem"] + [f"pt_{o}" for o in pt_opts] + ["tf_dm", "tf_rnnprop", "tf_swarm", "tf_scale"]
rows = []

for p in problems:
    base = p[:-5] if p.endswith("_test") else p

    pt_curves = pt.get(p, pt.get(base, {}))

    row = {"problem": p}
    for opt in pt_opts:
        v = ""
        if opt in pt_curves and pt_curves[opt]:
            v = float(pt_curves[opt][-1])
        row[f"pt_{opt}"] = v

    row["tf_dm"] = load_tf_final(tf_dm_dir, base)
    row["tf_rnnprop"] = load_tf_final(tf_rnn_dir, base)
    row["tf_swarm"] = load_swarm_final(tf_swarm_dir, base)
    row["tf_scale"] = load_scale_final(tf_scale_dir, base)
    rows.append(row)

with summary_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=headers)
    w.writeheader()
    w.writerows(rows)

with summary_md.open("w", encoding="utf-8") as f:
    f.write("# Benchmark Comparison\n\n")
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
    for r in rows:
        vals = []
        for h in headers:
            v = r[h]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        f.write("| " + " | ".join(vals) + " |\n")

    # Plain text table for quick terminal viewing.
    display_headers = headers
    display_rows = []
    for r in rows:
      line = []
      for h in display_headers:
        v = r[h]
        if isinstance(v, float):
          line.append(f"{v:.6f}")
        else:
          line.append(str(v))
      display_rows.append(line)

    widths = [len(h) for h in display_headers]
    for row in display_rows:
      for i, cell in enumerate(row):
        widths[i] = max(widths[i], len(cell))

    def fmt_row(cells):
      return " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells)))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(display_headers), sep]
    lines += [fmt_row(r) for r in display_rows]

    with summary_txt.open("w", encoding="utf-8") as f:
      f.write("\n".join(lines) + "\n")

    print("\nComparison Table:\n")
    print("\n".join(lines))
    print()

print(f"Wrote: {summary_csv}")
print(f"Wrote: {summary_md}")
print(f"Wrote: {summary_txt}")

training_rows = []
if training_csv.exists():
  with training_csv.open("r", encoding="utf-8", newline="") as f:
    training_rows = list(csv.DictReader(f))

training_headers = [
  "timestamp",
  "framework",
  "model",
  "checkpoint_path",
  "missing_before",
  "auto_train",
  "status",
  "train_command",
  "notes",
]

if training_rows:
  with training_md.open("w", encoding="utf-8") as f:
    f.write("# Missing Checkpoint Training Summary\n\n")
    f.write("| " + " | ".join(training_headers) + " |\n")
    f.write("|" + "|".join(["---"] * len(training_headers)) + "|\n")
    for r in training_rows:
      vals = [str(r.get(h, "")) for h in training_headers]
      f.write("| " + " | ".join(vals) + " |\n")

  table_rows = [[str(r.get(h, "")) for h in training_headers] for r in training_rows]
  widths = [len(h) for h in training_headers]
  for row in table_rows:
    for i, cell in enumerate(row):
      widths[i] = max(widths[i], len(cell))

  def fmt_training_row(cells):
    return " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells)))

  sep = "-+-".join("-" * w for w in widths)
  lines = [fmt_training_row(training_headers), sep]
  lines += [fmt_training_row(r) for r in table_rows]

  with training_txt.open("w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

  print("\nMissing Checkpoint Training Summary:\n")
  print("\n".join(lines))
  print()
else:
  with training_md.open("w", encoding="utf-8") as f:
    f.write("# Missing Checkpoint Training Summary\n\nNo missing checkpoints detected.\n")
  with training_txt.open("w", encoding="utf-8") as f:
    f.write("No missing checkpoints detected.\n")

print(f"Wrote: {training_csv}")
print(f"Wrote: {training_md}")
print(f"Wrote: {training_txt}")
PY

echo "[4/4] Done."
echo "  Output directory: $OUT_DIR"
echo "  - PyTorch JSON : $PT_JSON"
echo "  - PyTorch CSV  : $PT_CSV"
echo "  - Compare CSV  : $SUMMARY_CSV"
echo "  - Compare MD   : $SUMMARY_MD"
echo "  - Compare TXT  : $SUMMARY_TXT"
echo "  - Training CSV : $TRAINING_CSV"
echo "  - Training MD  : $TRAINING_MD"
echo "  - Training TXT : $TRAINING_TXT"
echo "  - Live Log     : $LIVE_LOG"
echo "  - Live Table   : $LIVE_TABLE"
