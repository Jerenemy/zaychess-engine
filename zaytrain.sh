#!/bin/bash
#SBATCH --job-name=zaychess_train
#SBATCH --partition=exx512
#SBATCH --nodelist=n91
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=output/sbatch_out/%x_%j.out
#SBATCH --error=output/sbatch_err/%x_%j.err
#SBATCH --account=pikthayer

set -euo pipefail

unset PYTHONPATH PYTHONHOME

mkdir -p output/sbatch_out output/sbatch_err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo

# --------- EDIT THESE TWO ----------
REPO="/zfshomes/jzay/chess/zaychess-engine"
CONDA_SH="$HOME/opt/mambaforge/etc/profile.d/conda.sh"
ENV_PREFIX="$HOME/opt/mambaforge/envs/zaychess-engine"
# -----------------------------------

# If your cluster hates that env due to glibc mismatch, this will fail
echo "GLIBC:"
ldd --version | head -n 1 || true
echo

# Persistent output location
OUTROOT="${OUTROOT:-$REPO/output/runs}"
RUN_NAME="run_${SLURM_JOB_ID}"
PERSIST_RUN_DIR="$OUTROOT/$RUN_NAME"
mkdir -p "$PERSIST_RUN_DIR"

# Scratch location
if [[ -n "${SLURM_TMPDIR:-}" && -d "${SLURM_TMPDIR:-}" ]]; then
  SCRATCH_BASE="$SLURM_TMPDIR"
else
  SCRATCH_BASE="/tmp/$USER"
fi

SCRATCH_RUN_DIR="$SCRATCH_BASE/$RUN_NAME"
mkdir -p "$SCRATCH_RUN_DIR"

echo "Repo: $REPO"
echo "Scratch: $SCRATCH_RUN_DIR"
echo "Output: $PERSIST_RUN_DIR"
echo

# Activate conda
if [[ ! -f "$CONDA_SH" ]]; then
  echo "Missing conda.sh at $CONDA_SH"
  exit 2
fi

source "$CONDA_SH"
conda activate "$ENV_PREFIX"

echo "Python: $(which python)"
python -V || true
echo

# Copy repo to scratch
rsync -a --delete "$REPO/" "$SCRATCH_RUN_DIR/repo/"

cd "$SCRATCH_RUN_DIR/repo"

echo "Repo root on scratch: $(pwd)"
echo "Top level:"
ls -la
echo
echo "Find train.py:"
find . -maxdepth 4 -name "train.py" -print
echo

# Verify expected path exists
if [[ ! -f "engine/train.py" ]]; then
  echo "Expected engine/train.py not found on scratch"
  echo "Your REPO path is probably wrong or you copied the wrong folder"
  exit 3
fi

# Output dir on scratch
SCRATCH_OUT="$SCRATCH_RUN_DIR/out"
mkdir -p "$SCRATCH_OUT"

# Threading hygiene
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Record GPU info
nvidia-smi || true
echo

# Periodic sync back so checkpoints survive node drama
(
  while true; do
    rsync -a "$SCRATCH_OUT/" "$PERSIST_RUN_DIR/" || true
    sleep 300
  done
) &
SYNC_PID=$!

# Hyperparams, override by exporting env vars before sbatch if you want
NUM_GENS="${NUM_GENS:-100}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
MCTS_STEPS="${MCTS_STEPS:-100}"
BUFFER_MAXLEN="${BUFFER_MAXLEN:-10000}"
BUFFER_BATCH_SIZE="${BUFFER_BATCH_SIZE:-1024}"
LR="${LR:-0.01}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

# Run training
python engine/train.py \
  --run-dir "$SCRATCH_OUT" \
  --num-gens "$NUM_GENS" \
  --num-epochs "$NUM_EPOCHS" \
  --mcts-steps "$MCTS_STEPS" \
  --buffer-maxlen "$BUFFER_MAXLEN" \
  --buffer-batch-size "$BUFFER_BATCH_SIZE" \
  --lr "$LR" \
  --momentum "$MOMENTUM" \
  --weight-decay "$WEIGHT_DECAY"

# Final sync and cleanup
kill "$SYNC_PID" || true
rsync -a "$SCRATCH_OUT/" "$PERSIST_RUN_DIR/"

echo
echo "Finished: $(date)"
echo "Saved to: $PERSIST_RUN_DIR"

