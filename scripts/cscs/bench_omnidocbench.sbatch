#!/bin/bash
#SBATCH --job-name=omnidocbench
#SBATCH --account=a127
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/bench_results/omnidoc_%x_%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/bench_results/omnidoc_%x_%j.err
#SBATCH --environment=/capstor/scratch/cscs/arthurperrin/envs/mmore-vlm.toml

# =============================================================================
# OmniDocBench evaluation on Clariden (GH200) for VLM-based PDF extractors.
#
# Usage:
#   sbatch --export=ALL,MODEL_ID=nvidia/Nemotron-Nano-12B-VL,TAG=nemotron \
#          scripts/slurm/bench_omnidocbench.sbatch
#
#   sbatch --export=ALL,MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct,TAG=qwen25vl \
#          scripts/slurm/bench_omnidocbench.sbatch
#
#   sbatch --export=ALL,MODEL_ID=meta-llama/Llama-3.2-11B-Vision-Instruct,TAG=llama32v \
#          scripts/slurm/bench_omnidocbench.sbatch
#
# Required env (set via --export or in your shell before sbatch):
#   MODEL_ID     HF model id of the VLM to evaluate
#   TAG          short label used in output paths
# Optional:
#   LIMIT        cap number of pages (smoke test), default = all
#   MAX_PIXELS   per-image pixel budget for the VLM (default 1_600_000)
# =============================================================================

set -euo pipefail

: "${MODEL_ID:?MODEL_ID is required (e.g. nvidia/Nemotron-Nano-12B-VL)}"
: "${TAG:?TAG is required (short label for this run)}"
LIMIT="${LIMIT:-0}"
MAX_PIXELS="${MAX_PIXELS:-1600000}"

# --- Filesystems --------------------------------------------------------------
SCRATCH=/capstor/scratch/cscs/$USER
IOPS=/iopsstor/scratch/cscs/$USER

export HF_HOME=$IOPS/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/datasets/hf
export HF_XET_HIGH_PERFORMANCE=1
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

DATASET_DIR=$SCRATCH/datasets/omnidocbench
RESULTS_DIR=$SCRATCH/bench_results/${TAG}_${SLURM_JOB_ID}
mkdir -p "$DATASET_DIR" "$RESULTS_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"

echo "==> Node:        $(hostname)"
echo "==> Model:       $MODEL_ID"
echo "==> Tag:         $TAG"
echo "==> Dataset dir: $DATASET_DIR"
echo "==> Results dir: $RESULTS_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv

# --- 0. Lightweight venv on top of NGC torch ---------------------------------
# Strategy: keep NGC's torch 2.6.0a0+nv24.12 (works, tuned for GH200) and only
# install python-only deps in a venv with --system-site-packages. Avoids the
# brittle aarch64+cu124 vLLM install path.
VENV="${VLLM_VENV:-$SCRATCH/envs/vllm-venv}"
if [ ! -f "$VENV/.ready" ]; then
    echo "==> Creating venv at $VENV and installing inference deps (one-time, ~3 min)"
    python -m venv --system-site-packages "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    pip install --upgrade pip
    # Use the latest transformers (Qwen2.5-VL, Llama-3.2-Vision, Nemotron-VL
    # all need recent releases). --upgrade-strategy=only-if-needed so we don't
    # touch torch/numpy/etc. from the NGC base.
    # transformers 5.x requires torch>=2.7 (imports TransformGetItemToIndex);
    # NGC ships 2.6.0a0+nv24.12 so we cap at 4.x. >=4.49 needed for Qwen2.5-VL.
    pip install --upgrade-strategy=only-if-needed --upgrade \
        "transformers>=4.49,<5.0" "accelerate" "rapidfuzz>=3.0" "jiwer" \
        "huggingface_hub" "Pillow"
    touch "$VENV/.ready"
else
    echo "==> Reusing venv at $VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"

# --- 1. Fetch OmniDocBench (once; cached on capstor) --------------------------
if [ ! -f "$DATASET_DIR/.ready" ]; then
    echo "==> Downloading OmniDocBench…"
    hf download opendatalab/OmniDocBench \
        --repo-type dataset \
        --local-dir "$DATASET_DIR"
    touch "$DATASET_DIR/.ready"
else
    echo "==> OmniDocBench already present, skipping download."
fi

# --- 2. Run extraction --------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"

EXTRA_ARGS=()
if [ "$LIMIT" -gt 0 ]; then EXTRA_ARGS+=(--limit "$LIMIT"); fi

"$VENV/bin/python" -u scripts/bench_omnidocbench.py \
    --dataset-dir   "$DATASET_DIR" \
    --model-id      "$MODEL_ID" \
    --tag           "$TAG" \
    --output-dir    "$RESULTS_DIR" \
    --max-pixels    "$MAX_PIXELS" \
    "${EXTRA_ARGS[@]}"

# --- 3. Score against gold ----------------------------------------------------
"$VENV/bin/python" -u scripts/bench_omnidocbench.py \
    --score-only \
    --dataset-dir "$DATASET_DIR" \
    --output-dir  "$RESULTS_DIR" \
    --tag         "$TAG"

echo "==> Done. Results in $RESULTS_DIR"
