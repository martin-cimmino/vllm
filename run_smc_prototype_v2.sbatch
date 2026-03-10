#!/bin/bash
#SBATCH --job-name=smc-prototype-v2
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_L13_018
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smc_prototype_v2_%j.out
#SBATCH --error=logs/smc_prototype_v2_%j.err

set -euo pipefail

cd /leonardo_scratch/fast/iGen_train/mcimmino
if [ -f .venv/bin/activate ]; then
	# shellcheck disable=SC1091
	source .venv/bin/activate
fi

export HF_HOME="/leonardo_scratch/fast/iGen_train/mcimmino/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Execute and forward all positional arguments to the Python script.
# Usage: sbatch vllm/run_smc_prototype_v2.sbatch -- --arg1 val1 --flag
exec python -u vllm/smc_prototype_v2.py "$@"
