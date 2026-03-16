#!/bin/bash
#SBATCH --job-name=smc-benchmark-v3
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_L13_018
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smc_benchmark_v3_%j.out
#SBATCH --error=logs/smc_benchmark_v3_%j.err

set -euo pipefail

cd /leonardo_scratch/fast/AIFAC_L13_018/mcimmino/vllm
if [ -f .venv/bin/activate ]; then
	# shellcheck disable=SC1091
	source .venv/bin/activate
fi

export HF_HOME="/leonardo_scratch/fast/AIFAC_L13_018/mcimmino/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Execute and forward all positional arguments to the Python script.
# Usage: sbatch run_smc_benchmark_v3.sbatch -- --all_problems --alpha 2.0
exec python -u smc_benchmark_v3.py "$@"
