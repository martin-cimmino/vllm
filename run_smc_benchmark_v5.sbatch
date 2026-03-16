#!/bin/bash
#SBATCH --job-name=smc_bench_v5
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_L13_018
#SBATCH --qos=qos_llm_min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/smc_bench_v5_%j.out
#SBATCH --error=logs/smc_bench_v5_%j.err

# ── Environment ───────────────────────────────────────────────────────
cd /leonardo_scratch/fast/AIFAC_L13_018/mcimmino/vllm
source .venv/bin/activate
module load cuda/12.2

export HF_DATASETS_CACHE="/leonardo_scratch/fast/AIFAC_L13_018/mcimmino/hf_cache"
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# ── Default parameters (override via -- args after sbatch script) ─────
MODEL=/leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16
METHOD=both
N_PARTICLES=16
ALPHA=5.0
ESS_THRESHOLD=0.9
TEMPERATURE=1.0
MAX_NEW_TOKENS=16384
SEED=42
ALPHA_RAMP_TOKENS=100
MAX_PROBLEMS=""
THINKING="--thinking"

# Parse optional overrides passed after '--'
while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD=$2; shift 2;;
        --n_particles) N_PARTICLES=$2; shift 2;;
        --alpha) ALPHA=$2; shift 2;;
        --ess_threshold) ESS_THRESHOLD=$2; shift 2;;
        --temperature) TEMPERATURE=$2; shift 2;;
        --max_new_tokens) MAX_NEW_TOKENS=$2; shift 2;;
        --seed) SEED=$2; shift 2;;
        --alpha_ramp_tokens) ALPHA_RAMP_TOKENS=$2; shift 2;;
        --max_problems) MAX_PROBLEMS="--max_problems $2"; shift 2;;
        --thinking) THINKING="--thinking"; shift;;
        *) shift;;
    esac
done

mkdir -p logs smc_results

echo "========================================================"
echo "  SMC Benchmark v5 — AIME 2025 Multi-Problem"
echo "  method=$METHOD  n=$N_PARTICLES  α=$ALPHA  τ=$ESS_THRESHOLD"
echo "  temp=$TEMPERATURE  seed=$SEED  max_new_tokens=$MAX_NEW_TOKENS"
echo "  alpha_ramp_tokens=$ALPHA_RAMP_TOKENS"
echo "  ${MAX_PROBLEMS:-all 30 problems}"
echo "========================================================"

python smc_benchmark_v5.py \
    --model "$MODEL" \
    --method "$METHOD" \
    --n_particles "$N_PARTICLES" \
    --alpha "$ALPHA" \
    --ess_threshold "$ESS_THRESHOLD" \
    --temperature "$TEMPERATURE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --seed "$SEED" \
    --alpha_ramp_tokens "$ALPHA_RAMP_TOKENS" \
    $MAX_PROBLEMS \
    $THINKING \
    --output_dir smc_results

echo "Job finished at $(date)"
