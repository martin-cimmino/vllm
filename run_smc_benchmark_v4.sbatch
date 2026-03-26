#!/bin/bash
#SBATCH --job-name=smc_bench_v4
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_L13_018
#SBATCH --qos=boost_qos_bprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/smc_bench_v4_%j.out
#SBATCH --error=logs/smc_bench_v4_%j.err

# ── Environment ───────────────────────────────────────────────────────
# cd /leonardo_scratch/fast/AIFAC_L13_018/mcimmino/vllm
source .venv/bin/activate
module load cuda/12.2

export HF_HOME="/leonardo_scratch/fast/AIFAC_L13_018/mcimmino/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# ── Default parameters (override via --arg) ───────────────────────────
MODEL=/leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16
N_PARTICLES=64
ALPHA=4.0
ESS_THRESHOLD=0.5
MAX_NEW_TOKENS=8192
SEED=17
PROBLEM_IDX=4
PROBLEM_FLAG=""
COMPARE=""
ALPHA_RAMP_TOKENS=100

# Parse optional overrides passed after '--'
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_particles) N_PARTICLES=$2; shift 2;;
        --alpha) ALPHA=$2; shift 2;;
        --ess_threshold) ESS_THRESHOLD=$2; shift 2;;
        --alpha_ramp_tokens) ALPHA_RAMP_TOKENS=$2; shift 2;;
        --max_new_tokens) MAX_NEW_TOKENS=$2; shift 2;;
        --seed) SEED=$2; shift 2;;
        --problem_idx) PROBLEM_IDX=$2; shift 2;;
        --all_problems) PROBLEM_FLAG="--all_problems"; shift;;
        --compare_baseline) COMPARE="--compare_baseline"; shift;;
        *) shift;;
    esac
done

mkdir -p logs smc_results

echo "========================================================"
echo "  SMC Benchmark v4 — Phase 2 Resampling"
echo "  n=$N_PARTICLES  α=$ALPHA  τ=$ESS_THRESHOLD  seed=$SEED"
echo "  problem_idx=$PROBLEM_IDX  max_new_tokens=$MAX_NEW_TOKENS"
echo "  compare_baseline=${COMPARE:-no}"
echo "========================================================"

python smc_benchmark_v4.py \
    --model "$MODEL" \
    --n_particles "$N_PARTICLES" \
    --alpha "$ALPHA" \
    --ess_threshold "$ESS_THRESHOLD" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --seed "$SEED" \
    ${PROBLEM_FLAG:-${PROBLEM_IDX:+--problem_idx $PROBLEM_IDX}} \
    $COMPARE \
    --output_dir smc_results \
    --alpha_ramp_tokens "$ALPHA_RAMP_TOKENS" \
    --thinking

echo "Job finished at $(date)"
