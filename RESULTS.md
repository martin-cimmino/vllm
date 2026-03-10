# Installation

## from source not recommended

```bash
module purge
module load cuda/12.2
module show cuda/12.2 # check that CUDA_HOME=/leonardo/prod/opt/compilers/cuda/12.6/none

export CUDA_HOME=/leonardo/prod/opt/compilers/cuda/12.6/none
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

which nvcc # nvidia cuda compiler should now be in the path
nvcc --version # check that nvcc returns Build cuda_12.6.r12.6/compiler.34841621_0

module load gcc/12.2.0
export CC=$(which gcc)
export CXX=$(which g++)

export TMPDIR=/leonardo_scratch/fast/iGen_train/mcimmino/tmp
export TORCH_CUDA_ARCH_LIST="8.0" # speed up the build significantly, avoid building for all architectures
```

## Recommended: using precompiled wheels (much faster and easier) as long as you don't need to modify C++ code

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
VLLM_PRECOMPILED_WHEEL_VARIANT=cu126 VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=cu126 # it's fine as long as we're only modifying Python code
uv pip install -r requirements.txt
```

check: https://docs.vllm.ai/en/latest/contributing/incremental_build/#generate-cmakeuserpresetsjson-using-the-helper-script


After setup, any future session only needs:

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')

from transformers import AutoModelForCausalLM, AutoConfig
print('Transformers: OK')

import lighteval
print('LightEval: OK')

import vllm
print(f'vLLM: {vllm.__version__}')
"
```

### Interactive Sessions (development & debugging)

```bash
# Activate env on the login node, then launch an interactive session
cd /leonardo_scratch/fast/iGen_train/mcimmino/vllm
source .venv/bin/activate

srun -N 1 --partition boost_usr_prod --nodes=1 \
     --account=AIFAC_L13_018 --qos boost_qos_dbg \
     --ntasks=1 --exclusive --time 00:30:00 \
     --gres=gpu:1 --pty /bin/bash

# Run Smoke test
python smoke_test.py --model /leonardo_scratch/fast/iGen_train/models/Domyn-Small-v0.2-bf16 --prompt "Hello world" --max_tokens 128

# Run smc prototype v2
python smc_prototype_v2.py
```