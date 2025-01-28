#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate llm-env

export CUDA_HOME=$CONDA_PREFIX
export NCCL_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

date;pwd

export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export RDVZ_ID=$RANDOM
echo "RDZV Endpoint $MASTER_ADDR:$MASTER_PORT"


RUN_NAME="llama3.2-1b-ultrachat_200k-sft-lora"
LOG_DIR="training_logs"
LOG_PATH="${LOG_DIR}/log_${RUN_NAME}.log"
# Make logging directories.
mkdir -p "${LOG_DIR}"

echo "Placing logs in: ${LOG_DIR}"

nvidia-smi

NUM_GPUs=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPUs per node: ${NUM_GPUs}"

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=WARN
# export TORCH_CPP_LOG_LEVEL=INFO
# export LOGLEVEL=INFO
export NCCL_ASYNC_ERROR_HANDLING=1


NUM_PROCS=${NUM_GPUs}


ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file=recipes/accelerate_configs/deepspeed_zero2.yaml \
    --num_machines 1 \
    --num_processes $NUM_PROCS \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank 0 \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    scripts/run_sft.py recipes/llama-3.2-1b/sft/config_lora.yaml > ${LOG_PATH} 2>&1


