#!/bin/bash

date;pwd

eval "$(conda shell.bash hook)"
conda activate dpo-llm-env

export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export RDVZ_ID=$RANDOM
echo "RDZV Endpoint $MASTER_ADDR:$MASTER_PORT"

lr=0.0007
beta=0.01
gamma_to_beta=1.4
RUN_NAME="llama3.2-1b-offline-simpo-${beta}-lr-${lr}-gamma-to-beta-${gamma_to_beta}-no-length-norm-v5"

export CUDA_VISIBLE_DEVICES="6,7"
NUM_GPUs=2

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=WARN
# export TORCH_CPP_LOG_LEVEL=INFO
# export LOGLEVEL=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_PROJECT="llama3-1b-v5"
export WANDB_MODE="offline"
export ACCELERATE_LOG_LEVEL="info"

LOG_DIR="simpo_1b_training_logs"
LOG_PATH="${LOG_DIR}/log_run_${RUN_NAME}.log"
# Make logging directories.
mkdir -p "${LOG_DIR}"

echo "Placing logs in: ${LOG_DIR}"

nvidia-smi

NUM_PROCS=${NUM_GPUs}

accelerate launch \
    --config_file=recipes/accelerate_configs/deepspeed_zero2.yaml \
    --num_machines 1 \
    --num_processes $NUM_PROCS \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank 0 \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    scripts/run_dpo.py recipes/llama-3.2-1b/simpo/config_qlora.yaml \
        --learning_rate=${lr} \
        --beta=${beta} \
        --gamma_beta_ratio=${gamma_to_beta} \
        --output_dir=/work/saeed/narval/simpo_1b_v5/${RUN_NAME} \
        --run_name=${RUN_NAME} > ${LOG_PATH} 2>&1

