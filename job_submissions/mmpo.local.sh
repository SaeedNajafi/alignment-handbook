#!/bin/bash

date;pwd

export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export RDVZ_ID=$RANDOM
echo "RDZV Endpoint $MASTER_ADDR:$MASTER_PORT"

RUN_NAME="llama3.2-1b-offline-mmpo-beta-0.05-lr-1.0e-6-re_ep_1.0-relu_coe-0.0-relu_eps-2.0"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NUM_GPUs=8

LOG_DIR="training_logs"
LOG_PATH="${LOG_DIR}/log_${RUN_NAME}.log"
# Make logging directories.
mkdir -p "${LOG_DIR}"

echo "Placing logs in: ${LOG_DIR}"
echo "GPUs per node: ${NUM_GPUs}"

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=WARN
# export TORCH_CPP_LOG_LEVEL=INFO
# export LOGLEVEL=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

NUM_PROCS=${NUM_GPUs}

WANDB_MODE=offline ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file=recipes/accelerate_configs/deepspeed_zero2.yaml \
    --num_machines 1 \
    --num_processes $NUM_PROCS \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank 0 \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    scripts/run_dpo.py recipes/llama-3.2-1b/mmpo/config_qlora.yaml > ${LOG_PATH} 2>&1