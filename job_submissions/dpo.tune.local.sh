#!/bin/bash

date;pwd

export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export RDVZ_ID=$RANDOM
echo "RDZV Endpoint $MASTER_ADDR:$MASTER_PORT"

export CUDA_VISIBLE_DEVICES="0"
NUM_GPUs=1

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=WARN
# export TORCH_CPP_LOG_LEVEL=INFO
# export LOGLEVEL=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_PROJECT="llama3-1b-v2"
export WANDB_MODE="offline"
export ACCELERATE_LOG_LEVEL="info"

NUM_PROCS=${NUM_GPUs}

LOG_DIR="dpo_1b_training_logs"
# Make logging directories.
mkdir -p "${LOG_DIR}"

echo "Placing logs in: ${LOG_DIR}"
echo "GPUs per node: ${NUM_GPUs}"

lrs=(0.0000005)
# betas=(0.05 0.1 0.5)
betas=(0.01)

for l in ${!lrs[@]};
do
    lr=${lrs[$l]}
    for b in ${!betas[@]};
    do
        beta=${betas[$b]}
        RUN_NAME="llama3-1b-offline-dpo-beta-${beta}-lr-${lr}-v2"
        accelerate launch \
            --config_file=recipes/accelerate_configs/deepspeed_zero2.yaml \
            --num_machines 1 \
            --num_processes $NUM_PROCS \
            --main_process_ip $MASTER_ADDR \
            --main_process_port $MASTER_PORT \
            --machine_rank 0 \
            --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
            scripts/run_dpo.py recipes/llama-3.2-1b/dpo/config_qlora.yaml \
                --learning_rate=${lr} \
                --beta=${beta} \
                --output_dir=/scratch/ssd004/scratch/snajafi/vector-backup/dpo_1b_lora_v2/${RUN_NAME} \
                --run_name=${RUN_NAME} > ${LOG_DIR}/log_${RUN_NAME}.log 2>&1
    done
done