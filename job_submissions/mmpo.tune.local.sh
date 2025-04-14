#!/bin/bash

date;pwd

eval "$(conda shell.bash hook)"
conda activate dpo-llm-env

export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export RDVZ_ID=$RANDOM
echo "RDZV Endpoint $MASTER_ADDR:$MASTER_PORT"

export CUDA_VISIBLE_DEVICES="2,3"
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

NUM_PROCS=${NUM_GPUs}

LOG_DIR="mmpo_1b_training_logs"
# Make logging directories.
mkdir -p "${LOG_DIR}"

echo "Placing logs in: ${LOG_DIR}"
echo "GPUs per node: ${NUM_GPUs}"

lrs=(0.0005)
betas=(0.01)
mmpo_relu_epsilon=(0.5)
reward_epsilons=(0.5)

for r_relu_eps_i in ${!mmpo_relu_epsilon[@]};
do
    r_relu_eps=${mmpo_relu_epsilon[$r_relu_eps_i]}
    for r_eps_i in ${!reward_epsilons[@]};
    do
        r_eps=${reward_epsilons[$r_eps_i]}
        for l in ${!lrs[@]};
        do
            lr=${lrs[$l]}
            for b in ${!betas[@]};
            do
                beta=${betas[$b]}
                RUN_NAME="llama3.2-1b-offline-mmpo-beta-${beta}-lr-${lr}-reward_eps_${r_eps}-relu-epsilon-${r_relu_eps}-v5-original-loss"
                accelerate launch \
                    --config_file=recipes/accelerate_configs/deepspeed_zero2.yaml \
                    --num_machines 1 \
                    --num_processes $NUM_PROCS \
                    --main_process_ip $MASTER_ADDR \
                    --main_process_port $MASTER_PORT \
                    --machine_rank 0 \
                    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
                    scripts/run_dpo.py recipes/llama-3.2-1b/mmpo/config_qlora.yaml \
                        --learning_rate=${lr} \
                        --beta=${beta} \
                        --mmpo_reward_epsilon=${r_eps} \
                        --mmpo_relu_epsilon=${r_relu_eps} \
                        --output_dir=/work/saeed/narval/mmpo_1b_v5/${RUN_NAME} \
                        --run_name=${RUN_NAME} > ${LOG_DIR}/log_${RUN_NAME}.log 2>&1
            done
        done
    done
done