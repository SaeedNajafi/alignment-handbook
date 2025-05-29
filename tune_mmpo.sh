#!/bin/bash

# lrs=(0.001 0.0005 0.0001 0.00005 0.00001 0.000001 0.000005)
# lrs=(0.001 0.0005)
# betas=(0.01 0.05 0.5 1.0 2.0)
# betas=(0.1)
# lrs=(0.0005)
lrs=(0.0005)
betas=(0.01)
for lr_i in ${!lrs[@]};
do
    lr=${lrs[$lr_i]}
    for betas_i in ${!betas[@]};
    do
        beta=${betas[$betas_i]}
        sbatch job_submissions/mmpo.slrm AVG_LOGPS=no LR=${lr} BETA=${beta} REWARD_EPSILON=0.9
    done
done