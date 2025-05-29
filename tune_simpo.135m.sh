#!/bin/bash

lrs=(0.0001)
# lrs=(0.0001 0.00005)
# betas=(0.01 0.05 0.5 1.0 2.0)
betas=(0.01)
for lr_i in ${!lrs[@]};
do
    lr=${lrs[$lr_i]}
    for betas_i in ${!betas[@]};
    do
        beta=${betas[$betas_i]}
        sbatch job_submissions/simpo.smollm.135m.slrm AVG_LOGPS=no LR=${lr} BETA=${beta} GAMMA_TO_BETA=1.6
    done
done