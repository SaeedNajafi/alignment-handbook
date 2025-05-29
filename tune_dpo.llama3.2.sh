#!/bin/bash

lrs=(0.0005)
# betas=(0.01 0.05 0.5 1.0 2.0)
betas=(0.01)
for lr_i in ${!lrs[@]};
do
    lr=${lrs[$lr_i]}
    for betas_i in ${!betas[@]};
    do
        beta=${betas[$betas_i]}
        sbatch job_submissions/dpo.slrm AVG_LOGPS=yes LR=${lr} BETA=${beta}
    done
done