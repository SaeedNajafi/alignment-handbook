#!/bin/bash

# lrs=(0.001 0.0001 0.0005 0.00001 0.00005 0.000001 0.000005)
# lrs=(0.0001 0.00005)
# betas=(0.01 0.05 0.5 1.0 2.0)
# betas=(0.1)
betas=(0.01)
lrs=(0.0001)
gamma_betas=(0.3 0.5 1.0 1.4 1.6)
for gamma_beta_i in ${!gamma_betas[@]};
do
    gamma_beta=${gamma_betas[$gamma_beta_i]}
    for lr_i in ${!lrs[@]};
    do
        lr=${lrs[$lr_i]}
        for betas_i in ${!betas[@]};
        do
            beta=${betas[$betas_i]}
            sbatch job_submissions/simpo.smollm.360m.slrm AVG_LOGPS=no LR=${lr} BETA=${beta} GAMMA_TO_BETA=${gamma_beta}
        done
    done
done