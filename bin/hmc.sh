#!/bin/bash

export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=16
export KMP_SETTINGS=TRUE
export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'

# export TF_XLA_FLAGS=“--tf_xla_cpu_global_jit”

export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

RUNNER='../l2hmc-qcd/run.py'

RUN_STEPS=2000
LATTICE_SHAPE="128, 16, 16, 2"

#NUM_STEPS=( 1 2 3 4 5 8 10 )
NUM_STEPS=( 1 2 3 4 5 8 10 15 )
#EPS_ARR=( 0.075 0.1 0.15 )
EPS_ARR=( 0.2 )
BETA_ARR=( 3. 3.5 4. 4.5 )

for beta in ${BETA_ARR[@]}
do
    for eps in ${EPS_ARR[@]}
    do
        for steps in ${NUM_STEPS[@]}
        do
            echo -e "\n"
            python3 ${RUNNER} \
                --hmc \
                --eps ${eps} \
                --beta ${beta} \
                --num_steps ${steps} \
                --run_steps ${RUN_STEPS} \
                --lattice_shape '128, 16, 16, 2'
            echo -e "\n"
        done
    done
done
