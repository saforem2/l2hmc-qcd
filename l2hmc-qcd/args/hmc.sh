#!/bin/bash

RUNNER_NP='../gauge_inference_np.py'
ARGS='./hmc.txt'

export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=8
export KMP_SETTINGS=TRUE

# python3 ${RUNNER_NP} @${ARGS}

B="--batch_size 1"
P="--print_steps 10"
R="--run_steps 20000"

python3 ${RUNNER_NP} ${B} ${P} ${R} --hmc --beta 5. --mix_samplers
python3 ${RUNNER_NP} ${B} ${P} ${R} --hmc --beta 6. --mix_samplers
python3 ${RUNNER_NP} ${B} ${P} ${R} --hmc --beta 7. --mix_samplers

