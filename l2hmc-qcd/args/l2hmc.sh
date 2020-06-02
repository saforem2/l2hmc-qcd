#!/bin/bash

RUNNER_NP='../gauge_inference_np.py'
ARGS='./l2hmc.txt'

export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=8
export KMP_SETTINGS=TRUE

python3 ${RUNNER_NP} @${ARGS}


# ARGS = "--batch_size 1 \
#         --print_steps 10 \
#         --run_steps 50000"


#python3 ${RUNNER_NP} ${P} ${R} --batch_size 16 --beta 4.
# python3 ${RUNNER_NP} ${P} ${R} ${M} ${NW} --beta 5. --mix_samplers
# python3 ${RUNNER_NP} ${ARGS} ${NW} --mix_samplers --beta 5.
# python3 ${RUNNER_NP} ${P} ${R} ${M} --beta 6. --mix_samplers
# python3 ${RUNNER_NP} ${P} ${R} ${M} --beta 7. --mix_samplers

#python3 ${RUNNER_NP} ${P} ${R} --batch_size 16 --beta 5. --mix_samplers
