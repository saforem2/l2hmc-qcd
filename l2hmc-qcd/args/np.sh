#!/bin/bash

RUNNER_NP='../gauge_inference_np.py'
ARGS_NP='./np.txt'

python3 ${RUNNER_NP} @${ARGS_NP}
