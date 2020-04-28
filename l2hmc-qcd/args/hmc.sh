#!/bin/bash

RUNNER_NP='../gauge_inference_np.py'
ARGS='./hmc.txt'

python3 ${RUNNER_NP} @${ARGS}
