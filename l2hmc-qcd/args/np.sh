#!/bin/bash

RUNNER_NP='../gauge_inference_np.py'
ARGS_NP='./np.txt'

ipython3 -m pudb ${RUNNER_NP} @${ARGS_NP}
