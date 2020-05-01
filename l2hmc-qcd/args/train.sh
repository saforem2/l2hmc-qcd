#!/bin/bash

TRAINER='../main.py'
ARGS='./gauge_args.txt'

export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=8
export KMP_SETTINGS=TRUE

python3 ${TRAINER} @${ARGS}
