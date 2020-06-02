#!/bin/bash

TRAINER='../main.py'
ARGS='./gauge_args.txt'

KMP_BLOCKTIME=0
OMP_NUM_THREADS=8
KMP_SETTINGS=TRUE
KMP_AFFINITY=granularity=fine,compact,1,0

# export TF_XLA_FLAGS=“--tf_xla_cpu_global_jit”

 TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

ipython3 -m pudb ${TRAINER} @${ARGS}
