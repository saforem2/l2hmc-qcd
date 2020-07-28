#!/bin/bash

# TRAINER='../main_eager.py'
TRAINER=../l2hmc-qcd/train.py
ARGS=./train_args.json

export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=1
export KMP_SETTINGS=TRUE
export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
export AUTOGRAPH_VERBOSITY=10

# export TF_XLA_FLAGS=“--tf_xla_cpu_global_jit”

export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

python3 -m pudb ${TRAINER} --json_file=./train_args.json
