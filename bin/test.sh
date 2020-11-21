#!/bin/bash

export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=1
export KMP_SETTINGS=TRUE
export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
export AUTOGRAPH_VERBOSITY=10

# export TF_XLA_FLAGS=“--tf_xla_cpu_global_jit”

export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

TEST_FILE='../l2hmc-qcd/tests/test_training.py'

which -a python3

python3 -m pudb ${TEST_FILE} --test_all
