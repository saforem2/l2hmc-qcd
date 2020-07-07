#/bin/bash

RUNNER='../l2hmc-qcd/run.py'
# ARGS="$@"
# ARGS='./inference_args.txt'

export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=1
export KMP_SETTINGS=TRUE
export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'

# export TF_XLA_FLAGS=“--tf_xla_cpu_global_jit”

export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"


LOG_DIR=$(tail -n 1 ./log_dirs.txt)


# ipython3 -m pudb ${RUNNER} --log_dir=log_dirs
# ipython3 -m pudb ${RUNNER} --log_dir=$LOG_DIR --run_steps 2000
python3 -m pudb ${RUNNER} --log_dir=$LOG_DIR --run_steps 2000
