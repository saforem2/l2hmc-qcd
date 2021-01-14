#!/bin/bash
#COBALT -A datascience

echo -e "\n"
echo "Starting cobalt job script..."

date

source /lus/theta-fs0/software/thetagpu/conda/tf_master/latest/mconda3/setup.sh

# export OMP_NUM_THREADS=1
# ====
# For CPU might want to try:
# export OMP_NUM_THREADS=64 

# export KMP_SETTINGS=1
export OMPI_MCA_opal_cuda_support=true
export NCCL_DEBUG=INFO
export KMP_SETTINGS=TRUE
export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
export AUTOGRAPH_VERBOSITY=10

# ====
# NOTE: `--tf_xla_enable_xla_devices` required for enabling XLA compilation
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices"

# ====
# Doesn't seem to be necessary/useful?
# export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_BASE}"

export TF_ENABLE_AUTO_MIXED_PRECISION=1

export PATH=$PATH:$HOME/.local/bin
export PYTHONPATH=/lus/theta-fs0/software/thetagpu/conda/tf_master/latest/mconda3/lib/python3.8/site-packages:$PYTHONPATH

echo COBALT_NODEFILE=$COBALT_NODEFILE
echo COBALT_JOBID=$COBALT_JOBID

# git log -1 --pretty >> git_log.txt

# ====
# get current folder containing this script
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
echo DIR=$DIR

# ====
# Check if `l2hmc-qcd/` exists in current directory
# If not, clone the repo, checkout `dev` branch, pull
L2HMC_DIR="$DIR/l2hmc-qcd"
if [[ ! -d $L2HMC_DIR ]]; then
    git clone https://github.com/saforem2/l2hmc-qcd
    git -C ${L2HMC_DIR} checkout dev
    git -C ${L2HMC_DIR} pull
fi

# ====
# Specify location of training and configuration scripts
TRAINER="$L2HMC_DIR/l2hmc-qcd/train.py"
JSON_FILE="$DIR/train_configs.json"

# ====
# Check if `./log_dirs.txt` exists,
# if so, it contains the path to the 
# directory of the most recent training run
LOG_DIR_FILE="$DIR/log_dirs.txt"

# ====
# Specify GPU information for thetaGPU
# total num of gpus = (num nodes * 8 gpus / node)
NODES=$(cat $COBALT_NODEFILE | wc -l)
PPN=8
PROCS=$((NODES * PPN))
echo NODES=$NODES  PPN=$PPN  PROCS=$PROCS

echo python3: $(which python3)

# ====
# Extra options for autotuning performance of `horovod`
#-x HOROVOD_AUTOTUNE=1 \
#-x HOROVOD_AUTOTUNE_LOG=./autotune_log.csv \

# ====
# If `./log_dirs.txt` exists, read the last line of the file
# which should contain the directory (`log_dir`) of the most 
# recent training run.  If `log_dir` contains a training 
# checkpoint, try to load it and resume training.
if [[ -f $LOG_DIR_FILE ]]; then
    LOG_DIR=$(tail -n 1 $LOG_DIR_FILE)
    echo "**************************************************************"
    echo "LOADING MODEL FROM CHECKPOINT..." $(date)
    echo "**************************************************************"
    mpirun -n $PROCS -npernode $PPN --verbose -hostfile $COBALT_NODEFILE \
        --allow-run-as-root -bind-to none -map-by slot \
        -x TF_XLA_FLAGS -x LD_LIBRARY_PATH -x PATH \
        -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=^docker0,lo \
        python3 $TRAINER --json_file=$JSON_FILE --log_dir=$LOG_DIR
else
    echo "**************************************************************"
    echo "STARTING A NEW TRAINING RUN:" $(date)
    echo "**************************************************************"
    mpirun -n $PROCS -npernode $PPN  -hostfile $COBALT_NODEFILE \
        --allow-run-as-root -bind-to none -map-by slot \
        -x TF_XLA_FLAGS -x LD_LIBRARY_PATH -x PATH \
        -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=^docker0,lo \
        python3 $TRAINER --json_file=$JSON_FILE
fi

echo -e "\n"
echo "=================================================================================================================="
exit $status
