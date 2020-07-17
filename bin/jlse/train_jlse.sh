#!/bin/zsh
#COBALT -n 1
#COBALT -t 06:00
#COBALT -q dgx

# export MODULEPATH=/soft/modulefiles:/usr/share/Modules/modulefiles:/etc/modulefiles:/usr/share/modulefiles
# source $HOME/.jlse_env
source $HOME/.env

env >> env.txt
echo "PATH: $PATH" >> env.txt

# source $HOME/jlse_env.sh

echo -e "\n"
echo -e "\n"
echo "======================================================================"
echo "Starting cobalt job script..."
date


module load cuda

export https_proxy="https://proxy:3128"
export http_proxy="http://proxy:3128"
export ftp_proxy="ftp://proxy:3128"

# source $HOME/jlse_env.sh
# source $HOME/network_setup.sh

conda activate hvdNCCL

# NODES=$( < $COBALT_NODEFILE wc -l )
# (( PROCS=$((NODES * 8)) ))
lspci | grep -i nvidia >> gpu_list.txt
PROCS=$(wc -l < gpu_list.txt)

echo "HOSTNAME: $HOSTNAME"
echo "PROCS: $PROCS"

export OMPI_MCA_opal_cuda_support=true
export NCCL_DEBUG=INFO
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=1
export KMP_SETTINGS=TRUE
export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
export AUTOGRAPH_VERBOSITY=10

export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/soft/compilers/cuda/cuda-10.0.130"

TRAINER=$HOME/l2hmc-qcd/l2hmc-qcd/train.py

LOG_DIR_FILE=./log_dirs.txt
if [[ -f "$LOG_DIR_FILE" ]]; then
    horovodrun -np ${PROCS} --verbose --autotune \
        python3 ${TRAINER} --json_file=./train_args.json \
            --log_dir=$(tail -n 1 $LOG_DIR_FILE)
else
    horovodrun -np ${PROCS} --verbose --autotune \
        python3 ${TRAINER} --json_file=./train_args.json
fi

echo -e "\n"
echo "======================================================================"
exit $status
