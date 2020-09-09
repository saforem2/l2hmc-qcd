#!/bin/bash
#COBALT -n 16 
#COBALT -t 12:00
#COBALT -A DLHMC
#COBALT -q default
#COBALT --attrs nox11:pubnet

echo -e "\n"
echo -e "\n"
echo "-----------------------------------------------------------------------"
echo “Starting Cobalt job script...”
date

# Here I use the nodefile to determine the number of nodes,
# and multiply by 2 to put one rank on each GPU
# NODES=$(cat $COBALT_NODEFILE | wc -l)
# let PROCS=$((NODES*2))
NODES=$( < $COBALT_NODEFILE wc -l )
(( PROCS=$((NODES*2)) ))


export AUTOGRAPH_VERBOSITY=10
export NCCL_DEBUG=INFO

TRAINER=/lus/theta-fs0/projects/DLHMC/l2hmc-qcd/l2hmc-qcd/train.py

export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/soft/datascience/DL_frameworks/installation/cuda/cuda-10.1.243

unset LD_PRELOAD

export INSTALL_DIR=/soft/datascience/DL_frameworks/installation
export CUDA_DIR=${INSTALL_DIR}/cuda/cuda-10.1.243

export LD_LIBRARY_PATH=${CUDA_DIR}/lib:${CUDA_DIR}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_DIR}/bin:$PATH

export LD_LIBRARY_PATH=${INSTALL_DIR}/lib/:${INSTALL_DIR}/lib64/:$LD_LIBRARY_PATH
export PATH=${INSTALL_DIR}/bin/:$PATH

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/soft/compilers/gcc/7.1.0/lib64

LOG_DIR_FILE=./log_dirs.txt
if [[ -f "$LOG_DIR_FILE" ]]; then
    LOG_DIR=$(tail -n 1 ${LOG_DIR_FILE})
else
    LOG_DIR="None"
fi


LOG_DIR=$( tail -n 1 ./log_dirs.txt ) 

env >> env.txt
python3 --version >> env.txt

mpirun -np ${PROCS} \
    python3 ${TRAINER} \
        --log_dir=$LOG_DIR \
        --gpu \
        --horovod \
        --eager_execution \
        --save_train_data \
        --eps 0.1 \
        --num_steps 5 \
        --batch_size 1024 \
        --time_size 16 \
        --space_size 16 \
        --beta_init 3.5 \
        --beta_final 3.5 \
        --run_steps 2000 \
        --train_steps 20000 \
        --save_steps 5000 \
        --logging_steps 50 \
        --print_steps 10 \
        --hmc_start \
        --hmc_steps 2000 \
        --warmup_lr \
        --warmup_steps 5000 \
        --lr_init 0.001 \
        --lr_decay_steps 2500 \
        --decay_rate 0.96 \
        --plaq_weight 10. \
        --charge_weight 0.1 \
        --network_type 'GaugeNetwork' \
        --units '2048, 1024, 1024, 1024, 1024, 2048'
status=$?

echo "mpirun status is $status"
echo -e "\n"
echo "-----------------------------------------------------------------------"
exit $status
