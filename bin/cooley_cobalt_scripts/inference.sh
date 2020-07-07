#!/bin/bash
#COBALT -n 1
#COBALT -t 02:00
#COBALT -A DLHMC
#COBALT -q pubnet-debug
#COBALT --attrs pubnet:nox11

echo "-----------------------------------------------------------------------"
echo “Starting Cobalt job script...”
date

# Here I use the nodefile to determine the number of nodes,
# and multiply by 2 to put one rank on each GPU
# NODES=$(cat $COBALT_NODEFILE | wc -l)
NODES=$( < $COBALT_NODEFILE wc -l )
# let PROCS=$((NODES*2))
(( PROCS=$((NODES*2)) ))



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

env > env.txt

#export PATH=/soft/libraries/mpi/mvapich2/gcc/bin/:${PATH}

RUNNER='/lus/theta-fs0/projects/DLHMC/l2hmc-qcd/l2hmc-qcd/run.py'
HMC_ARGS='--hmc --beta 4. --eps 0.1 --num_steps 2 --run_steps 2000'
LATTICE_SHAPE="--lattice_shape '1024, 16, 16, 2'"
LOG_DIR=$( tail -n 1 ./log_dirs.txt ) 


python3 --version >> env.txt

python3 ${RUNNER} --run_steps 2000 --log_dir=${LOG_DIR}
python3 ${RUNNER} ${HMC_ARGS} --lattice_shape '1024, 16, 16, 2'
python3 ${RUNNER} \
    --hmc \
    --beta 4. \
    --num_steps 2 \
    --run_steps 2000 \
    --lattice_shape '1024, 16, 16, 2'
