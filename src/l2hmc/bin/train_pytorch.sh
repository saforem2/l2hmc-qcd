#!/bin/sh

echo -e "\n"
echo "Starting cobalt job script..."
date

# module load conda/2021-11-30
eval "$(/lus/grand/projects/DLHMC/conda/2021-11-30/bin/conda shell.zsh hook)"

export OMPI_MCA_opal_cuda_support=true
export NCCL_DEBUG=INFO
export KMP_SETTINGS=TRUE
export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
export OMP_NUM_THREADS=16
# export NUMEXPR_MAX_THREADS=256

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")

SRC="../"
LOGDIR="${SRC}/logs/pytorch"
LOGFILE="$LOGDIR/train_pytorch_ngpu${NGPU}_$TSTAMP.log"
if [ ! -d "${LOGDIR}" ]; then
  mkdir ${LOGDIR}

# SRC="/lus/grand/projects/DLHMC/projects/l2hmc-qcd/src/l2hmc"
# LOGDIR=/lus/grand/projects/DLHMC/projects/l2hmc-qcd/src/l2hmc/logs/pytorch

EXEC="${SRC}/main.py"

NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F "," '{print NF}')

echo "************************************"
echo "STARTING A NEW RUN ON ${NGPU} GPUs"
echo "DATE: ${TSTAMP}"
echo "EXEC: ${EXEC}"
echo "Writing logs to $LOGFILE"
echo "************************************"

ACCELERATE_CONFIG="${SRC}/conf/accelerate/gpu/accelerate${NGPU}.yaml"

accelerate launch --config_file="${ACCELERATE_CONFIG}" \
  "${EXEC}" framework=pytorch "$@" > ${LOGFILE} 2>&1 &
