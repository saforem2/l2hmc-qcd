#!/bin/sh

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: ${TSTAMP}"

# module load conda/2021-11-30
eval "$(/lus/grand/projects/DLHMC/conda/2021-11-30/bin/conda shell.zsh hook)"

export OMPI_MCA_opal_cuda_support=true
export NCCL_DEBUG=INFO
export KMP_SETTINGS=TRUE
export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
export OMP_NUM_THREADS=16
# export NUMEXPR_MAX_THREADS=256

NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F "," '{print NF}')

SRC="/lus/grand/projects/DLHMC/projects/l2hmc-qcd/src/l2hmc"
EXEC="${SRC}/main.py"
LOGDIR="${SRC}/logs/pytorch"
LOGFILE="${LOGDIR}/${TSTAMP}_train_pytorch_ngpu${NGPU}.log"


echo "************************************"
echo "STARTING A NEW RUN ON ${NGPU} GPUs"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DATE: ${TSTAMP}"
echo "EXEC: ${EXEC}"
echo "Writing logs to $LOGFILE"
echo "************************************"

echo $LOGFILE >> "${SRC}/logs/latest"
echo $LOGFILE >> "${SRC}/logs/pytorch/latest"

if [ ! -d "${LOGDIR}" ]; then
  mkdir -p ${LOGDIR}
fi


ACCELERATE_CONFIG="${SRC}/conf/accelerate/gpu/accelerate${NGPU}.yaml"
# if [ -f $ACCELERATE_CONFIG ]; then
accelerate launch --config_file="${ACCELERATE_CONFIG}" \
  "${EXEC}" framework=pytorch "$@" > ${LOGFILE} 2>&1 &
