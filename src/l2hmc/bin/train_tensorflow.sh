#!/bin/sh

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: ${TSTAMP}"

eval "$(/lus/grand/projects/DLHMC/conda/2021-11-30/bin/conda shell.zsh hook)"

export OMP_NUM_THREADS=16
export NCCL_DEBUG=INFO
export KMP_SETTINGS=TRUE
export OMPI_MCA_opal_cuda_support=true
export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices"

export TF_ENABLE_AUTO_MIXED_PRECISION=1

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
# echo DIR=$DIR

# export NUMEXPR_MAX_THREADS=256

SRC="/lus/grand/projects/DLHMC/projects/l2hmc-qcd/src/l2hmc"
# SRC=".."
# LOGDIR=/lus/grand/projects/DLHMC/projects/l2hmc-qcd/src/l2hmc/logs/tensorflow
# LOGDIR="${SRC}/logs/pytorch"
# LOGDIR="../logs/tensorflow"
# LOGDIR="../logs/tensorflow"
LOGDIR="${SRC}/logs/tensorflow"
LOGFILE="${LOGDIR}/${TSTAMP}_train_tf_ngpu${NGPU}.log"
# LOGFILE="$LOGDIR/train_tf_ngpu${NGPU}_$TSTAMP.log"
if [ ! -d "${LOGDIR}" ]; then
  mkdir -p ${LOGDIR}
fi

EXEC="${SRC}/main.py"

NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F "," '{print NF}')

echo "*************************************************************"
echo "STARTING A NEW RUN ON ${NGPU} GPUs"
echo "DATE: ${TSTAMP}"
echo "EXEC: ${EXEC}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Writing logs to $LOGFILE"
echo "*************************************************************"

echo $LOGFILE >> "${SRC}/logs/latest"
echo $LOGFILE >> "${SRC}/logs/tensorflow/latest"

mpirun -np ${NGPU} -H localhost:${NGPU} \
    --verbose \
    --allow-run-as-root \
    -bind-to none \
    -map-by slot \
    -x CUDA_VISIBLE_DEVICES \
    -x TF_ENABLE_AUTO_MIXED_PRECISION \
    -x OMPI_MCA_opal_cuda_support \
    -x KMP_SETTINGS \
    -x KMP_AFFINITY \
    -x AUTOGRAPH_VERBOSITY \
    -x TF_XLA_FLAGS \
    -x LD_LIBRARY_PATH \
    -x PATH \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=^docker0,lo \
    python3 ${EXEC} framework=tensorflow $@ > ${LOGFILE} 2>&1 &
