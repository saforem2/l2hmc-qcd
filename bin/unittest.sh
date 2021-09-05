#!/bin/sh
#COBALT -A DLHMC

echo -e "\n"
echo "Starting cobalt job script..."

date

module load conda/2021-06-28
conda activate /lus/grand/projects/DLHMC/conda/2021-06-28

export OMPI_MCA_opal_cuda_support=true
export NCCL_DEBUG=INFO
export KMP_SETTINGS=TRUE
export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
# export AUTOGRAPH_VERBOSITY=10

export OMP_NUM_THREADS=16

# ====
# NOTE: `--tf_xla_enable_xla_devices` required for enabling XLA compilation
# export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices"
# unset TF_XLA_FLAGS

# ====
# Doesn't seem to be necessary/useful?
# export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_BASE}"

export TF_ENABLE_AUTO_MIXED_PRECISION=1

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
echo DIR=$DIR


#$TEST_SCRIPT="$DIR/../l2hmc-qcd/tests/test_training.py"
TEST_SCRIPT="/lus/grand/projects/DLHMC/l2hmc-qcd/l2hmc-qcd/tests/test.py"

# TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
# LOGFILE="test1gpu_$TSTAMP.log"
NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F "," '{print NF}')

# export CUDA_VISIBLE_DEVICES=0,1
echo "********************************************"
echo "STARTING A NEW TESTING RUN ON ${NGPU} GPUs"
echo "********************************************"

echo "Writing logs to $LOGFILE"

CONFIGS="test_configs.json"

# ====
# Check if `./log_dirs.txt` exists,
# if so, it contains the path to the 
# directory of the most recent training run
LOG_DIR_FILE="$DIR/log_dirs.txt"

DEVICE_LOG="cuda_visible_devices.log"

echo $CUDA_VISIBLE_DEVICES >> $DEVICE_LOG


if [[ -f $CONFIGS ]]; then
    echo "*************************************"
    echo "Loading configs from: $CONFIGS"
    echo "*************************************"
    if [[ -f $LOG_DIR_FILE ]]; then
        mpirun -np ${NGPU} -H localhost:${NGPU} \
            --verbose --allow-run-as-root -bind-to none -map-by slot \
            -x CUDA_VISIBLE_DEVICES \
            -x TF_ENABLE_AUTO_MIXED_PRECISION \
            -x OMPI_MCA_opal_cuda_support \
            -x KMP_SETTINGS -x KMP_AFFINITY -x AUTOGRAPH_VERBOSITY \
            -x TF_XLA_FLAGS -x LD_LIBRARY_PATH -x PATH \
            -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=^docker0,lo \
            python3 ${TEST_SCRIPT} \
                --configs_file=${CONFIGS} \
                --restore_from=${LOG_DIR}
    else
        mpirun -np ${NGPU} -H localhost:${NGPU} \
            --verbose --allow-run-as-root -bind-to none -map-by slot \
            -x CUDA_VISIBLE_DEVICES \
            -x TF_ENABLE_AUTO_MIXED_PRECISION \
            -x OMPI_MCA_opal_cuda_support \
            -x KMP_SETTINGS -x KMP_AFFINITY -x AUTOGRAPH_VERBOSITY \
            -x TF_XLA_FLAGS -x LD_LIBRARY_PATH -x PATH \
            -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=^docker0,lo \
            python3 ${TEST_SCRIPT} \
                --configs_file=${CONFIGS}
    fi
else
    echo "****************************"
    echo "USING DEFAULT CONFIGS" 
    echo "****************************"
    if  [[ -f $LOG_DIR_FILE ]]; then
        mpirun -np ${NGPU} -H localhost:${NGPU} \
            --verbose --allow-run-as-root -bind-to none -map-by slot \
            -x CUDA_VISIBLE_DEVICES \
            -x TF_ENABLE_AUTO_MIXED_PRECISION \
            -x OMPI_MCA_opal_cuda_support \
            -x KMP_SETTINGS -x KMP_AFFINITY -x AUTOGRAPH_VERBOSITY \
            -x TF_XLA_FLAGS -x LD_LIBRARY_PATH -x PATH \
            -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=^docker0,lo \
            python3 $TEST_SCRIPT \
                --restore_from=${LOG_DIR}
    else
        mpirun -np ${NGPU} -H localhost:${NGPU} \
            --verbose --allow-run-as-root -bind-to none -map-by slot \
            -x CUDA_VISIBLE_DEVICES \
            -x TF_ENABLE_AUTO_MIXED_PRECISION \
            -x OMPI_MCA_opal_cuda_support \
            -x KMP_SETTINGS -x KMP_AFFINITY -x AUTOGRAPH_VERBOSITY \
            -x TF_XLA_FLAGS -x LD_LIBRARY_PATH -x PATH \
            -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=^docker0,lo \
            python3 ${TEST_SCRIPT}
    fi
fi

echo -e "\n"
echo "=================================================================================================================="
exit $status
