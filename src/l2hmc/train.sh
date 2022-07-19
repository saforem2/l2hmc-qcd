#!/bin/bash
#COBALT -n 1
#COBALT -q single-gpu
#COBALT -A DLHMC
#COBALT --attrs filesystems=home,theta-fs0,grand,eagle
#COBALT -O l2hmc

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: ${TSTAMP}"

# ---- Get job information -----------------------------------------------
NRANKS=$(wc -l < ${COBALT_NODEFILE})
NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))

. /etc/profile

# Load conda module and activate base environment
# eval "$(/lus/theta-fs0/software/thetagpu/conda/2022-07-01/mconda3/bin/conda shell.zsh hook)"
module load conda/2022-07-01
conda activate base

# ---- Environment settings -----------------------------------------------
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
EXEC="${DIR}/main.py"
PARENT=$(dirname $DIR)
ROOT=$(dirname $PARENT)
# ROOT=$(dirname $PARENT)

HOST=$(hostname)
LOGDIR="${DIR}/logs"
LOGFILE="${LOGDIR}/${TSTAMP}-${HOST}_ngpu${NGPUS}.log"
if [ ! -d "${LOGDIR}" ]; then
  mkdir -p ${LOGDIR}
fi

echo $LOGFILE >> "${DIR}/logs/latest"

# echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "DIR=${DIR}"
echo "EXEC=${EXEC}"
echo "PARENT=${PARENT}"
echo "ROOT=${ROOT}"
echo "LOGDIR=${LOGDIR}"
echo "LOGFILE=${LOGFILE}"

conda run python3 -m pip install --upgrade pip

if [ -d "${ROOT}/venv/" ]; then
  source "${ROOT}/venv/bin/activate"
  conda run python3 -m pip install -e "${ROOT}" --no-deps
fi


# if [ -d "${ROOT}/venv/" ]; then
#   source "${ROOT}/venv/bin/activate"
#   python3 -m pip install -e "${ROOT}" --no-deps
# fi


# ---- Install required packages ------------------------------------------
conda run python3 -m pip install \
  hydra-core \
  hydra_colorlog \
  arviz \
  ipython \
  pyright \
  celerite \
  joblib \
  xarray \
  seaborn \
  bokeh \
  nodejs \
  h5py \
  matplotx \
  torchviz \

python3 -m pip install --pre --upgrade aim
python3 -m pip install --pre --upgrade wandb


# ---- Environment settings -----------------------------------------------
export NCCL_DEBUG=INFO
export KMP_SETTINGS=TRUE
export OMP_NUM_THREADS=16
export OMPI_MCA_opal_cuda_support=true
export TF_ENABLE_AUTO_MIXED_PRECISION=1
# export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"
# /lus/theta-fs0/software/thetagpu/cuda/TensorRT-8.2.5.1/lib
# :/lus/theta-fs0/software/thetagpu/cuda/nccl_2.12.12-1+cuda11.6_x86_64/lib
# :/lus/theta-fs0/software/thetagpu/cuda/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib64
# :/usr/local/cuda-11.4/lib64
# :/lus/theta-fs0/software/thetagpu/ucx/ucx-1.12.1_gcc-9.4.0/lib
# :/lus/theta-fs0/software/thetagpu/openmpi/openmpi-4.1.4_ucx-1.12.1_gcc-9.4.0/lib 
# export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
# export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices"



# ---- Print job information ---------------------------------------------
echo "********************************************************************"
echo "STARTING A NEW RUN ON ${NGPU} GPUs"
echo "DATE: ${TSTAMP}"
echo "NRANKS: $NRANKS"
echo "NGPUS PER RANK: ${NGPU_PER_RANK}"
echo "NGPUS TOTAL: ${NGPUS}"
echo "EXEC: ${EXEC}"
echo "Writing logs to $LOGFILE"
echo "python3: $(which python3)"
echo "l2hmc: $(python3 -c 'import l2hmc; print(l2hmc.__file__)')"
echo "********************************************************************"

# ---- Run Job --------------------------------------------
# WIDTH=$COLUMNS nohup \
#   mpirun -np ${NGPUS} \
#   -hostfile ${COBALT_NODEFILE} \
#   --verbose \
python3 ${EXEC} mode=debug framework=tensorflow  #> ${LOGFILE} 2>&1 &
# if [ -f ${EXEC} ]; then
#   # WIDTH=$COLUMNS nohup \
#     # mpirun -np ${NGPUS} \
#     # -hostfile ${COBALT_NODEFILE} \
#     # --verbose \
#   python3 ${EXEC} mode=debug framework=tensorflow $@ > ${LOGFILE} 2>&1 &
# fi
# exit 0
