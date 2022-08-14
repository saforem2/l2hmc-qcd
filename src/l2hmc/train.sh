#!/bin/bash
#COBALT -n 1
#COBALT -q single-gpu
#COBALT -A DLHMC
#COBALT --attrs filesystems=home,theta-fs0,grand,eagle

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
HOST=$(hostname)
echo "Job started at: ${TSTAMP}"

# ---- Specify directories and executable for experiment ------------------
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
MAIN="${DIR}/main.py"
PARENT=$(dirname $DIR)
ROOT=$(dirname $PARENT)
# ROOT=$(dirname $PARENT)

LOGDIR="${DIR}/logs"
LOGFILE="${LOGDIR}/${TSTAMP}-${HOST}_ngpu${NGPUS}.log"
if [ ! -d "${LOGDIR}" ]; then
  mkdir -p ${LOGDIR}
fi

echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  Job started at: ${TSTAMP} on ${HOST}                         ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"

# Keep track of latest logfile for easy access
echo $LOGFILE >> "${DIR}/logs/latest"

# Double check everythings in the right spot
echo "DIR=${DIR}"
echo "MAIN=${MAIN}"
echo "PARENT=${PARENT}"
echo "ROOT=${ROOT}"
echo "LOGDIR=${LOGDIR}"
echo "LOGFILE=${LOGFILE}"

NCPUS=$(getconf _NPROCESSORS_ONLN)
# Load conda module and activate base environment
# CONDA_EXEC="/lus/theta-fs0/software/thetagpu/conda/2022-07-01/mconda3/bin/conda"
# if [[ -f ${CONDA_EXEC} ]]; then
  # eval "$(/lus/theta-fs0/software/thetagpu/conda/2022-07-01/mconda3/bin/conda shell.zsh hook)"
# fi

# # ---- Check if running on Linux ---------------------------------
# if [[ $(uname) == Linux* ]]; then
# ---- Check if running on ThetaGPU ----------------------------
if [[ $(hostname) == theta* ]]; then
  NRANKS=$(wc -l < ${COBALT_NODEFILE})
  HOSTFILE=${COBALT_NODEFILE}
  NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
  NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
  MPI_COMMAND=$(which mpirun)
  MPI_FLAGS="--verbose \
    -n ${NGPUS} \
    --hostfile ${HOSTFILE} \
    -npernode ${NGPU_PER_RANK} \
    -x PATH \
    -x LD_LIBRARY_PATH"
  module load conda/2022-07-01
  conda activate base
# ---- Check if running on Polaris -----------------------------
elif [[ $(hostname) == x* ]]; then
  NRANKS=$(wc -l < ${PBS_NODEFILE})
  HOSTFILE=${PBS_NODEFILE}
  NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
  NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
  MPI_COMMAND=$(which mpiexec)
  MPI_FLAGS="--verbose \
    --envall \
    -n ${NGPUS} \
    --ppn ${NGPU_PER_RANK} \
    --hostfile ${HOSTFILE}"
  module load conda/2022-07-19
  conda activate base
# ---- Check if running on MacOS --------------------------------
elif [[ $(uname) == Darwin* ]]; then
  # ---- Check if environment has an mpirun executable ----------
  if [[ -x $(which mpirun) ]]; then
    MPI_COMMAND=$(which mpirun)
    MPI_FLAGS="-np ${NCPUS}"
  fi
# ---- Otherwise, run without MPI -------------------------------
else
    MPI_COMMAND=""
    MPI_FLAGS=""
    echo "HOSTNAME: $(hostname)"
fi


# -----------------------------------------------------------
# 1. Check if a virtual environment exists in project root: 
#    `l2hmc-qcd/`
#
# 2. If so, activate environment and make sure we have an 
#    editable install
# -----------------------------------------------------------
VENV_DIR="${ROOT}/venv/"
# if [ -d ${VENV_DIR} ]; then
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Found venv at: ${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
  python3 -m pip install -e "${ROOT}" --no-deps
else
  echo "Creating new venv at: ${VENV_DIR}"
  python3 -m venv "${ROOT}/venv/" --system-site-packages
  source "${VENV_DIR}/bin/activate"
  python3 -m pip install -e "${ROOT}" --no-deps
fi

python3 -m pip install --upgrade pip
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
  accelerate \
  matplotx \
  torchviz

python3 -m pip install --pre --upgrade aim
python3 -m pip install --pre --upgrade wandb

# ---- Environment settings -----------------------------------------------
export NCCL_DEBUG=INFO
export KMP_SETTINGS=TRUE
export OMP_NUM_THREADS=16
export OMPI_MCA_opal_cuda_support=true
export TF_ENABLE_AUTO_MIXED_PRECISION=1
# export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"
# export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
# export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices"

EXEC="${MPI_COMMAND} ${MPI_FLAGS} $(which python3) ${MAIN}"


# ---- Print job information -------------------------------------------------
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  STARTING A NEW RUN ON ${NGPU} GPUs ${NCPUS}   ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
printf '%.s─' $(seq 1 $(tput cols))
echo "┃  - DATE: ${TSTAMP}"
echo "┃  - NCPUS: ${NCPUS}"
echo "┃  - NRANKS: ${NRANKS}"
echo "┃  - NGPUS PER RANK: ${NGPU_PER_RANK}"
echo "┃  - NGPUS TOTAL: ${NGPUS}"
echo "┃  - MAIN: ${MAIN}"
echo "┃  - Writing logs to ${LOGFILE}"
echo "┃  - python3: $(which python3)"
echo "┃  - mpirun: $(which mpirun)"
echo "┃  - l2hmc: $(python3 -c 'import l2hmc; print(l2hmc.__file__)')"
echo "┃  - exec: ${EXEC}"
# echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
printf '%.s─' $(seq 1 $(tput cols))

# Run executable command
${EXEC} $@ > ${LOGFILE}

# if [[ -x "${MPI_COMMAND}" ]]; then
#   ${MPI_COMMAND} ${MPI_FLAGS} $(which python3) ${MAIN} $@ > ${LOGFILE}
# else
#   $(which python3) ${MAIN} $@ > ${LOGFILE}
# fi
# $EXEC

# # if [ -f ${MAIN} ]; then
# # ---- Run Job --------------------------------------------
# if (( ${NGPUS} > 1 )); then
#   ${MPI_COMMAND} ${MPI_FLAGS} python3 ${MAIN} $@ > ${LOGFILE}
# else
#   NCPUS=$(getconf _NPROCESSORS_ONLN)
#   MPI_COMMAND=$(which mpirun)
#   if [ -x "${MPI_COMMAND}" ]; then
#     ${MPI_COMMAND} -np ${NCPUS} python3 ${MAIN} $@ > ${LOGFILE}
#   else
#     python3 ${MAIN} $@ > ${LOGFILE}
#   fi
# fi

exit

# vim:tw=4
