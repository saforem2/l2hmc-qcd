#!/bin/bash -l
# -------------------------------------------------------
#PBS -k doe
#PBS -e <path for stderr>

##PBS -V exports all the environment variables in your environnment to the compute node
# The rest is an example of how an MPI job might be set up
# echo Working directory is $PBS_O_WORKDIR
# cd $PBS_O_WORKDIR

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
HOST=$(hostname)
# echo "Job ID: ${PBS_JOBID}"

# ---- Specify directories and executable for experiment ------------------
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
MAIN="${DIR}/main.py"
PARENT=$(dirname $DIR)
ROOT=$(dirname $PARENT)

echo "Job started at: ${TSTAMP} on ${HOST}"
echo "Job running in: ${DIR}"

NCPUS=$(getconf _NPROCESSORS_ONLN)

# if [[ ! -z "${CONDA_EXE}" ]] ; # && [[ $(hostname) == theta* ]] ; then

#   if [[ $(hostname) == theta* ]]; then
#     module load conda/2022-07-01 ; conda activate base
#     VENV_DIR="${ROOT}/venvs/thetaGPU/2022-07-01"  # -libaio"

#   elif [[ $(hostname) == x* ]]; then
#     module load conda ; conda activate base
#     VENV_DIR="${ROOT}/venvs/polaris/2023-01-10"
#   fi
# fi


# ---- Check if running on ThetaGPU ----------------------------
if [[ $(hostname) == theta* ]]; then
  MACHINE="ThetaGPU"
  HOSTFILE="${COBALT_NODEFILE}"
  # if [[ ! -z "${CONDA_EXE}" ]] ; then
  module load conda/2022-07-01 ; conda activate base
  conda activate /lus/grand/projects/datascience/foremans/locations/thetaGPU/miniconda3/envs/2022-07-01
  VENV_DIR="${ROOT}/venvs/thetaGPU/2022-07-01-deepspeed"  # -libaio"
  # module load conda/2023-01-11 ; conda activate base
  # VENV_DIR="${ROOT}/venvs/thetaGPU/2023-01-11"  # -libaio"
  # module load conda/2023-01-11 ; conda activate base
  # VENV_DIR="${ROOT}/venvs/thetaGPU/2023-01-11"
  # fi

  NRANKS=$(wc -l < ${HOSTFILE})
  # HOSTFILE=${COBALT_NODEFILE}
  NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
  NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
  MPI_COMMAND=$(which mpirun)
  export CFLAGS="-I${CONDA_PREFIX}/include/"
  export LDFLAGS="-L${CONDA_PREFIX}/lib/"
  # -x PYTHONSTARTUP \
  MPI_FLAGS="-n ${NGPUS} \
    --hostfile ${HOSTFILE} \
    -npernode ${NGPU_PER_RANK} \
    -x CFLAGS \
    -x LDFLAGS \
    -x PYTHONUSERBASE \
    -x http_proxy \
    -x https_proxy \
    -x PATH \
    -x LD_LIBRARY_PATH"

# ---- Check if running on Polaris -----------------------------
elif [[ $(hostname) == x* ]]; then
  # echo "----------------------"
  # echo "| Running on Polaris |"
  # echo "----------------------"
  MACHINE="Polaris"

  # if [[ ! -z "${CONDA_EXE}" ]] ; then
    # module load conda/2023-01-10; conda activate base
    # VENV_DIR="${ROOT}/venvs/polaris/2023-01-10"
  module load conda/2022-09-08; conda activate base
  VENV_DIR="${ROOT}/venvs/polaris/2022-09-08"
  # fi

  NRANKS=$(wc -l < ${PBS_NODEFILE})
  HOSTFILE=${PBS_NODEFILE}
  NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
  NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
  MPI_COMMAND=$(which mpiexec)
  # --cpu-bind verbose,list:0,8,16,24 \
  export CFLAGS="-I${CONDA_PREFIX}/include/"
  export LDFLAGS="-L${CONDA_PREFIX}/lib/"
  MPI_FLAGS="--envall \
    --verbose \
    -n ${NGPUS} \
    --depth ${NCPUS} \
    --ppn ${NGPU_PER_RANK} \
    --hostfile ${HOSTFILE}"
  # module load conda/2022-09-08; conda activate base
  unset MPICH_GPU_SUPPORT_ENABLED
  export IBV_FORK_SAFE=1
  export NCCL_COLLNET_ENABLE=1

# ---- Check if running on MacOS --------------------------------
else
  MACHINE=$(hostname)
  VENV_DIR="${ROOT}/venv/"
  if [[ $(uname) == Darwin* ]]; then
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
fi

# -----------------------------------------------------------
# 1. Locate virtual envronment to use:
#     a. Look for custom venv, unique to specific resource 
#         (e.g. `l2hmc-qcd/venvs/polaris/2022-09-08`)
#
#     b. Otherwise, look for generic environment at:
#         `l2hmc-qcd/venv/`
#
# 2. Perform Editable install
# -----------------------------------------------------------
if [[ ! -z "${VIRTUAL_ENV}" ]] ; then
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Found venv at: ${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
  else
    if [[ -f "${ROOT}/venv/bin/activate" ]]; then
      echo "Found venv at: ${ROOT}/venv/, using that"
      source "${VENV_DIR}/bin/activate"
    else
      echo "Creating new venv at: ${VENV_DIR}"
      python3 -m venv "${ROOT}/venv/" --system-site-packages
      source "${VENV_DIR}/bin/activate"
      python3 -m pip install --upgrade pip setuptools wheel
      python3 -m pip install -e "${ROOT}"
    fi
  fi
fi

# ---- Environment settings -----------------------------------------------
export OMP_NUM_THREADS=$NCPUS
export WIDTH=$COLUMNS
export COLUMNS=$COLUMNS
echo "WIDTH: ${COLUMNS}"
export NCCL_DEBUG=ERROR

export WANDB_CACHE_DIR="${ROOT}/.cache/wandb"
# export WANDB_
# export KMP_SETTINGS=TRUE
# export OMPI_MCA_opal_cuda_support=TRUE
# export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"
# export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'

export TF_ENABLE_AUTO_MIXED_PRECISION=1
# export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices"

LOGDIR="${DIR}/logs"
LOGFILE="${LOGDIR}/${TSTAMP}-${HOST}_ngpu${NGPUS}_ncpu${NCPUS}.log"
if [ ! -d "${LOGDIR}" ]; then
  mkdir -p ${LOGDIR}
fi


# Keep track of latest logfile for easy access
echo $LOGFILE >> "${DIR}/logs/latest"

# Double check everythings in the right spot
# printf '%.s─' $(seq 1 $(tput cols))

# -------------------------------
# CONSTRUCT EXECUTABLE TO BE RAN
# -------------------------------
EXEC="${MPI_COMMAND} ${MPI_FLAGS} $(which python3) ${MAIN}"


# ---- Print job information -------------------------------------------------
echo -e '\n'
# printf '%.s─' $(seq 1 $(tput cols))
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "┃ STARTING A NEW RUN @ ${MACHINE} ON ${NGPUS} GPUs ${NCPUS} CPUS  "
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "┃  - DIR=${DIR}"
echo "┃  - MAIN=${MAIN}"
echo "┃  - PARENT=${PARENT}"
echo "┃  - ROOT=${ROOT}"
echo "┃  - LOGDIR=${LOGDIR}"
echo "┃  - LOGFILE=${LOGFILE}"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "┃  - hostname: $(hostname)"
echo "┃  - DATE: ${TSTAMP}"
echo "┃  - NCPUS: ${NCPUS}"
echo "┃  - NRANKS: ${NRANKS}"
echo "┃  - NGPUS PER RANK: ${NGPU_PER_RANK}"
echo "┃  - NGPUS TOTAL: ${NGPUS}"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "┃  - MAIN: ${MAIN}"
echo "┃  - Writing logs to ${LOGFILE}"
echo "┃  - python3: $(which python3)"
echo "┃  - mpirun: ${MPI_COMMAND}"
echo "┃  - l2hmc: $(python3 -c 'import l2hmc; print(l2hmc.__file__)')"
echo "┃  - exec: ${EXEC} $@"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# printf '%.s─' $(seq 1 $(tput cols))
echo -e '\n'

echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo '┃ To view output: `tail -f $(tail -1 logs/latest)`'
echo "┃ Latest logfile: $(tail -1 ./logs/latest)"
echo "┃ tail -f $(tail -1 logs/latest)"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Run executable command
${EXEC} $@ > ${LOGFILE} & #; ret_code=$?

# if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# wait
