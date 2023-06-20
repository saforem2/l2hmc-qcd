#!/bin/bash -l

# TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
# HOST=$(hostname)
# echo "Job started at: ${TSTAMP} on ${HOST}"
# echo "Job running in: ${DIR}"


# ---- Specify directories and executable for experiment ------------------
# TMP=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
# DIR=$(dirname "${TMP}")
# MAIN="${DIR}/main.py"
# PARENT=$(dirname "$DIR")
# ROOT=$(dirname "$PARENT")

# Resolve path to current file
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
PARENT=$(dirname "${DIR}")
MAIN="${PARENT}/main.py"
GRANDPARENT=$(dirname "${PARENT}")
ROOT=$(dirname "${GRANDPARENT}")

NCPUS=$(getconf _NPROCESSORS_ONLN)

thetaGPU220701() {
  module load conda/2022-07-01 ; conda activate base
  conda activate \
    /lus/grand/projects/datascience/foremans/locations/thetaGPU/miniconda3/envs/2022-07-01
}

thetaGPU230426() {
  module load conda/2023-01-11
  conda activate base
  conda activate /lus/grand/projects/datascience/foremans/locations/thetaGPU/miniconda3/envs/2023-04-26
}

setupConda() {
  CONDA_DATE="$1"
  module load "conda/${CONDA_DATE}"
  conda activate base
  echo "Using: $(which python3)"
}

setupCondaCustom() {
  CONDA_PATH="$2"
  conda activate "$CONDA_PATH"
  echo "Using $(which python3)"
}


venvSetup() {
  VENV_DIR="$1"
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Found venv at ${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
  else
    echo "No venv found"
    mkdir -p "${VENV_DIR}"
    echo "Making new venv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}" --system-site-packages
    source "${VENV_DIR}/bin/activate"
    python3 -m pip install --upgrade pip setuptools wheel
    python3 -m pip install -e "${ROOT}[dev]"
  fi
}

# ┏━━━━━━━━━━┓
# ┃ ThetaGPU ┃
# ┗━━━━━━━━━━┛
setupThetaGPU() {
  if [[ $(hostname) == theta* ]]; then
    export MACHINE="ThetaGPU"
    HOSTFILE="${COBALT_NODEFILE}"
    export NVME_PATH="/raid/scratch/"
    # module load conda/2022-07-01 ; conda activate base
    # thetaGPU220701
    thetaGPU230426
    VENV_DIR="${ROOT}/venvs/thetaGPU/2023-04-26"
    venvSetup "$VENV_DIR"
    # -- MPI / Comms Setup ----------------------------------
    echo "HOSTFILE: ${HOSTFILE}"
    echo "COBALT_NODEFILE: ${COBALT_NODEFILE}"
    NHOSTS=$(wc -l < "${COBALT_NODEFILE}")
    NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
    NGPUS=$((NHOSTS * NGPU_PER_RANK))
    MPI_COMMAND=$(which mpirun)
    MPI_DEFAULTS="\
      --verbose \
      --hostfile ${HOSTFILE} \
      -x CFLAGS \
      -x LDFLAGS \
      -x PYTHONUSERBASE \
      -x http_proxy \
      -x https_proxy \
      -x PATH \
      -x LD_LIBRARY_PATH"
    MPI_ELASTIC="\
      -n ${NGPUS} \
      -npernode ${NGPU_PER_RANK}"
  else
    echo "Unexpected hostname: $(hostname)"
  fi
}

# ┏━━━━━━━━━┓
# ┃ Polaris ┃
# ┗━━━━━━━━━┛
setupPolaris()  {
  if [[ $(hostname) == x* ]]; then
    export MACHINE="Polaris"
    HOSTFILE="${PBS_NODEFILE}"
    export IBV_FORK_SAFE=1
    export NVME_PATH="/local/scratch/"
    # -----------------------------------------------
    # module load conda/2022-09-08; conda activate base
    # VENV_DIR="${ROOT}/venvs/polaris/2022-09-08"
    module load conda/2023-01-10-unstable
    conda activate base
    VENV_DIR="${ROOT}/venvs/polaris/2023-01-10"
    venvSetup "$VENV_DIR"
    # -----------------------------------------------
    NRANKS=$(wc -l < "${PBS_NODEFILE}")
    NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
    NGPUS=$((NHOSTS * NGPU_PER_RANK))
    MPI_COMMAND=$(which mpiexec)
    # -----------------------------------------------
    MPI_DEFAULTS="\
      --envall \
      --verbose \
      --hostfile ${HOSTFILE}"
    MPI_ELASTIC="\
      -n ${NGPUS} \
      --ppn ${NGPU_PER_RANK}"
  else
    echo "Unexpected hostname: $(hostname)"
  fi
}


setupJob() {
  # ---- Environment settings -----------------------------------------------
  export OMP_NUM_THREADS=$NCPUS
  export WIDTH=$COLUMNS
  export COLUMNS=$COLUMNS
  echo "WIDTH: ${COLUMNS}"
  export NCCL_DEBUG=ERROR
  export MACHINE="${MACHINE}"
  export CFLAGS="-I${CONDA_PREFIX}/include/"
  export LDFLAGS="-L${CONDA_PREFIX}/lib/"
  export WANDB_CACHE_DIR="${ROOT}/.cache/wandb"
  export TORCH_EXTENSIONS_DIR="${ROOT}/.cache/torch_extensions"
  mkdir -p "${ROOT}/.cache/{wandb,torch_extensions}"
  # export KMP_SETTINGS=TRUE
  # export OMPI_MCA_opal_cuda_support=TRUE
  # export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
  # export PATH="${CONDA_PREFIX}/bin:${PATH}"
  export TF_ENABLE_AUTO_MIXED_PRECISION=1
  # export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices"
  export NVME_PATH="${NVME_PATH}"
  export MPI_DEFAULTS="${MPI_DEFAULTS}"
  export MPI_ELASTIC="${MPI_ELASTIC}"
  export MPI_COMMAND="${MPI_COMMAND}"
  PYTHON_EXECUTABLE="$(which python3)"
  export PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}"
  echo "USING PYTHON: $(which python3)"
  echo "CFLAGS: ${CFLAGS}"
  echo "LDFLAGS: ${LDFLAGS}"
  # -------------------------------
  # CONSTRUCT EXECUTABLE TO BE RAN
  # -------------------------------
  EXEC="${MPI_COMMAND} ${MPI_DEFAULTS} ${MPI_ELASTIC} $(which python3) ${MAIN}"
  export EXEC="$EXEC"
}
