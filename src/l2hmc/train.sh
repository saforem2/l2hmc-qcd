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
PARENT=$(dirname "$DIR")
ROOT=$(dirname "$PARENT")

echo "Job started at: ${TSTAMP} on ${HOST}"
echo "Job running in: ${DIR}"

NCPUS=$(getconf _NPROCESSORS_ONLN)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Machine Specific Configuration ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# - Because different ALCF resources have different software 
# stacks, we setup our environment differently depending on the
# machine in use.
#
# - In particular, we must:
#   1. Identify the machine / ALCF resource being used
#   2. Load the proper conda module and activate the base env
#   3. Identify the proper mpi executable
#   4. Specify the correct flags to pass to our mpi executable
#
# - We consider three cases:
#   1. If $(hostname) startswith theta*, we're on ThetaGPU
#   2. If $(hostname) startswith x*, we're on Polaris
# ---------------------------------------------------------------
#
# ┏━━━━━━━━━━┓
# ┃ ThetaGPU ┃
# ┗━━━━━━━━━━┛
setupThetaGPU() {
  if [[ $(hostname) == theta* ]]; then
    export MACHINE="ThetaGPU"
    HOSTFILE="${COBALT_NODEFILE}"

    # -- Python / Conda setup -------------------------------------------------
    module load conda/2022-07-01 ; conda activate base
    conda activate \
      /lus/grand/projects/datascience/foremans/locations/thetaGPU/miniconda3/envs/2022-07-01
    VENV_DIR="../../venvs/thetaGPU/2022-07-01-deepspeed"
    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
      # source "${VENV_DIR}/bin/activate"
      source ../../venvs/thetaGPU/2022-07-01-deepspeed/bin/activate
    else
      echo "No venv found"
      # python3 -m pip install -e "${ROOT}[dev]"
    fi
    export CFLAGS="-I${CONDA_PREFIX}/include/"
    export LDFLAGS="-L${CONDA_PREFIX}/lib/"
    # -------------------------------------------------------------------------

    # -- MPI / Comms Setup ----------------------------------
    NRANKS=$(wc -l < "${HOSTFILE}")
    NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
    NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
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
    # MPIEXEC="${MPI_COMMAND} ${MPI_DEFAULTS} ${MPI_ELASTIC}"
    # -------------------------------------------------------
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
    # -----------------------------------------------
    module load conda/2022-09-08; conda activate base
    VENV_DIR="../../venvs/polaris/2022-09-08"
    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
      source "${VENV_DIR}/bin/activate"
    else
      echo "No venv found"
      # python3 -m pip install -e "${ROOT}[dev]"
    fi
    export CFLAGS="-I${CONDA_PREFIX}/include/"
    export LDFLAGS="-L${CONDA_PREFIX}/lib/"
    export IBV_FORK_SAFE=1
    # -----------------------------------------------
    NRANKS=$(wc -l < "${HOSTFILE}")
    # HOSTFILE=${HOSTFILE}
    NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
    NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
    MPI_COMMAND=$(which mpiexec)
    # -----------------------------------------------
    MPI_DEFAULTS="\
      --envall \
      --verbose \
      --hostfile ${HOSTFILE}"
    MPI_ELASTIC="\
      -n ${NGPUS} \
      --ppn ${NGPU_PER_RANK}"
    # MPIEXEC="${MPI_COMMAND} ${MPI_FLAGS} ${MPI_ELASTIC}"
  else
    echo "Unexpected hostname: $(hostname)"
  fi
}


# ┏━━━━━━━━━┓
# ┃ ??????? ┃
# ┗━━━━━━━━━┛
# else
#   MACHINE=$(hostname)
#   VENV_DIR="${ROOT}/venv/"
#   if [[ $(uname) == Darwin* ]]; then
#     # Check if environment has an mpirun executable
#     if [[ -x $(which mpirun) ]]; then
#       MPI_COMMAND=$(which mpirun)
#       MPI_FLAGS="-np ${NCPUS}"
#     fi
#   # Otherwise, run without MPI
#   else
#       MPI_COMMAND=""
#       MPI_FLAGS=""
#       echo "HOSTNAME: $(hostname)"
#   fi
# fi
#
#
#
setupJob() {
  # ---- Environment settings -----------------------------------------------
  export OMP_NUM_THREADS=$NCPUS
  export WIDTH=$COLUMNS
  export COLUMNS=$COLUMNS
  echo "WIDTH: ${COLUMNS}"
  export NCCL_DEBUG=ERROR
  export MACHINE="${MACHINE}"

  export WANDB_CACHE_DIR="${ROOT}/.cache/wandb"
  # export KMP_SETTINGS=TRUE
  # export OMPI_MCA_opal_cuda_support=TRUE
  # export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"
  # export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'

  export TF_ENABLE_AUTO_MIXED_PRECISION=1
  # export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices"

  LOGDIR="${DIR}/logs"
  LOGFILE="${LOGDIR}/${TSTAMP}-${HOST}_ngpu${NGPUS}_ncpu${NCPUS}.log"
  export LOGFILE=$LOGFILE
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
  EXEC="${MPI_COMMAND} ${MPI_DEFAULTS} ${MPI_ELASTIC} $(which python3) ${MAIN}"

  # export EXEC="$EXEC $@"
}

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Elastic Training:                              ┃
# ┃  Use all available GPUs on all available nodes ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
elasticDistributed() {
  NRANKS=$(wc -l < "${HOSTFILE}")
  NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
  NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
  echo "\
    Running on ${NRANKS} ranks \
    with ${NGPU_PER_RANK} GPUs each \
    for a total of ${NGPUS} GPUs"
  EXEC="\
    ${MPI_COMMAND} \
    ${MPI_DEFAULTS} \
    ${MPI_ELASTIC} \
    $(which python3) \
    ${MAIN}"
  # export EXEC="${EXEC} "$@""
  ${EXEC} "$@"
}


# ┏━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Print Job Information ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━┛
printJobInfo() {
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
  echo '┃  - exec: "${EXEC} $@"'
  echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo '┃ To view output: `tail -f $(tail -1 logs/latest)`'
  echo "┃ Latest logfile: $(tail -1 ./logs/latest)"
  echo "┃ tail -f $(tail -1 logs/latest)"
  echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# main() {
if [[ $(hostname) == theta* ]]; then
  echo "Setting up ThetaGPU from $(hostname)"
  setupThetaGPU
elif [[ $(hostname) == x* ]]; then
  echo "Setting up Polaris from $(hostname)"
  setupPolaris
else
  echo "Unexpected hostname $(hostname)"
fi

setupJob
printJobInfo
${EXEC} "$@" > ${LOGFILE} &
# }


# Run executable command
# ${EXEC} $@ > ${LOGFILE} & #; ret_code=$?

# if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# wait
