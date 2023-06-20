#!/bin/bash -l
# ----------------------------------------------------------------------------
#PBS -k doe
#PBS -V exports all the environment variables in your environnment to the
#compute node The rest is an example of how an MPI job might be set up
# echo Working directory is $PBS_O_WORKDIR
# cd $PBS_O_WORKDIR
# ----------------------------------------------------------------------------

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
HOST=$(hostname)
echo "Job started at: ${TSTAMP} on ${HOST}"
echo "Job running in: ${DIR}"

# HERE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
# DIR=$(dirname "$HERE")
#
resolveDir() {
  SOURCE=${BASH_SOURCE[0]}
  while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
    DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
    SOURCE=$(readlink "$SOURCE")
    [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
  done
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  HERE="${DIR}/train.sh"
  # SETUP="${DIR}/setup.sh"
  # source "$SETUP"
  PARENT=$(dirname "${DIR}")
  MAIN="${PARENT}/main.py"
  GRANDPARENT=$(dirname "${PARENT}")
  ROOT=$(dirname "${GRANDPARENT}")
  echo "DIR: ${DIR}"
  echo "HERE: ${HERE}"
  echo "PARENT: ${PARENT}"
  echo "GRANDPARENT: ${GRANDPARENT}"
  echo "ROOT: ${ROOT}"
}

#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ Make sure we're not already running; if so, exit here ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
killIfRunning() {
  PIDS=$(ps aux | grep -E 'mpi.+main.+py' | grep -v grep | awk '{print $2}')
  if [ -n "${PIDS}" ]; then
    echo "Already running! Exiting!"
    exit 1
  fi
}


setupLogs() {
  LOGDIR="${PARENT}/logs"
  LOGFILE="${LOGDIR}/${TSTAMP}-${HOST}_ngpu${NGPUS}_ncpu${NCPUS}.log"
  export LOGDIR="${LOGDIR}"
  export LOGFILE=$LOGFILE
  if [ ! -d "${LOGDIR}" ]; then
    mkdir -p "${LOGDIR}"
  fi
  # Keep track of latest logfile for easy access
  echo "$LOGFILE" >> "${LOGDIR}/latest"
}


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Elastic Training:                              ┃
# ┃  Use all available GPUs on all available nodes ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# elasticDistributed() {
#   NRANKS=$(wc -l < "${HOSTFILE}")
#   NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
#   NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
#   echo "\
#     Running on ${NRANKS} ranks \
#     with ${NGPU_PER_RANK} GPUs each \
#     for a total of ${NGPUS} GPUs"
#   EXEC="\
#     ${MPI_COMMAND} \
#     ${MPI_DEFAULTS} \
#     ${MPI_ELASTIC} \
#     $(which python3) \
#     ${MAIN}"
#   export EXEC="${EXEC}"
#   ${EXEC} "$@"
# }


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
  echo "┃  - exec: ${EXEC}" && echo "ARGS: " && echo "$@"
  echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo '┃ To view output: `tail -f $(tail -1 logs/latest)`'  # noqa
  echo "┃ Latest logfile: $(tail -1 ./logs/latest)"
  echo "┃ tail -f $(tail -1 logs/latest)"
  echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

setupEnv() {
  # source ./setup.sh with helper functions for getting setup @ ALCF
  SETUP_FILE="${DIR}/setup.sh"
  if [[ -f "${SETUP_FILE}" ]]; then
    echo "source-ing ${SETUP_FILE}"
    # shellcheck source=./setup.sh
    source "${SETUP_FILE}"
  else
    echo "ERROR: UNABLE TO SOURCE ${SETUP_FILE}"
  fi
  if [[ $(hostname) == theta* ]]; then
    echo "Setting up ThetaGPU from $(hostname)"
    setupThetaGPU
  elif [[ $(hostname) == x* ]]; then
    echo "Setting up Polaris from $(hostname)"
    setupPolaris
  else
    echo "Unexpected hostname $(hostname)"
  fi
}

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ SETUP CONDA + MPI ENVIRONMENT @ ALCF ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
setup() {
  resolveDir
  killIfRunning
  setupEnv
  setupJob
  setupLogs
  export NODE_RANK=0
  export NNODES=$NRANKS
  export GPUS_PER_NODE=$NGPU_PER_RANK
  export WORLD_SIZE=$NGPUS
  # export LC_ALL=$(locale -a | grep UTF-8)
  printJobInfo "$@" | tee -a "${LOGFILE}"
}

setup "$@"
${EXEC} "$@" > "${LOGFILE}" 2>&1 &

# wait ;
