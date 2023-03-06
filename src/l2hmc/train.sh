#!/bin/bash -l
# ----------------------------------------------------------------------------
#PBS -k doe
#PBS -V exports all the environment variables in your environnment to the 
#compute node The rest is an example of how an MPI job might be set up
# echo Working directory is $PBS_O_WORKDIR
# cd $PBS_O_WORKDIR
# ----------------------------------------------------------------------------

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
# source ./setup.sh with helper functions for getting setup @ ALCF
source "${DIR}/setup.sh"

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
