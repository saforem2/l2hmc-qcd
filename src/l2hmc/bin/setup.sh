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
#
TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
HOST=$(hostname)

# Resolve path to current file
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
PARENT=$(dirname "${DIR}")
MAIN="${PARENT}/__main__.py"
GRANDPARENT=$(dirname "${PARENT}")
ROOT=$(dirname "${GRANDPARENT}")
MAIN="$PARENT/__main__.py"
SETUP_SCRIPT="$DIR/setup.sh"
TRAIN_SCRIPT="$DIR/train.sh"
NCPUS=$(getconf _NPROCESSORS_ONLN)


function join_by {
    local d=${1-} f=${2-};
    if shift 2; then
        printf %s "$f" "${@/#/$d}"
    fi
}

function pprint() {
    OUTPUT="$*"
    echo "┌─────────────────────────────────────────────────────────────────────┐"
    echo "│ [setup.sh]:  ${OUTPUT[*]}"
    echo "└─────────────────────────────────────────────────────────────────────┘"
}
function loadCondaEnv() {
    if [[ "${CONDA_EXE}" ]]; then
        echo "Already inside ${CONDA_EXE}, exiting!"
    else
        MODULE_STR="$1"
        module load "conda/${MODULE_STR}"
        conda activate base
    fi
}

function setupVenv() {
    VENV_DIR="$1"
    if [[ -d "${VENV_DIR}" ]]; then
        echo "Found venv at: ${VENV_DIR}"
        source "${VENV_DIR}/bin/activate"
    else
        echo "Skipping setupVenv() on $(hostname)"
    fi
}

function setupPython() {
    local conda_date=$1
    local venv_path=$2
    if [[ "${CONDA_EXE}" ]]; then
        echo "Caught CONDA_EXE: ${CONDA_EXE}"
    else
        loadCondaEnv "${conda_date}"
    fi
    if [[ "${VIRTUAL_ENV}" ]]; then
        echo "Caught VIRTUAL_ENV: ${VIRTUAL_ENV}"
    else
        setupVenv "${venv_path}"
    fi
}

# ┏━━━━━━━━━━┓
# ┃ ThetaGPU ┃
# ┗━━━━━━━━━━┛
function setupThetaGPU() {
    if [[ $(hostname) == theta* ]]; then
        export MACHINE="ThetaGPU"
        export NVME_PATH="/raid/scratch/"
        HOSTFILE=${HOSTFILE:-${COBALT_NODEFILE}}
        CONDA_DATE="2023-01-11"
        VENV_DIR="${ROOT}/venvs/${MACHINE}/${CONDA_DATE}"
        setupPython "${CONDA_DATE}" "${VENV_DIR}"
        NHOSTS=$(wc -l < "${HOSTFILE}")
        NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
        NGPUS=$((NHOSTS * NGPU_PER_HOST))
        LAUNCH="mpirun -n $NGPUS -N $NGPU_PER_HOST --hostfile $HOSTFILE -x PATH -x LD_LIBRARY_PATH"
    else
        echo "[setupThetaGPU]: Unexpected hostname $(hostname)"
    fi
}

# ┏━━━━━━━━━━┓
# ┃ ThetaGPU ┃
# ┗━━━━━━━━━━┛
function setupPolaris() {
    if [[ $(hostname) == x3* ]]; then
        export MACHINE="polaris"
        export NVME_PATH="/raid/scratch/"
        HOSTFILE=${HOSTFILE:-${PBS_NODEFILE}}
        CONDA_DATE="2023-10-04"
        VENV_DIR="${ROOT}/venvs/${MACHINE}/${CONDA_DATE}"
        setupPython "${CONDA_DATE}" "${VENV_DIR}"
        NHOSTS=$(wc -l < "${HOSTFILE}")
        NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
        NGPUS=$((NHOSTS * NGPU_PER_HOST))
        LAUNCH="mpiexec --verbose --envall -n $NGPUS -ppn $NGPU_PER_HOST --hostfile ${HOSTFILE}"
        # alias mpilaunch="$(which mpiexec) --verbose --envall -n $NGPUS -ppn $NGPU_PER_HOST --hostfile ${HOSTFILE}"
        # alias mpilaunch="${LAUNCH}"
    else
        echo "[setupPolaris]: Unexpected hostname $(hostname)"
    fi
}

# ┏━━━━━━━┓
# ┃ NERSC ┃
# ┗━━━━━━━┛
function setupPerlmutter() {
    if [[ $(hostname) == login* || $(hostname) == nid* ]]; then
        export MACHINE="Perlmutter"
        SLURM_NODES=$(scontrol show hostname "${SLURM_NODELIST}")
        SLURM_NODEFILE="${HOME}/.slurm-nodefile"
        printf "%s\n" "${SLURM_NODES[@]}" > "${SLURM_NODEFILE}"
        export HOSTFILE="${HOSTFILE:-${SLURM_NODEFILE}}"
        [ "$SLURM_JOB_ID" ] \
            && echo "Caught SLURM_JOB_ID: ${SLURM_JOB_ID}" \
            || echo "!!!!!! Running without SLURM allocation !!!!!!!!"
        module load libfabric cudatoolkit pytorch/2.0.1
        export NODELIST="${SLURM_NODELIST:-$(hostname)}"
        export NHOSTS="${SLURM_NNODES:-1}"
        export NGPU_PER_HOST="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
        export NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
        LAUNCH="srun -N ${NHOSTS} -n ${NGPUS} -l u"
    else
        echo "[setupPerlmutter]: Unexpected hostname $(hostname)"
    fi
}

function setupLogs() {
    LOGDIR="${PARENT}/logs"
    LOGFILE="${LOGDIR}/${TSTAMP}-${HOST}_ngpu${NGPUS}_ncpu${NCPUS}.log"
    export LOGDIR="${LOGDIR}"
    export LOGFILE=$LOGFILE
    if [ ! -d "${LOGDIR}" ]; then
        mkdir -p "${LOGDIR}"
    fi
    echo "Writing to logfile: ${LOGFILE}"
    # Keep track of latest logfile for easy access
    echo "$LOGFILE" >> "${LOGDIR}/latest"
}

function printJobInfo() {
    ARGS=$*
    HOSTS_ARR=$(cat "${HOSTFILE:-$(hostname)}")
    # if [[ "${HOSTFILE}" ]]; then
    #     HOSTS_ARR=$(cat "${HOSTFILE}")
    #     HOSTS=$(join_by ' ' "${HOSTS_ARR}")
    # elif [[ "${SLURM_NODELIST}" ]]; then
    #     HOSTS="${SLURM_NODELIST}"
    # else
    #     echo "[printJobInfo][WARNING]: HOSTFILE not set, using resources on localhost!"
    #     HOSTS=$(hostname)
    # fi
    echo "┌─────────────────────────────────────────────────────────────────────"  #┐"
    echo "│ [setup.sh]: Job started at: ${TSTAMP} on ${HOST} by ${USER}"
    echo "│ [setup.sh]: Job running in: ${DIR}"
    echo "└─────────────────────────────────────────────────────────────────────"  #┘"
    echo "┌─────────────────────────────────────────────────────────────────────"  #┐"
    echo "│ [setup.sh]: DIR=${DIR}"
    echo "│ [setup.sh]: MAIN=${MAIN}"
    echo "│ [setup.sh]: SETUP_SCRIPT=${SETUP_SCRIPT}"
    echo "│ [setup.sh]: TRAIN_SCRIPT=${TRAIN_SCRIPT}"
    echo "│ [setup.sh]: PARENT=${PARENT}"
    echo "│ [setup.sh]: ROOT=${ROOT}"
    echo "│ [setup.sh]: LOGDIR=${LOGDIR}"
    echo "│ [setup.sh]: LOGFILE=${LOGFILE}"
    echo "└─────────────────────────────────────────────────────────────────────"  #┘"
    # echo "┌─────────────────────────────────────────────────────────────────────"  # ┐"
    # echo "│ [setup.sh]: [Hosts][${HOSTFILE}]: "
    # echo "│ [setup.sh]:   ${HOSTS[*]}"
    # echo "└──────────────────────────────────────────────────────────────────┘"
    echo "┌─────────────────────────────────────────────────────────────────────"  # ┐"
    echo "│ [setup.sh]: HOSTS: "
    echo "│   $(join_by ', ' $HOSTS_ARR)"
    echo "│ [setup.sh]: Using ${NHOSTS} hosts from ${HOSTFILE}"
    echo "│ [setup.sh]: With ${NGPU_PER_HOST} GPUs per host"
    echo "│ [setup.sh]: For a total of: ${NGPUS} GPUs"
    echo "└─────────────────────────────────────────────────────────────────────"  #┘"
    echo "┌─────────────────────────────────────────────────────────────────────"  #┐"
    echo "│ [setup.sh]: Using python: $(which python3)"
    echo "│ [setup.sh]: ARGS: ${ARGS[*]}"
    echo "│ [setup.sh]: LAUNCH: ${LAUNCH} python3 ${MAIN} ${ARGS[*]}"
    echo "└─────────────────────────────────────────────────────────────────────"  #┘"
    echo "┌─────────────────────────────────────────────────────────────────────"  #┐"
    echo "│ [setup.sh]: Writing logs to ${LOGFILE}"
    echo '│ [setup.sh]: To view output: `tail -f $(tail -1 logs/latest)`'  # noqa
    echo "│ [setup.sh]: Latest logfile: $(tail -1 ./logs/latest)"
    echo "│ [setup.sh]: tail -f $(tail -1 logs/latest)"
    echo "└─────────────────────────────────────────────────────────────────────"  #┘"
    echo -e "\n"
}

function setupJob() {
    if [[ $(hostname) == x3* ]]; then
        setupPolaris
    elif [[ $(hostname) == thetagpu* ]]; then
        setupThetaGPU
    elif [[ $(hostname) == nid* || $(hostname) == login* ]]; then
        setupPelmutter
        HOSTS="${SLURM_NODELIST}"
    else
        echo "[setupJob]: Unexpected hostname $(hostname)"
        exit 1
    fi
    export NHOSTS
    export NGPU_PER_HOST
    export NGPUS
    export HOSTFILE
    export LAUNCH
    setupLogs
}
