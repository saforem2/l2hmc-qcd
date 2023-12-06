#!/bin/bash -l
# ----------------------------------------------------------------------------
#PBS -k doe
#PBS -V exports all the environment variables in your environnment to the
#compute node The rest is an example of how an MPI job might be set up
# echo Working directory is $PBS_O_WORKDIR
# cd $PBS_O_WORKDIR
# ----------------------------------------------------------------------------
#

function resolveDir() {
    # Resolve path to current file
    SOURCE=${BASH_SOURCE[0]}
    while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
        DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
        SOURCE=$(readlink "$SOURCE")
        [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
    done
    DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
    PARENT=$(dirname "${DIR}")
    MAIN="$DIR/__main__.py"
    SETUP_SCRIPT="$DIR/bin/setup.sh"
    TRAIN_SCRIPT="$DIR/bin/train.sh"
    ROOT=$(dirname "${PARENT}")
}


function sourceFile() {
    fpath="${1:-${DIR}/setup.sh}"
    # SETUP_FILE="${DIR}/setup.sh"
    if [[ -f "${fpath}" ]]; then
        echo "source-ing ${fpath}"
        # shellcheck source=./setup.sh
        source "${fpath}"
    else
        echo "ERROR: UNABLE TO SOURCE ${fpath}"
    fi
}

#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ Make sure we're not already running; if so, exit here ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
function killIfRunning() {
    PIDS=$(ps aux | grep -E "$USER.+mpi.+.+python3.+__main__.+py" | grep -v grep | awk '{print $2}')
    if [ -n "${PIDS}" ]; then
        echo "Already running! Exiting!"
        exit 1
    fi
}


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ SETUP CONDA + MPI ENVIRONMENT @ ALCF ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
function setup() {
    local args="$*"
    resolveDir
    killIfRunning
    sourceFile "${DIR}/setup.sh"
    setupJob
    EXEC="${LAUNCH} python3 ${MAIN} ${args}"  # | tee -a ${LOGFILE}"
    export EXEC
    pprint "EXEC: ${EXEC}"
    printJobInfo "$@" | tee -a "${LOGFILE}"
    # export NODE_RANK=0
    export NNODES=$NHOSTS
    export GPUS_PER_NODE=$NGPU_PER_HOST
    export WORLD_SIZE=$NGPUS
}

setup "$@"
# ${EXEC} | tee -a "${LOGFILE}"
${EXEC} >> "${LOGFILE}" 2>&1 &
# wait $!
