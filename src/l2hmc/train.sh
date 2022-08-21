#!/bin/bash
#COBALT -n 1
#COBALT -q single-gpu
#COBALT -A DLHMC
#COBALT --attrs filesystems=home,theta-fs0,grand,eagle
# -------------------------------------------------------
# UG Section 2.5, page UG-24 Job Submission Options
# Add another # at the beginning of the line to comment out a line
# NOTE: adding a switch to the command line will override values in this file.

# These options are MANDATORY at ALCF; Your qsub will fail if you don't provide them.
#PBS -A <short project name>
#PBS -l walltime=HH:MM:SS

# Highly recommended 
# The first 15 characters of the job name are displayed in the qstat output:
#PBS -N <name>

# If you need a queue other than the default (uncomment to use)
##PBS -q <queue name>
# Controlling the output of your application
# UG Sec 3.3 page UG-40 Managing Output and Error Files
# By default, PBS spools your output on the compute node and then uses scp to move it the
# destination directory after the job finishes.  Since we have globally mounted file systems
# it is highly recommended that you use the -k option to write directly to the destination
# the doe stands for direct, output, error
#PBS -o <path for stdout>
#PBS -k doe
#PBS -e <path for stderr>
# Setting job dependencies
# UG Section 6.2, page UG-107 Using Job Dependencies
# There are many options for how to set up dependancies;  afterok will give behavior similar
# to Cobalt (uncomment to use)
##PBS depend=afterok:<jobid>:<jobid>

# Environment variables (uncomment to use)
# Section 6.12, page UG-126 Using Environment Variables
# Sect 2.59.7, page RG-231 Enviornment variables PBS puts in the job environment
##PBS -v <variable list>
## -v a=10, "var2='A,B'", c=20, HOME=/home/zzz
##PBS -V exports all the environment variables in your environnment to the compute node
# The rest is an example of how an MPI job might be set up
# echo Working directory is $PBS_O_WORKDIR
# cd $PBS_O_WORKDIR

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
HOST=$(hostname)
echo "Job ID: ${PBS_JOBID}"
echo "Job started at: ${TSTAMP}"

# ---- Specify directories and executable for experiment ------------------
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
echo "DIR:$DIR"
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
  # cd /lus/grand/projects/DLHMC/foremans/l2hmc-qcd/src/l2hmc/
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
  # cd /lus/grand/projects/datascience/foremans/polaris/projects/l2hmc-qcd/src/l2hmc
# ---- Check if running on MacOS --------------------------------
else
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

python3 -m pip install --upgrade aim
python3 -m pip install --pre --upgrade wandb

# ---- Environment settings -----------------------------------------------
export NCCL_DEBUG=INFO
export KMP_SETTINGS=TRUE
export OMP_NUM_THREADS=16
# export OMPI_MCA_opal_cuda_support=TRUE
# export TF_ENABLE_AUTO_MIXED_PRECISION=1
# export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"
# export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
# export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices"

EXEC="${MPI_COMMAND} ${MPI_FLAGS} $(which python3) ${MAIN}"


# ---- Print job information -------------------------------------------------
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  STARTING A NEW RUN ON ${NGPU} GPUs ${NCPUS}   ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
echo -e '\n'
printf '%.s─' $(seq 1 $(tput cols))
echo "┃  - DATE: ${TSTAMP}"
echo "┃  - NCPUS: ${NCPUS}"
echo "┃  - NRANKS: ${NRANKS}"
echo "┃  - NGPUS PER RANK: ${NGPU_PER_RANK}"
echo "┃  - NGPUS TOTAL: ${NGPUS}"
echo "┃  - MAIN: ${MAIN}"
echo "┃  - Writing logs to ${LOGFILE}"
echo "┃  - python3: $(which python3)"
echo "┃  - mpirun: ${MPI_COMMAND}"
echo "┃  - l2hmc: $(python3 -c 'import l2hmc; print(l2hmc.__file__)')"
echo "┃  - exec: ${EXEC} $@"
printf '%.s─' $(seq 1 $(tput cols))
echo -e '\n'

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
