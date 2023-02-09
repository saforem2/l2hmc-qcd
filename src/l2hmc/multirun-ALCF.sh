#!/bin/bash -l

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
MAIN="./main.py"

echo "Job started at: ${TSTAMP} on $(hostname)"
echo "Job running in: ${DIR}"

export WANDB_CACHE_DIR="../../.cache/wandb"

DEFAULTS="\
  mode=debug \
  conv=none \
  restore=false \
  save=false \
  seed=1234"

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
      source "${VENV_DIR}/bin/activate"
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

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Use 1 GPU on a single node ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
singleDevice() {
  echo "Running on 1 GPU of ${MACHINE}"
  EXEC="$(which python3) ${MAIN} ${DEFAULTS}"
  # export EXEC="${EXEC} "$@""
  CUDA_VISIBLE_DEVICES=0 ${EXEC} "$@"
}

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Use 2 GPUs on a single node ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
twoDevices() {
  echo "Running on 2 GPUs of ${MACHINE}"
  EXEC="\
    ${MPI_COMMAND} \
    ${MPI_DEFAULTS} \
    -n 2 \
    $(which python3) \
    ${MAIN} \
    ${DEFAULTS}"
  # export EXEC="${EXEC} "$@""
  CUDA_VISIBLE_DEVICES=0,1 ${EXEC} "$@"
}


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Use 4 GPUs on a single node ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
fourDevices() {
  echo "Running on 4 GPUs"
  EXEC="\
    ${MPI_COMMAND} \
    ${MPI_DEFAULTS} \
    -n 4 \
    $(which python3) \
    ${MAIN} \
    ${DEFAULTS}"
  # export EXEC="${EXEC} "$@""
  CUDA_VISIBLE_DEVICES=0,1,2,3 ${EXEC} "$@"
}


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Use all available GPUs on a single  node ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
fullNode() {
  NRANKS=1
  NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
  NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
  echo "Running on 1 rank(s) \
    with ${NGPU_PER_RANK} GPU(s) each \
    for a total of ${NGPUS} GPUs"
  EXEC="\
    ${MPI_COMMAND} \
    ${MPI_DEFAULTS} \
    -n ${NGPUS} \
    $(which python3) \
    ${MAIN} \
    ${DEFAULTS}"
  # export EXEC="${EXEC} "$@""
  ${EXEC} "$@"
}


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Use all available GPUs on all available nodes ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
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
    ${MAIN} \
    ${DEFAULTS}"
  # export EXEC="${EXEC} "$@""
  ${EXEC} "$@"
}

# ┏━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ TensorFlow + Horovod ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━┛
runTensorFlow() {
  TF_ARGS="framework=tensorflow backend=horovod precision=float32"
  singleDevice ${TF_ARGS}
  twoDevices ${TF_ARGS}
  fourDevices ${TF_ARGS}
  fullNode ${TF_ARGS}
  elasticDistributed ${TF_ARGS}
}

# ┏━━━━━━━━━━━━━━━━┓
# ┃ PyTorch        ┃
# ┃   + DDP        ┃
# ┃   + Horovod    ┃
# ┃   + DeepSpeed  ┃
# ┗━━━━━━━━━━━━━━━━┛
runPyTorch() {
  BACKENDS=("DDP" "deepspeed" "horovod")
  PRECISIONS=("fp16" "float32")
  for BE in "${BACKENDS[@]}"; do
    for PREC in "${PRECISIONS[@]}"; do
      PT_ARGS="framework=pytorch backend=${BE} precision=${PREC}"
      singleDevice ${PT_ARGS}
      twoDevices ${PT_ARGS}
      fourDevices ${PT_ARGS}
      fullNode ${PT_ARGS}
      elasticDistributed ${PT_ARGS}
    done
  done
}


if [[ $(hostname) == theta* ]]; then
  setupThetaGPU
elif [[ $(hostname) == x* ]]; then
  setupPolaris
else
  echo "Unexpected hostname $(hostname)"
fi

echo "Running TensorFlow on $(hostname)"
runTensorFlow

echo "Running PyTorch on $(hostname)"
runPyTorch
