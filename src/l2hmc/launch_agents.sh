#!/bin/bash

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")

if [[ $(hostname) == theta* ]]; then
  NRANKS=$(wc -l < ${COBALT_NODEFILE})
  HOSTFILE=${COBALT_NODEFILE}
  NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
  NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
  MPI_COMMAND="mpirun -np $NGPUS -npernode $NGPU_PER_RANK ./set_affinity_gpu_polaris.sh"
  echo "-----------------------"
  echo "| Running on ThetaGPU |"
  echo "-----------------------"
  echo "HOSTNAME: $(hostname)"
  # Load conda module and activate base environment
  eval "$(/lus/theta-fs0/software/thetagpu/conda/2022-07-01/mconda3/bin/conda shell.zsh hook)"
  module load conda/2022-07-01
  conda activate base
elif [[ $(hostname) == x* ]]; then
  NRANKS=$(wc -l < ${PBS_NODEFILE})
  HOSTFILE=${PBS_NODEFILE}
  NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
  NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
  MPI_COMMAND="mpiexec -n $NGPUS --ppn $NGPU_PER_RANK -d 16 --envall ./set_affinity_gpu_polaris.sh"
  # MPI_COMMAND=$(which mpiexec)
  # MPI_FLAGS="-n ${NGPUS} --ppn ${NGPU_PER_RANK} --envall --hostfile ${HOSTFILE}"
  echo "-----------------------"
  echo "| Running on Polaris |"
  echo "-----------------------"
  echo "HOSTNAME: $(hostname)"
  module load conda/2022-07-19
  conda activate base
else
  echo "HOSTNAME: $(hostname)"
fi

# RANKS=$(cat $HOSTFILE)
# NGPU_PER_RANK=$(nvidia-smi -L | wc -l)

# EXEC="$L2HMC_DIR/launch_on_rank.sh"

SWEEP_ID=$@
echo SWEEP ID: ${SWEEP_ID}

if (( ${NGPUS} > 1 )); then
  ${MPI_COMMAND} wandb agent "l2hmc-qcd/l2hmc-qcd/${SWEEP_ID}"
else
  python3 wandb agent ${SWEEPID}
fi
