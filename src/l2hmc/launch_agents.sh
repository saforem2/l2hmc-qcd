#!/bin/bash

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")

if [[ $(hostname) == theta* ]]; then
  HOSTFILE=${COBALT_NODEFILE}

elif [[ $(hostname) == x* ]]; then
  HOSTFILE=${PBS_NODEFILE}
else
  echo "HOSTNAME: $(hostname)"
fi

RANKS=$(cat $HOSTFILE)
NGPU_PER_RANK=$(nvidia-smi -L | wc -l)

SWEEP_ID=$@
export SWEEPDIR="logs/sweeps/sweep-${SWEEP_ID}"
mkdir -p ${SWEEPDIR}

for rank in $RANKS; do
  OLD_HOST=$(hostname)
  ssh $rank
  echo "$(hostname) (from ${OLD_HOST})"
  for ((i=1;i<=$NGPU_PER_RANK;i++)); do
    TSTAMP=$(date "+%Y-%m-%d-%H%M%S-%s")
    LOGFILE="${SWEEPDIR}/${HOST}:gpu-${i}-${TSTAMP}.log"
    echo "Launching agent on ${HOST}:GPU-${i}"
    echo "Writing logs to: ${LOGFILE}"
    CUDA_VISIBLE_DEVICES=$i wandb agent "l2hmc-qcd/l2hmc-qcd/${SWEEP_ID}" > ${LOGFILE} 2>&1 &
  done
done
