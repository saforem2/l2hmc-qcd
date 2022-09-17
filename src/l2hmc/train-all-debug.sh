#!/bin/bash

LOGDIR="./logs/debug"
mkdir -p "${LOGDIR}"

START=$(date +%s)

./train.sh mode=debug framework=pytorch backend='DDP' > "${LOGDIR}/${START}/train-pt-ddp.log"
./train.sh mode=debug framework=pytorch backend='horovod' init_wandb=false init_aim=false > "${LOGDIR}/${START}/train-pt-hvd.log"
./train.sh mode=debug framework=tensorflow backend='horovod' init_wandb=false init_aim=false > "${LOGDIR}/${START}/train-tf-hvd.log"
