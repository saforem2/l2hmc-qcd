#!/bin/bash
#COBALT -A datascience

echo -e "\n"
echo "Starting cobalt job script..."

date
# source /lus/theta-fs0/software/thetagpu/conda/tf_master/2020-12-17/mconda3/setup.sh

# eval "$(lus/theta-fs0/software/thetagpu/conda/tf_master/2020-12-17/mconda3/condabin/conda
# shell.bash hook)"
# source ~/tf_hvd_env.sh
# source ~/conda_zsh_setup.sh

export OMPI_MCA_opal_cuda_support=true
export NCCL_DEBUG=INFO
export KMP_SETTINGS=TRUE
export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
export AUTOGRAPH_VERBOSITY=10
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices"
export TF_ENABLE_AUTO_MIXED_PRECISION=1
export PATH=$PATH:$HOME/.local/bin
export PYTHONPATH=/lus/theta-fs0/software/thetagpu/conda/tf_master/2020-12-17/mconda3/lib/python3.8/site-packages:$PYTHONPATH
echo python3: $(which python3)
#     mpirun -n $PROCS -npernode $PPN --verbose -hostfile $COBALT_NODEFILE \
#         --allow-run-as-root -bind-to none -map-by slot \
#         -x TF_XLA_FLAGS -x LD_LIBRARY_PATH \
#         -x NCCL_DEBUG=INFO -x PATH \
#         -x NCCL_SOCKET_IFNAME=^docker0,lo \
#         python3 $TRAINER --json_file=$JSON_FILE
# '/lus/theta-fs0/projects/DLHMC/thetaGPU/training/2020_11_30/t16x16_b2048_lf10_bi5_bf6_4ranks'
# '/lus/theta-fs0/projects/DLHMC/thetaGPU/training/2020_12_04/t16x16_b2048_lf10_bi4_bf5_4ranks_REP'
# '/lus/theta-fs0/projects/DLHMC/thetaGPU/training/2020_12_05/t16x16_b2048_lf10_bi5_bf6_4ranks_REP'
# '/lus/theta-fs0/projects/DLHMC/thetaGPU/training/2020_12_10'
# '/lus/theta-fs0/projects/DLHMC/thetaGPU/training/2020_12_15/t16x16bi4bf5b2048epsDynamic'
RUN_DIRS=(
    '/lus/theta-fs0/projects/DLHMC/thetaGPU/training/2020_11_29/t16x16_b2048_lf10_bi4_bf5_4ranks' 
    '/lus/theta-fs0/projects/DLHMC/thetaGPU/training/2020_12_02/t16x16_b2048_lf10_bi6_bf7_4ranks' 
    '/lus/theta-fs0/projects/DLHMC/thetaGPU/training/2020_12_02/t16x16_b2048_lf10_bi4_bf5_4ranks_rep' 
    '/lus/theta-fs0/projects/DLHMC/thetaGPU/training/2020_12_16/t16x16b2048bi4bf5xSplit' 
    '/lus/theta-fs0/projects/DLHMC/thetaGPU/training/2020_12_16/t16x16b2048bi5bf6epsDynamic' 
    '/lus/theta-fs0/projects/DLHMC/thetaGPU/training/2020_12_17/t16x16b2048bi4bf5xSplitCombined' 
    '/lus/theta-fs0/projects/DLHMC/thetaGPU/training/2020_12_18/t16x16b2048bi4bf5xSplit_REP'
)

# for d in "${RUN_DIRS[@]}"; do
#     # LOG_DIR=$(tail -n 1 "$d/log_dirs.txt"
#     if [[ -f "$d/log_dirs.txt" ]]; then
#         LOG_DIR=$(tail -n 1 $d/log_dirs.txt)
#         RANK=$(($COUNTER % 7))
#         export CUDA_VISIBLE_DEVICES="$RANK"
#
#         echo LOG_DIR: $LOG_DIR
#         echo RANK: $RANK
#         echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
#         mpirun -np 1 -H localhost:1 \
#             --allow-run-as-root -bind-to none -map-by slot \
#             -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
#             -x LD_LIBRARY_PATH -x PATH \
#             -x NCCL_SOCKET_IFNAME=^docker0,lo \
#             python3 $d/l2hmc-qcd/l2hmc-qcd/inference.py --log_dir=$LOG_DIR \
#                 --run_steps=125000 \
#                 --therm_frac=0.2 \
#                 --num_chains=8 \
#                 --train_steps=1000
#         >> inference_rank$RANK.log 2>&1 &
#         COUNTER=$[$COUNTER + 1]
#         echo COUNTER: $COUNTER
#     fi
#     # echo $(tail -n 1 "$d/log_dirs.txt"
# done

# ------------------------------------
# cd ~/thetaGPU/l2hmc-qcd && git pull && cd -
#
# for d in "${RUN_DIRS[@]}"; do
#     PROJECT_DIR=${d}/l2hmc-qcd/l2hmc-qcd
#     cp ~/thetaGPU/l2hmc-qcd/l2hmc-qcd/{inference.py,config.py} $PROJECT_DIR
#     cp
#     ~/thetaGPU/l2hmc-qcd/l2hmc-qcd/utils/{__init__.py,file_io.py,inference_utils.py,plotting_utils.py,data_utils.py} $PROJECT_DIR/utils/
#     # mv $PROJECT_DIR/{inference.py,config.py} $PROJECT_DIR/orig
#     # mkdir $PROJECT_DIR/utils/orig
#     # mv $PROJECT_DIR/utils/{inference_utils.py,file_io.py,__init__.py,plotting_utils.py,data_utils.py} $PROJECT_DIR/utils/orig
#     # cp ~/thetaGPU/l2hmc-qcd/l2hmc-qcd/{inference.py,config.py} $PROJECT_DIR
#     # cp ~/l2hmc-qcd/l2hmc-qcd/utils/{inference_utils.py,file_io.py,__init__.py,plotting_utils.py,data_utils.py} $PROJECT_DIR/utils/
# done


# LOG_DIR=$(tail -n 1 "${RUN_DIRS[0]}"/log_dirs.txt)
# RANK=0
# export CUDA_VISIBLE_DEVICES="$RANK"
# echo LOG_DIR: $LOG_DIR
# echo RANK: $RANK
# echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
# mpirun -np 1 -H localhost:1 \
#     --allow-run-as-root -bind-to none -map-by slot \
#     -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
#     -x LD_LIBRARY_PATH -x PATH \ -x NCCL_SOCKET_IFNAME=^docker0,lo \
#     python3 $d/l2hmc-qcd/l2hmc-qcd/inference.py \
#         --log_dir=$LOG_DIR \
#         --run_steps=125000 \
#         --therm_frac=0.2 \
#         --num_chains=8 \
#         --train_steps=1000 >> inference_rank$RANK.log 2>&1 &
#
#
#         LOG_DIR=$(tail -n 1 $d/log_dirs.txt)
#         RANK=$(($COUNTER % 7))
#         export CUDA_VISIBLE_DEVICES="$RANK"

# RUN_DIR=$RUN_DIRS[${RANK}]
# LOG_DIR_FILE=${RUN_DIRS[$RANK]}/log_dirs.txt
# PROJECT_DIR=${RUN_DIRS[${RANK}]}/l2hmc-qcd
# RUN_DIR=/lus/theta-fs0/projects/DLHMC/thetaGPU/training/2020_11_29/t16x16_b2048_lf10_bi4_bf5_4ranks
RANK=0
RUN_DIR=${RUN_DIRS[1]}
LOG_DIR_FILE=$RUN_DIR/log_dirs.txt
LOG_DIR=$(tail -n 1 $LOG_DIR_FILE)
PROJECT_DIR=${RUN_DIR}/l2hmc-qcd/l2hmc-qcd
export CUDA_VISIBLE_DEVICES="$RANK"
echo LOG_DIR: $LOG_DIR
echo PROJECT_DIR: $PROJECT_DIR
echo RANK: $RANK
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
mpirun -np 1 -H localhost:1 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo \
    python3 ${PROJECT_DIR}/inference.py \
        --log_dir=$LOG_DIR \
        --run_steps=125000 \
        --therm_frac=0.2 \
        --num_chains=8 \
        --batch_size=128 \
        --train_steps=1000 >> inference_rank$RANK.log 2>&1 &


RANK=2
RUN_DIR=${RUN_DIRS[2]}
LOG_DIR_FILE=$RUN_DIR/log_dirs.txt
LOG_DIR=$(tail -n 1 $LOG_DIR_FILE)
PROJECT_DIR=${RUN_DIR}/l2hmc-qcd/l2hmc-qcd
export CUDA_VISIBLE_DEVICES="$RANK"
echo LOG_DIR: $LOG_DIR
echo PROJECT_DIR: $PROJECT_DIR
echo RANK: $RANK
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
mpirun -np 1 -H localhost:1 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo \
    python3 ${PROJECT_DIR}/inference.py \
        --log_dir=$LOG_DIR \
        --run_steps=125000 \
        --therm_frac=0.2 \
        --num_chains=8 \
        --batch_size=128 \
        --train_steps=1000 >> inference_rank$RANK.log 2>&1 &

RANK=3
RUN_DIR=${RUN_DIRS[3]}
LOG_DIR_FILE=$RUN_DIR/log_dirs.txt
LOG_DIR=$(tail -n 1 $LOG_DIR_FILE)
PROJECT_DIR=${RUN_DIR}/l2hmc-qcd/l2hmc-qcd
export CUDA_VISIBLE_DEVICES="$RANK"
echo LOG_DIR: $LOG_DIR
echo PROJECT_DIR: $PROJECT_DIR
echo RANK: $RANK
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
mpirun -np 1 -H localhost:1 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo \
    python3 ${PROJECT_DIR}/inference.py \
        --log_dir=$LOG_DIR \
        --run_steps=125000 \
        --therm_frac=0.2 \
        --num_chains=8 \
        --batch_size=128 \
        --train_steps=1000 >> inference_rank$RANK.log 2>&1 &

RANK=5
RUN_DIR=${RUN_DIRS[4]}
LOG_DIR_FILE=$RUN_DIR/log_dirs.txt
LOG_DIR=$(tail -n 1 $LOG_DIR_FILE)
PROJECT_DIR=${RUN_DIR}/l2hmc-qcd/l2hmc-qcd
export CUDA_VISIBLE_DEVICES="$RANK"
echo LOG_DIR: $LOG_DIR
echo PROJECT_DIR: $PROJECT_DIR
echo RANK: $RANK
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
mpirun -np 1 -H localhost:1 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo \
    python3 ${PROJECT_DIR}/inference.py \
        --log_dir=$LOG_DIR \
        --run_steps=125000 \
        --therm_frac=0.2 \
        --num_chains=8 \
        --batch_size=128 \
        --train_steps=1000 >> inference_rank$RANK.log 2>&1 &

RANK=6
RUN_DIR=${RUN_DIRS[5]}
LOG_DIR_FILE=$RUN_DIR/log_dirs.txt
LOG_DIR=$(tail -n 1 $LOG_DIR_FILE)
PROJECT_DIR=${RUN_DIR}/l2hmc-qcd/l2hmc-qcd
export CUDA_VISIBLE_DEVICES="$RANK"
echo LOG_DIR: $LOG_DIR
echo PROJECT_DIR: $PROJECT_DIR
echo RANK: $RANK
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
mpirun -np 1 -H localhost:1 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo \
    python3 ${PROJECT_DIR}/inference.py \
        --log_dir=$LOG_DIR \
        --run_steps=125000 \
        --therm_frac=0.2 \
        --num_chains=8 \
        --batch_size=128 \
        --train_steps=1000 >> inference_rank$RANK.log 2>&1 &

RANK=7
RUN_DIR=${RUN_DIRS[6]}
LOG_DIR_FILE=$RUN_DIR/log_dirs.txt
LOG_DIR=$(tail -n 1 $LOG_DIR_FILE)
PROJECT_DIR=${RUN_DIR}/l2hmc-qcd/l2hmc-qcd
export CUDA_VISIBLE_DEVICES="$RANK"
echo LOG_DIR: $LOG_DIR
echo PROJECT_DIR: $PROJECT_DIR
echo RANK: $RANK
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
mpirun -np 1 -H localhost:1 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo \
    python3 ${PROJECT_DIR}/inference.py \
        --log_dir=$LOG_DIR \
        --run_steps=125000 \
        --therm_frac=0.2 \
        --num_chains=8 \
        --batch_size=128 \
        --train_steps=1000 >> inference_rank$RANK.log 2>&1 &


RANK=8
RUN_DIR=${RUN_DIRS[7]}
LOG_DIR_FILE=$RUN_DIR/log_dirs.txt
LOG_DIR=$(tail -n 1 $LOG_DIR_FILE)
PROJECT_DIR=${RUN_DIR}/l2hmc-qcd/l2hmc-qcd
export CUDA_VISIBLE_DEVICES="$RANK"
echo LOG_DIR: $LOG_DIR
echo PROJECT_DIR: $PROJECT_DIR
echo RANK: $RANK
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
mpirun -np 1 -H localhost:1 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo \
    python3 ${PROJECT_DIR}/inference.py \
        --log_dir=$LOG_DIR \
        --run_steps=125000 \
        --therm_frac=0.2 \
        --num_chains=8 \
        --batch_size=128 \
        --train_steps=1000 >> inference_rank$RANK.log 2>&1 &


# RANK=7
# RUN_DIR=${RUN_DIRS[8]}
# LOG_DIR_FILE=$RUN_DIR/log_dirs.txt
# LOG_DIR=$(tail -n 1 $LOG_DIR_FILE)
# PROJECT_DIR=${RUN_DIR}/l2hmc-qcd/l2hmc-qcd
# export CUDA_VISIBLE_DEVICES="$RANK"
# echo LOG_DIR: $LOG_DIR
# echo PROJECT_DIR: $PROJECT_DIR
# echo RANK: $RANK
# echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
# mpirun -np 1 -H localhost:1 \
#     --allow-run-as-root -bind-to none -map-by slot \
#     -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
#     -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo \
#     python3 ${PROJECT_DIR}/inference.py \
#         --log_dir=$LOG_DIR \
#         --run_steps=125000 \
#         --therm_frac=0.2 \
#         --num_chains=8 \
#         --batch_size=128 \
#         --train_steps=1000 >> inference_rank$RANK.log 2>&1 &

# RANK=6
# RANK=6
# RUN_DIR=/lus/theta-fs0/projects/DLHMC/thetaGPU/training/2020_12_18/t16x16b2048bi4bf5xSplit_REP
# LOG_DIR_FILE=$RUN_DIR/log_dirs.txt
# LOG_DIR=$(tail -n 1 $LOG_DIR_FILE)
# PROJECT_DIR=${RUN_DIR}/l2hmc-qcd
# export CUDA_VISIBLE_DEVICES="$RANK"
# echo LOG_DIR: $LOG_DIR
# echo PROJECT_DIR: $PROJECT_DIR
# echo RANK: $RANK
# echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
# mpirun -np 1 -H localhost:1 \
#     --allow-run-as-root -bind-to none -map-by slot \
#     -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
#     -x LD_LIBRARY_PATH -x PATH \ -x NCCL_SOCKET_IFNAME=^docker0,lo \
#     python3 ${PROJECT_DIR}/l2hmc-qcd/inference.py \
#         --log_dir=$LOG_DIR \
#         --run_steps=125000 \
#         --therm_frac=0.2 \
#         --num_chains=8 \
#         --train_steps=1000 >> inference_rank$RANK.log 2>&1 &
#
# RANK=7
# RUN_DIR=/lus/theta-fs0/projects/DLHMC/thetaGPU/training/2020_11_29/t16x16_b2048_lf10_bi4_bf5_4ranks
# LOG_DIR_FILE=$RUN_DIR/log_dirs.txt
# LOG_DIR=$(tail -n 1 $LOG_DIR_FILE)
# PROJECT_DIR=${RUN_DIR}/l2hmc-qcd
# export CUDA_VISIBLE_DEVICES="$RANK"
# echo LOG_DIR: $LOG_DIR
# echo PROJECT_DIR: $PROJECT_DIR
# echo RANK: $RANK
# echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
# mpirun -np 1 -H localhost:1 \
#     --allow-run-as-root -bind-to none -map-by slot \
#     -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
#     -x LD_LIBRARY_PATH -x PATH \ -x NCCL_SOCKET_IFNAME=^docker0,lo \
#     python3 ${PROJECT_DIR}/l2hmc-qcd/inference.py \
#         --log_dir=$LOG_DIR \
#         --run_steps=125000 \
#         --therm_frac=0.2 \
#         --num_chains=8 \
#         --train_steps=1000 >> inference_rank$RANK.log 2>&1 &
#



#
# RANK=7
# RUN_DIR=$RUN_DIRS[${RANK}]
# LOG_DIR_FILE=${RUN_DIRS[$RANK]}/log_dirs.txt
# LOG_DIR=$(tail -n 1 $LOG_DIR_FILE)
# PROJECT_DIR=${RUN_DIR}/l2hmc-qcd
# # PROJECT_DIR=${RUN_DIRS[${RANK}]}/l2hmc-qcd
# export CUDA_VISIBLE_DEVICES="$RANK"
# echo LOG_DIR: $LOG_DIR
# echo PROJECT_DIR: $PROJECT_DIR
# echo RANK: $RANK
# echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
# mpirun -np 1 -H localhost:1 \
#     --allow-run-as-root -bind-to none -map-by slot \
#     -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
#     -x LD_LIBRARY_PATH -x PATH \ -x NCCL_SOCKET_IFNAME=^docker0,lo \
#     python3 ${PROJECT_DIR}/l2hmc-qcd/inference.py \
#         --log_dir=$LOG_DIR \
#         --run_steps=125000 \
#         --therm_frac=0.2 \
#         --num_chains=8 \
#         --train_steps=1000 >> inference_rank$RANK.log 2>&1 &

# for d in "${RUN_DIRS[@]}"; do
#     # LOG_DIR=$(tail -n 1 "$d/log_dirs.txt"
#     if [[ -f "$d/log_dirs.txt" ]]; then
#         LOG_DIR=$(tail -n 1 $d/log_dirs.txt)
#         RANK=$(($COUNTER % 7))
#         export CUDA_VISIBLE_DEVICES="$RANK"
#
#         echo LOG_DIR: $LOG_DIR
#         echo RANK: $RANK
#         echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
#         mpirun -np 1 -H localhost:1 \
#             --allow-run-as-root -bind-to none -map-by slot \
#             -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
#             -x LD_LIBRARY_PATH -x PATH \
#             -x NCCL_SOCKET_IFNAME=^docker0,lo \
#             python3 $d/l2hmc-qcd/l2hmc-qcd/inference.py --log_dir=$LOG_DIR \
#                 --run_steps=125000 \
#                 --therm_frac=0.2 \
#                 --num_chains=8 \
#                 --train_steps=1000
#         >> inference_rank$RANK.log 2>&1 &
#         COUNTER=$[$COUNTER + 1]
#         echo COUNTER: $COUNTER
#     fi
#     # echo $(tail -n 1 "$d/log_dirs.txt"
# done

echo -e "\n"
echo "=================================================================================================================="
exit $status

