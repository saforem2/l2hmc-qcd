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
export PYTHONPATH=/lus/theta-fs0/software/thetagpu/conda/tf_master/2020-12-23/mconda3/lib/python3.8/site-packages:$PYTHONPATH
echo python3: $(which python3)

RANK=0
export CUDA_VISIBLE_DEVICES="$RANK"
echo RANK: $RANK, CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
mpirun -np 1 -H localhost:1 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo \
    python3 ../hmc.py --json_file=../../bin/hmc_configs.json \
        --run_loop --run_steps 125000 > hmc_rank$RANK.log &

RANK=1
export CUDA_VISIBLE_DEVICES="$RANK"
echo RANK: $RANK, CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
mpirun -np 1 -H localhost:1 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo \
    python3 ../hmc.py --json_file=../../bin/hmc_configs.json \
        --run_loop --run_steps 125000 > hmc_rank$RANK.log &

RANK=2
export CUDA_VISIBLE_DEVICES="$RANK"
echo RANK: $RANK, CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
mpirun -np 1 -H localhost:1 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo \
    python3 ../hmc.py --json_file=../../bin/hmc_configs.json \
        --run_loop --run_steps 125000 > hmc_rank$RANK.log &

RANK=3
export CUDA_VISIBLE_DEVICES="$RANK"
echo RANK: $RANK, CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
mpirun -np 1 -H localhost:1 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo \
    python3 ../hmc.py --json_file=../../bin/hmc_configs.json \
        --run_loop --run_steps 125000 > hmc_rank$RANK.log &

RANK=4
export CUDA_VISIBLE_DEVICES="$RANK"
echo RANK: $RANK, CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
mpirun -np 1 -H localhost:1 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo \
    python3 ../hmc.py --json_file=../../bin/hmc_configs.json \
        --run_loop --run_steps 125000 > hmc_rank$RANK.log &

RANK=5
export CUDA_VISIBLE_DEVICES="$RANK"
echo RANK: $RANK, CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
mpirun -np 1 -H localhost:1 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo \
    python3 ../hmc.py --json_file=../../bin/hmc_configs.json \
        --run_loop --run_steps 125000 > hmc_rank$RANK.log &

RANK=6
export CUDA_VISIBLE_DEVICES="$RANK"
echo RANK: $RANK, CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
mpirun -np 1 -H localhost:1 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo \
    python3 ../hmc.py --json_file=../../bin/hmc_configs.json \
        --run_loop --run_steps 125000 > hmc_rank$RANK.log &

RANK=7
export CUDA_VISIBLE_DEVICES="$RANK"
echo RANK: $RANK, CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
mpirun -np 1 -H localhost:1 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x CUDA_VISIBLE_DEVICES -x TF_XLA_FLAGS -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo \
    python3 ../hmc.py --json_file=../../bin/hmc_configs.json \
        --run_loop --run_steps 125000 > hmc_rank$RANK.log &
