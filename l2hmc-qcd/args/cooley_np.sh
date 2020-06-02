#!/bin/bash

# '/Users/saforem2/l2hmc-qcd/gauge_logs/cooley_logs/2020_05_13/L16_b2048_lf2_GaugeNetwork_aw00_qw01_pw01_dp025_f32_1747_1'
# '/Users/saforem2/l2hmc-qcd/gauge_logs/cooley_logs/2020_05_14/L16_b2048_lf2_GaugeNetwork_aw00_qw01_pw01_dp05_f32_2213_1'
# '/Users/saforem2/l2hmc-qcd/gauge_logs/cooley_logs/2020_05_06/L16_b1024_lf2_GaugeNetwork_aw00_qw01_dp025_f32_2305_1'
# '/Users/saforem2/DLHMC/l2hmc-qcd/gauge_logs/2020_05_20/L16_b1024_lf5_GaugeNetwork_aw00_qw01_dp05_eps_fixed_f32_2138_1'

LOG_DIRS=(
    '/Users/saforem2/l2hmc-qcd/gauge_logs/cooley_logs/2020_05_30/L16_b1024_lf2_GaugeNetwork_aw00_qw01_pw01_dp05_eps_fixed_f32_1629_1'
)

BETAS=(
    4.
    4.5
    5.
)

PARAMS="--hmc_start --run_steps 10000 --init 'rand' --batch_size 32 --print_steps 10"
NET_WEIGHTS="-xsw 1. -xtw 1. -xqw 1. -vsw 1. -vtw 1. -vqw 1."

RUNNER_NP='../gauge_inference_np.py'

# --log_dir=${logdir} \
# --hmc_start \
# --run_steps 10000 \
# --init 'rand' \
# --beta ${beta} \
# --batch_size 32 \
# --print_steps 10 \
# -xsw 1. -xtw 1. -xqw 1. \
# -vsw 1. -vtw 1. -vqw 1.

for logdir in ${LOG_DIRS[*]}
do
    for beta in ${BETAS[*]}
    do
        python3 ${RUNNER_NP} \
            --log_dir=${logdir} \
            --hmc_start \
            --run_steps 10000 \
            --init 'rand' \
            --beta ${beta} \
            --batch_size 32 \
            --print_steps 10 \
            -xsw 1. -xtw 1. -xqw 1. \
            -vsw 1. -vtw 1. -vqw 1.
    done
dong
