#!/bin/bash

RUNNER='../gauge_inference.py'

export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=8
export KMP_SETTINGS=TRUE

ARGS='--run_steps 100 --plot_chains 5 --print_steps 50'
HMC_ARGS='--run_steps 5000 --plot_chains 10 --print_steps 10 --hmc'
LOG_DIR='../../gauge_logs/gce_logs/2020_05_14/L16_b1024_lf2_GaugeNetwork_aw00_qw01_pw01_dp05_f32_0203/'

betas=(
    5.
    5.5
    6.
    6.5
    7.
)
for beta in ${betas[*]}
do
    python3 ${RUNNER} --beta $beta --log_dir ${LOG_DIR} ${ARGS}
    # python3 ${RUNNER} --beta $beta --log_dir ${LOG_DIR} ${HMC_ARGS}
    # python3 ${RUNNER} --beta $beta --log_dir ${LOG_DIR} ${HMC_ARGS} --eps 0.1
    # python3 ${RUNNER} --beta $beta --log_dir ${LOG_DIR} ${HMC_ARGS} --eps 0.2
done

# python3 ${RUNNER} --beta 5. --log_dir ${LOG_DIR} ${ARGS}
# python3 ${RUNNER} --beta 5. --log_dir ${LOG_DIR} ${HMC_ARGS}
# python3 ${RUNNER} --beta 5. --log_dir ${LOG_DIR} --eps 0.1 ${HMC_ARGS}
# python3 ${RUNNER} --beta 5. --log_dir ${LOG_DIR} --eps 0.2 ${HMC_ARGS}


# ARGS = "--batch_size 1 \
#         --print_steps 10 \
#         --run_steps 50000"


#python3 ${RUNNER_NP} ${P} ${R} --batch_size 16 --beta 4.
# python3 ${RUNNER_NP} ${P} ${R} ${M} ${NW} --beta 5. --mix_samplers
# python3 ${RUNNER_NP} ${ARGS} ${NW} --mix_samplers --beta 5.
# python3 ${RUNNER_NP} ${P} ${R} ${M} --beta 6. --mix_samplers
# python3 ${RUNNER_NP} ${P} ${R} ${M} --beta 7. --mix_samplers

#python3 ${RUNNER_NP} ${P} ${R} --batch_size 16 --beta 5. --mix_samplers
