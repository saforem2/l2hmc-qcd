RUNNER_TF=/Users/saforem2/ANL/l2hmc-qcd/l2hmc-qcd/gauge_inference.py
RUNNER_NP=/Users/saforem2/ANL/l2hmc-qcd/l2hmc-qcd/gauge_inference_np.py
# LOG_DIR=/Users/saforem2/ANL/l2hmc-qcd/gauge_logs/2020_01_29/L8_b64_lf1_qw0_f32_0337

###################
#      NUMPY
##################@
python3 ${RUNNER_NP} \
    --run_steps 50000 \
    --batch_size 1 \
    -xsw 1 \
    -xtw 1 \
    -xqw 1 \
    -vsw 1 \
    -vtw 1 \
    -vqw 1

python3 ${RUNNER_NP} \
    --run_steps 50000 \
    --batch_size 1 \
    -xsw 0 \
    -xtw 0 \
    -xqw 0 \
    -vsw 0 \
    -vtw 0 \
    -vqw 0

#######################
#      TENSORFLOW
##################@####
python3 ${RUNNER_TF} \
    --run_steps 10000 \
    --x_scale_weight 1 \
    --x_translation_weight 1 \
    --x_transformation_weight 1 \
    --v_scale_weight 1 \
    --v_translation_weight 1 \
    --v_transformation_weight 1

python3 ${RUNNER_TF} \
    --run_steps 10000 \
    --x_scale_weight 0 \
    --x_translation_weight 0 \
    --x_transformation_weight 0 \
    --v_scale_weight 0 \
    --v_translation_weight 0 \
    --v_transformation_weight 0
