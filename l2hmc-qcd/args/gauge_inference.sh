RUNNER_TF='../gauge_inference.py'
RUNNER_NP='../gauge_inference_np.py'

# LOG_DIR='../../gauge_logs/2020_01_28/L8_b64_lf1_qw0_f32_1759'

###################
#      NUMPY
##################@
python3 ${RUNNER_NP} \
    --run_steps 25000 \
    --batch_size 1 \
    -xsw 1 \
    -xtw 1 \
    -xqw 1 \
    -vsw 1 \
    -vtw 1 \
    -vqw 1

python3 ${RUNNER_NP} \
    --run_steps 25000 \
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
