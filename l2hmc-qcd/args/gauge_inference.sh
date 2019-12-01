runner=/Users/saforem2/ANL/l2hmc-qcd/l2hmc-qcd/gauge_inference.py

# (1, 1, 1, 1, 1, 1)
python3 ${runner} \
    --run_steps 2000 \
    --beta_inference 4. \
    --samples_init 'random' \
    --x_scale_weight 1. --x_translation_weight 1. --x_transformation_weight 1.\
    --v_scale_weight 1.  --v_translation_weight 1. --v_transformation_weight 1.

# (1, 1, 1, 0, 0, 0)
python3 ${runner} \
    --run_steps 2000 \
    --beta_inference 4. \
    --samples_init 'random' \
    --x_scale_weight 1. --x_translation_weight 1. --x_transformation_weight 1.\
    --v_scale_weight 0. --v_translation_weight 0. --v_transformation_weight 0.

# (0, 0, 0, 1, 1, 1)
python3 ${runner} \
    --run_steps 2000 \
    --beta_inference 4. \
    --samples_init 'random' \
    --x_scale_weight 0. --x_translation_weight 0. --x_transformation_weight 0.\
    --v_scale_weight 1. --v_translation_weight 1. --v_transformation_weight 1.

# (1, 0, 0, 0, 0, 0)
python3 ${runner} \
    --run_steps 2000 \
    --beta_inference 4. \
    --samples_init 'random' \
    --x_scale_weight 1. --x_translation_weight 0. --x_transformation_weight 0.\
    --v_scale_weight 0. --v_translation_weight 0. --v_transformation_weight 0.

# (0, 1, 0, 0, 0, 0)
python3 ${runner} \
    --run_steps 2000 \
    --beta_inference 4. \
    --samples_init 'random' \
    --x_scale_weight 0. --x_translation_weight 1. --x_transformation_weight 0.\
    --v_scale_weight 0. --v_translation_weight 0. --v_transformation_weight 0.

# (0, 0, 1, 0, 0, 0)
python3 ${runner} \
    --run_steps 2000 \
    --beta_inference 4. \
    --samples_init 'random' \
    --x_scale_weight 0. --x_translation_weight 0. --x_transformation_weight 1.\
    --v_scale_weight 0. --v_translation_weight 0. --v_transformation_weight 0.

# (0, 0, 0, 1, 0, 0)
python3 ${runner} \
    --run_steps 2000 \
    --beta_inference 4. \
    --samples_init 'random' \
    --x_scale_weight 0. --x_translation_weight 0. --x_transformation_weight 0.\
    --v_scale_weight 1. --v_translation_weight 0. --v_transformation_weight 0.

# (0, 0, 0, 0, 1, 0)
python3 ${runner} \
    --run_steps 2000 \
    --beta_inference 4. \
    --samples_init 'random' \
    --x_scale_weight 0. --x_translation_weight 0. --x_transformation_weight 0.\
    --v_scale_weight 0. --v_translation_weight 1. --v_transformation_weight 0.

# (0, 0, 0, 0, 0, 1)
python3 ${runner} \
    --run_steps 2000 \
    --beta_inference 4. \
    --samples_init 'random' \
    --x_scale_weight 0. --x_translation_weight 0. --x_transformation_weight 0.\
    --v_scale_weight 0. --v_translation_weight 0. --v_transformation_weight 1.

# (0, 0, 0, 0, 0, 0)
python3 ${runner} \
    --run_steps 2000 \
    --beta_inference 4. \
    --samples_init 'random' \
    --x_scale_weight 0. --x_translation_weight 0. --x_transformation_weight 0.\
    --v_scale_weight 0. --v_translation_weight 0. --v_transformation_weight 0.
