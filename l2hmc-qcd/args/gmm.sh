TRAIN_SCRIPT='../gmm_main.py'
INFERENCE_SCRIPT='../gmm_inference.py'

which python3

python3 ${train_script} @gmm_args.txt

python3 ${inference_script} @gmm_inference_args.txt
