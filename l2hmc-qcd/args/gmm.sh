train_script=/Users/saforem2/ANL/l2hmc-qcd/l2hmc-qcd/gmm_main.py
inference_script=/Users/saforem2/ANL/l2hmc-qcd/l2hmc-qcd/gmm_inference.py

which python3

python3 ${train_script} @gmm_args.txt

python3 ${inference_script} @gmm_inference_args.txt
