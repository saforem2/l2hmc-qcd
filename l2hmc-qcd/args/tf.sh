#!/bin/bash

RUNNER_TF='../gauge_inference.py'
ARGS_TF='./tf.txt'

python3 ${RUNNER_TF} @${ARGS_TF}
