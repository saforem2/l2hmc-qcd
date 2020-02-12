#!/bin/bash

TRAINER='../main.py'
ARGS='./gauge_args.txt'

python3 ${TRAINER} @${ARGS}
