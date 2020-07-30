"""
inference_jlse.py

Run inference on trained models by looping over a list of `log_dirs`
"""
import os
import sys

from run import run
from config import BASE_DIR
from utils.attr_dict import AttrDict


def loop_over_log_dirs():
    rld1 = os.path.join(BASE_DIR, 'gauge_logs_eager', '2020_07')
    rld2 = os.path.join(BASE_DIR, 'gauge_logs_eager', '2020_06')
    ld1 = [
        os.path.join(rld1, i) for i in os.listdir(rld1)
        if os.path.isdir(os.path.join(rld1, i))
    ]
    ld2 = [
        os.path.join(rld2, i) for i in os.listdir(rld2)
        if os.path.isdir(os.path.join(rld2, i))
    ]

    log_dirs = ld1 + ld2
    for log_dir in log_dirs:
        args = AttrDict({
            'hmc': False,
            'run_steps': 2000,
            'overwrite': True,
            'log_dir': log_dir,
        })

        try:
            run(args, log_dir, random_start=True)
        except:
            pass
