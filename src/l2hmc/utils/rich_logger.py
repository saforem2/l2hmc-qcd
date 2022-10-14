"""
utils/rich_logger.py
"""
from __future__ import absolute_import, annotations, division, print_function


LOGGER_FIELDS = {
    'era': {},
    'epoch': {},
    'tstep': {
        # 'format': '{:^1g}',
        # 'name': r'tstep',
    },
    'estep': {
        # 'format': '{:^1g}',
        # 'name': 'estep',
    },
    'hstep': {
        # 'format': '{:^1g}',
        # 'name': 'hstep',
    },
    'dt': {
        'format': '{:^3.4f}',
    },
    'beta': {'format': '{:^3.2f}'},
    'loss': {
        'format': '{:^3.4f}',
        'goal': 'lower_is_better',
    },
    'dQsin': {
        'format': '{:^3.4f}',
        'goal': 'higher_is_better',
    },
    'dQint': {
        'format': '{:^3.4f}',
        'goal': 'higher_is_better',
    },
    'energy': {'format': '{:^3.4f}'},
    'logprob': {'format': '{:^3.4f}'},
    'logdet': {'format': '{:^3.4f}'},
    'sldf': {'format': '{:^3.4f}'},
    'sldb': {'format': '{:^3.4f}'},
    'sld': {'format': '{:^3.4f}'},
    'xeps': {'format': '{:^3.4f}'},
    'veps': {'format': '{:^3.4f}'},
    'acc': {'format': '{:^3.4f}'},
    'acc_mask': {'format': '{:^3.4f}'},
    'sumlogdet': {'format': '{:^3.4f}'},
    'plaqs': {'format': '{:^3.4f}'},
    'intQ': {'format': '{:^3.4f}'},
    'sinQ': {'format': '{:^3.4f}'},
    'lr': {'format': '{:^3.4f}'},
}
