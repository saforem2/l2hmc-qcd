"""
config.py

Implements methods for dealing with config files stored in json format.

Author: Sam Foreman (github: @saforem2)
Date: 04/09/2019
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from attr_dict import AttrDict


def get_config_from_json(json_file):
    """Get the config from a json file.

    Args:
        json_file

    Returns:
        config (AttrDict): Attribute dictionary object.
        config_dict: Config dictionary.
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # Convert the dictionary to an AttrDict
    config = AttrDict(config_dict)

    return config, config_dict


def process_config(json_file):
    """Process config; create directories for summaries and checkpoints."""
    config, _ = get_config_from_json(json_file)
    config.summary_dir = os.path.join(
        '../experiments', config.exp_name, 'summary/'
    )
    config.checkpoint_dir = os.path.join(
        '../experiments', config.exp_name, 'checkpoint/'
    )

    return config
