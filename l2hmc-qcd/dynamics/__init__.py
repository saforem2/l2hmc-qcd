import os
import sys

from collections import namedtuple

import tensorflow as tf

from config import MonteCarloStates, NET_WEIGHTS_HMC, NET_WEIGHTS_L2HMC, State
from utils.attr_dict import AttrDict

modulepath = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(modulepath)

