from collections import namedtuple
from config import NP_FLOAT, TF_FLOAT, TF_INT, Weights
from utils.seed_dict import seeds, xnet_seeds, vnet_seeds

SNAME = 'scale_layer'
SCOEFF = 'coeff_scale'
TNAME = 'translation_layer'
QNAME = 'transformation_layer'
QCOEFF = 'coeff_transformation'

NetworkConfig = namedtuple('NetworkConfig', [
    'type', 'units', 'dropout_prob', 'activation_fn'
])
