"""
network.py

Implements network architecture used to train L2HMC algorithm on 2D U(1)
lattice gauge model.

Author: Sam Foreman (github: @saforem2)
Date: 07/22/2019
"""
import numpy as np
import tensorflow as tf
from .network_utils import (custom_dense, batch_norm,
                            add_elements_to_collection)
from .conv_net3d import ConvNet3D
from .conv_net2d import ConvNet2D
from .generic_net import GenericNet
import utils.file_io as io
from tensorflow.keras import backend as K
#  import utils.file_io as io

from globals import GLOBAL_SEED, TF_FLOAT, NP_FLOAT


np.random.seed(GLOBAL_SEED)

if '2.' not in tf.__version__:
    tf.set_random_seed(GLOBAL_SEED)


class FullNet(tf.keras.Model):
    """Complete network used for training L2HMC model."""
    def __init__(self, model_name, **kwargs):
        """
        Initialization method.

        Args:
            model_name: Name of the model.
            kwargs: Keyword arguments used to specify specifics of
                convolutional structure.
        """
        super(FullNet, self).__init__(name=model_name)

        kwargs['name_scope'] = 'x_conv_block'
        network_arch = kwargs.get('network_arch', 'conv3D')

        if network_arch == 'conv2D':
            self.x_conv_net = ConvNet2D('ConvNet2Dx', **kwargs)

            kwargs['name_scope'] = 'v_conv_block'
            self.v_conv_net = ConvNet2D('ConvNet2Dv', **kwargs)

        elif network_arch == 'conv3D':
            self.x_conv_net = ConvNet3D('ConvNet3Dx', **kwargs)

            kwargs['name_scope'] = 'v_conv_block'
            self.v_conv_net = ConvNet3D('ConvNet3Dv', **kwargs)

        else:
            self.x_conv_net = self.v_conv_net = None

        kwargs['name_scope'] = 'generic_block'
        self.generic_block = GenericNet("GenericNet", **kwargs)

    def call(self, inputs, train_phase):
        v, x, t = inputs

        if self.x_conv_net is not None:
            v = self.v_conv_net(v, train_phase)
            x = self.x_conv_net(x, train_phase)

        scale, translation, transformation = self.generic_block([v, x, t],
                                                                train_phase)

        return scale, translation, transformation
